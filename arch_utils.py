"""
Architecture Abstraction Utilities

다양한 transformer 아키텍처(BERT, XLM-R, ModernBERT, GTE 등)에 대해
레이어 접근, 토크나이저 pruning, 임베딩 pruning을 범용적으로 처리한다.
"""

import copy
import json
import os

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig


# ── Layer Access ──────────────────────────────────────────────

def get_layers(model, layer_accessor):
    """dot-separated 경로로 모델의 레이어 리스트에 접근한다."""
    obj = model
    for attr in layer_accessor.split("."):
        obj = getattr(obj, attr)
    return obj


def set_layers(model, layer_accessor, new_layers):
    """dot-separated 경로의 레이어 리스트를 교체한다."""
    parts = layer_accessor.split(".")
    obj = model
    for attr in parts[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, parts[-1], new_layers)


def discover_layer_accessor(model):
    """모델의 레이어 리스트 경로를 자동 감지한다."""
    candidates = [
        "encoder.layer",       # BERT, XLM-R, GTE
        "encoder.layers",      # some models
        "layers",              # ModernBERT
        "transformer.layer",   # some models
        "transformer.layers",  # some models
    ]
    for path in candidates:
        try:
            layers = get_layers(model, path)
            if isinstance(layers, nn.ModuleList) and len(layers) > 0:
                return path
        except AttributeError:
            continue
    raise ValueError(f"Could not detect layer accessor for {type(model).__name__}")


# ── Layer Pruning (Architecture-Agnostic) ─────────────────────

def prune_layers(model, layer_indices, layer_accessor=None):
    """모델에서 지정된 레이어만 유지하고 나머지를 제거한다.

    Deep copy 없이 in-place로 동작한다. 호출 전에 deepcopy를 사용하라.
    """
    if layer_accessor is None:
        layer_accessor = discover_layer_accessor(model)

    layers = get_layers(model, layer_accessor)
    kept = nn.ModuleList([layers[i] for i in layer_indices])
    set_layers(model, layer_accessor, kept)
    model.config.num_hidden_layers = len(layer_indices)

    return model


def create_pruned_student(teacher_model_id, layer_indices, layer_accessor=None,
                          trust_remote_code=False):
    """Teacher 모델을 로드하고 레이어를 pruning하여 student를 생성한다.

    Returns:
        (student_model, tokenizer) tuple
    """
    config = AutoConfig.from_pretrained(teacher_model_id,
                                         trust_remote_code=trust_remote_code)
    model = AutoModel.from_pretrained(teacher_model_id,
                                       trust_remote_code=trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_id,
                                               trust_remote_code=trust_remote_code)

    if layer_accessor is None:
        layer_accessor = discover_layer_accessor(model)

    student = copy.deepcopy(model)
    student = prune_layers(student, layer_indices, layer_accessor)

    return student, tokenizer


# ── Hidden Dimension Reduction ────────────────────────────────

def reduce_hidden_dim(model, new_hidden_dim, new_intermediate_size=None,
                      trust_remote_code=False):
    """Hidden dimension을 축소한다.

    새 config으로 모델을 생성한 뒤, 기존 가중치를 슬라이싱하여 복사한다.
    Attention head 수는 new_hidden_dim에 맞게 자동 조정된다.

    Args:
        model: HuggingFace transformer 모델 (layer pruning 완료 상태)
        new_hidden_dim: 목표 hidden dimension
        new_intermediate_size: 목표 FFN intermediate size. None이면 비례 축소.
        trust_remote_code: custom code 모델 지원 여부

    Returns:
        축소된 모델
    """
    old_hidden = model.config.hidden_size
    if new_hidden_dim >= old_hidden:
        return model

    old_inter = getattr(model.config, 'intermediate_size', old_hidden * 4)
    if new_intermediate_size is None:
        ratio = new_hidden_dim / old_hidden
        new_intermediate_size = max(64, (int(old_inter * ratio) // 64) * 64)

    # 새 config 생성
    new_config = copy.deepcopy(model.config)
    new_config.hidden_size = new_hidden_dim
    new_config.intermediate_size = new_intermediate_size

    # Attention head 수 비례 조정
    # 제약: (1) hidden_dim % n_heads == 0
    #       (2) GQA인 경우 n_heads % n_kv_heads == 0
    ratio = new_hidden_dim / old_hidden
    old_n_kv = getattr(new_config, 'num_key_value_heads', None)

    if hasattr(new_config, 'num_attention_heads'):
        n_heads = getattr(new_config, 'num_attention_heads')
        if n_heads is not None:
            n_heads = max(1, int(n_heads * ratio))
            while new_hidden_dim % n_heads != 0 and n_heads > 1:
                n_heads -= 1
            new_config.num_attention_heads = n_heads

    if old_n_kv is not None and hasattr(new_config, 'num_key_value_heads'):
        n_heads = getattr(new_config, 'num_attention_heads', n_heads)
        n_kv = max(1, int(old_n_kv * ratio))
        # n_kv는 n_heads의 약수여야 함
        while n_kv > 1 and (n_heads % n_kv != 0 or new_hidden_dim % n_kv != 0):
            n_kv -= 1
        new_config.num_key_value_heads = n_kv

    # head_dim이 별도 config인 모델 (Qwen3 등): hidden_dim/num_heads로 재계산
    if hasattr(new_config, 'head_dim') and new_config.head_dim is not None:
        new_heads = getattr(new_config, 'num_attention_heads', 1)
        new_config.head_dim = new_hidden_dim // new_heads

    # 새 모델 생성 (랜덤 초기화)
    new_model = AutoModel.from_config(new_config, trust_remote_code=trust_remote_code)

    # 기존 가중치를 슬라이싱하여 복사
    old_sd = model.state_dict()
    new_sd = new_model.state_dict()

    copied, skipped = 0, 0
    for key in new_sd:
        if key not in old_sd:
            skipped += 1
            continue
        old_t = old_sd[key]
        new_t = new_sd[key]
        if old_t.shape == new_t.shape:
            new_sd[key] = old_t.clone()
            copied += 1
        else:
            # 각 차원을 new shape에 맞게 슬라이싱
            slices = tuple(
                slice(0, min(s_new, s_old))
                for s_new, s_old in zip(new_t.shape, old_t.shape)
            )
            sliced = old_t[slices]
            if sliced.shape == new_t.shape:
                new_sd[key] = sliced.clone()
                copied += 1
            else:
                # 크기가 정확히 맞지 않으면 가능한 부분만 복사
                target_slices = tuple(slice(0, s) for s in sliced.shape)
                new_sd[key][target_slices] = sliced.clone()
                copied += 1

    new_model.load_state_dict(new_sd)
    print(f"  Hidden dim reduction: {old_hidden} → {new_hidden_dim} "
          f"(intermediate: {old_inter} → {new_intermediate_size}, "
          f"weights copied: {copied}, new: {skipped})")

    # Special token ID 보존
    for attr in ["pad_token_id", "bos_token_id", "eos_token_id",
                 "cls_token_id", "sep_token_id", "unk_token_id", "mask_token_id"]:
        old_val = getattr(model.config, attr, None)
        if old_val is not None:
            setattr(new_model.config, attr, old_val)

    return new_model


# ── Tokenizer Type Detection ─────────────────────────────────

def detect_tokenizer_type(tokenizer):
    """토크나이저의 타입(Unigram/BPE/WordPiece)을 감지한다."""
    tok_json = json.loads(tokenizer.backend_tokenizer.to_str())
    return tok_json["model"]["type"]


# ── Vocab Pruning (Tokenizer-Agnostic) ────────────────────────

def prune_tokenizer_and_embeddings(model, tokenizer, keep_ids, save_dir):
    """토크나이저와 모델 임베딩을 동시에 pruning한다.

    Unigram, BPE, WordPiece 토크나이저를 모두 지원한다.

    Args:
        model: HuggingFace transformer 모델
        tokenizer: HuggingFace tokenizer
        keep_ids: 유지할 토큰 ID 리스트 (정렬된 상태)
        save_dir: pruned tokenizer를 저장할 디렉토리

    Returns:
        pruned model (임베딩이 축소된 상태)
    """
    os.makedirs(save_dir, exist_ok=True)

    # old → new ID 매핑
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(keep_ids)}

    # 토크나이저 pruning
    tok_json = json.loads(tokenizer.backend_tokenizer.to_str())
    model_type = tok_json["model"]["type"]

    if model_type == "Unigram":
        tok_json = _prune_unigram(tok_json, keep_ids, old_to_new)
    elif model_type == "BPE":
        tok_json = _prune_bpe(tok_json, keep_ids, old_to_new)
    elif model_type == "WordPiece":
        tok_json = _prune_wordpiece(tok_json, keep_ids, old_to_new)
    else:
        print(f"  WARNING: Unknown tokenizer type '{model_type}', skipping tokenizer pruning")
        tokenizer.save_pretrained(save_dir)
        model = _prune_embeddings(model, keep_ids)
        return model

    # added_tokens 재매핑
    if "added_tokens" in tok_json:
        new_added = []
        for at in tok_json["added_tokens"]:
            old_id = at["id"]
            if old_id in old_to_new:
                at["id"] = old_to_new[old_id]
                new_added.append(at)
        tok_json["added_tokens"] = new_added

    # post_processor의 special_tokens IDs도 재매핑
    pp = tok_json.get("post_processor")
    if pp and "special_tokens" in pp:
        for token_name, token_info in pp["special_tokens"].items():
            if "ids" in token_info:
                token_info["ids"] = [
                    old_to_new[oid] for oid in token_info["ids"]
                    if oid in old_to_new
                ]

    # 저장: 먼저 원본 tokenizer config 저장, 그다음 pruned 파일 덮어쓰기
    tokenizer.save_pretrained(save_dir)

    # pruned tokenizer.json 덮어쓰기
    tok_json_path = os.path.join(save_dir, "tokenizer.json")
    with open(tok_json_path, "w", encoding="utf-8") as f:
        json.dump(tok_json, f, ensure_ascii=False)

    # added_tokens.json 갱신
    added_tokens_path = os.path.join(save_dir, "added_tokens.json")
    if os.path.exists(added_tokens_path):
        with open(added_tokens_path, "r", encoding="utf-8") as f:
            added_tokens = json.load(f)
        new_added_tokens = {}
        for token_str, old_id in added_tokens.items():
            if old_id in old_to_new:
                new_added_tokens[token_str] = old_to_new[old_id]
        with open(added_tokens_path, "w", encoding="utf-8") as f:
            json.dump(new_added_tokens, f, ensure_ascii=False)

    # tokenizer_config.json의 added_tokens_decoder도 갱신
    tok_config_path = os.path.join(save_dir, "tokenizer_config.json")
    if os.path.exists(tok_config_path):
        with open(tok_config_path, "r", encoding="utf-8") as f:
            tok_config = json.load(f)
        if "added_tokens_decoder" in tok_config:
            new_decoder = {}
            for old_id_str, token_info in tok_config["added_tokens_decoder"].items():
                old_id = int(old_id_str)
                if old_id in old_to_new:
                    new_decoder[str(old_to_new[old_id])] = token_info
            tok_config["added_tokens_decoder"] = new_decoder
        with open(tok_config_path, "w", encoding="utf-8") as f:
            json.dump(tok_config, f, ensure_ascii=False, indent=2)

    # 임베딩 pruning
    model = _prune_embeddings(model, keep_ids)

    return model


def _prune_unigram(tok_json, keep_ids, old_to_new):
    """Unigram (SentencePiece) 토크나이저 pruning."""
    old_vocab = tok_json["model"]["vocab"]  # [[piece, score], ...]
    new_vocab = []
    for old_id in keep_ids:
        if old_id < len(old_vocab):
            new_vocab.append(old_vocab[old_id])
    tok_json["model"]["vocab"] = new_vocab

    # unk_id 재매핑
    old_unk_id = tok_json["model"].get("unk_id")
    if old_unk_id is not None and old_unk_id in old_to_new:
        tok_json["model"]["unk_id"] = old_to_new[old_unk_id]

    return tok_json


def _prune_bpe(tok_json, keep_ids, old_to_new):
    """BPE 토크나이저 pruning."""
    old_vocab = tok_json["model"]["vocab"]  # {"token": id, ...}

    # keep_ids에 해당하는 토큰만 유지, ID 재매핑
    keep_ids_set = set(keep_ids)
    new_vocab = {}
    for token, old_id in old_vocab.items():
        if old_id in keep_ids_set:
            new_vocab[token] = old_to_new[old_id]
    tok_json["model"]["vocab"] = new_vocab

    # 유지된 토큰 집합
    kept_tokens = set(new_vocab.keys())

    # merges 필터링: 입력 토큰 2개와 결과 토큰이 모두 vocab에 있는 merge만 유지
    if "merges" in tok_json["model"]:
        new_merges = []
        for merge in tok_json["model"]["merges"]:
            if isinstance(merge, list):
                parts = merge
            else:
                parts = merge.split(" ")
            if len(parts) == 2:
                merged_token = parts[0] + parts[1]
                if (parts[0] in kept_tokens
                    and parts[1] in kept_tokens
                    and merged_token in kept_tokens):
                    new_merges.append(merge)
        tok_json["model"]["merges"] = new_merges

    return tok_json


def _prune_wordpiece(tok_json, keep_ids, old_to_new):
    """WordPiece 토크나이저 pruning."""
    old_vocab = tok_json["model"]["vocab"]  # {"token": id, ...}

    keep_ids_set = set(keep_ids)
    new_vocab = {}
    for token, old_id in old_vocab.items():
        if old_id in keep_ids_set:
            new_vocab[token] = old_to_new[old_id]
    tok_json["model"]["vocab"] = new_vocab

    return tok_json


def _prune_embeddings(model, keep_ids):
    """모델의 word embedding과 config의 special token ID를 pruning한다."""
    old_emb = model.get_input_embeddings()
    old_weight = old_emb.weight.data

    new_vocab_size = len(keep_ids)
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(keep_ids)}

    # config의 special token ID 재매핑
    for attr in ["pad_token_id", "bos_token_id", "eos_token_id",
                 "cls_token_id", "sep_token_id", "unk_token_id",
                 "mask_token_id", "decoder_start_token_id"]:
        old_id = getattr(model.config, attr, None)
        if old_id is not None:
            if old_id in old_to_new:
                setattr(model.config, attr, old_to_new[old_id])
            else:
                setattr(model.config, attr, None)

    # padding_idx 재매핑
    padding_idx = getattr(old_emb, 'padding_idx', None)
    if padding_idx is not None:
        padding_idx = old_to_new.get(padding_idx, None)
    new_emb = nn.Embedding(new_vocab_size, old_weight.shape[1],
                            padding_idx=padding_idx)

    for new_id, old_id in enumerate(keep_ids):
        if old_id < old_weight.shape[0]:
            new_emb.weight.data[new_id] = old_weight[old_id]

    model.set_input_embeddings(new_emb)
    model.config.vocab_size = new_vocab_size

    return model


# ── Corpus-based Token Collection ─────────────────────────────

def collect_corpus_tokens(tokenizer, texts=None, max_vocab=None,
                          vocab_keep_ratio=None):
    """코퍼스에서 실제 사용되는 토큰을 수집한다.

    Args:
        tokenizer: HuggingFace tokenizer
        texts: 코퍼스 텍스트 리스트. None이면 기본 다국어 샘플 사용.
        max_vocab: 최대 vocab 크기. None이면 코퍼스에 나타난 모든 토큰 유지.
        vocab_keep_ratio: 빈도 누적 기준 상위 비율만 유지 (0.0~1.0).
                          예: 0.99 → 빈도순 정렬 후 누적합이 전체의 99%가 될 때까지의
                          토큰만 유지. 나머지 희소 토큰 제거.
                          max_vocab과 동시 사용 시 vocab_keep_ratio가 먼저 적용됨.

    Returns:
        유지할 토큰 ID 리스트 (정렬됨)
    """
    from collections import Counter

    if texts is None:
        texts = _get_default_multilingual_samples()

    # 빈도 계산
    freq = Counter()
    batch_size = 1000
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(batch, add_special_tokens=True, truncation=True,
                            max_length=128)
        for ids in encoded["input_ids"]:
            freq.update(ids)

    # 특수 토큰은 무조건 유지
    keep_ids = set(tokenizer.all_special_ids)

    # 기본 문자/구두점 보장
    basic_chars = list("0123456789.,!?;:'\"-()[]{}/@#$%^&*+=<>~_ \t\n")
    for ch in basic_chars:
        ids = tokenizer.encode(ch, add_special_tokens=False)
        keep_ids.update(ids)

    # BPE 토크나이저: byte-level fallback 토큰을 반드시 유지
    # GPT-2 스타일 BPE는 첫 256개 vocab entry가 byte 토큰
    tok_type = detect_tokenizer_type(tokenizer)
    if tok_type == "BPE":
        tok_json = json.loads(tokenizer.backend_tokenizer.to_str())
        vocab = tok_json["model"]["vocab"]
        # byte-level 토큰은 보통 단일 문자 (길이 1-2) 중 ID가 작은 것들
        # 안전하게 첫 256개 + 모든 단일문자 토큰 유지
        for token, tid in vocab.items():
            if tid < 256 or len(token) <= 1:
                keep_ids.add(tid)

    if vocab_keep_ratio is not None:
        # 빈도 누적 기준: 빈도순 정렬 후 누적합이 전체의 ratio가 될 때까지 유지
        total_freq = sum(freq.values())
        target_freq = total_freq * vocab_keep_ratio
        corpus_tokens = sorted(freq.keys(), key=lambda t: freq[t], reverse=True)
        cumsum = 0
        for tid in corpus_tokens:
            keep_ids.add(tid)
            cumsum += freq[tid]
            if cumsum >= target_freq:
                break
        n_kept_corpus = len([t for t in corpus_tokens if t in keep_ids])
        n_removed = len(corpus_tokens) - n_kept_corpus
        actual_coverage = sum(freq[t] for t in keep_ids if t in freq) / max(total_freq, 1) * 100
        vocab_size = getattr(tokenizer, 'vocab_size', len(tokenizer.get_vocab()))
        print(f"  Vocab: {len(keep_ids):,} / {vocab_size:,} tokens kept "
              f"(cumulative freq ratio={vocab_keep_ratio}, removed {n_removed} rare tokens, "
              f"coverage={actual_coverage:.1f}%)")
    elif max_vocab is not None:
        remaining = max_vocab - len(keep_ids)
        if remaining > 0:
            for tid, _ in freq.most_common():
                if tid not in keep_ids:
                    keep_ids.add(tid)
                    if len(keep_ids) >= max_vocab:
                        break
        coverage = sum(freq[t] for t in keep_ids if t in freq) / max(sum(freq.values()), 1) * 100
        print(f"  Vocab: {len(keep_ids):,} tokens (target {max_vocab:,}, coverage={coverage:.1f}%)")
    else:
        keep_ids.update(freq.keys())
        vocab_size = getattr(tokenizer, 'vocab_size', len(tokenizer.get_vocab()))
        print(f"  Vocab: {len(keep_ids):,} / {vocab_size:,} tokens kept")

    return sorted(keep_ids)


def _get_default_multilingual_samples():
    """최소한의 다국어 샘플."""
    return [
        "예약 좀 해줘", "지난번 주문 뭐였지?", "안녕하세요 반갑습니다",
        "Book a table for me", "What did I order last time?", "Hello how are you",
        "予約をお願いします", "前回の注文は何でしたか", "こんにちは元気ですか",
        "帮我预约一下", "上次我点了什么", "你好你好吗",
        "Reserva una mesa", "Qué pedí la última vez", "Hola cómo estás",
        "Réservez une table", "Qu'est-ce que j'ai commandé", "Bonjour comment allez-vous",
        "Reservieren Sie einen Tisch", "Was habe ich bestellt", "Hallo wie geht es",
    ] * 10


# ── SentenceTransformer Wrapping ──────────────────────────────

def save_as_sentence_transformer(model, tokenizer, save_path):
    """HF 모델을 SentenceTransformer 포맷으로 저장한다.

    Custom code 모델(GTE 등)은 auto_map과 custom .py 파일을 보존한다.
    표준 모델(BERT, ModernBERT 등)은 auto_map을 제거하여 깔끔하게 저장한다.
    """
    from sentence_transformers import SentenceTransformer, models as st_models
    import shutil
    import glob as _glob

    # 임시 HF 모델 저장
    hf_tmp = os.path.join(save_path, "_hf_tmp")
    os.makedirs(hf_tmp, exist_ok=True)
    model.save_pretrained(hf_tmp)
    tokenizer.save_pretrained(hf_tmp)

    # custom code 모델 여부 확인
    config_path = os.path.join(hf_tmp, "config.json")
    is_custom = False
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        is_custom = "auto_map" in config

    if is_custom:
        # Custom code 모델: HF 캐시에서 custom .py 파일 복사
        _copy_custom_code_files(model, hf_tmp)
    else:
        # 표준 모델: _name_or_path 정리
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            config.pop("_name_or_path", None)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

    # SentenceTransformer 구성
    # Windows에서 signal.SIGALRM 미지원 문제 우회
    os.environ["HF_HUB_TRUST_REMOTE_CODE"] = "1"
    word_model = st_models.Transformer(
        hf_tmp,
        config_args={"trust_remote_code": True},
        model_args={"trust_remote_code": True},
        tokenizer_args={"trust_remote_code": True},
    )
    pool_model = st_models.Pooling(
        word_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
    )
    st_model = SentenceTransformer(modules=[word_model, pool_model])
    st_model.save(save_path)

    # 임시 디렉토리 정리
    shutil.rmtree(hf_tmp, ignore_errors=True)

    return st_model


def _copy_custom_code_files(model, target_dir):
    """모델의 custom code 파일(.py)을 target 디렉토리로 복사한다."""
    import shutil
    import glob as _glob

    # model._name_or_path에서 원본 경로 추출
    source_path = getattr(model.config, '_name_or_path', None)
    if not source_path:
        return

    # HuggingFace 캐시에서 .py 파일 찾기
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    # model ID를 캐시 디렉토리 형식으로 변환
    model_cache_name = "models--" + source_path.replace("/", "--")
    model_cache_dir = os.path.join(cache_dir, model_cache_name)

    if os.path.exists(model_cache_dir):
        # snapshots 디렉토리에서 최신 스냅샷 찾기
        snapshots_dir = os.path.join(model_cache_dir, "snapshots")
        if os.path.exists(snapshots_dir):
            snapshots = os.listdir(snapshots_dir)
            if snapshots:
                latest = os.path.join(snapshots_dir, snapshots[-1])
                for py_file in _glob.glob(os.path.join(latest, "*.py")):
                    fname = os.path.basename(py_file)
                    dest = os.path.join(target_dir, fname)
                    if not os.path.exists(dest):
                        shutil.copy2(py_file, dest)
                        print(f"  Copied custom code: {fname}")


# ── Architecture Visualization (for Model Cards) ─────────────

def generate_architecture_diagram(teacher_config, layer_indices, vocab_size,
                                   pruned_vocab_size=None):
    """모델 아키텍처 pruning 과정을 ASCII 다이어그램으로 시각화한다."""
    t = teacher_config
    n_layers = t["num_layers"]
    n_kept = len(layer_indices)
    hidden = t["hidden_dim"]
    orig_vocab = t["vocab_size"]
    p_vocab = pruned_vocab_size or orig_vocab

    # Teacher architecture
    lines = []
    lines.append("```")
    lines.append(f"{'='*62}")
    lines.append(f"  TEACHER: {t['short_name']}  →  STUDENT: {n_kept}L / {p_vocab:,} vocab")
    lines.append(f"{'='*62}")
    lines.append("")

    # Side by side: Teacher | Student
    lines.append(f"  {'TEACHER':^27}    {'STUDENT':^27}")
    lines.append(f"  {'─'*27}    {'─'*27}")
    lines.append("")

    # Input
    lines.append(f"  ┌─────────────────────────┐    ┌─────────────────────────┐")
    lines.append(f"  │   Input Tokens          │    │   Input Tokens          │")
    lines.append(f"  └────────────┬────────────┘    └────────────┬────────────┘")
    lines.append(f"               │                              │")

    # Embeddings
    lines.append(f"  ┌────────────┴────────────┐    ┌────────────┴────────────┐")
    lines.append(f"  │  Embeddings             │    │  Embeddings (pruned)    │")
    lines.append(f"  │  vocab: {orig_vocab:>7,}         │    │  vocab: {p_vocab:>7,}         │")
    lines.append(f"  │  dim: {hidden:>4}              │    │  dim: {hidden:>4}              │")
    lines.append(f"  └────────────┬────────────┘    └────────────┬────────────┘")
    lines.append(f"               │                              │")

    # Layers
    kept_set = set(layer_indices)
    for i in range(n_layers):
        is_kept = i in kept_set
        new_idx = layer_indices.index(i) if is_kept else None

        teacher_layer = f"  │  Layer {i:>2}               │"
        if is_kept:
            student_layer = f"  │  Layer {new_idx:>2} ← L{i:<2}        │"
            arrow = " ──►"
        else:
            student_layer = f"  │{'':25}│"
            arrow = "  ╳ "

        if i == 0:
            lines.append(f"  ┌─────────────────────────┐    ┌─────────────────────────┐")
        lines.append(f"{teacher_layer}{arrow}{student_layer}")

        # 다음 레이어 또는 닫기 전 구분선
        if i < n_layers - 1:
            if i + 1 in kept_set or i in kept_set:
                lines.append(f"  ├─────────────────────────┤    ├─────────────────────────┤")
            else:
                lines.append(f"  ├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤    │{'':25}│")
        else:
            # 마지막 kept 레이어 이후 student 박스 닫기
            lines.append(f"  └────────────┬────────────┘    └────────────┬────────────┘")

    lines.append(f"               │                              │")
    lines.append(f"  ┌────────────┴────────────┐    ┌────────────┴────────────┐")
    lines.append(f"  │  Mean Pooling           │    │  Mean Pooling           │")
    lines.append(f"  │  → {hidden}d embedding       │    │  → {hidden}d embedding       │")
    lines.append(f"  └─────────────────────────┘    └─────────────────────────┘")
    lines.append("")

    # Size comparison
    from config import estimate_size
    teacher_size = estimate_size(
        list(range(n_layers)), hidden, orig_vocab, t["intermediate_size"]
    )
    student_size = estimate_size(
        layer_indices, hidden, p_vocab, t["intermediate_size"]
    )
    reduction = (1 - student_size["fp32_mb"] / teacher_size["fp32_mb"]) * 100

    lines.append(f"  Size: {teacher_size['fp32_mb']}MB (FP32)           →  {student_size['fp32_mb']}MB (FP32)")
    lines.append(f"  Params: {teacher_size['total_params']:,}        →  {student_size['total_params']:,}")
    lines.append(f"  Reduction: {reduction:.1f}%")
    lines.append(f"{'='*62}")
    lines.append("```")

    return "\n".join(lines)
