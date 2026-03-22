"""
Teacher 모델에서 layer pruning + corpus-based vocab pruning으로 경량 student 모델을 생성한다.

전략:
  - Layer pruning: teacher의 12개 레이어 중 선택 (2~6개)
  - Vocab pruning: 18개 타겟 언어 코퍼스에서 실제 사용되는 토큰만 유지 (250K → ~40-50K)
  - FP32 유지 (양자화 없음) → safetensors 50~100MB

Usage:
    python create_students.py                  # 모든 실험 생성
    python create_students.py --only L6_uniform L4_uniform  # 특정 실험만
    python create_students.py --no-prune       # vocab pruning 없이 (원본 vocab)
"""

import argparse
import copy
import json
import os
import shutil

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, models
from transformers import AutoConfig, AutoModel, AutoTokenizer
from tokenizers import Tokenizer as HFTokenizer
from datasets import load_dataset

from config import EXPERIMENTS, STUDENTS_DIR, TEACHER_MODEL, TARGET_LANGUAGES


# ── Corpus-based Vocab Pruning ──────────────────────────────────

def collect_corpus_tokens(tokenizer, max_per_lang=5000, cache_dir="data/distill_corpus",
                          max_vocab=None):
    """18개 타겟 언어의 MASSIVE 코퍼스를 토큰화하여 실제 사용 토큰만 수집한다.

    Args:
        max_vocab: 최대 vocab 크기. None이면 모든 코퍼스 토큰 유지.
                   숫자면 빈도 기반으로 상위 N개만 유지.
    """
    from collections import Counter

    # 캐시된 코퍼스 로드
    cache_file = os.path.join(cache_dir, f"texts_{max_per_lang}.txt")

    if os.path.exists(cache_file):
        print(f"  Loading cached corpus: {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        print("  Loading MASSIVE dataset for vocab collection...")
        os.makedirs(cache_dir, exist_ok=True)
        texts = []
        MASSIVE_LANG_MAP = {
            "ko": "ko", "en": "en", "ja": "ja", "zh": "zh-CN",
            "es": "es", "fr": "fr", "de": "de", "pt": "pt",
            "it": "it", "ru": "ru", "ar": "ar", "hi": "hi",
            "th": "th", "vi": "vi", "id": "id", "tr": "tr",
            "nl": "nl", "pl": "pl",
        }
        for lang, subset in MASSIVE_LANG_MAP.items():
            try:
                ds = load_dataset("mteb/amazon_massive_intent", subset, split="train")
                lang_texts = [row["text"] for row in ds if row.get("text")][:max_per_lang]
                texts.extend(lang_texts)
                print(f"    {lang}: {len(lang_texts)} sentences")
            except Exception as e:
                print(f"    {lang}: failed - {e}")

        with open(cache_file, "w", encoding="utf-8") as f:
            for t in texts:
                f.write(t.strip() + "\n")

    print(f"  Corpus: {len(texts):,} sentences")

    # 코퍼스 전체를 토큰화하여 빈도 계산
    freq = Counter()
    batch_size = 1000
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(batch, add_special_tokens=True, truncation=True, max_length=128)
        for ids in encoded["input_ids"]:
            freq.update(ids)

    # 특수 토큰은 무조건 유지
    keep_ids = set(tokenizer.all_special_ids)

    # 기본 문자/구두점 토큰도 보장
    basic_chars = list("0123456789.,!?;:'\"-()[]{}/@#$%^&*+=<>~_ \t\n")
    for ch in basic_chars:
        ids = tokenizer.encode(ch, add_special_tokens=False)
        keep_ids.update(ids)

    if max_vocab is not None:
        # 빈도 기반: 상위 max_vocab개만 유지
        remaining = max_vocab - len(keep_ids)
        if remaining > 0:
            top_tokens = freq.most_common()
            for tid, count in top_tokens:
                if tid not in keep_ids:
                    keep_ids.add(tid)
                    if len(keep_ids) >= max_vocab:
                        break
        coverage = sum(freq[t] for t in keep_ids if t in freq) / sum(freq.values()) * 100
        print(f"  Vocab: {len(keep_ids):,} tokens (top-{max_vocab}, coverage={coverage:.1f}%)")
    else:
        # 전체 코퍼스 토큰 유지
        keep_ids.update(freq.keys())
        print(f"  Vocab: {len(keep_ids):,} / {tokenizer.vocab_size:,} tokens kept")

    return sorted(keep_ids)


def prune_tokenizer_and_model(model, tokenizer, keep_ids, save_dir):
    """모델의 embedding과 tokenizer.json을 동시에 pruning한다.

    Unigram 토크나이저의 vocab 리스트에서 사용하지 않는 토큰을 제거하고,
    모델의 word embedding도 같은 순서로 축소한다.
    """
    # ── 1. 토크나이저 vocab 재구성 ──
    # HuggingFace fast tokenizer의 내부 JSON 구조
    tok_json = json.loads(tokenizer.backend_tokenizer.to_str())
    old_vocab = tok_json["model"]["vocab"]  # [[piece, score], ...]

    # keep_ids에 해당하는 vocab만 추출 (순서 유지)
    new_vocab = []
    old_to_new = {}
    for new_id, old_id in enumerate(keep_ids):
        if old_id < len(old_vocab):
            new_vocab.append(old_vocab[old_id])
            old_to_new[old_id] = new_id

    tok_json["model"]["vocab"] = new_vocab

    # unk_id 재매핑
    old_unk_id = tok_json["model"].get("unk_id", 3)
    if old_unk_id in old_to_new:
        tok_json["model"]["unk_id"] = old_to_new[old_unk_id]

    # added_tokens 재매핑
    if "added_tokens" in tok_json:
        new_added = []
        for at in tok_json["added_tokens"]:
            old_id = at["id"]
            if old_id in old_to_new:
                at["id"] = old_to_new[old_id]
                new_added.append(at)
        tok_json["added_tokens"] = new_added

    # 새 tokenizer.json 저장
    tok_json_path = os.path.join(save_dir, "tokenizer.json")
    with open(tok_json_path, "w", encoding="utf-8") as f:
        json.dump(tok_json, f, ensure_ascii=False)

    # tokenizer_config.json도 복사 (vocab_size는 나중에 config.json에서 관리)
    tokenizer.save_pretrained(save_dir)
    # 방금 저장한 pruned tokenizer.json으로 덮어쓰기
    with open(tok_json_path, "w", encoding="utf-8") as f:
        json.dump(tok_json, f, ensure_ascii=False)

    # ── 2. 모델 embedding pruning ──
    old_weight = model.embeddings.word_embeddings.weight.data
    new_vocab_size = len(keep_ids)
    new_emb = nn.Embedding(new_vocab_size, old_weight.shape[1])

    for new_id, old_id in enumerate(keep_ids):
        new_emb.weight.data[new_id] = old_weight[old_id]

    model.embeddings.word_embeddings = new_emb
    model.config.vocab_size = new_vocab_size

    return model


# ── Student 생성 ────────────────────────────────────────────────

def create_student(teacher_st, layer_indices, save_path, keep_ids=None):
    """Teacher의 특정 레이어만 복사하여 student 모델을 생성한다."""
    teacher_hf = teacher_st[0].auto_model
    tokenizer = teacher_st[0].tokenizer

    # Student config
    student_config = copy.deepcopy(teacher_hf.config)
    student_config.num_hidden_layers = len(layer_indices)

    # Student 모델 생성
    student_hf = type(teacher_hf)(student_config)

    # 임베딩 복사
    student_hf.embeddings.load_state_dict(teacher_hf.embeddings.state_dict())

    # 선택된 레이어 복사
    for new_idx, old_idx in enumerate(layer_indices):
        student_hf.encoder.layer[new_idx].load_state_dict(
            teacher_hf.encoder.layer[old_idx].state_dict()
        )

    # HF 모델로 임시 저장
    hf_path = os.path.join(save_path, "_hf_model")
    os.makedirs(hf_path, exist_ok=True)

    if keep_ids is not None:
        # Vocab pruning: tokenizer.json + embedding 동시 pruning
        orig_vocab = student_hf.config.vocab_size
        student_hf = prune_tokenizer_and_model(student_hf, tokenizer, keep_ids, hf_path)
        print(f"  Vocab pruned: {orig_vocab:,} → {student_hf.config.vocab_size:,}")
        student_hf.save_pretrained(hf_path)
    else:
        student_hf.save_pretrained(hf_path)
        tokenizer.save_pretrained(hf_path)

    # SentenceTransformer로 재구성 (mean pooling)
    word_model = models.Transformer(hf_path)
    pool_model = models.Pooling(
        word_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
    )
    student_st = SentenceTransformer(modules=[word_model, pool_model])
    student_st.save(save_path)

    # 임시 HF 디렉토리 정리
    shutil.rmtree(hf_path, ignore_errors=True)

    return student_st


def estimate_size(layer_indices, hidden_dim=384, vocab_size=40000):
    """FP32 기준 예상 모델 크기 (MB) 추정."""
    embed_params = vocab_size * hidden_dim
    embed_params += hidden_dim  # LayerNorm
    embed_params += 514 * hidden_dim  # position embeddings
    embed_params += 2 * hidden_dim  # token_type

    layer_params = (
        3 * hidden_dim * hidden_dim
        + hidden_dim * hidden_dim
        + 2 * hidden_dim * (hidden_dim * 4)
        + 4 * hidden_dim
    )
    total_params = embed_params + len(layer_indices) * layer_params

    fp32_mb = total_params * 4 / (1024 ** 2)

    return {
        "fp32_mb": round(fp32_mb, 1),
        "total_params": total_params,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", help="특정 실험만 실행")
    parser.add_argument("--no-prune", action="store_true",
                        help="Vocab pruning 생략 (원본 250K vocab)")
    parser.add_argument("--max-vocab", type=int, default=None,
                        help="최대 vocab 크기 (빈도 기반 필터링)")
    args = parser.parse_args()

    os.makedirs(STUDENTS_DIR, exist_ok=True)

    experiments = EXPERIMENTS
    if args.only:
        experiments = [e for e in experiments if e["name"] in args.only]

    # Teacher 로드
    print(f"Loading teacher: {TEACHER_MODEL}")
    teacher = SentenceTransformer(TEACHER_MODEL)
    print(f"  Layers: {teacher[0].auto_model.config.num_hidden_layers}")
    print(f"  Hidden: {teacher[0].auto_model.config.hidden_size}")
    print()

    # Corpus-based vocab pruning (한 번만)
    keep_ids = None
    if not args.no_prune:
        print("Collecting corpus tokens for vocab pruning...")
        tokenizer = teacher[0].tokenizer
        keep_ids = collect_corpus_tokens(tokenizer, max_vocab=args.max_vocab)
        print()

    # Student 생성
    for exp in experiments:
        name = exp["name"]
        layers = exp["layer_indices"]
        save_path = os.path.join(STUDENTS_DIR, name)

        v_size = len(keep_ids) if keep_ids else 250002
        size = estimate_size(layers, vocab_size=v_size)
        print(f"Creating {name}: layers={layers}")
        print(f"  {exp['description']}")
        print(f"  Estimated FP32 size: {size['fp32_mb']}MB")
        print(f"  Total params: {size['total_params']:,}")

        student = create_student(teacher, layers, save_path, keep_ids=keep_ids)

        # Sanity check
        test_sentences = ["Hello world", "안녕하세요", "こんにちは"]
        embeddings = student.encode(test_sentences)
        print(f"  Sanity check: output shape = {embeddings.shape}")

        # 실제 safetensors 크기 확인
        st_path = os.path.join(save_path, "model.safetensors")
        if os.path.exists(st_path):
            size_mb = os.path.getsize(st_path) / (1024 ** 2)
            print(f"  model.safetensors: {size_mb:.1f}MB")
        print(f"  Saved to {save_path}")
        print()

    print("Done! All student models created.")
    print("Next step: python run_mteb.py")


if __name__ == "__main__":
    main()
