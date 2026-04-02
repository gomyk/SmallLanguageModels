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
        "layers",              # ModernBERT, EuroBERT
        "model.layers",        # some wrapped models
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


def _unwrap_peft_model(model):
    """PEFT 모델(LoRA 등)에서 base transformer 모델을 추출한다.

    JinaEmbeddingsV5Model 등 PeftMixedModel로 래핑된 모델에서
    내부의 실제 transformer (예: EuroBertModel)를 꺼낸다.
    LoRA 어댑터는 제거되며, base 가중치만 유지된다.
    (student는 distillation으로 재학습하므로 LoRA 미반영 OK)
    """
    model_class = type(model).__name__

    # PEFT 라이브러리가 없으면 그대로 반환
    try:
        from peft import PeftModel, PeftMixedModel
    except ImportError:
        return model

    if not isinstance(model, (PeftModel, PeftMixedModel)):
        return model

    # PeftMixedModel: base_model.model → 실제 모델
    if isinstance(model, PeftMixedModel):
        base = model.base_model
        if hasattr(base, 'model'):
            inner = base.model
            _fix_config_after_peft_unwrap(inner)
            print(f"  Unwrapped PEFT: {model_class} → {type(inner).__name__}")
            return inner
        _fix_config_after_peft_unwrap(base)
        print(f"  Unwrapped PEFT: {model_class} → {type(base).__name__}")
        return base

    # PeftModel: merge_and_unload로 LoRA 병합 시도
    if isinstance(model, PeftModel):
        try:
            merged = model.merge_and_unload()
            _fix_config_after_peft_unwrap(merged)
            print(f"  Unwrapped PEFT: {model_class} → {type(merged).__name__} (merged)")
            return merged
        except Exception:
            base = model.base_model
            inner = base.model if hasattr(base, 'model') else base
            _fix_config_after_peft_unwrap(inner)
            return inner

    return model


def _fix_config_after_peft_unwrap(model):
    """PEFT unwrap 후 config의 auto_map/architectures/model_type을 base 모델로 수정한다.

    JinaEmbeddingsV5Model → EuroBertModel 등, PEFT wrapper 대신
    실제 base 모델 클래스를 가리키도록 변경하여 저장/재로드 시 문제를 방지한다.
    """
    config = model.config
    base_class_name = type(model).__name__  # e.g., "EuroBertModel"
    base_module = type(model).__module__    # e.g., "transformers_modules.xxx.modeling_eurobert"

    # base 모델의 config_class에서 정보 추출
    base_config_class = getattr(type(model), 'config_class', None)
    if base_config_class:
        base_config_name = base_config_class.__name__  # e.g., "EuroBertConfig"
        base_config_module = base_config_class.__module__
        # model_type 교체 (class attribute → instance attribute shadow)
        if hasattr(base_config_class, 'model_type'):
            config.model_type = base_config_class.model_type
    else:
        base_config_name = None
        base_config_module = None

    # auto_map에서 PEFT wrapper → base 모델/config 클래스로 교체
    if hasattr(config, 'auto_map') and config.auto_map:
        new_auto_map = {}
        for key, value in config.auto_map.items():
            if key == "AutoModel" and "." in value:
                module_file = base_module.rsplit(".", 1)[-1]
                new_auto_map[key] = f"{module_file}.{base_class_name}"
            elif key == "AutoConfig" and base_config_name and "." in value:
                config_module_file = base_config_module.rsplit(".", 1)[-1]
                new_auto_map[key] = f"{config_module_file}.{base_config_name}"
            else:
                new_auto_map[key] = value
        config.auto_map = new_auto_map

    # architectures도 base 모델로 변경
    if hasattr(config, 'architectures'):
        config.architectures = [base_class_name]


def create_pruned_student(teacher_model_id, layer_indices, layer_accessor=None,
                          trust_remote_code=False):
    """Teacher 모델을 로드하고 레이어를 pruning하여 student를 생성한다.

    PEFT 모델(jina-v5 등)은 자동으로 base 모델을 추출한다.

    Returns:
        (student_model, tokenizer) tuple
    """
    config = AutoConfig.from_pretrained(teacher_model_id,
                                         trust_remote_code=trust_remote_code)
    model = AutoModel.from_pretrained(teacher_model_id,
                                       trust_remote_code=trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_id,
                                               trust_remote_code=trust_remote_code)

    # PEFT 모델이면 base transformer 추출
    model = _unwrap_peft_model(model)

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
    # PEFT unwrap 후 config의 auto_map이 PEFT wrapper를 가리키면
    # AutoModel.from_config가 실패할 수 있으므로 모델 클래스 직접 사용 fallback
    try:
        new_model = AutoModel.from_config(new_config, trust_remote_code=trust_remote_code)
    except (AttributeError, TypeError):
        new_model = type(model)(new_config)

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


def reduce_hidden_dim_pca(model, tokenizer, new_hidden_dim, corpus_texts,
                          new_intermediate_size=None, trust_remote_code=False,
                          n_samples=2000, batch_size=64, max_length=128):
    """PCA 기반 hidden dimension 축소.

    코퍼스에서 hidden states를 수집하고 PCA로 최적 projection 방향을 계산한 뒤,
    가중치를 PCA 공간으로 변환하여 차원을 축소한다.

    단순 슬라이싱 대비 PCA의 장점:
      - 가장 정보량이 높은 방향을 보존
      - 축소 후 explained variance 비율로 품질 확인 가능

    Args:
        model: HuggingFace transformer 모델 (layer pruning 완료 상태)
        tokenizer: HuggingFace tokenizer
        new_hidden_dim: 목표 hidden dimension
        corpus_texts: PCA 계산용 코퍼스 텍스트 리스트
        new_intermediate_size: FFN intermediate size. None이면 비례 축소.
        trust_remote_code: custom code 모델 지원 여부
        n_samples: PCA에 사용할 샘플 수
        batch_size: 인코딩 배치 크기
        max_length: 최대 시퀀스 길이

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

    device = next(model.parameters()).device

    # ── 1. Hidden states 수집 ──
    print(f"  PCA: collecting hidden states ({min(len(corpus_texts), n_samples)} samples)...")
    model.eval()
    all_hidden = []
    with torch.no_grad():
        for i in range(0, min(len(corpus_texts), n_samples), batch_size):
            batch = corpus_texts[i:i + batch_size]
            encoded = tokenizer(
                batch, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            ).to(device)
            output = model(**encoded)
            h = output.last_hidden_state  # [B, L, D]
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # [B, D]
            all_hidden.append(pooled.cpu().float())

    H = torch.cat(all_hidden, dim=0)  # [N, D]

    # ── 2. PCA 계산 ──
    H_centered = H - H.mean(0, keepdim=True)
    _, S, Vt = torch.linalg.svd(H_centered, full_matrices=False)

    # V_proj: [D, K] — D차원을 K차원으로 projection
    V_proj = Vt[:new_hidden_dim].T  # [D, K]

    explained_var = (S[:new_hidden_dim] ** 2).sum() / (S ** 2).sum()
    print(f"  PCA: {old_hidden}d → {new_hidden_dim}d "
          f"(explained variance: {explained_var:.1%}, "
          f"samples: {H.shape[0]})")

    # ── 3. 새 config + 모델 생성 ──
    new_config = copy.deepcopy(model.config)
    new_config.hidden_size = new_hidden_dim
    new_config.intermediate_size = new_intermediate_size

    # Attention head 수 조정 (reduce_hidden_dim과 동일 로직)
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
        while n_kv > 1 and (n_heads % n_kv != 0 or new_hidden_dim % n_kv != 0):
            n_kv -= 1
        new_config.num_key_value_heads = n_kv

    if hasattr(new_config, 'head_dim') and new_config.head_dim is not None:
        new_heads = getattr(new_config, 'num_attention_heads', 1)
        new_config.head_dim = new_hidden_dim // new_heads

    try:
        new_model = AutoModel.from_config(new_config, trust_remote_code=trust_remote_code)
    except (AttributeError, TypeError):
        new_model = type(model)(new_config)

    # ── 4. PCA projection으로 가중치 변환 ──
    V = V_proj.to(device)  # [D, K]
    old_sd = model.state_dict()
    new_sd = new_model.state_dict()

    copied, projected, skipped = 0, 0, 0
    for key in new_sd:
        if key not in old_sd:
            skipped += 1
            continue

        old_t = old_sd[key]
        new_t = new_sd[key]

        if old_t.shape == new_t.shape:
            new_sd[key] = old_t.clone()
            copied += 1
            continue

        # 1D: bias, RMSNorm/LayerNorm weight
        if old_t.dim() == 1:
            if old_t.shape[0] == old_hidden and new_t.shape[0] == new_hidden_dim:
                # Norm 가중치: PCA 공간에서는 기본값이 더 안정적
                # (distillation에서 fine-tune 됨)
                if "norm" in key.lower() or "ln" in key.lower():
                    new_sd[key] = torch.ones(new_hidden_dim, dtype=old_t.dtype,
                                             device=old_t.device)
                else:
                    # Bias: PCA projection 적용
                    new_sd[key] = (V.T.cpu().float() @ old_t.float()).to(old_t.dtype)
                projected += 1
            elif old_t.shape[0] == old_inter and new_t.shape[0] == new_intermediate_size:
                new_sd[key] = old_t[:new_intermediate_size].clone()
                copied += 1
            else:
                new_sd[key] = old_t[:new_t.shape[0]].clone()
                copied += 1
            continue

        # 2D: linear layers, embeddings
        if old_t.dim() == 2:
            out_dim, in_dim = old_t.shape
            new_out, new_in = new_t.shape
            t = old_t.float()

            # input 차원이 hidden_dim인 경우 → PCA projection
            if in_dim == old_hidden and new_in == new_hidden_dim:
                t = t @ V.cpu().float()  # [out, D] @ [D, K] → [out, K]
            elif in_dim != new_in:
                t = t[:, :new_in]

            # output 차원이 hidden_dim인 경우 → PCA projection
            if out_dim == old_hidden and new_out == new_hidden_dim:
                t = V.T.cpu().float() @ t  # [K, D] @ [D, ?] → [K, ?]
            elif out_dim != new_out:
                t = t[:new_out]

            new_sd[key] = t.to(old_t.dtype)
            projected += 1
            continue

        # 기타 차원: 슬라이싱 fallback
        slices = tuple(
            slice(0, min(s_new, s_old))
            for s_new, s_old in zip(new_t.shape, old_t.shape)
        )
        sliced = old_t[slices]
        if sliced.shape == new_t.shape:
            new_sd[key] = sliced.clone()
        copied += 1

    # Shape mismatch fallback: PCA가 처리하지 못한 차원 (head_dim 변경 등)
    for key in new_sd:
        if key not in old_sd:
            continue
        new_t = new_sd[key]
        # 원본 shape와 비교하여 여전히 불일치하는 경우 슬라이싱/패딩
        if new_t.shape != new_model.state_dict()[key].shape:
            target_shape = new_model.state_dict()[key].shape
            slices = tuple(
                slice(0, min(s_src, s_tgt))
                for s_src, s_tgt in zip(new_t.shape, target_shape)
            )
            fixed = torch.zeros(target_shape, dtype=new_t.dtype, device=new_t.device)
            src_slices = tuple(slice(0, min(s_src, s_tgt))
                               for s_src, s_tgt in zip(new_t.shape, target_shape))
            fixed[src_slices] = new_t[slices]
            new_sd[key] = fixed

    new_model.load_state_dict(new_sd)
    print(f"  PCA dim reduction: {old_hidden} → {new_hidden_dim} "
          f"(intermediate: {old_inter} → {new_intermediate_size}, "
          f"projected: {projected}, copied: {copied}, new: {skipped})")

    # Special token ID 보존
    for attr in ["pad_token_id", "bos_token_id", "eos_token_id",
                 "cls_token_id", "sep_token_id", "unk_token_id", "mask_token_id"]:
        old_val = getattr(model.config, attr, None)
        if old_val is not None:
            setattr(new_model.config, attr, old_val)

    return new_model


def reduce_hidden_dim_activation(model, tokenizer, new_hidden_dim, corpus_texts,
                                  new_intermediate_size=None, trust_remote_code=False,
                                  n_samples=2000, batch_size=64, max_length=128):
    """활성화 기반 hidden dimension 축소.

    코퍼스를 돌려 각 hidden dimension의 중요도를 측정하고,
    중요도가 높은 상위 K개 차원만 선별하여 축소한다.

    PCA와 달리 회전(projection) 없이 실제 차원을 선택하므로:
      - 원본 가중치 구조가 보존됨
      - 활성화 빈도/크기가 높은 차원 = 실제 사용되는 차원만 유지
      - Attention head 구조에 영향이 적음

    중요도 점수: 각 차원의 평균 절대 활성화 값 + 분산 (결합)

    Args:
        model: HuggingFace transformer 모델 (layer pruning 완료 상태)
        tokenizer: HuggingFace tokenizer
        new_hidden_dim: 목표 hidden dimension
        corpus_texts: 중요도 계산용 코퍼스 텍스트 리스트
        new_intermediate_size: FFN intermediate size. None이면 비례 축소.
        trust_remote_code: custom code 모델 지원 여부
        n_samples: 중요도 계산에 사용할 샘플 수
        batch_size: 인코딩 배치 크기
        max_length: 최대 시퀀스 길이

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

    device = next(model.parameters()).device

    # ── 1. Hidden states 수집 → 차원별 중요도 계산 ──
    print(f"  Activation pruning: collecting hidden states "
          f"({min(len(corpus_texts), n_samples)} samples)...")
    model.eval()

    # 차원별 통계 누적
    dim_sum = torch.zeros(old_hidden, dtype=torch.float64)
    dim_sq_sum = torch.zeros(old_hidden, dtype=torch.float64)
    dim_abs_sum = torch.zeros(old_hidden, dtype=torch.float64)
    total_count = 0

    with torch.no_grad():
        for i in range(0, min(len(corpus_texts), n_samples), batch_size):
            batch = corpus_texts[i:i + batch_size]
            encoded = tokenizer(
                batch, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            ).to(device)
            output = model(**encoded)
            h = output.last_hidden_state  # [B, L, D]
            mask = encoded["attention_mask"].unsqueeze(-1).float()

            # 모든 토큰의 활성화를 사용 (마스킹 적용)
            h_masked = (h * mask).cpu().double()  # [B, L, D]
            token_counts = mask.sum(dim=1).cpu().double()  # [B, 1]

            # 배치의 mean pooled representation도 함께 사용
            pooled = h_masked.sum(dim=1) / token_counts.clamp(min=1e-9)  # [B, D]

            dim_sum += pooled.sum(dim=0)
            dim_sq_sum += (pooled ** 2).sum(dim=0)
            dim_abs_sum += pooled.abs().sum(dim=0)
            total_count += pooled.shape[0]

    # ── 2. 차원별 중요도 점수 계산 ──
    dim_mean = dim_sum / total_count
    dim_var = (dim_sq_sum / total_count) - dim_mean ** 2
    dim_abs_mean = dim_abs_sum / total_count

    # 중요도 = 정규화된 (평균 절대값 + 표준편차)
    # 둘 다 높을수록 해당 차원이 활발하게 사용됨
    abs_norm = dim_abs_mean / dim_abs_mean.max().clamp(min=1e-9)
    std_norm = dim_var.sqrt() / dim_var.sqrt().max().clamp(min=1e-9)
    importance = abs_norm + std_norm

    # 상위 K개 차원 선택
    _, keep_indices = importance.topk(new_hidden_dim)
    keep_indices = keep_indices.sort().values  # 정렬하여 순서 유지
    keep_indices_cpu = keep_indices.long()

    # 통계 출력
    kept_importance = importance[keep_indices_cpu].sum()
    total_importance = importance.sum()
    print(f"  Activation pruning: {old_hidden}d → {new_hidden_dim}d "
          f"(importance retained: {kept_importance/total_importance:.1%}, "
          f"samples: {total_count})")
    print(f"    Top dims: {keep_indices_cpu[:10].tolist()}... "
          f"Bottom dropped: {importance.topk(old_hidden).indices[-5:].tolist()}")

    # ── 3. Intermediate dimension도 활성화 기반 선택 ──
    # FFN의 중간 차원에 대해서도 동일한 로직 적용
    inter_keep_indices = None
    if new_intermediate_size < old_inter:
        # FFN의 intermediate states 수집은 별도 hook 필요 → 가중치 L1 norm으로 대체
        # FFN up/gate projection의 output 가중치 크기로 중요도 추정
        old_sd = model.state_dict()
        inter_importance = torch.zeros(old_inter, dtype=torch.float64)
        inter_count = 0
        for key, weight in old_sd.items():
            if weight.dim() == 2:
                out_dim, in_dim = weight.shape
                if out_dim == old_inter and in_dim == old_hidden:
                    # Up/gate projection: [inter, hidden]
                    inter_importance += weight.float().abs().sum(dim=1).double().cpu()
                    inter_count += 1
        if inter_count > 0:
            inter_importance /= inter_count
            _, inter_keep_indices = inter_importance.topk(new_intermediate_size)
            inter_keep_indices = inter_keep_indices.sort().values.long()
            kept_inter_imp = inter_importance[inter_keep_indices].sum()
            total_inter_imp = inter_importance.sum()
            print(f"    Intermediate: {old_inter} → {new_intermediate_size} "
                  f"(importance retained: {kept_inter_imp/total_inter_imp:.1%})")

    # ── 4. 새 config + 모델 생성 ──
    new_config = copy.deepcopy(model.config)
    new_config.hidden_size = new_hidden_dim
    new_config.intermediate_size = new_intermediate_size

    # Attention head 수 조정
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
        while n_kv > 1 and (n_heads % n_kv != 0 or new_hidden_dim % n_kv != 0):
            n_kv -= 1
        new_config.num_key_value_heads = n_kv

    if hasattr(new_config, 'head_dim') and new_config.head_dim is not None:
        new_heads = getattr(new_config, 'num_attention_heads', 1)
        new_config.head_dim = new_hidden_dim // new_heads

    try:
        new_model = AutoModel.from_config(new_config, trust_remote_code=trust_remote_code)
    except (AttributeError, TypeError):
        new_model = type(model)(new_config)

    # ── 5. 선별된 차원만 가중치 복사 ──
    ki = keep_indices_cpu  # hidden dim indices
    ii = inter_keep_indices  # intermediate dim indices (or None)
    old_sd = model.state_dict()
    new_sd = new_model.state_dict()

    copied, selected, skipped = 0, 0, 0
    for key in new_sd:
        if key not in old_sd:
            skipped += 1
            continue

        old_t = old_sd[key]
        new_t = new_sd[key]

        if old_t.shape == new_t.shape:
            new_sd[key] = old_t.clone()
            copied += 1
            continue

        # 1D: bias, norm weights
        if old_t.dim() == 1:
            if old_t.shape[0] == old_hidden and new_t.shape[0] == new_hidden_dim:
                new_sd[key] = old_t[ki].clone()
                selected += 1
            elif ii is not None and old_t.shape[0] == old_inter and new_t.shape[0] == new_intermediate_size:
                new_sd[key] = old_t[ii].clone()
                selected += 1
            else:
                new_sd[key] = old_t[:new_t.shape[0]].clone()
                copied += 1
            continue

        # 2D: linear layers, embeddings
        if old_t.dim() == 2:
            out_dim, in_dim = old_t.shape
            new_out, new_in = new_t.shape
            t = old_t

            # Input dim 선별
            if in_dim == old_hidden and new_in == new_hidden_dim:
                t = t[:, ki]
            elif ii is not None and in_dim == old_inter and new_in == new_intermediate_size:
                t = t[:, ii]
            elif in_dim != new_in:
                t = t[:, :new_in]

            # Output dim 선별
            if out_dim == old_hidden and new_out == new_hidden_dim:
                t = t[ki]
            elif ii is not None and out_dim == old_inter and new_out == new_intermediate_size:
                t = t[ii]
            elif out_dim != new_out:
                t = t[:new_out]

            new_sd[key] = t.clone()
            selected += 1
            continue

        # 기타: 슬라이싱 fallback
        slices = tuple(
            slice(0, min(s_new, s_old))
            for s_new, s_old in zip(new_t.shape, old_t.shape)
        )
        sliced = old_t[slices]
        if sliced.shape == new_t.shape:
            new_sd[key] = sliced.clone()
        copied += 1

    new_model.load_state_dict(new_sd)
    print(f"  Activation dim reduction: {old_hidden} → {new_hidden_dim} "
          f"(intermediate: {old_inter} → {new_intermediate_size}, "
          f"selected: {selected}, copied: {copied}, new: {skipped})")

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

def _remap_post_processor_ids(pp, old_to_new):
    """post_processor 내 special_tokens ID를 재귀적으로 재매핑한다.

    Sequence 타입 post_processor는 processors 리스트 안에 중첩될 수 있으므로
    재귀적으로 탐색하여 모든 special_tokens의 ids를 변환한다.
    """
    if "special_tokens" in pp:
        for token_name, token_info in pp["special_tokens"].items():
            if "ids" in token_info:
                token_info["ids"] = [
                    old_to_new[oid] for oid in token_info["ids"]
                    if oid in old_to_new
                ]
    # Sequence 타입: processors 리스트 내 각 processor도 재귀 처리
    if "processors" in pp:
        for proc in pp["processors"]:
            _remap_post_processor_ids(proc, old_to_new)


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

    # post_processor의 special_tokens IDs도 재매핑 (중첩 Sequence 포함)
    pp = tok_json.get("post_processor")
    if pp:
        _remap_post_processor_ids(pp, old_to_new)

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

    # BPE merge 역추적: keep_ids에 포함된 토큰을 만드는 데 필요한
    # 중간 subword 토큰과 merge rule을 모두 보존한다.
    # 이 단계가 없으면 pruned 토크나이저가 일부 토큰을 조립할 수 없어
    # byte-level fallback으로 과도하게 분해된다.
    if tok_type == "BPE":
        keep_ids = _backtrack_bpe_merges(tokenizer, keep_ids)

    return sorted(keep_ids)


def _backtrack_bpe_merges(tokenizer, keep_ids):
    """BPE merge rule을 역추적하여 필요한 중간 토큰을 모두 keep_ids에 추가한다.

    BPE 토큰 X가 merge "A B → AB"로 생성되었다면, A와 B도 vocab에 있어야
    토크나이저가 X를 올바르게 조립할 수 있다. A, B 역시 다른 merge의 결과일 수
    있으므로 재귀적으로 역추적한다.

    Args:
        tokenizer: HuggingFace tokenizer (BPE)
        keep_ids: 유지할 토큰 ID set 또는 list

    Returns:
        역추적 후 확장된 keep_ids (sorted list)
    """
    tok_json = json.loads(tokenizer.backend_tokenizer.to_str())
    vocab = tok_json["model"]["vocab"]
    merges_raw = tok_json["model"]["merges"]

    # merge → result token ID 매핑
    token_to_merge_idx = {}  # result_token_id → merge_idx
    merge_inputs = {}        # merge_idx → (a_id, b_id)

    for idx, merge in enumerate(merges_raw):
        if isinstance(merge, list):
            a, b = merge[0], merge[1]
        else:
            parts = merge.split(" ", 1)
            if len(parts) != 2:
                continue
            a, b = parts
        result = a + b
        if result in vocab:
            result_id = vocab[result]
            token_to_merge_idx[result_id] = idx
            merge_inputs[idx] = (vocab.get(a), vocab.get(b))

    # BFS 역추적
    needed = set(keep_ids)
    queue = list(needed)
    visited = set(needed)

    while queue:
        tid = queue.pop()
        if tid in token_to_merge_idx:
            merge_idx = token_to_merge_idx[tid]
            a_id, b_id = merge_inputs[merge_idx]
            for dep_id in (a_id, b_id):
                if dep_id is not None and dep_id not in visited:
                    visited.add(dep_id)
                    needed.add(dep_id)
                    queue.append(dep_id)

    added = len(needed) - len(keep_ids) if isinstance(keep_ids, set) else len(needed) - len(set(keep_ids))
    if added > 0:
        print(f"  BPE merge backtrack: +{added:,} intermediate tokens "
              f"({len(needed):,} total)")

    return sorted(needed)


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


def _patch_config_json_for_base_model(model, save_dir):
    """save_pretrained 후 config.json을 base 모델 클래스 기준으로 패치한다.

    PretrainedConfig.to_dict()가 __class__.model_type을 강제 사용하므로,
    PEFT unwrap 후 instance 속성 변경만으로는 부족하다.
    save_pretrained 후 파일을 직접 수정하여 base 모델 정보로 교체한다.
    """
    config_path = os.path.join(save_dir, "config.json")
    if not os.path.exists(config_path):
        return

    # base 모델 정보가 config에 설정되어 있는지 확인
    base_config_class = getattr(type(model), 'config_class', None)
    if not base_config_class:
        return

    # 현재 config의 class와 base 모델의 config_class가 다르면 패치 필요
    if type(model.config) == base_config_class:
        return

    base_model_name = type(model).__name__
    base_config_name = base_config_class.__name__
    base_model_module = type(model).__module__.rsplit(".", 1)[-1]
    base_config_module = base_config_class.__module__.rsplit(".", 1)[-1]
    base_model_type = getattr(base_config_class, 'model_type', None)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    patched = False

    if base_model_type and cfg.get("model_type") != base_model_type:
        cfg["model_type"] = base_model_type
        patched = True

    if "auto_map" in cfg:
        am = cfg["auto_map"]
        if "AutoModel" in am:
            am["AutoModel"] = f"{base_model_module}.{base_model_name}"
        if "AutoConfig" in am:
            am["AutoConfig"] = f"{base_config_module}.{base_config_name}"
        patched = True

    if cfg.get("architectures") != [base_model_name]:
        cfg["architectures"] = [base_model_name]
        patched = True

    if patched:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"  Patched config.json: model_type={base_model_type}, "
              f"AutoModel={base_model_name}")


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

    # PEFT unwrap된 모델: save_pretrained가 원본 클래스 정보를 쓰므로
    # config.json을 직접 패치하여 base 모델 클래스로 교체
    _patch_config_json_for_base_model(model, hf_tmp)

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
