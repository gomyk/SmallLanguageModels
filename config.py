"""
Multi-Teacher Student Model Compression Pipeline - Configuration

여러 teacher 모델에서 layer pruning + vocab pruning으로 경량 student를 생성하고
MTEB 벤치마크(Classification, Clustering, STS)로 평가한다.
"""

# ── Teacher Models ────────────────────────────────────────────
TEACHERS = {
    "minilm": {
        "model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "short_name": "MiniLM-L12",
        "hidden_dim": 384,
        "num_layers": 12,
        "intermediate_size": 1536,
        "vocab_size": 250002,
        "layer_accessor": "encoder.layer",
        "tokenizer_type": "unigram",
        "trust_remote_code": False,
    },
    "modernbert": {
        "model_id": "answerdotai/ModernBERT-base",
        "short_name": "ModernBERT",
        "hidden_dim": 768,
        "num_layers": 22,
        "intermediate_size": 1152,
        "vocab_size": 50368,
        "layer_accessor": "layers",
        "tokenizer_type": "bpe",
        "trust_remote_code": False,
    },
    "gte": {
        "model_id": "alibaba-NLP/gte-multilingual-base",
        "short_name": "GTE-multilingual",
        "hidden_dim": 768,
        "num_layers": 12,
        "intermediate_size": 3072,
        "vocab_size": 250048,
        "layer_accessor": "encoder.layer",
        "tokenizer_type": "unigram",
        "trust_remote_code": True,
    },
    "me5": {
        "model_id": "intfloat/multilingual-e5-base",
        "short_name": "mE5-base",
        "hidden_dim": 768,
        "num_layers": 12,
        "intermediate_size": 3072,
        "vocab_size": 250002,
        "layer_accessor": "encoder.layer",
        "tokenizer_type": "unigram",
        "trust_remote_code": False,
    },
    "me5s": {
        "model_id": "intfloat/multilingual-e5-small",
        "short_name": "mE5-small",
        "hidden_dim": 384,
        "num_layers": 12,
        "intermediate_size": 1536,
        "vocab_size": 250037,
        "layer_accessor": "encoder.layer",
        "tokenizer_type": "unigram",
        "trust_remote_code": False,
    },
    "gemma_emb": {
        "model_id": "google/embeddinggemma-300m",
        "short_name": "EmbeddingGemma-300M",
        "hidden_dim": 768,
        "num_layers": 24,
        "intermediate_size": 1152,
        "vocab_size": 262144,
        "layer_accessor": "layers",
        "tokenizer_type": "unigram",
        "trust_remote_code": False,
        "num_attention_heads": 3,
        "num_kv_heads": 1,
        "head_dim": 256,
        "has_glu": True,
        "is_decoder": True,
        "license": "gemma",
        "license_notice": (
            "This model is a derivative of Google's Gemma. "
            "Gemma is provided under and subject to the Gemma Terms of Use "
            "found at [ai.google.dev/gemma/terms](https://ai.google.dev/gemma/terms). "
            "Use of this model must comply with the "
            "[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy)."
        ),
    },
    "qwen3": {
        "model_id": "Qwen/Qwen3-0.6B",
        "short_name": "Qwen3-0.6B",
        "hidden_dim": 1024,
        "num_layers": 28,
        "intermediate_size": 3072,
        "vocab_size": 151936,
        "layer_accessor": "layers",
        "tokenizer_type": "bpe",
        "trust_remote_code": False,
        "num_attention_heads": 16,
        "num_kv_heads": 8,
        "head_dim": 128,
        "has_glu": True,
        "is_decoder": True,
    },
    "mmbert": {
        "model_id": "jhu-clsp/mmBERT-small",
        "short_name": "mmBERT-small",
        "hidden_dim": 384,
        "num_layers": 22,
        "intermediate_size": 1152,
        "vocab_size": 256000,
        "layer_accessor": "layers",
        "tokenizer_type": "bpe",
        "trust_remote_code": True,
    },
    "jina_v5": {
        "model_id": "jinaai/jina-embeddings-v5-text-nano",
        "short_name": "Jina-v5-nano",
        "hidden_dim": 768,
        "num_layers": 12,
        "intermediate_size": 3072,
        "vocab_size": 128256,
        "layer_accessor": "layers",
        "tokenizer_type": "bpe",
        "trust_remote_code": True,
        "has_glu": True,
        "is_decoder": True,  # RoPE 사용, position/token_type embedding 없음
        "is_peft_model": True,  # PeftMixedModel → EuroBertModel 추출 필요
        "model_kwargs": {"default_task": "text-matching"},
        "license": "cc-by-nc-4.0",
        "license_notice": (
            "This model is a derivative of Jina AI's jina-embeddings-v5-text-nano. "
            "The original model is provided under CC BY-NC 4.0 license. "
            "See [jina-embeddings-v5-text-nano](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano) "
            "for details."
        ),
    },
}

# 하위 호환 (기존 스크립트)
TEACHER_MODEL = TEACHERS["minilm"]["model_id"]
TEACHER_HIDDEN_DIM = TEACHERS["minilm"]["hidden_dim"]
TEACHER_NUM_LAYERS = TEACHERS["minilm"]["num_layers"]


# ── Layer Index Generation ────────────────────────────────────

def make_uniform_indices(num_layers, target_count):
    """균등 간격으로 레이어 인덱스를 생성한다."""
    return [round(i * (num_layers - 1) / (target_count - 1)) for i in range(target_count)]


def generate_experiments(teacher_key):
    """Teacher별 실험 설정을 생성한다 (L4, L6)."""
    t = TEACHERS[teacher_key]
    n = t["num_layers"]
    return [
        {
            "name": f"{teacher_key}_L6_uniform",
            "teacher": teacher_key,
            "description": f"6 layers, evenly spaced from {t['short_name']} ({n}L)",
            "layer_indices": make_uniform_indices(n, 6),
        },
        {
            "name": f"{teacher_key}_L4_uniform",
            "teacher": teacher_key,
            "description": f"4 layers, evenly spaced from {t['short_name']} ({n}L)",
            "layer_indices": make_uniform_indices(n, 4),
        },
    ]


def generate_me5_experiments():
    """mE5 teacher용 3×3 실험 설정을 생성한다 (L2/L4/L6 × vocab 99%/97%/95%)."""
    t = TEACHERS["me5"]
    n = t["num_layers"]
    layer_configs = [
        (2, make_uniform_indices(n, 2)),
        (4, make_uniform_indices(n, 4)),
        (6, make_uniform_indices(n, 6)),
    ]
    vocab_ratios = [0.99, 0.97, 0.95]
    experiments = []
    for layer_count, layer_indices in layer_configs:
        for ratio in vocab_ratios:
            pct = int(ratio * 100)
            experiments.append({
                "name": f"me5_L{layer_count}_p{pct}",
                "teacher": "me5",
                "description": (
                    f"{layer_count} layers uniform, {pct}% vocab retention "
                    f"from {t['short_name']} ({n}L)"
                ),
                "layer_indices": layer_indices,
                "vocab_keep_ratio": ratio,
            })
    return experiments


# 기존 MiniLM 실험 (하위 호환)
EXPERIMENTS = [
    {
        "name": "L6_uniform",
        "teacher": "minilm",
        "description": "6 layers, evenly spaced (general-purpose)",
        "layer_indices": [0, 2, 4, 7, 9, 11],
    },
    {
        "name": "L4_uniform",
        "teacher": "minilm",
        "description": "4 layers, evenly spaced (compact)",
        "layer_indices": [0, 4, 7, 11],
    },
    {
        "name": "L3_uniform",
        "teacher": "minilm",
        "description": "3 layers, evenly spaced (ultra-compact)",
        "layer_indices": [0, 6, 11],
    },
    {
        "name": "L6_top",
        "teacher": "minilm",
        "description": "6 layers, top half (semantic-focused)",
        "layer_indices": [6, 7, 8, 9, 10, 11],
    },
    {
        "name": "L4_top",
        "teacher": "minilm",
        "description": "4 layers, top quarter (semantic-focused compact)",
        "layer_indices": [8, 9, 10, 11],
    },
    {
        "name": "L6_bottom",
        "teacher": "minilm",
        "description": "6 layers, bottom half (syntactic-focused)",
        "layer_indices": [0, 1, 2, 3, 4, 5],
    },
    {
        "name": "L2_ends",
        "teacher": "minilm",
        "description": "2 layers, first + last (minimal)",
        "layer_indices": [0, 11],
    },
]

# 50MB Compact 실험 (기존)
COMPACT_EXPERIMENTS = [
    {
        "name": "L3_compact",
        "teacher": "minilm",
        "description": "3 layers uniform + 20K vocab (~50MB target)",
        "layer_indices": [0, 6, 11],
    },
    {
        "name": "L4_compact",
        "teacher": "minilm",
        "description": "4 layers uniform + 20K vocab (~57MB target)",
        "layer_indices": [0, 4, 7, 11],
    },
    {
        "name": "L6_compact",
        "teacher": "minilm",
        "description": "6 layers bottom + 20K vocab (~71MB target)",
        "layer_indices": [0, 1, 2, 3, 4, 5],
    },
]


# ── MTEB Evaluation Tasks ────────────────────────────────────
MTEB_TASK_GROUPS = {
    "Classification": [
        "AmazonCounterfactualClassification",
        "Banking77Classification",
        "ImdbClassification",
        "MTOPDomainClassification",
        "MassiveIntentClassification",
        "MassiveScenarioClassification",
        "ToxicConversationsClassification",
        "TweetSentimentExtractionClassification",
    ],
    "Clustering": [
        "ArXivHierarchicalClusteringP2P",
        "ArXivHierarchicalClusteringS2S",
        "BiorxivClusteringP2P.v2",
        "MedrxivClusteringP2P.v2",
        "MedrxivClusteringS2S.v2",
        "StackExchangeClustering.v2",
        "StackExchangeClusteringP2P.v2",
        "TwentyNewsgroupsClustering.v2",
    ],
    "STS": [
        "BIOSSES",
        "SICK-R",
        "STS12",
        "STS13",
        "STS14",
        "STS15",
        "STS17",
        "STS22.v2",
        "STSBenchmark",
    ],
}

# 전체 태스크 리스트 (flat)
MTEB_TASKS = []
for _group_tasks in MTEB_TASK_GROUPS.values():
    MTEB_TASKS.extend(_group_tasks)

# Massive 태스크 제외 버전 (오래 걸리는 태스크 제외)
MTEB_EXCLUDE_MASSIVE = ["MassiveIntentClassification", "MassiveScenarioClassification"]

def get_mteb_task_groups(exclude=None):
    """MTEB 태스크 그룹을 반환한다. exclude에 태스크 이름 리스트를 넘기면 제외."""
    if not exclude:
        return MTEB_TASK_GROUPS
    filtered = {}
    for group, tasks in MTEB_TASK_GROUPS.items():
        filtered_tasks = [t for t in tasks if t not in exclude]
        if filtered_tasks:
            filtered[group] = filtered_tasks
    return filtered

# 기존 하위 호환: quick eval
QUICK_EVAL_LANGS = ["eng", "kor", "zho", "spa", "ara"]


# ── Target Languages (16개) ───────────────────────────────────
TARGET_LANGUAGES = [
    "ko", "en", "ja", "zh", "es", "fr", "de", "pt",
    "it", "ru", "ar", "hi", "th", "vi", "id", "pl",
]

LANG_TO_ISO3 = {
    "ko": "kor", "en": "eng", "ja": "jpn", "zh": "zho",
    "es": "spa", "fr": "fra", "de": "deu", "pt": "por",
    "it": "ita", "ru": "rus", "ar": "ara", "hi": "hin",
    "th": "tha", "vi": "vie", "id": "ind", "pl": "pol",
}


# ── Distillation Data Sources ─────────────────────────────────
# MTEB 태스크 데이터셋에서 텍스트를 추출하여 distillation 학습에 사용
DISTILL_DATASETS = {
    # Classification datasets (train only — test는 MTEB 평가용)
    "amazon_counterfactual": {
        "hf_id": "mteb/amazon_counterfactual",
        "text_fields": ["text"],
        "splits": ["train"],
    },
    "banking77": {
        "hf_id": "mteb/banking77",
        "text_fields": ["text"],
        "splits": ["train"],
    },
    "imdb": {
        "hf_id": "mteb/imdb",
        "text_fields": ["text"],
        "splits": ["train"],
    },
    "mtop_domain": {
        "hf_id": "mteb/mtop_domain",
        "text_fields": ["text"],
        "splits": ["train"],
    },
    "massive_intent": {
        "hf_id": "mteb/amazon_massive_intent",
        "text_fields": ["text"],
        "splits": ["train"],
        "subsets": ["en", "ko", "ja", "zh-CN", "es", "fr", "de"],
    },
    "massive_scenario": {
        "hf_id": "mteb/amazon_massive_scenario",
        "text_fields": ["text"],
        "splits": ["train"],
        "subsets": ["en", "ko", "ja", "zh-CN", "es", "fr", "de"],
    },
    "toxic_conversations": {
        "hf_id": "mteb/toxic_conversations_50k",
        "text_fields": ["text"],
        "splits": ["train"],
    },
    "tweet_sentiment": {
        "hf_id": "mteb/tweet_sentiment_extraction",
        "text_fields": ["text"],
        "splits": ["train"],
    },
    # STS datasets (train only)
    "stsb": {
        "hf_id": "mteb/stsbenchmark-sts",
        "text_fields": ["sentence1", "sentence2"],
        "splits": ["train"],
    },
    # Large-scale NLI datasets (for bigger distillation corpus)
    "snli": {
        "hf_id": "stanfordnlp/snli",
        "text_fields": ["premise", "hypothesis"],
        "splits": ["train"],
    },
    "multi_nli": {
        "hf_id": "nyu-mll/multi_nli",
        "text_fields": ["premise", "hypothesis"],
        "splits": ["train"],
    },
}


# ── Paths ─────────────────────────────────────────────────────
STUDENTS_DIR = "students"
RESULTS_DIR = "results"
EXPORT_DIR = "exported"


def get_teacher_students_dir(teacher_key):
    """Teacher별 student 저장 디렉토리."""
    import os
    return os.path.join(STUDENTS_DIR, teacher_key)


def get_teacher_results_dir(teacher_key):
    """Teacher별 결과 저장 디렉토리."""
    import os
    return os.path.join(RESULTS_DIR, teacher_key)


# ── Size Estimation ───────────────────────────────────────────

def estimate_size(layer_indices, hidden_dim=384, vocab_size=40000, intermediate_size=None,
                  num_attention_heads=None, num_kv_heads=None, head_dim=None,
                  has_glu=False, is_decoder=False):
    """FP32 기준 예상 모델 크기 (MB) 추정.

    GQA (Grouped Query Attention), GLU (Gated Linear Unit), 디코더 모델을 지원한다.
    head_dim이 별도로 지정된 모델 (Qwen3 등)도 정확하게 추정한다.

    Args:
        layer_indices: 유지할 레이어 인덱스 리스트
        hidden_dim: hidden dimension
        vocab_size: vocabulary 크기
        intermediate_size: FFN intermediate dimension
        num_attention_heads: Q head 수 (GQA용, None이면 standard MHA)
        num_kv_heads: K/V head 수 (GQA용, None이면 num_attention_heads 사용)
        head_dim: attention head dimension. None이면 hidden_dim // num_attention_heads.
                  Qwen3 등 head_dim이 별도 config인 모델은 명시 필요.
        has_glu: GLU/SwiGLU 활성화 사용 여부 (gate + up + down = 3배)
        is_decoder: 디코더 모델 여부 (position/token_type embedding 제외)
    """
    if intermediate_size is None:
        intermediate_size = hidden_dim * 4

    # Embedding params
    embed_params = vocab_size * hidden_dim  # word embeddings
    if is_decoder:
        embed_params += hidden_dim  # final RMSNorm/LayerNorm
    else:
        embed_params += hidden_dim  # LayerNorm
        embed_params += 514 * hidden_dim  # position embeddings (approx)
        embed_params += 2 * hidden_dim  # token_type

    # Attention params (GQA + 별도 head_dim 지원)
    if num_attention_heads and num_kv_heads:
        hd = head_dim if head_dim else (hidden_dim // num_attention_heads)
        q_dim = num_attention_heads * hd
        kv_dim = num_kv_heads * hd
        attn_params = (
            hidden_dim * q_dim      # Q projection
            + hidden_dim * kv_dim   # K projection
            + hidden_dim * kv_dim   # V projection
            + q_dim * hidden_dim    # O projection
        )
    else:
        attn_params = 4 * hidden_dim * hidden_dim  # standard MHA: Q, K, V, O

    # FFN params (GLU 지원)
    ffn_multiplier = 3 if has_glu else 2
    ffn_params = ffn_multiplier * hidden_dim * intermediate_size

    layer_params = attn_params + ffn_params + 4 * hidden_dim  # + norms
    total_params = embed_params + len(layer_indices) * layer_params

    fp32_mb = total_params * 4 / (1024 ** 2)

    return {
        "fp32_mb": round(fp32_mb, 1),
        "total_params": total_params,
    }


def _estimate_for_teacher(teacher_key, layer_indices, vocab_size=None,
                          hidden_dim=None, intermediate_size=None):
    """Teacher config에서 추정 파라미터를 자동 추출하여 estimate_size를 호출한다.

    Hidden dim이 축소된 경우 num_heads, num_kv_heads, head_dim도 비례 조정한다.
    """
    t = TEACHERS[teacher_key]
    h = hidden_dim if hidden_dim is not None else t["hidden_dim"]
    inter = intermediate_size if intermediate_size is not None else t["intermediate_size"]
    v = vocab_size if vocab_size is not None else t["vocab_size"]

    n_heads = t.get("num_attention_heads")
    n_kv = t.get("num_kv_heads")
    hd = t.get("head_dim")  # Qwen3 등에서 별도 head_dim

    # Hidden dim이 변경된 경우 attention 구조 비례 조정
    if h != t["hidden_dim"] and n_heads:
        ratio = h / t["hidden_dim"]
        n_heads = max(1, int(n_heads * ratio))
        while h % n_heads != 0 and n_heads > 1:
            n_heads -= 1
        if n_kv:
            n_kv = max(1, int(n_kv * ratio))
            # GQA: n_heads % n_kv == 0 이어야 함
            while n_kv > 1 and (n_heads % n_kv != 0 or h % n_kv != 0):
                n_kv -= 1
        # head_dim도 비례 축소 (Qwen3처럼 별도 head_dim이 있는 경우)
        if hd:
            hd = h // n_heads

    return estimate_size(
        layer_indices, h, v, inter,
        num_attention_heads=n_heads,
        num_kv_heads=n_kv,
        head_dim=hd,
        has_glu=t.get("has_glu", False),
        is_decoder=t.get("is_decoder", False),
    )


def calculate_target_vocab(hidden_dim, num_layers, intermediate_size=None,
                           target_mb=50.0):
    """목표 사이즈에 맞는 vocab 크기를 역산한다.

    768d 모델은 레이어 파라미터만으로 50MB를 초과할 수 있다.
    이 경우 최소 vocab(5000)을 반환하고 실제 사이즈가 타겟을 넘을 수 있다.
    """
    if intermediate_size is None:
        intermediate_size = hidden_dim * 4

    layer_params = (
        3 * hidden_dim * hidden_dim
        + hidden_dim * hidden_dim
        + 2 * hidden_dim * intermediate_size
        + 4 * hidden_dim
    )

    overhead_params = (
        hidden_dim           # LayerNorm
        + 514 * hidden_dim   # position embeddings
        + 2 * hidden_dim     # token_type
    )

    total_non_vocab = overhead_params + num_layers * layer_params
    target_bytes = target_mb * (1024 ** 2)
    available_for_vocab = (target_bytes / 4) - total_non_vocab

    min_vocab = 5000  # 최소한의 토크나이저 품질 보장
    max_vocab = int(available_for_vocab / hidden_dim)

    if max_vocab < min_vocab:
        layer_only_mb = (total_non_vocab * 4) / (1024 ** 2)
        min_total_mb = ((total_non_vocab + min_vocab * hidden_dim) * 4) / (1024 ** 2)
        print(f"  WARNING: Layer params alone = {layer_only_mb:.1f}MB "
              f"(target={target_mb}MB). Min achievable = {min_total_mb:.1f}MB "
              f"with {min_vocab:,} vocab.")
        return min_vocab

    return max_vocab


def find_optimal_config(teacher_key, max_params=20_000_000, max_fp32_mb=50.0,
                        min_layers=4, estimated_vocab_size=None,
                        corpus_vocab_size=None):
    """크기 제약을 만족하는 최적 모델 설정을 탐색한다.

    Layer, hidden dim, vocab size를 joint 최적화한다.
    우선순위: hidden dim 보존 > 레이어 수 > vocab 크기.

    전략:
      1. 각 레이어 수 (teacher → min_layers)에 대해:
         a. 원본 hidden dim을 유지할 수 있는지 확인
         b. 가능하면 남은 예산으로 vocab 최대화
         c. 불가능하면 hidden dim 축소 후 vocab 최대화
      2. hidden dim × num_layers가 가장 큰 설정 선택 (품질 우선)

    Args:
        teacher_key: TEACHERS dict의 키
        max_params: 최대 파라미터 수
        max_fp32_mb: 최대 FP32 모델 크기 (MB)
        min_layers: 최소 레이어 수
        estimated_vocab_size: vocab 상한 (None이면 corpus_vocab_size 또는 원본 사용)
        corpus_vocab_size: 코퍼스에 등장한 전체 토큰 수 (vocab 상한으로 사용)

    Returns:
        dict with layer_indices, hidden_dim, intermediate_size,
              target_vocab, needs_hidden_reduction
    """
    t = TEACHERS[teacher_key]

    param_limit_from_mb = int(max_fp32_mb * 1024 * 1024 / 4)
    effective_max = min(max_params, param_limit_from_mb)

    hidden_dim = t["hidden_dim"]
    inter_size = t["intermediate_size"]

    # Vocab 상한: 코퍼스에 등장한 토큰 수 또는 명시적 값
    vocab_cap = estimated_vocab_size or corpus_vocab_size or t["vocab_size"]

    # vocab 하한: estimated가 있으면 그 값 (강제 토큰 등으로 줄일 수 없는 경우)
    vocab_floor = estimated_vocab_size if estimated_vocab_size else 3000

    def _calc_vocab_budget(teacher_key, layer_indices, h, inter):
        """주어진 layer/hidden 설정에서 남은 예산으로 vocab 최대화."""
        s_zero = _estimate_for_teacher(teacher_key, layer_indices, 0,
                                       hidden_dim=h, intermediate_size=inter)
        remaining = effective_max - s_zero["total_params"]
        max_v = int(remaining / h) if h > 0 else 0
        return min(max(max_v, 0), vocab_cap)

    # 우선순위: hidden_dim 보존 > vocab 최대화 > 레이어 수 (min 유지)
    # min_layers부터 위로 올려가며 원본 hidden_dim 유지 가능한지 확인
    indices = make_uniform_indices(t["num_layers"], min_layers)

    # 1단계: min_layers에서 원본 hidden dim 유지 가능?
    v_budget = _calc_vocab_budget(teacher_key, indices, hidden_dim, inter_size)
    if v_budget >= vocab_floor:
        return {
            "layer_indices": indices,
            "hidden_dim": hidden_dim,
            "intermediate_size": inter_size,
            "target_vocab": v_budget,
            "needs_hidden_reduction": False,
        }

    # 2단계: min_layers에서 hidden dim 축소 + vocab 최대화
    # Binary search: vocab >= vocab_floor 을 보장하면서 최대 hidden_dim
    lo, hi = 64, hidden_dim - 64
    best_h = 64
    while lo <= hi:
        mid = ((lo + hi) // 2 // 64) * 64
        if mid < 64:
            mid = 64
        ratio = mid / hidden_dim
        scaled_inter = max(64, (int(inter_size * ratio) // 64) * 64)
        v_budget = _calc_vocab_budget(teacher_key, indices, mid, scaled_inter)
        if v_budget >= vocab_floor:
            best_h = mid
            lo = mid + 64
        else:
            hi = mid - 64

    best_ratio = best_h / hidden_dim
    best_inter = max(64, (int(inter_size * best_ratio) // 64) * 64)
    v_budget = _calc_vocab_budget(teacher_key, indices, best_h, best_inter)

    return {
        "layer_indices": indices,
        "hidden_dim": best_h,
        "intermediate_size": best_inter,
        "target_vocab": max(v_budget, vocab_floor),
        "needs_hidden_reduction": best_h < hidden_dim,
    }
