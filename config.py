"""
Intent Classifier Student Model - Experiment Configuration

Teacher 모델에서 layer pruning으로 다양한 student 후보를 생성하고
MTEB 벤치마크로 최적 구조를 탐색한다.
"""

# ── Teacher Model ──────────────────────────────────────────────
TEACHER_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TEACHER_HIDDEN_DIM = 384
TEACHER_NUM_LAYERS = 12

# ── Student Experiments ────────────────────────────────────────
# 각 실험은 teacher의 12개 레이어 중 어떤 레이어를 가져올지 정의
# "uniform": 균등 간격, "top": 상위 레이어, "bottom": 하위 레이어
EXPERIMENTS = [
    {
        "name": "L6_uniform",
        "description": "6 layers, evenly spaced (general-purpose)",
        "layer_indices": [0, 2, 4, 7, 9, 11],
    },
    {
        "name": "L4_uniform",
        "description": "4 layers, evenly spaced (compact)",
        "layer_indices": [0, 4, 7, 11],
    },
    {
        "name": "L3_uniform",
        "description": "3 layers, evenly spaced (ultra-compact)",
        "layer_indices": [0, 6, 11],
    },
    {
        "name": "L6_top",
        "description": "6 layers, top half (semantic-focused)",
        "layer_indices": [6, 7, 8, 9, 10, 11],
    },
    {
        "name": "L4_top",
        "description": "4 layers, top quarter (semantic-focused compact)",
        "layer_indices": [8, 9, 10, 11],
    },
    {
        "name": "L6_bottom",
        "description": "6 layers, bottom half (syntactic-focused)",
        "layer_indices": [0, 1, 2, 3, 4, 5],
    },
    {
        "name": "L2_ends",
        "description": "2 layers, first + last (minimal)",
        "layer_indices": [0, 11],
    },
]

# ── 50MB Compact Experiments (20K vocab) ─────────────────────
COMPACT_EXPERIMENTS = [
    {
        "name": "L3_compact",
        "description": "3 layers uniform + 20K vocab (~50MB target)",
        "layer_indices": [0, 6, 11],
    },
    {
        "name": "L4_compact",
        "description": "4 layers uniform + 20K vocab (~57MB target)",
        "layer_indices": [0, 4, 7, 11],
    },
    {
        "name": "L6_compact",
        "description": "6 layers bottom + 20K vocab (~71MB target)",
        "layer_indices": [0, 1, 2, 3, 4, 5],
    },
]

# ── Target Languages (18개) ───────────────────────────────────
TARGET_LANGUAGES = [
    "ko", "en", "ja", "zh", "es", "fr", "de", "pt",
    "it", "ru", "ar", "hi", "th", "vi", "id", "tr", "nl", "pl",
]

# ISO 639-3 mapping (MTEB에서 사용)
LANG_TO_ISO3 = {
    "ko": "kor", "en": "eng", "ja": "jpn", "zh": "zho",
    "es": "spa", "fr": "fra", "de": "deu", "pt": "por",
    "it": "ita", "ru": "rus", "ar": "ara", "hi": "hin",
    "th": "tha", "vi": "vie", "id": "ind", "tr": "tur",
    "nl": "nld", "pl": "pol",
}

# ── MTEB Evaluation ───────────────────────────────────────────
# 다국어 분류 성능에 가장 관련 높은 MTEB 태스크
MTEB_TASKS = [
    "MassiveIntentClassification",     # 60개 의도, 51개 언어 - 직접적 관련
    "MassiveScenarioClassification",   # 18개 시나리오, 51개 언어
]

# 빠른 평가용 언어 서브셋 (다양한 문자 체계 커버, ISO 639-3)
QUICK_EVAL_LANGS = ["eng", "kor", "zho", "spa", "ara"]

# ── Paths ─────────────────────────────────────────────────────
STUDENTS_DIR = "students"
RESULTS_DIR = "results"
EXPORT_DIR = "exported"
