---
language: ["ko", "en", "ja", "zh", "es", "fr", "de", "pt", "it", "ru", "ar", "hi", "th", "vi", "id", "tr", "nl", "pl"]
tags:
- sentence-transformers
- multilingual
- layer-pruning
- vocab-pruning
- gte-multilingual
library_name: sentence-transformers
pipeline_tag: sentence-similarity
license: apache-2.0
---

# gte_L6_uniform

Lightweight sentence encoder created from `alibaba-NLP/gte-multilingual-base` via layer pruning + vocabulary pruning.

## Model Details

| Property | Value |
|---|---|
| Teacher | alibaba-NLP/gte-multilingual-base |
| Architecture | GTE-multilingual (pruned) |
| Hidden dim | 768 |
| Layers | 6 / 12 |
| Layer indices | [0, 2, 4, 7, 9, 11] |
| Strategy | 6 layers, evenly spaced from GTE-multilingual (12L) |
| Parameters | 234,919,680 |
| Model size (FP32) | 349.7MB |
| Distilled | No |

## Architecture

```
==============================================================
  TEACHER: GTE-multilingual  →  STUDENT: 6L / 63,531 vocab
==============================================================

            TEACHER                        STUDENT          
  ───────────────────────────    ───────────────────────────

  ┌─────────────────────────┐    ┌─────────────────────────┐
  │   Input Tokens          │    │   Input Tokens          │
  └────────────┬────────────┘    └────────────┬────────────┘
               │                              │
  ┌────────────┴────────────┐    ┌────────────┴────────────┐
  │  Embeddings             │    │  Embeddings (pruned)    │
  │  vocab: 250,048         │    │  vocab:  63,531         │
  │  dim:  768              │    │  dim:  768              │
  └────────────┬────────────┘    └────────────┬────────────┘
               │                              │
  ┌─────────────────────────┐    ┌─────────────────────────┐
  │  Layer  0               │ ──►  │  Layer  0 ← L0         │
  ├─────────────────────────┤    ├─────────────────────────┤
  │  Layer  1               │  ╳   │                         │
  ├─────────────────────────┤    ├─────────────────────────┤
  │  Layer  2               │ ──►  │  Layer  1 ← L2         │
  ├─────────────────────────┤    ├─────────────────────────┤
  │  Layer  3               │  ╳   │                         │
  ├─────────────────────────┤    ├─────────────────────────┤
  │  Layer  4               │ ──►  │  Layer  2 ← L4         │
  ├─────────────────────────┤    ├─────────────────────────┤
  │  Layer  5               │  ╳   │                         │
  ├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤    │                         │
  │  Layer  6               │  ╳   │                         │
  ├─────────────────────────┤    ├─────────────────────────┤
  │  Layer  7               │ ──►  │  Layer  3 ← L7         │
  ├─────────────────────────┤    ├─────────────────────────┤
  │  Layer  8               │  ╳   │                         │
  ├─────────────────────────┤    ├─────────────────────────┤
  │  Layer  9               │ ──►  │  Layer  4 ← L9         │
  ├─────────────────────────┤    ├─────────────────────────┤
  │  Layer 10               │  ╳   │                         │
  ├─────────────────────────┤    ├─────────────────────────┤
  │  Layer 11               │ ──►  │  Layer  5 ← L11        │
  └────────────┬────────────┘    └────────────┬────────────┘
               │                              │
  ┌────────────┴────────────┐    ┌────────────┴────────────┐
  │  Mean Pooling           │    │  Mean Pooling           │
  │  → 768d embedding       │    │  → 768d embedding       │
  └─────────────────────────┘    └─────────────────────────┘

  Size: 1058.2MB (FP32)           →  349.7MB (FP32)
  Params: 277,405,440        →  91,674,624
  Reduction: 67.0%
==============================================================
```

## Quick Start

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("gte_L6_uniform", trust_remote_code=True)

sentences = [
    "Hello, how are you?",
    "안녕하세요",
    "Bonjour, comment allez-vous?",
]

embeddings = model.encode(sentences)
print(embeddings.shape)  # (3, 768)
```

## MTEB Evaluation Results

**Overall Average: 42.78%**

| Task Group | Average |
|---|---|
| Classification | 51.3% |
| Clustering | 31.28% |
| STS | 45.42% |

### Classification

| Task | Average | Details |
|---|---|---|
| AmazonCounterfactualClassification | 62.03% | en: 65.04%, en-ext: 63.25%, de: 62.73% |
| Banking77Classification | 58.65% | default: 58.65% |
| ImdbClassification | 63.58% | default: 63.58% |
| MTOPDomainClassification | 61.22% | en: 70.06%, es: 64.08%, hi: 60.95% |
| MassiveIntentClassification | 30.15% | zh-CN: 49.78%, en: 47.57%, ja: 45.14% |
| MassiveScenarioClassification | 31.92% | zh-CN: 54.49%, en: 50.72%, ja: 47.17% |
| ToxicConversationsClassification | 57.02% | default: 57.02% |
| TweetSentimentExtractionClassification | 45.87% | default: 45.87% |

### Clustering

| Task | Average | Details |
|---|---|---|
| ArXivHierarchicalClusteringP2P | 53.65% | default: 53.65% |
| ArXivHierarchicalClusteringS2S | 45.3% | default: 45.3% |
| BiorxivClusteringP2P.v2 | 21.28% | default: 21.28% |
| MedrxivClusteringP2P.v2 | 26.07% | default: 26.07% |
| MedrxivClusteringS2S.v2 | 21.24% | default: 21.24% |
| StackExchangeClustering.v2 | 39.07% | default: 39.07% |
| StackExchangeClusteringP2P.v2 | 32.7% | default: 32.7% |
| TwentyNewsgroupsClustering.v2 | 10.91% | default: 10.91% |

### STS

| Task | Average | Details |
|---|---|---|
| BIOSSES | 49.91% | default: 49.91% |
| SICK-R | 51.42% | default: 51.42% |
| STS12 | 39.09% | default: 39.09% |
| STS13 | 51.12% | default: 51.12% |
| STS14 | 45.69% | default: 45.69% |
| STS15 | 60.2% | default: 60.2% |
| STS17 | 18.02% | es-es: 61.34%, en-en: 59.81%, ko-ko: 50.21% |
| STS22.v2 | 38.98% | zh: 62.9%, es: 58.01%, fr: 55.34% |
| STSBenchmark | 54.35% | default: 54.35% |



## Training

Created via **layer pruning + vocabulary pruning** (no additional training):

1. **Teacher**: `alibaba-NLP/gte-multilingual-base` (12 layers, 768d)
2. **Layer selection**: `[0, 2, 4, 7, 9, 11]` - 6 layers, evenly spaced from GTE-multilingual (12L)
3. **Vocab pruning**: Corpus-based filtering for target languages


## Supported Languages (18)

ko, en, ja, zh, es, fr, de, pt, it, ru, ar, hi, th, vi, id, tr, nl, pl
