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

# gte_L4_uniform

Lightweight sentence encoder created from `alibaba-NLP/gte-multilingual-base` via layer pruning + vocabulary pruning.

## Model Details

| Property | Value |
|---|---|
| Teacher | alibaba-NLP/gte-multilingual-base |
| Architecture | GTE-multilingual (pruned) |
| Hidden dim | 768 |
| Layers | 4 / 12 |
| Layer indices | [0, 4, 7, 11] |
| Strategy | 4 layers, evenly spaced from GTE-multilingual (12L) |
| Parameters | 220,757,760 |
| Model size (FP32) | 277.7MB |
| Distilled | No |

## Architecture

```
==============================================================
  TEACHER: GTE-multilingual  →  STUDENT: 4L / 57,376 vocab
==============================================================

            TEACHER                        STUDENT          
  ───────────────────────────    ───────────────────────────

  ┌─────────────────────────┐    ┌─────────────────────────┐
  │   Input Tokens          │    │   Input Tokens          │
  └────────────┬────────────┘    └────────────┬────────────┘
               │                              │
  ┌────────────┴────────────┐    ┌────────────┴────────────┐
  │  Embeddings             │    │  Embeddings (pruned)    │
  │  vocab: 250,048         │    │  vocab:  57,376         │
  │  dim:  768              │    │  dim:  768              │
  └────────────┬────────────┘    └────────────┬────────────┘
               │                              │
  ┌─────────────────────────┐    ┌─────────────────────────┐
  │  Layer  0               │ ──►  │  Layer  0 ← L0         │
  ├─────────────────────────┤    ├─────────────────────────┤
  │  Layer  1               │  ╳   │                         │
  ├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤    │                         │
  │  Layer  2               │  ╳   │                         │
  ├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤    │                         │
  │  Layer  3               │  ╳   │                         │
  ├─────────────────────────┤    ├─────────────────────────┤
  │  Layer  4               │ ──►  │  Layer  1 ← L4         │
  ├─────────────────────────┤    ├─────────────────────────┤
  │  Layer  5               │  ╳   │                         │
  ├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤    │                         │
  │  Layer  6               │  ╳   │                         │
  ├─────────────────────────┤    ├─────────────────────────┤
  │  Layer  7               │ ──►  │  Layer  2 ← L7         │
  ├─────────────────────────┤    ├─────────────────────────┤
  │  Layer  8               │  ╳   │                         │
  ├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤    │                         │
  │  Layer  9               │  ╳   │                         │
  ├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤    │                         │
  │  Layer 10               │  ╳   │                         │
  ├─────────────────────────┤    ├─────────────────────────┤
  │  Layer 11               │ ──►  │  Layer  3 ← L11        │
  └────────────┬────────────┘    └────────────┬────────────┘
               │                              │
  ┌────────────┴────────────┐    ┌────────────┴────────────┐
  │  Mean Pooling           │    │  Mean Pooling           │
  │  → 768d embedding       │    │  → 768d embedding       │
  └─────────────────────────┘    └─────────────────────────┘

  Size: 1058.2MB (FP32)           →  277.7MB (FP32)
  Params: 277,405,440        →  72,785,664
  Reduction: 73.8%
==============================================================
```

## Quick Start

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("gte_L4_uniform", trust_remote_code=True)

sentences = [
    "Hello, how are you?",
    "안녕하세요",
    "Bonjour, comment allez-vous?",
]

embeddings = model.encode(sentences)
print(embeddings.shape)  # (3, 768)
```

## MTEB Evaluation Results

**Overall Average: 45.57%**

| Task Group | Average |
|---|---|
| Classification | 55.62% |
| Clustering | 30.05% |
| STS | 50.42% |

### Classification

| Task | Average | Details |
|---|---|---|
| AmazonCounterfactualClassification | 65.24% | en: 67.1%, en-ext: 66.57%, de: 65.49% |
| Banking77Classification | 68.58% | default: 68.58% |
| ImdbClassification | 63.28% | default: 63.28% |
| MTOPDomainClassification | 68.67% | en: 78.94%, es: 71.23%, hi: 69.49% |
| MassiveIntentClassification | 35.71% | zh-CN: 56.02%, en: 55.74%, ja: 51.76% |
| MassiveScenarioClassification | 37.58% | zh-CN: 61.24%, en: 59.46%, ja: 53.4% |
| ToxicConversationsClassification | 57.84% | default: 57.84% |
| TweetSentimentExtractionClassification | 48.1% | default: 48.1% |

### Clustering

| Task | Average | Details |
|---|---|---|
| ArXivHierarchicalClusteringP2P | 50.97% | default: 50.97% |
| ArXivHierarchicalClusteringS2S | 43.38% | default: 43.38% |
| BiorxivClusteringP2P.v2 | 20.78% | default: 20.78% |
| MedrxivClusteringP2P.v2 | 26.37% | default: 26.37% |
| MedrxivClusteringS2S.v2 | 20.98% | default: 20.98% |
| StackExchangeClustering.v2 | 34.36% | default: 34.36% |
| StackExchangeClusteringP2P.v2 | 31.55% | default: 31.55% |
| TwentyNewsgroupsClustering.v2 | 12.03% | default: 12.03% |

### STS

| Task | Average | Details |
|---|---|---|
| BIOSSES | 42.61% | default: 42.61% |
| SICK-R | 55.11% | default: 55.11% |
| STS12 | 47.97% | default: 47.97% |
| STS13 | 65.61% | default: 65.61% |
| STS14 | 57.02% | default: 57.02% |
| STS15 | 64.76% | default: 64.76% |
| STS17 | 17.95% | es-es: 68.69%, en-en: 63.86%, ko-ko: 55.96% |
| STS22.v2 | 40.55% | zh: 65.02%, es: 58.47%, it: 55.59% |
| STSBenchmark | 62.23% | default: 62.23% |



## Training

Created via **layer pruning + vocabulary pruning** (no additional training):

1. **Teacher**: `alibaba-NLP/gte-multilingual-base` (12 layers, 768d)
2. **Layer selection**: `[0, 4, 7, 11]` - 4 layers, evenly spaced from GTE-multilingual (12L)
3. **Vocab pruning**: Corpus-based filtering for target languages


## Supported Languages (18)

ko, en, ja, zh, es, fr, de, pt, it, ru, ar, hi, th, vi, id, tr, nl, pl
