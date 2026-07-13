---
language: ["ko", "en", "ja", "zh", "es", "fr", "de", "pt", "it", "ru", "ar", "hi", "th", "vi", "id", "tr", "nl", "pl"]
tags:
- sentence-transformers
- multilingual
- layer-pruning
- vocab-pruning
- minilm-l12
library_name: sentence-transformers
pipeline_tag: sentence-similarity
license: apache-2.0
---

# L4_uniform

Lightweight sentence encoder created from `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` via layer pruning + vocabulary pruning.

## Model Details

| Property | Value |
|---|---|
| Teacher | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 |
| Architecture | MiniLM-L12 (pruned) |
| Hidden dim | 384 |
| Layers | 4 / 12 |
| Layer indices | [0, 4, 7, 11] |
| Strategy | 4 layers, evenly spaced (compact) |
| Parameters | 103,283,328 |
| Model size (FP32) | 84.6MB |
| Distilled | No |

## Architecture

```
==============================================================
  TEACHER: MiniLM-L12  →  STUDENT: 4L / 38,755 vocab
==============================================================

            TEACHER                        STUDENT          
  ───────────────────────────    ───────────────────────────

  ┌─────────────────────────┐    ┌─────────────────────────┐
  │   Input Tokens          │    │   Input Tokens          │
  └────────────┬────────────┘    └────────────┬────────────┘
               │                              │
  ┌────────────┴────────────┐    ┌────────────┴────────────┐
  │  Embeddings             │    │  Embeddings (pruned)    │
  │  vocab: 250,002         │    │  vocab:  38,755         │
  │  dim:  384              │    │  dim:  384              │
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
  │  → 384d embedding       │    │  → 384d embedding       │
  └─────────────────────────┘    └─────────────────────────┘

  Size: 448.0MB (FP32)           →  84.6MB (FP32)
  Params: 117,451,392        →  22,164,480
  Reduction: 81.1%
==============================================================
```

## Quick Start

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("L4_uniform", trust_remote_code=True)

sentences = [
    "Hello, how are you?",
    "안녕하세요",
    "Bonjour, comment allez-vous?",
]

embeddings = model.encode(sentences)
print(embeddings.shape)  # (3, 384)
```

## MTEB Evaluation Results

**Overall Average: 49.02%**

| Task Group | Average |
|---|---|
| Classification | 56.87% |
| Clustering | 32.04% |
| STS | 57.15% |

### Classification

| Task | Average | Details |
|---|---|---|
| AmazonCounterfactualClassification | 67.02% | en: 70.31%, en-ext: 68.1%, de: 65.73% |
| Banking77Classification | 69.18% | default: 69.18% |
| ImdbClassification | 59.38% | default: 59.38% |
| MTOPDomainClassification | 71.48% | en: 80.02%, es: 73.78%, hi: 71.07% |
| MassiveIntentClassification | 36.9% | en: 58.41%, zh-CN: 58.07%, ja: 56.73% |
| MassiveScenarioClassification | 39.51% | zh-CN: 63.96%, en: 62.71%, ja: 59.84% |
| ToxicConversationsClassification | 62.02% | default: 62.02% |
| TweetSentimentExtractionClassification | 49.43% | default: 49.43% |

### Clustering

| Task | Average | Details |
|---|---|---|
| ArXivHierarchicalClusteringP2P | 49.93% | default: 49.93% |
| ArXivHierarchicalClusteringS2S | 46.08% | default: 46.08% |
| BiorxivClusteringP2P.v2 | 21.47% | default: 21.47% |
| MedrxivClusteringP2P.v2 | 26.05% | default: 26.05% |
| MedrxivClusteringS2S.v2 | 22.94% | default: 22.94% |
| StackExchangeClustering.v2 | 41.23% | default: 41.23% |
| StackExchangeClusteringP2P.v2 | 32.19% | default: 32.19% |
| TwentyNewsgroupsClustering.v2 | 16.43% | default: 16.43% |

### STS

| Task | Average | Details |
|---|---|---|
| BIOSSES | 45.64% | default: 45.64% |
| SICK-R | 62.01% | default: 62.01% |
| STS12 | 57.85% | default: 57.85% |
| STS13 | 65.48% | default: 65.48% |
| STS14 | 60.39% | default: 60.39% |
| STS15 | 73.93% | default: 73.93% |
| STS17 | 46.29% | en-en: 76.54%, es-es: 75.88%, ko-ko: 62.72% |
| STS22.v2 | 37.34% | zh: 57.86%, es: 54.85%, fr: 51.41% |
| STSBenchmark | 65.38% | default: 65.38% |



## Training

Created via **layer pruning + vocabulary pruning** (no additional training):

1. **Teacher**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (12 layers, 384d)
2. **Layer selection**: `[0, 4, 7, 11]` - 4 layers, evenly spaced (compact)
3. **Vocab pruning**: Corpus-based filtering for target languages


## Supported Languages (18)

ko, en, ja, zh, es, fr, de, pt, it, ru, ar, hi, th, vi, id, tr, nl, pl
