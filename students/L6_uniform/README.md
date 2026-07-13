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

# L6_uniform

Lightweight sentence encoder created from `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` via layer pruning + vocabulary pruning.

## Model Details

| Property | Value |
|---|---|
| Teacher | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 |
| Architecture | MiniLM-L12 (pruned) |
| Hidden dim | 384 |
| Layers | 6 / 12 |
| Layer indices | [0, 2, 4, 7, 9, 11] |
| Strategy | 6 layers, evenly spaced (general-purpose) |
| Parameters | 106,825,344 |
| Model size (FP32) | 98.1MB |
| Distilled | No |

## Architecture

```
==============================================================
  TEACHER: MiniLM-L12  →  STUDENT: 6L / 38,775 vocab
==============================================================

            TEACHER                        STUDENT          
  ───────────────────────────    ───────────────────────────

  ┌─────────────────────────┐    ┌─────────────────────────┐
  │   Input Tokens          │    │   Input Tokens          │
  └────────────┬────────────┘    └────────────┬────────────┘
               │                              │
  ┌────────────┴────────────┐    ┌────────────┴────────────┐
  │  Embeddings             │    │  Embeddings (pruned)    │
  │  vocab: 250,002         │    │  vocab:  38,775         │
  │  dim:  384              │    │  dim:  384              │
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
  │  → 384d embedding       │    │  → 384d embedding       │
  └─────────────────────────┘    └─────────────────────────┘

  Size: 448.0MB (FP32)           →  98.1MB (FP32)
  Params: 117,451,392        →  25,714,176
  Reduction: 78.1%
==============================================================
```

## Quick Start

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("L6_uniform", trust_remote_code=True)

sentences = [
    "Hello, how are you?",
    "안녕하세요",
    "Bonjour, comment allez-vous?",
]

embeddings = model.encode(sentences)
print(embeddings.shape)  # (3, 384)
```

## MTEB Evaluation Results

**Overall Average: 53.16%**

| Task Group | Average |
|---|---|
| Classification | 58.9% |
| Clustering | 34.38% |
| STS | 64.74% |

### Classification

| Task | Average | Details |
|---|---|---|
| AmazonCounterfactualClassification | 68.27% | de: 69.95%, en: 69.4%, en-ext: 68.78% |
| Banking77Classification | 73.53% | default: 73.53% |
| ImdbClassification | 60.64% | default: 60.64% |
| MTOPDomainClassification | 75.11% | en: 84.23%, es: 78.09%, th: 76.25% |
| MassiveIntentClassification | 37.62% | en: 62.82%, zh-CN: 60.31%, ja: 58.99% |
| MassiveScenarioClassification | 41.45% | en: 68.66%, zh-CN: 67.6%, ja: 63.51% |
| ToxicConversationsClassification | 61.36% | default: 61.36% |
| TweetSentimentExtractionClassification | 53.21% | default: 53.21% |

### Clustering

| Task | Average | Details |
|---|---|---|
| ArXivHierarchicalClusteringP2P | 50.12% | default: 50.12% |
| ArXivHierarchicalClusteringS2S | 46.66% | default: 46.66% |
| BiorxivClusteringP2P.v2 | 25.42% | default: 25.42% |
| MedrxivClusteringP2P.v2 | 28.32% | default: 28.32% |
| MedrxivClusteringS2S.v2 | 25.33% | default: 25.33% |
| StackExchangeClustering.v2 | 44.13% | default: 44.13% |
| StackExchangeClusteringP2P.v2 | 33.07% | default: 33.07% |
| TwentyNewsgroupsClustering.v2 | 22.01% | default: 22.01% |

### STS

| Task | Average | Details |
|---|---|---|
| BIOSSES | 57.32% | default: 57.32% |
| SICK-R | 69.91% | default: 69.91% |
| STS12 | 66.88% | default: 66.88% |
| STS13 | 71.42% | default: 71.42% |
| STS14 | 68.52% | default: 68.52% |
| STS15 | 79.84% | default: 79.84% |
| STS17 | 53.52% | en-en: 82.46%, es-es: 78.22%, ko-ko: 66.78% |
| STS22.v2 | 40.57% | zh: 59.49%, es: 58.65%, fr: 57.4% |
| STSBenchmark | 74.69% | default: 74.69% |



## Training

Created via **layer pruning + vocabulary pruning** (no additional training):

1. **Teacher**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (12 layers, 384d)
2. **Layer selection**: `[0, 2, 4, 7, 9, 11]` - 6 layers, evenly spaced (general-purpose)
3. **Vocab pruning**: Corpus-based filtering for target languages


## Supported Languages (18)

ko, en, ja, zh, es, fr, de, pt, it, ru, ar, hi, th, vi, id, tr, nl, pl
