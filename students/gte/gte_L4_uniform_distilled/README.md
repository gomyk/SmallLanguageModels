---
language: ["ko", "en", "ja", "zh", "es", "fr", "de", "pt", "it", "ru", "ar", "hi", "th", "vi", "id", "tr", "nl", "pl"]
tags:
- sentence-transformers
- multilingual
- layer-pruning
- vocab-pruning
- knowledge-distillation
- gte-multilingual
library_name: sentence-transformers
pipeline_tag: sentence-similarity
license: apache-2.0
---

# gte_L4_uniform_distilled (Distilled)

Lightweight sentence encoder created from `alibaba-NLP/gte-multilingual-base` via layer pruning + vocabulary pruning + knowledge distillation.

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
| Distilled | Yes |

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

model = SentenceTransformer("gte_L4_uniform_distilled", trust_remote_code=True)

sentences = [
    "Hello, how are you?",
    "안녕하세요",
    "Bonjour, comment allez-vous?",
]

embeddings = model.encode(sentences)
print(embeddings.shape)  # (3, 768)
```

## MTEB Evaluation Results

**Overall Average: 56.97%**

| Task Group | Average |
|---|---|
| Classification | 63.0% |
| Clustering | 35.73% |
| STS | 70.49% |

### Classification

| Task | Average | Details |
|---|---|---|
| AmazonCounterfactualClassification | 69.34% | en-ext: 74.06%, en: 72.1%, de: 69.15% |
| Banking77Classification | 82.7% | default: 82.7% |
| ImdbClassification | 60.94% | default: 60.94% |
| MTOPDomainClassification | 81.66% | en: 89.15%, es: 84.21%, fr: 82.21% |
| MassiveIntentClassification | 40.9% | en: 70.82%, zh-CN: 68.47%, ja: 66.5% |
| MassiveScenarioClassification | 46.01% | en: 76.68%, zh-CN: 75.65%, ja: 72.94% |
| ToxicConversationsClassification | 63.41% | default: 63.41% |
| TweetSentimentExtractionClassification | 59.05% | default: 59.05% |

### Clustering

| Task | Average | Details |
|---|---|---|
| ArXivHierarchicalClusteringP2P | 52.87% | default: 52.87% |
| ArXivHierarchicalClusteringS2S | 47.21% | default: 47.21% |
| BiorxivClusteringP2P.v2 | 25.97% | default: 25.97% |
| MedrxivClusteringP2P.v2 | 29.68% | default: 29.68% |
| MedrxivClusteringS2S.v2 | 24.9% | default: 24.9% |
| StackExchangeClustering.v2 | 43.5% | default: 43.5% |
| StackExchangeClusteringP2P.v2 | 34.78% | default: 34.78% |
| TwentyNewsgroupsClustering.v2 | 26.93% | default: 26.93% |

### STS

| Task | Average | Details |
|---|---|---|
| BIOSSES | 67.24% | default: 67.24% |
| SICK-R | 73.92% | default: 73.92% |
| STS12 | 73.6% | default: 73.6% |
| STS13 | 76.98% | default: 76.98% |
| STS14 | 75.26% | default: 75.26% |
| STS15 | 84.9% | default: 84.9% |
| STS17 | 58.52% | en-en: 83.29%, es-es: 79.59%, ko-ko: 70.34% |
| STS22.v2 | 44.26% | zh: 68.37%, es: 61.22%, it: 60.83% |
| STSBenchmark | 79.77% | default: 79.77% |


## Distillation Impact

| Task | Before | After | Delta |
|---|---|---|---|
| AmazonCounterfactualClassification | 65.24% | 69.34% | +4.1%p |
| ArXivHierarchicalClusteringP2P | 50.97% | 52.87% | +1.9%p |
| ArXivHierarchicalClusteringS2S | 43.38% | 47.21% | +3.83%p |
| Banking77Classification | 68.58% | 82.7% | +14.12%p |
| BiorxivClusteringP2P.v2 | 20.78% | 25.97% | +5.19%p |
| BIOSSES | 42.61% | 67.24% | +24.63%p |
| ImdbClassification | 63.28% | 60.94% | -2.34%p |
| MassiveIntentClassification | 35.71% | 40.9% | +5.19%p |
| MassiveScenarioClassification | 37.58% | 46.01% | +8.43%p |
| MedrxivClusteringP2P.v2 | 26.37% | 29.68% | +3.31%p |
| MedrxivClusteringS2S.v2 | 20.98% | 24.9% | +3.92%p |
| MTOPDomainClassification | 68.67% | 81.66% | +12.99%p |
| SICK-R | 55.11% | 73.92% | +18.81%p |
| StackExchangeClustering.v2 | 34.36% | 43.5% | +9.14%p |
| StackExchangeClusteringP2P.v2 | 31.55% | 34.78% | +3.23%p |
| STS12 | 47.97% | 73.6% | +25.63%p |
| STS13 | 65.61% | 76.98% | +11.37%p |
| STS14 | 57.02% | 75.26% | +18.24%p |
| STS15 | 64.76% | 84.9% | +20.14%p |
| STS17 | 17.95% | 58.52% | +40.57%p |
| STS22.v2 | 40.55% | 44.26% | +3.71%p |
| STSBenchmark | 62.23% | 79.77% | +17.54%p |
| ToxicConversationsClassification | 57.84% | 63.41% | +5.57%p |
| TweetSentimentExtractionClassification | 48.1% | 59.05% | +10.95%p |
| TwentyNewsgroupsClustering.v2 | 12.03% | 26.93% | +14.9%p |


## Training

### Stage 1: Layer Pruning
- Teacher: `alibaba-NLP/gte-multilingual-base` (12 layers, 768d)
- Selected layers: `[0, 4, 7, 11]` (4 layers, evenly spaced from GTE-multilingual (12L))
- Vocabulary pruning applied

### Stage 2: Knowledge Distillation
- **Method**: MSE + Cosine Similarity loss
- **Data**: MTEB Classification/Clustering/STS task datasets
- **Optimizer**: AdamW (lr=2e-5, weight_decay=0.01)
- **Schedule**: Cosine annealing over 3 epochs


## Supported Languages (18)

ko, en, ja, zh, es, fr, de, pt, it, ru, ar, hi, th, vi, id, tr, nl, pl
