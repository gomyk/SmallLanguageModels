---
language: ["ko", "en", "ja", "zh", "es", "fr", "de", "pt", "it", "ru", "ar", "hi", "th", "vi", "id", "tr", "nl", "pl"]
tags:
- sentence-transformers
- multilingual
- layer-pruning
- vocab-pruning
- knowledge-distillation
- modernbert
library_name: sentence-transformers
pipeline_tag: sentence-similarity
license: apache-2.0
---

# modernbert_L6_uniform_distilled (Distilled)

Lightweight sentence encoder created from `answerdotai/ModernBERT-base` via layer pruning + vocabulary pruning + knowledge distillation.

## Model Details

| Property | Value |
|---|---|
| Teacher | answerdotai/ModernBERT-base |
| Architecture | ModernBERT (pruned) |
| Hidden dim | 768 |
| Layers | 6 / 22 |
| Layer indices | [0, 4, 8, 13, 17, 21] |
| Strategy | 6 layers, evenly spaced from ModernBERT (22L) |
| Parameters | 63,870,720 |
| Model size (FP32) | 176.0MB |
| Distilled | Yes |

## Architecture

```
==============================================================
  TEACHER: ModernBERT  →  STUDENT: 6L / 27,279 vocab
==============================================================

            TEACHER                        STUDENT          
  ───────────────────────────    ───────────────────────────

  ┌─────────────────────────┐    ┌─────────────────────────┐
  │   Input Tokens          │    │   Input Tokens          │
  └────────────┬────────────┘    └────────────┬────────────┘
               │                              │
  ┌────────────┴────────────┐    ┌────────────┴────────────┐
  │  Embeddings             │    │  Embeddings (pruned)    │
  │  vocab:  50,368         │    │  vocab:  27,279         │
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
  ├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤    │                         │
  │  Layer  7               │  ╳   │                         │
  ├─────────────────────────┤    ├─────────────────────────┤
  │  Layer  8               │ ──►  │  Layer  2 ← L8         │
  ├─────────────────────────┤    ├─────────────────────────┤
  │  Layer  9               │  ╳   │                         │
  ├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤    │                         │
  │  Layer 10               │  ╳   │                         │
  ├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤    │                         │
  │  Layer 11               │  ╳   │                         │
  ├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤    │                         │
  │  Layer 12               │  ╳   │                         │
  ├─────────────────────────┤    ├─────────────────────────┤
  │  Layer 13               │ ──►  │  Layer  3 ← L13        │
  ├─────────────────────────┤    ├─────────────────────────┤
  │  Layer 14               │  ╳   │                         │
  ├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤    │                         │
  │  Layer 15               │  ╳   │                         │
  ├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤    │                         │
  │  Layer 16               │  ╳   │                         │
  ├─────────────────────────┤    ├─────────────────────────┤
  │  Layer 17               │ ──►  │  Layer  4 ← L17        │
  ├─────────────────────────┤    ├─────────────────────────┤
  │  Layer 18               │  ╳   │                         │
  ├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤    │                         │
  │  Layer 19               │  ╳   │                         │
  ├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤    │                         │
  │  Layer 20               │  ╳   │                         │
  ├─────────────────────────┤    ├─────────────────────────┤
  │  Layer 21               │ ──►  │  Layer  5 ← L21        │
  └────────────┬────────────┘    └────────────┬────────────┘
               │                              │
  ┌────────────┴────────────┐    ┌────────────┴────────────┐
  │  Mean Pooling           │    │  Mean Pooling           │
  │  → 768d embedding       │    │  → 768d embedding       │
  └─────────────────────────┘    └─────────────────────────┘

  Size: 495.8MB (FP32)           →  176.0MB (FP32)
  Params: 129,980,160        →  46,138,368
  Reduction: 64.5%
==============================================================
```

## Quick Start

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("modernbert_L6_uniform_distilled", trust_remote_code=True)

sentences = [
    "Hello, how are you?",
    "안녕하세요",
    "Bonjour, comment allez-vous?",
]

embeddings = model.encode(sentences)
print(embeddings.shape)  # (3, 768)
```

## MTEB Evaluation Results

**Overall Average: 40.35%**

| Task Group | Average |
|---|---|
| Classification | 50.57% |
| Clustering | 29.91% |
| STS | 40.56% |

### Classification

| Task | Average | Details |
|---|---|---|
| AmazonCounterfactualClassification | 66.11% | en: 69.96%, en-ext: 68.7%, de: 63.61% |
| Banking77Classification | 61.69% | default: 61.69% |
| ImdbClassification | 53.73% | default: 53.73% |
| MTOPDomainClassification | 54.47% | en: 71.35%, es: 57.77%, de: 54.3% |
| MassiveIntentClassification | 31.3% | en: 53.06%, zh-CN: 47.36%, zh-TW: 41.65% |
| MassiveScenarioClassification | 32.3% | en: 57.38%, zh-CN: 50.25%, zh-TW: 43.81% |
| ToxicConversationsClassification | 57.77% | default: 57.77% |
| TweetSentimentExtractionClassification | 47.19% | default: 47.19% |

### Clustering

| Task | Average | Details |
|---|---|---|
| ArXivHierarchicalClusteringP2P | 49.8% | default: 49.8% |
| ArXivHierarchicalClusteringS2S | 48.45% | default: 48.45% |
| BiorxivClusteringP2P.v2 | 11.05% | default: 11.05% |
| MedrxivClusteringP2P.v2 | 21.71% | default: 21.71% |
| MedrxivClusteringS2S.v2 | 21.75% | default: 21.75% |
| StackExchangeClustering.v2 | 42.7% | default: 42.7% |
| StackExchangeClusteringP2P.v2 | 32.54% | default: 32.54% |
| TwentyNewsgroupsClustering.v2 | 11.24% | default: 11.24% |

### STS

| Task | Average | Details |
|---|---|---|
| BIOSSES | 42.43% | default: 42.43% |
| SICK-R | 53.89% | default: 53.89% |
| STS12 | 43.95% | default: 43.95% |
| STS13 | 42.51% | default: 42.51% |
| STS14 | 40.74% | default: 40.74% |
| STS15 | 53.89% | default: 53.89% |
| STS17 | 27.51% | en-en: 60.2%, es-es: 58.08%, ko-ko: 44.39% |
| STS22.v2 | 18.53% | zh: 47.25%, es: 39.83%, fr: 35.19% |
| STSBenchmark | 41.58% | default: 41.58% |


## Distillation Impact

| Task | Before | After | Delta |
|---|---|---|---|
| AmazonCounterfactualClassification | 59.33% | 66.11% | +6.78%p |
| ArXivHierarchicalClusteringP2P | 50.19% | 49.8% | -0.39%p |
| ArXivHierarchicalClusteringS2S | 46.96% | 48.45% | +1.49%p |
| Banking77Classification | 35.01% | 61.69% | +26.68%p |
| BiorxivClusteringP2P.v2 | 12.62% | 11.05% | -1.57%p |
| BIOSSES | 33.84% | 42.43% | +8.59%p |
| ImdbClassification | 55.05% | 53.73% | -1.32%p |
| MassiveIntentClassification | 25.86% | 31.3% | +5.44%p |
| MassiveScenarioClassification | 26.28% | 32.3% | +6.02%p |
| MedrxivClusteringP2P.v2 | 22.13% | 21.71% | -0.42%p |
| MedrxivClusteringS2S.v2 | 19.43% | 21.75% | +2.32%p |
| MTOPDomainClassification | 43.24% | 54.47% | +11.23%p |
| SICK-R | 46.99% | 53.89% | +6.9%p |
| StackExchangeClustering.v2 | 34.26% | 42.7% | +8.44%p |
| StackExchangeClusteringP2P.v2 | 31.01% | 32.54% | +1.53%p |
| STS12 | 35.32% | 43.95% | +8.63%p |
| STS13 | 33.7% | 42.51% | +8.81%p |
| STS14 | 37.07% | 40.74% | +3.67%p |
| STS15 | 49.85% | 53.89% | +4.04%p |
| STS17 | 23.34% | 27.51% | +4.17%p |
| STS22.v2 | 24.05% | 18.53% | -5.52%p |
| STSBenchmark | 39.82% | 41.58% | +1.76%p |
| ToxicConversationsClassification | 52.6% | 57.77% | +5.17%p |
| TweetSentimentExtractionClassification | 38.42% | 47.19% | +8.77%p |
| TwentyNewsgroupsClustering.v2 | 9.11% | 11.24% | +2.13%p |


## Training

### Stage 1: Layer Pruning
- Teacher: `answerdotai/ModernBERT-base` (22 layers, 768d)
- Selected layers: `[0, 4, 8, 13, 17, 21]` (6 layers, evenly spaced from ModernBERT (22L))
- Vocabulary pruning applied

### Stage 2: Knowledge Distillation
- **Method**: MSE + Cosine Similarity loss
- **Data**: MTEB Classification/Clustering/STS task datasets
- **Optimizer**: AdamW (lr=2e-5, weight_decay=0.01)
- **Schedule**: Cosine annealing over 3 epochs


## Supported Languages (18)

ko, en, ja, zh, es, fr, de, pt, it, ru, ar, hi, th, vi, id, tr, nl, pl
