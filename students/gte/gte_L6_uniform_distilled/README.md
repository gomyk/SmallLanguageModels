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

# gte_L6_uniform_distilled (Distilled)

Lightweight sentence encoder created from `alibaba-NLP/gte-multilingual-base` via layer pruning + vocabulary pruning + knowledge distillation.

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
| Distilled | Yes |

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

model = SentenceTransformer("gte_L6_uniform_distilled", trust_remote_code=True)

sentences = [
    "Hello, how are you?",
    "안녕하세요",
    "Bonjour, comment allez-vous?",
]

embeddings = model.encode(sentences)
print(embeddings.shape)  # (3, 768)
```

## MTEB Evaluation Results

**Overall Average: 60.55%**

| Task Group | Average |
|---|---|
| Classification | 65.06% |
| Clustering | 39.6% |
| STS | 75.15% |

### Classification

| Task | Average | Details |
|---|---|---|
| AmazonCounterfactualClassification | 72.89% | en-ext: 77.6%, en: 76.42%, de: 71.09% |
| Banking77Classification | 84.6% | default: 84.6% |
| ImdbClassification | 61.96% | default: 61.96% |
| MTOPDomainClassification | 85.06% | en: 91.5%, es: 87.99%, fr: 84.93% |
| MassiveIntentClassification | 42.29% | en: 72.77%, zh-CN: 70.33%, ja: 68.31% |
| MassiveScenarioClassification | 47.31% | en: 77.55%, zh-CN: 76.14%, ja: 73.62% |
| ToxicConversationsClassification | 65.05% | default: 65.05% |
| TweetSentimentExtractionClassification | 61.34% | default: 61.34% |

### Clustering

| Task | Average | Details |
|---|---|---|
| ArXivHierarchicalClusteringP2P | 55.28% | default: 55.28% |
| ArXivHierarchicalClusteringS2S | 50.15% | default: 50.15% |
| BiorxivClusteringP2P.v2 | 31.01% | default: 31.01% |
| MedrxivClusteringP2P.v2 | 32.96% | default: 32.96% |
| MedrxivClusteringS2S.v2 | 30.57% | default: 30.57% |
| StackExchangeClustering.v2 | 47.42% | default: 47.42% |
| StackExchangeClusteringP2P.v2 | 35.8% | default: 35.8% |
| TwentyNewsgroupsClustering.v2 | 33.61% | default: 33.61% |

### STS

| Task | Average | Details |
|---|---|---|
| BIOSSES | 73.46% | default: 73.46% |
| SICK-R | 78.01% | default: 78.01% |
| STS12 | 77.31% | default: 77.31% |
| STS13 | 82.59% | default: 82.59% |
| STS14 | 80.24% | default: 80.24% |
| STS15 | 87.62% | default: 87.62% |
| STS17 | 67.67% | en-en: 86.24%, es-es: 82.71%, ko-ko: 74.85% |
| STS22.v2 | 45.07% | fr-pl: 73.25%, zh: 64.44%, es: 63.42% |
| STSBenchmark | 84.39% | default: 84.39% |


## Distillation Impact

| Task | Before | After | Delta |
|---|---|---|---|
| AmazonCounterfactualClassification | 62.03% | 72.89% | +10.86%p |
| ArXivHierarchicalClusteringP2P | 53.65% | 55.28% | +1.63%p |
| ArXivHierarchicalClusteringS2S | 45.3% | 50.15% | +4.85%p |
| Banking77Classification | 58.65% | 84.6% | +25.95%p |
| BiorxivClusteringP2P.v2 | 21.28% | 31.01% | +9.73%p |
| BIOSSES | 49.91% | 73.46% | +23.55%p |
| ImdbClassification | 63.58% | 61.96% | -1.62%p |
| MassiveIntentClassification | 30.15% | 42.29% | +12.14%p |
| MassiveScenarioClassification | 31.92% | 47.31% | +15.39%p |
| MedrxivClusteringP2P.v2 | 26.07% | 32.96% | +6.89%p |
| MedrxivClusteringS2S.v2 | 21.24% | 30.57% | +9.33%p |
| MTOPDomainClassification | 61.22% | 85.06% | +23.84%p |
| SICK-R | 51.42% | 78.01% | +26.59%p |
| StackExchangeClustering.v2 | 39.07% | 47.42% | +8.35%p |
| StackExchangeClusteringP2P.v2 | 32.7% | 35.8% | +3.1%p |
| STS12 | 39.09% | 77.31% | +38.22%p |
| STS13 | 51.12% | 82.59% | +31.47%p |
| STS14 | 45.69% | 80.24% | +34.55%p |
| STS15 | 60.2% | 87.62% | +27.42%p |
| STS17 | 18.02% | 67.67% | +49.65%p |
| STS22.v2 | 38.98% | 45.07% | +6.09%p |
| STSBenchmark | 54.35% | 84.39% | +30.04%p |
| ToxicConversationsClassification | 57.02% | 65.05% | +8.03%p |
| TweetSentimentExtractionClassification | 45.87% | 61.34% | +15.47%p |
| TwentyNewsgroupsClustering.v2 | 10.91% | 33.61% | +22.7%p |


## Training

### Stage 1: Layer Pruning
- Teacher: `alibaba-NLP/gte-multilingual-base` (12 layers, 768d)
- Selected layers: `[0, 2, 4, 7, 9, 11]` (6 layers, evenly spaced from GTE-multilingual (12L))
- Vocabulary pruning applied

### Stage 2: Knowledge Distillation
- **Method**: MSE + Cosine Similarity loss
- **Data**: MTEB Classification/Clustering/STS task datasets
- **Optimizer**: AdamW (lr=2e-5, weight_decay=0.01)
- **Schedule**: Cosine annealing over 3 epochs


## Supported Languages (18)

ko, en, ja, zh, es, fr, de, pt, it, ru, ar, hi, th, vi, id, tr, nl, pl
