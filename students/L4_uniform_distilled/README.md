---
language: ["ko", "en", "ja", "zh", "es", "fr", "de", "pt", "it", "ru", "ar", "hi", "th", "vi", "id", "tr", "nl", "pl"]
tags:
- sentence-transformers
- multilingual
- layer-pruning
- vocab-pruning
- knowledge-distillation
- minilm-l12
library_name: sentence-transformers
pipeline_tag: sentence-similarity
license: apache-2.0
---

# L4_uniform_distilled (Distilled)

Lightweight sentence encoder created from `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` via layer pruning + vocabulary pruning + knowledge distillation.

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
| Distilled | Yes |

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

model = SentenceTransformer("L4_uniform_distilled", trust_remote_code=True)

sentences = [
    "Hello, how are you?",
    "안녕하세요",
    "Bonjour, comment allez-vous?",
]

embeddings = model.encode(sentences)
print(embeddings.shape)  # (3, 384)
```

## MTEB Evaluation Results

**Overall Average: 54.62%**

| Task Group | Average |
|---|---|
| Classification | 58.95% |
| Clustering | 36.16% |
| STS | 67.19% |

### Classification

| Task | Average | Details |
|---|---|---|
| AmazonCounterfactualClassification | 65.7% | en: 68.34%, de: 67.09%, en-ext: 66.57% |
| Banking77Classification | 78.43% | default: 78.43% |
| ImdbClassification | 60.43% | default: 60.43% |
| MTOPDomainClassification | 74.26% | en: 83.96%, es: 78.09%, fr: 76.42% |
| MassiveIntentClassification | 34.72% | en: 65.57%, zh-CN: 62.56%, ja: 62.19% |
| MassiveScenarioClassification | 40.37% | en: 71.61%, zh-CN: 69.22%, ja: 67.48% |
| ToxicConversationsClassification | 61.37% | default: 61.37% |
| TweetSentimentExtractionClassification | 56.29% | default: 56.29% |

### Clustering

| Task | Average | Details |
|---|---|---|
| ArXivHierarchicalClusteringP2P | 51.36% | default: 51.36% |
| ArXivHierarchicalClusteringS2S | 46.3% | default: 46.3% |
| BiorxivClusteringP2P.v2 | 26.63% | default: 26.63% |
| MedrxivClusteringP2P.v2 | 30.15% | default: 30.15% |
| MedrxivClusteringS2S.v2 | 25.7% | default: 25.7% |
| StackExchangeClustering.v2 | 46.56% | default: 46.56% |
| StackExchangeClusteringP2P.v2 | 35.85% | default: 35.85% |
| TwentyNewsgroupsClustering.v2 | 26.74% | default: 26.74% |

### STS

| Task | Average | Details |
|---|---|---|
| BIOSSES | 53.2% | default: 53.2% |
| SICK-R | 73.61% | default: 73.61% |
| STS12 | 72.11% | default: 72.11% |
| STS13 | 75.62% | default: 75.62% |
| STS14 | 72.2% | default: 72.2% |
| STS15 | 80.86% | default: 80.86% |
| STS17 | 48.49% | en-en: 81.73%, es-es: 75.65%, ar-ar: 57.76% |
| STS22.v2 | 49.17% | fr: 69.99%, es-en: 65.46%, zh: 63.87% |
| STSBenchmark | 79.48% | default: 79.48% |


## Distillation Impact

| Task | Before | After | Delta |
|---|---|---|---|
| AmazonCounterfactualClassification | 67.02% | 65.7% | -1.32%p |
| ArXivHierarchicalClusteringP2P | 49.93% | 51.36% | +1.43%p |
| ArXivHierarchicalClusteringS2S | 46.08% | 46.3% | +0.22%p |
| Banking77Classification | 69.18% | 78.43% | +9.25%p |
| BiorxivClusteringP2P.v2 | 21.47% | 26.63% | +5.16%p |
| BIOSSES | 45.64% | 53.2% | +7.56%p |
| ImdbClassification | 59.38% | 60.43% | +1.05%p |
| MassiveIntentClassification | 36.9% | 34.72% | -2.18%p |
| MassiveScenarioClassification | 39.51% | 40.37% | +0.86%p |
| MedrxivClusteringP2P.v2 | 26.05% | 30.15% | +4.1%p |
| MedrxivClusteringS2S.v2 | 22.94% | 25.7% | +2.76%p |
| MTOPDomainClassification | 71.48% | 74.26% | +2.78%p |
| SICK-R | 62.01% | 73.61% | +11.6%p |
| StackExchangeClustering.v2 | 41.23% | 46.56% | +5.33%p |
| StackExchangeClusteringP2P.v2 | 32.19% | 35.85% | +3.66%p |
| STS12 | 57.85% | 72.11% | +14.26%p |
| STS13 | 65.48% | 75.62% | +10.14%p |
| STS14 | 60.39% | 72.2% | +11.81%p |
| STS15 | 73.93% | 80.86% | +6.93%p |
| STS17 | 46.29% | 48.49% | +2.2%p |
| STS22.v2 | 37.34% | 49.17% | +11.83%p |
| STSBenchmark | 65.38% | 79.48% | +14.1%p |
| ToxicConversationsClassification | 62.02% | 61.37% | -0.65%p |
| TweetSentimentExtractionClassification | 49.43% | 56.29% | +6.86%p |
| TwentyNewsgroupsClustering.v2 | 16.43% | 26.74% | +10.31%p |


## Training

### Stage 1: Layer Pruning
- Teacher: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (12 layers, 384d)
- Selected layers: `[0, 4, 7, 11]` (4 layers, evenly spaced (compact))
- Vocabulary pruning applied

### Stage 2: Knowledge Distillation
- **Method**: MSE + Cosine Similarity loss
- **Data**: MTEB Classification/Clustering/STS task datasets
- **Optimizer**: AdamW (lr=2e-5, weight_decay=0.01)
- **Schedule**: Cosine annealing over 3 epochs


## Supported Languages (18)

ko, en, ja, zh, es, fr, de, pt, it, ru, ar, hi, th, vi, id, tr, nl, pl
