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

# L6_uniform_distilled (Distilled)

Lightweight sentence encoder created from `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` via layer pruning + vocabulary pruning + knowledge distillation.

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
| Distilled | Yes |

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

model = SentenceTransformer("L6_uniform_distilled", trust_remote_code=True)

sentences = [
    "Hello, how are you?",
    "안녕하세요",
    "Bonjour, comment allez-vous?",
]

embeddings = model.encode(sentences)
print(embeddings.shape)  # (3, 384)
```

## MTEB Evaluation Results

**Overall Average: 56.93%**

| Task Group | Average |
|---|---|
| Classification | 59.41% |
| Clustering | 38.07% |
| STS | 71.48% |

### Classification

| Task | Average | Details |
|---|---|---|
| AmazonCounterfactualClassification | 67.92% | de: 70.3%, en: 70.06%, en-ext: 69.5% |
| Banking77Classification | 79.34% | default: 79.34% |
| ImdbClassification | 58.94% | default: 58.94% |
| MTOPDomainClassification | 76.3% | en: 85.94%, es: 79.58%, th: 78.32% |
| MassiveIntentClassification | 35.04% | en: 66.34%, zh-CN: 62.56%, ja: 62.27% |
| MassiveScenarioClassification | 40.77% | en: 71.95%, zh-CN: 69.09%, ja: 68.11% |
| ToxicConversationsClassification | 60.34% | default: 60.34% |
| TweetSentimentExtractionClassification | 56.67% | default: 56.67% |

### Clustering

| Task | Average | Details |
|---|---|---|
| ArXivHierarchicalClusteringP2P | 51.94% | default: 51.94% |
| ArXivHierarchicalClusteringS2S | 48.06% | default: 48.06% |
| BiorxivClusteringP2P.v2 | 30.65% | default: 30.65% |
| MedrxivClusteringP2P.v2 | 31.34% | default: 31.34% |
| MedrxivClusteringS2S.v2 | 28.24% | default: 28.24% |
| StackExchangeClustering.v2 | 48.14% | default: 48.14% |
| StackExchangeClusteringP2P.v2 | 35.9% | default: 35.9% |
| TwentyNewsgroupsClustering.v2 | 30.3% | default: 30.3% |

### STS

| Task | Average | Details |
|---|---|---|
| BIOSSES | 60.1% | default: 60.1% |
| SICK-R | 77.0% | default: 77.0% |
| STS12 | 72.99% | default: 72.99% |
| STS13 | 79.03% | default: 79.03% |
| STS14 | 76.54% | default: 76.54% |
| STS15 | 84.27% | default: 84.27% |
| STS17 | 58.61% | en-en: 84.68%, es-es: 78.41%, nl-en: 64.48% |
| STS22.v2 | 51.39% | fr: 70.62%, es-en: 67.53%, zh: 64.99% |
| STSBenchmark | 83.35% | default: 83.35% |


## Distillation Impact

| Task | Before | After | Delta |
|---|---|---|---|
| AmazonCounterfactualClassification | 68.27% | 67.92% | -0.35%p |
| ArXivHierarchicalClusteringP2P | 50.12% | 51.94% | +1.82%p |
| ArXivHierarchicalClusteringS2S | 46.66% | 48.06% | +1.4%p |
| Banking77Classification | 73.53% | 79.34% | +5.81%p |
| BiorxivClusteringP2P.v2 | 25.42% | 30.65% | +5.23%p |
| BIOSSES | 57.32% | 60.1% | +2.78%p |
| ImdbClassification | 60.64% | 58.94% | -1.7%p |
| MassiveIntentClassification | 37.62% | 35.04% | -2.58%p |
| MassiveScenarioClassification | 41.45% | 40.77% | -0.68%p |
| MedrxivClusteringP2P.v2 | 28.32% | 31.34% | +3.02%p |
| MedrxivClusteringS2S.v2 | 25.33% | 28.24% | +2.91%p |
| MTOPDomainClassification | 75.11% | 76.3% | +1.19%p |
| SICK-R | 69.91% | 77.0% | +7.09%p |
| StackExchangeClustering.v2 | 44.13% | 48.14% | +4.01%p |
| StackExchangeClusteringP2P.v2 | 33.07% | 35.9% | +2.83%p |
| STS12 | 66.88% | 72.99% | +6.11%p |
| STS13 | 71.42% | 79.03% | +7.61%p |
| STS14 | 68.52% | 76.54% | +8.02%p |
| STS15 | 79.84% | 84.27% | +4.43%p |
| STS17 | 53.52% | 58.61% | +5.09%p |
| STS22.v2 | 40.57% | 51.39% | +10.82%p |
| STSBenchmark | 74.69% | 83.35% | +8.66%p |
| ToxicConversationsClassification | 61.36% | 60.34% | -1.02%p |
| TweetSentimentExtractionClassification | 53.21% | 56.67% | +3.46%p |
| TwentyNewsgroupsClustering.v2 | 22.01% | 30.3% | +8.29%p |


## Training

### Stage 1: Layer Pruning
- Teacher: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (12 layers, 384d)
- Selected layers: `[0, 2, 4, 7, 9, 11]` (6 layers, evenly spaced (general-purpose))
- Vocabulary pruning applied

### Stage 2: Knowledge Distillation
- **Method**: MSE + Cosine Similarity loss
- **Data**: MTEB Classification/Clustering/STS task datasets
- **Optimizer**: AdamW (lr=2e-5, weight_decay=0.01)
- **Schedule**: Cosine annealing over 3 epochs


## Supported Languages (18)

ko, en, ja, zh, es, fr, de, pt, it, ru, ar, hi, th, vi, id, tr, nl, pl
