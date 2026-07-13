---
language: ["ko", "en", "ja", "zh", "es", "fr", "de", "pt", "it", "ru", "ar", "hi", "th", "vi", "id", "pl"]
tags:
- sentence-transformers
- multilingual
- model-compression
- layer-pruning
- vocab-pruning
- knowledge-distillation
- me5-small
library_name: sentence-transformers
pipeline_tag: sentence-similarity
license: apache-2.0
---

# me5s_compressed_v3_distilled (Distilled)

Compact multilingual sentence encoder compressed from `intfloat/multilingual-e5-small` (9x compression).

## Model Details

| Property | Value |
|---|---|
| Base model | `intfloat/multilingual-e5-small` |
| Architecture | bert (encoder) |
| Hidden dim | 384 (from 384) |
| Layers | 4 (from 12) |
| Intermediate | 1536 |
| Attention heads | 12 |
| Vocab size | 15,424 (from 250,037) |
| Parameters | ~13.2M |
| Model size (FP32) | 51.0MB |
| Compression | 9x |
| Distilled | Yes |

## Quick Start

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("me5s_compressed_v3_distilled", trust_remote_code=True)

sentences = [
    "Hello, how are you?",
    "안녕하세요, 잘 지내세요?",
    "こんにちは、元気ですか？",
    "你好，你好吗？",
]

embeddings = model.encode(sentences)
print(embeddings.shape)  # (4, 384)
```

## MTEB Evaluation Results

**Overall Average: 54.46%**

| Task Group | Average |
|---|---|
| Classification | 58.28% |
| Clustering | 31.08% |
| STS | 70.1% |

### Classification

| Task | Average | Details |
|---|---|---|
| AmazonCounterfactualClassification | 68.95% | de: 72.16%, en-ext: 71.78%, en: 71.12%, ja: 60.74% |
| Banking77Classification | 66.96% | default: 66.96% |
| ImdbClassification | 59.98% | default: 59.98% |
| MTOPDomainClassification | 81.96% | en: 86.71%, es: 84.21%, hi: 81.69%, th: 80.77%, de: 80.09% |
| MassiveIntentClassification | 33.24% | en: 60.91%, ja: 56.73%, zh-CN: 56.17%, pt: 55.75%, it: 54.64% |
| MassiveScenarioClassification | 40.6% | en: 67.11%, zh-CN: 65.44%, ja: 64.59%, de: 62.88%, ko: 62.48% |
| ToxicConversationsClassification | 54.49% | default: 54.49% |
| TweetSentimentExtractionClassification | 60.05% | default: 60.05% |

### Clustering

| Task | Average | Details |
|---|---|---|
| ArXivHierarchicalClusteringP2P | 49.68% | default: 49.68% |
| ArXivHierarchicalClusteringS2S | 45.56% | default: 45.56% |
| BiorxivClusteringP2P.v2 | 19.6% | default: 19.6% |
| MedrxivClusteringP2P.v2 | 24.83% | default: 24.83% |
| MedrxivClusteringS2S.v2 | 22.03% | default: 22.03% |
| StackExchangeClustering.v2 | 39.38% | default: 39.38% |
| StackExchangeClusteringP2P.v2 | 31.8% | default: 31.8% |
| TwentyNewsgroupsClustering.v2 | 15.77% | default: 15.77% |

### STS

| Task | Average | Details |
|---|---|---|
| BIOSSES | 72.57% | default: 72.57% |
| SICK-R | 74.69% | default: 74.69% |
| STS12 | 73.58% | default: 73.58% |
| STS13 | 73.43% | default: 73.43% |
| STS14 | 73.35% | default: 73.35% |
| STS15 | 82.21% | default: 82.21% |
| STS17 | 59.08% | en-en: 84.4%, es-es: 79.74%, ko-ko: 72.11%, ar-ar: 67.11%, fr-en: 64.15% |
| STS22.v2 | 45.12% | fr: 67.71%, es: 63.98%, es-en: 61.84%, en: 60.59%, it: 60.2% |
| STSBenchmark | 77.7% | default: 77.7% |
| STSBenchmarkMultilingualSTS | 69.24% | en: 77.7%, es: 73.92%, fr: 73.89%, pt: 71.41%, it: 70.75% |



## Training

### Stage 1: Model Compression
- **Teacher**: `intfloat/multilingual-e5-small` (12L, 384d)
- **Compression**: Layer pruning + Vocab pruning
- **Result**: 4L / 384d / 15,424 vocab

### Stage 2: Knowledge Distillation
- **Method**: MSE + Cosine Similarity loss
- **Data**: MTEB Classification/Clustering/STS task datasets
- **Optimizer**: AdamW (lr=2e-5, weight_decay=0.01)
- **Schedule**: Cosine annealing over 3 epochs


## Supported Languages (16)

ko, en, ja, zh, es, fr, de, pt, it, ru, ar, hi, th, vi, id, pl
