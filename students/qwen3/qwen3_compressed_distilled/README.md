---
language: ["ko", "en", "ja", "zh", "es", "fr", "de", "pt", "it", "ru", "ar", "hi", "th", "vi", "id", "tr", "nl", "pl"]
tags:
- sentence-transformers
- multilingual
- model-compression
- layer-pruning
- vocab-pruning
- knowledge-distillation
- progressive-distillation
- qwen3-0.6b
library_name: sentence-transformers
pipeline_tag: sentence-similarity
license: apache-2.0
---

# qwen3_compressed_distilled (Distilled)

Compact multilingual sentence encoder compressed from `Qwen/Qwen3-0.6B` (48x compression).

## Model Details

| Property | Value |
|---|---|
| Base model | `Qwen/Qwen3-0.6B` |
| Architecture | qwen3 (decoder) |
| Hidden dim | 448 (from 1024) |
| Layers | 4 (from 28) |
| Intermediate | 1344 |
| Attention heads | 7 |
| KV heads | 1 |
| Vocab size | 7,341 (from 151,936) |
| Parameters | ~12.4M |
| Model size (FP32) | 47.1MB |
| Compression | 48x |
| Distilled | Yes (2-stage) |

## Quick Start

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("qwen3_compressed_distilled", trust_remote_code=True)

sentences = [
    "Hello, how are you?",
    "안녕하세요, 잘 지내세요?",
    "こんにちは、元気ですか？",
    "你好，你好吗？",
]

embeddings = model.encode(sentences)
print(embeddings.shape)  # (4, 448)
```

## MTEB Evaluation Results

**Overall Average: 32.29%**

| Task Group | Average |
|---|---|
| Classification | 39.54% |
| Clustering | 29.66% |
| STS | 27.68% |

### Classification

| Task | Average | Details |
|---|---|---|
| AmazonCounterfactualClassification | 57.59% | en: 61.01%, en-ext: 57.63%, de: 57.34%, ja: 54.4% |
| Banking77Classification | 33.06% | default: 33.06% |
| ImdbClassification | 51.77% | default: 51.77% |
| MTOPDomainClassification | 40.28% | es: 49.82%, en: 49.41%, fr: 41.34%, th: 38.1%, de: 36.01% |
| MassiveIntentClassification | 18.75% | zh-CN: 39.84%, en: 32.94%, ko: 31.09%, ja: 30.64%, fr: 28.88% |
| MassiveScenarioClassification | 22.01% | zh-CN: 53.77%, en: 42.48%, ja: 41.01%, ko: 40.22%, es: 36.93% |
| ToxicConversationsClassification | 51.26% | default: 51.26% |
| TweetSentimentExtractionClassification | 41.62% | default: 41.62% |

### Clustering

| Task | Average | Details |
|---|---|---|
| ArXivHierarchicalClusteringP2P | 48.97% | default: 48.97% |
| ArXivHierarchicalClusteringS2S | 49.66% | default: 49.66% |
| BiorxivClusteringP2P.v2 | 7.42% | default: 7.42% |
| MedrxivClusteringP2P.v2 | 19.2% | default: 19.2% |
| MedrxivClusteringS2S.v2 | 20.79% | default: 20.79% |
| StackExchangeClustering.v2 | 45.32% | default: 45.32% |
| StackExchangeClusteringP2P.v2 | 33.99% | default: 33.99% |
| TwentyNewsgroupsClustering.v2 | 11.97% | default: 11.97% |

### STS

| Task | Average | Details |
|---|---|---|
| BIOSSES | 26.5% | default: 26.5% |
| SICK-R | 36.86% | default: 36.86% |
| STS12 | 37.37% | default: 37.37% |
| STS13 | 22.61% | default: 22.61% |
| STS14 | 19.72% | default: 19.72% |
| STS15 | 37.43% | default: 37.43% |
| STS17 | 17.77% | en-en: 37.23%, ko-ko: 33.84%, es-es: 32.55%, ar-ar: 28.19%, fr-en: 14.73% |
| STSBenchmark | 23.14% | default: 23.14% |


## Distillation Impact

| Task | Before | After | Delta |
|---|---|---|---|
| AmazonCounterfactualClassification | 53.9% | 57.59% | +3.69%p |
| ArXivHierarchicalClusteringP2P | 46.3% | 48.97% | +2.67%p |
| ArXivHierarchicalClusteringS2S | 46.53% | 49.66% | +3.13%p |
| BIOSSES | 1.1% | 26.5% | +25.4%p |
| Banking77Classification | 9.18% | 33.06% | +23.88%p |
| BiorxivClusteringP2P.v2 | 5.86% | 7.42% | +1.56%p |
| ImdbClassification | 50.25% | 51.77% | +1.52%p |
| MTOPDomainClassification | 28.9% | 40.28% | +11.38%p |
| MassiveIntentClassification | 15.91% | 18.75% | +2.84%p |
| MassiveScenarioClassification | 17.94% | 22.01% | +4.07%p |
| MedrxivClusteringP2P.v2 | 17.63% | 19.2% | +1.57%p |
| MedrxivClusteringS2S.v2 | 18.42% | 20.79% | +2.37%p |
| SICK-R | 22.32% | 36.86% | +14.54%p |
| STS12 | 7.97% | 37.37% | +29.4%p |
| STS13 | 18.76% | 22.61% | +3.85%p |
| STS14 | 10.4% | 19.72% | +9.32%p |
| STS15 | 24.3% | 37.43% | +13.13%p |
| STS17 | 10.58% | 17.77% | +7.19%p |
| STSBenchmark | 7.96% | 23.14% | +15.18%p |
| StackExchangeClustering.v2 | 40.91% | 45.32% | +4.41%p |
| StackExchangeClusteringP2P.v2 | 30.96% | 33.99% | +3.03%p |
| ToxicConversationsClassification | 50.35% | 51.26% | +0.91%p |
| TweetSentimentExtractionClassification | 34.47% | 41.62% | +7.15%p |
| TwentyNewsgroupsClustering.v2 | 9.52% | 11.97% | +2.45%p |


## Training

### Stage 1: Model Compression
- **Teacher**: `Qwen/Qwen3-0.6B` (28L, 1024d, 596M params)
- **Compression**: Layer pruning → Hidden dim reduction → Vocab pruning
- **Result**: 4L / 448d / 7,341 vocab

### Stage 2: Two-Stage Knowledge Distillation
Compression ratio 48x requires progressive distillation:

1. **Stage 1**: Teacher (596M) → Intermediate (~119M)
   - MSE + Cosine Similarity loss
   - MTEB task datasets (Classification/Clustering/STS)
2. **Stage 2**: Intermediate → Final Student (12.4M)
   - Same training objective
   - AdamW (lr=2e-5, weight_decay=0.01), Cosine annealing


## Supported Languages (18)

ko, en, ja, zh, es, fr, de, pt, it, ru, ar, hi, th, vi, id, tr, nl, pl
