---
language: ["ko", "en", "ja", "zh", "es", "fr", "de", "pt", "it", "ru", "ar", "hi", "th", "vi", "id", "pl"]
tags:
- sentence-transformers
- multilingual
- model-compression
- layer-pruning
- vocab-pruning
- knowledge-distillation
- jina-v5-nano
library_name: sentence-transformers
pipeline_tag: sentence-similarity
license: cc-by-nc-4.0
---

# jina_v5_compressed_distilled (Distilled)

Compact multilingual sentence encoder compressed from `jinaai/jina-embeddings-v5-text-nano` (11x compression).

## Model Details

| Property | Value |
|---|---|
| Base model | `jinaai/jina-embeddings-v5-text-nano` |
| Architecture | eurobert (decoder) |
| Hidden dim | 384 (from 768) |
| Layers | 6 (from 12) |
| Intermediate | 1536 |
| Attention heads | 6 |
| KV heads | 6 |
| Vocab size | 12,796 (from 128,256) |
| Parameters | ~19.1M |
| Model size (FP32) | 72.8MB |
| Compression | 11x |
| Distilled | Yes |

## Quick Start

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("jina_v5_compressed_distilled", trust_remote_code=True)

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

**Overall Average: 44.66%**

| Task Group | Average |
|---|---|
| Classification | 51.79% |
| Clustering | 29.68% |
| STS | 51.64% |

### Classification

| Task | Average | Details |
|---|---|---|
| AmazonCounterfactualClassification | 63.38% | en: 66.87%, en-ext: 66.55%, de: 60.79%, ja: 59.3% |
| Banking77Classification | 63.49% | default: 63.49% |
| ImdbClassification | 64.78% | default: 64.78% |
| MTOPDomainClassification | 55.74% | en: 72.0%, es: 66.08%, fr: 62.46%, de: 55.36%, hi: 48.9% |
| MassiveIntentClassification | 25.49% | en: 54.22%, fr: 46.33%, de: 45.61%, es: 45.01%, zh-CN: 38.76% |
| MassiveScenarioClassification | 31.92% | en: 64.35%, de: 58.19%, fr: 57.73%, es: 57.39%, zh-CN: 53.1% |
| ToxicConversationsClassification | 57.49% | default: 57.49% |
| TweetSentimentExtractionClassification | 52.05% | default: 52.05% |

### Clustering

| Task | Average | Details |
|---|---|---|
| ArXivHierarchicalClusteringP2P | 47.9% | default: 47.9% |
| ArXivHierarchicalClusteringS2S | 46.85% | default: 46.85% |
| BiorxivClusteringP2P.v2 | 11.99% | default: 11.99% |
| MedrxivClusteringP2P.v2 | 21.74% | default: 21.74% |
| MedrxivClusteringS2S.v2 | 20.75% | default: 20.75% |
| StackExchangeClustering.v2 | 41.72% | default: 41.72% |
| StackExchangeClusteringP2P.v2 | 33.77% | default: 33.77% |
| TwentyNewsgroupsClustering.v2 | 12.74% | default: 12.74% |

### STS

| Task | Average | Details |
|---|---|---|
| BIOSSES | 47.78% | default: 47.78% |
| SICK-R | 62.78% | default: 62.78% |
| STS12 | 60.82% | default: 60.82% |
| STS13 | 55.85% | default: 55.85% |
| STS14 | 56.38% | default: 56.38% |
| STS15 | 70.11% | default: 70.11% |
| STS17 | 19.02% | es-es: 69.59%, en-en: 66.09%, ar-ar: 51.58%, ko-ko: 50.01%, nl-en: 15.48% |
| STS22.v2 | 33.29% | fr-pl: 84.52%, zh: 58.95%, es: 51.17%, ar: 50.81%, it: 50.63% |
| STSBenchmark | 58.71% | default: 58.71% |


## Distillation Impact

| Task | Before | After | Delta |
|---|---|---|---|
| AmazonCounterfactualClassification | 59.99% | 63.38% | +3.39%p |
| ArXivHierarchicalClusteringP2P | 47.49% | 47.9% | +0.41%p |
| ArXivHierarchicalClusteringS2S | 46.99% | 46.85% | -0.14%p |
| BIOSSES | 32.24% | 47.78% | +15.54%p |
| Banking77Classification | 45.48% | 63.49% | +18.01%p |
| BiorxivClusteringP2P.v2 | 10.67% | 11.99% | +1.32%p |
| ImdbClassification | 53.35% | 64.78% | +11.43%p |
| MTOPDomainClassification | 46.78% | 55.74% | +8.96%p |
| MassiveIntentClassification | 28.79% | 25.49% | -3.3%p |
| MassiveScenarioClassification | 29.45% | 31.92% | +2.47%p |
| MedrxivClusteringP2P.v2 | 21.78% | 21.74% | -0.04%p |
| MedrxivClusteringS2S.v2 | 20.71% | 20.75% | +0.04%p |
| SICK-R | 50.98% | 62.78% | +11.8%p |
| STS12 | 42.29% | 60.82% | +18.53%p |
| STS13 | 47.69% | 55.85% | +8.16%p |
| STS14 | 48.34% | 56.38% | +8.04%p |
| STS15 | 54.33% | 70.11% | +15.78%p |
| STS17 | 16.67% | 19.02% | +2.35%p |
| STS22.v2 | 30.46% | 33.29% | +2.83%p |
| STSBenchmark | 45.65% | 58.71% | +13.06%p |
| StackExchangeClustering.v2 | 38.42% | 41.72% | +3.3%p |
| StackExchangeClusteringP2P.v2 | 32.47% | 33.77% | +1.3%p |
| ToxicConversationsClassification | 52.72% | 57.49% | +4.77%p |
| TweetSentimentExtractionClassification | 38.51% | 52.05% | +13.54%p |
| TwentyNewsgroupsClustering.v2 | 9.41% | 12.74% | +3.33%p |


## Training

### Stage 1: Model Compression
- **Teacher**: `jinaai/jina-embeddings-v5-text-nano` (12L, 768d)
- **Compression**: Layer pruning + Vocab pruning
- **Result**: 6L / 384d / 12,796 vocab

### Stage 2: Knowledge Distillation
- **Method**: MSE + Cosine Similarity loss
- **Data**: MTEB Classification/Clustering/STS task datasets
- **Optimizer**: AdamW (lr=2e-5, weight_decay=0.01)
- **Schedule**: Cosine annealing over 3 epochs

## License

This model is a derivative of Jina AI's jina-embeddings-v5-text-nano. The original model is provided under CC BY-NC 4.0 license. See [jina-embeddings-v5-text-nano](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano) for details.


## Supported Languages (16)

ko, en, ja, zh, es, fr, de, pt, it, ru, ar, hi, th, vi, id, pl
