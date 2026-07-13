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
- embeddinggemma-300m
library_name: sentence-transformers
pipeline_tag: sentence-similarity
license: gemma
---

# gemma_emb_compressed_distilled (Distilled)

Compact multilingual sentence encoder compressed from `google/embeddinggemma-300m` (24x compression).

## Model Details

| Property | Value |
|---|---|
| Base model | `google/embeddinggemma-300m` |
| Architecture | gemma3_text (decoder) |
| Hidden dim | 384 (from 768) |
| Layers | 4 (from 24) |
| Intermediate | 576 |
| Attention heads | 1 |
| KV heads | 1 |
| Vocab size | 19,485 (from 262,144) |
| Parameters | ~12.5M |
| Model size (FP32) | 47.7MB |
| Compression | 24x |
| Distilled | Yes (2-stage) |

## Quick Start

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("gemma_emb_compressed_distilled", trust_remote_code=True)

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

**Overall Average: 25.82%**

| Task Group | Average |
|---|---|
| Classification | 34.81% |
| Clustering | 28.29% |
| STS | 15.62% |

### Classification

| Task | Average | Details |
|---|---|---|
| AmazonCounterfactualClassification | 54.73% | en-ext: 56.48%, en: 56.1%, de: 56.03%, ja: 50.33% |
| Banking77Classification | 12.88% | default: 12.88% |
| ImdbClassification | 51.34% | default: 51.34% |
| MTOPDomainClassification | 33.78% | th: 35.0%, es: 34.41%, fr: 33.67%, en: 33.27%, hi: 33.19% |
| MassiveIntentClassification | 15.53% | zh-CN: 54.26%, ja: 45.35%, zh-TW: 45.06%, ko: 40.29%, th: 17.34% |
| MassiveScenarioClassification | 20.81% | zh-CN: 66.69%, ja: 53.96%, zh-TW: 53.52%, ko: 49.29%, vi: 22.58% |
| ToxicConversationsClassification | 51.68% | default: 51.68% |
| TweetSentimentExtractionClassification | 37.77% | default: 37.77% |

### Clustering

| Task | Average | Details |
|---|---|---|
| ArXivHierarchicalClusteringP2P | 47.02% | default: 47.02% |
| ArXivHierarchicalClusteringS2S | 48.8% | default: 48.8% |
| BiorxivClusteringP2P.v2 | 8.06% | default: 8.06% |
| MedrxivClusteringP2P.v2 | 18.8% | default: 18.8% |
| MedrxivClusteringS2S.v2 | 18.2% | default: 18.2% |
| StackExchangeClustering.v2 | 40.68% | default: 40.68% |
| StackExchangeClusteringP2P.v2 | 34.68% | default: 34.68% |
| TwentyNewsgroupsClustering.v2 | 10.04% | default: 10.04% |

### STS

| Task | Average | Details |
|---|---|---|
| BIOSSES | 20.21% | default: 20.21% |
| SICK-R | 23.92% | default: 23.92% |
| STS12 | 12.98% | default: 12.98% |
| STS13 | 16.57% | default: 16.57% |
| STS14 | 1.39% | default: 1.39% |
| STS15 | 22.27% | default: 22.27% |
| STS17 | 26.2% | en-en: 42.85%, es-es: 41.09%, en-tr: 34.02%, ar-ar: 33.99%, it-en: 27.46% |
| STS22.v2 | 13.11% | fr-pl: 39.44%, en: 25.46%, es: 24.75%, de-fr: 21.99%, ar: 15.92% |
| STSBenchmark | 3.94% | default: 3.94% |


## Distillation Impact

| Task | Before | After | Delta |
|---|---|---|---|
| AmazonCounterfactualClassification | 59.01% | 54.73% | -4.28%p |
| ArXivHierarchicalClusteringP2P | 45.54% | 47.02% | +1.48%p |
| ArXivHierarchicalClusteringS2S | 45.1% | 48.8% | +3.7%p |
| BIOSSES | -0.64% | 20.21% | +20.85%p |
| Banking77Classification | 19.07% | 12.88% | -6.19%p |
| BiorxivClusteringP2P.v2 | 8.07% | 8.06% | -0.01%p |
| ImdbClassification | 52.55% | 51.34% | -1.21%p |
| MTOPDomainClassification | 38.89% | 33.78% | -5.11%p |
| MassiveIntentClassification | 22.16% | 15.53% | -6.63%p |
| MassiveScenarioClassification | 23.12% | 20.81% | -2.31%p |
| MedrxivClusteringP2P.v2 | 19.06% | 18.8% | -0.26%p |
| MedrxivClusteringS2S.v2 | 17.57% | 18.2% | +0.63%p |
| SICK-R | 30.8% | 23.92% | -6.88%p |
| STS12 | 23.59% | 12.98% | -10.61%p |
| STS13 | 19.19% | 16.57% | -2.62%p |
| STS14 | 11.24% | 1.39% | -9.85%p |
| STS15 | 30.55% | 22.27% | -8.28%p |
| STS17 | 12.2% | 26.2% | +14.0%p |
| STS22.v2 | 15.19% | 13.11% | -2.08%p |
| STSBenchmark | 15.56% | 3.94% | -11.62%p |
| StackExchangeClustering.v2 | 41.53% | 40.68% | -0.85%p |
| StackExchangeClusteringP2P.v2 | 33.43% | 34.68% | +1.25%p |
| ToxicConversationsClassification | 50.12% | 51.68% | +1.56%p |
| TweetSentimentExtractionClassification | 36.28% | 37.77% | +1.49%p |
| TwentyNewsgroupsClustering.v2 | 8.85% | 10.04% | +1.19%p |


## Training

### Stage 1: Model Compression
- **Teacher**: `google/embeddinggemma-300m` (24L, 768d, 303M params)
- **Compression**: Layer pruning → Hidden dim reduction → Vocab pruning
- **Result**: 4L / 384d / 19,485 vocab

### Stage 2: Two-Stage Knowledge Distillation
Compression ratio 24x requires progressive distillation:

1. **Stage 1**: Teacher (303M) → Intermediate (~61M)
   - MSE + Cosine Similarity loss
   - MTEB task datasets (Classification/Clustering/STS)
2. **Stage 2**: Intermediate → Final Student (12.5M)
   - Same training objective
   - AdamW (lr=2e-5, weight_decay=0.01), Cosine annealing

## License

This model is a derivative of Google's Gemma. Gemma is provided under and subject to the Gemma Terms of Use found at [ai.google.dev/gemma/terms](https://ai.google.dev/gemma/terms). Use of this model must comply with the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).


## Supported Languages (18)

ko, en, ja, zh, es, fr, de, pt, it, ru, ar, hi, th, vi, id, tr, nl, pl
