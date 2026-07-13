---
language: ["ko", "en", "ja", "zh", "es", "fr", "de", "pt", "it", "ru", "ar", "hi", "th", "vi", "id", "tr", "nl", "pl"]
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

# jina_v5_compressed_distilled_v2 (Distilled)

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
| Vocab size | 13,357 (from 128,256) |
| Parameters | ~19.3M |
| Model size (FP32) | 73.6MB |
| Compression | 11x |
| Distilled | Yes |

## Quick Start

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("jina_v5_compressed_distilled_v2", trust_remote_code=True)

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

**Overall Average: 49.22%**

| Task Group | Average |
|---|---|
| Classification | 56.74% |
| Clustering | 30.52% |
| STS | 60.39% |

### Classification

| Task | Average | Details |
|---|---|---|
| AmazonCounterfactualClassification | 65.26% | en-ext: 68.9%, en: 67.88%, de: 66.16%, ja: 58.09% |
| Banking77Classification | 75.51% | default: 75.51% |
| ImdbClassification | 73.14% | default: 73.14% |
| MTOPDomainClassification | 62.19% | en: 79.51%, es: 72.58%, fr: 70.21%, de: 66.62%, hi: 51.91% |
| MassiveIntentClassification | 29.89% | en: 63.28%, fr: 57.88%, de: 56.17%, es: 55.9%, zh-CN: 46.61% |
| MassiveScenarioClassification | 35.1% | en: 70.88%, de: 66.66%, fr: 65.81%, es: 65.47%, zh-CN: 58.37% |
| ToxicConversationsClassification | 58.11% | default: 58.11% |
| TweetSentimentExtractionClassification | 54.71% | default: 54.71% |

### Clustering

| Task | Average | Details |
|---|---|---|
| ArXivHierarchicalClusteringP2P | 48.7% | default: 48.7% |
| ArXivHierarchicalClusteringS2S | 46.39% | default: 46.39% |
| BiorxivClusteringP2P.v2 | 13.89% | default: 13.89% |
| MedrxivClusteringP2P.v2 | 22.72% | default: 22.72% |
| MedrxivClusteringS2S.v2 | 20.87% | default: 20.87% |
| StackExchangeClustering.v2 | 42.55% | default: 42.55% |
| StackExchangeClusteringP2P.v2 | 33.52% | default: 33.52% |
| TwentyNewsgroupsClustering.v2 | 15.52% | default: 15.52% |

### STS

| Task | Average | Details |
|---|---|---|
| BIOSSES | 57.99% | default: 57.99% |
| SICK-R | 68.92% | default: 68.92% |
| STS12 | 63.79% | default: 63.79% |
| STS13 | 62.56% | default: 62.56% |
| STS14 | 64.02% | default: 64.02% |
| STS15 | 75.28% | default: 75.28% |
| STS17 | 22.41% | en-en: 72.4%, es-es: 70.73%, ar-ar: 55.51%, ko-ko: 49.54%, nl-en: 16.0% |
| STSBenchmark | 68.16% | default: 68.16% |



## Training

### Stage 1: Model Compression
- **Teacher**: `jinaai/jina-embeddings-v5-text-nano` (12L, 768d)
- **Compression**: Layer pruning + Vocab pruning
- **Result**: 6L / 384d / 13,357 vocab

### Stage 2: Knowledge Distillation
- **Method**: MSE + Cosine Similarity loss
- **Data**: MTEB Classification/Clustering/STS task datasets
- **Optimizer**: AdamW (lr=2e-5, weight_decay=0.01)
- **Schedule**: Cosine annealing over 3 epochs

## License

This model is a derivative of Jina AI's jina-embeddings-v5-text-nano. The original model is provided under CC BY-NC 4.0 license. See [jina-embeddings-v5-text-nano](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano) for details.


## Supported Languages (18)

ko, en, ja, zh, es, fr, de, pt, it, ru, ar, hi, th, vi, id, tr, nl, pl
