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

# jina_v5_compressed_distilled_trainonly (Distilled)

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

model = SentenceTransformer("jina_v5_compressed_distilled_trainonly", trust_remote_code=True)

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

**Overall Average: 49.14%**

| Task Group | Average |
|---|---|
| Classification | 56.84% |
| Clustering | 30.06% |
| STS | 59.24% |

### Classification

| Task | Average | Details |
|---|---|---|
| AmazonCounterfactualClassification | 66.04% | en-ext: 70.06%, en: 69.39%, de: 66.78%, ja: 57.94% |
| Banking77Classification | 82.19% | default: 82.19% |
| ImdbClassification | 63.33% | default: 63.33% |
| MTOPDomainClassification | 65.07% | en: 82.2%, es: 76.7%, fr: 72.72%, de: 70.68%, hi: 51.51% |
| MassiveIntentClassification | 32.14% | en: 67.98%, zh-CN: 63.42%, fr: 62.54%, es: 62.28%, de: 62.15% |
| MassiveScenarioClassification | 36.59% | en: 75.02%, zh-CN: 71.63%, de: 71.11%, es: 69.78%, fr: 69.34% |
| ToxicConversationsClassification | 55.46% | default: 55.46% |
| TweetSentimentExtractionClassification | 53.94% | default: 53.94% |

### Clustering

| Task | Average | Details |
|---|---|---|
| ArXivHierarchicalClusteringP2P | 47.75% | default: 47.75% |
| ArXivHierarchicalClusteringS2S | 46.64% | default: 46.64% |
| BiorxivClusteringP2P.v2 | 13.31% | default: 13.31% |
| MedrxivClusteringP2P.v2 | 22.25% | default: 22.25% |
| MedrxivClusteringS2S.v2 | 20.72% | default: 20.72% |
| StackExchangeClustering.v2 | 41.26% | default: 41.26% |
| StackExchangeClusteringP2P.v2 | 33.35% | default: 33.35% |
| TwentyNewsgroupsClustering.v2 | 15.22% | default: 15.22% |

### STS

| Task | Average | Details |
|---|---|---|
| BIOSSES | 54.23% | default: 54.23% |
| SICK-R | 69.97% | default: 69.97% |
| STS12 | 65.85% | default: 65.85% |
| STS13 | 66.36% | default: 66.36% |
| STS14 | 67.47% | default: 67.47% |
| STS15 | 76.89% | default: 76.89% |
| STS17 | 23.22% | en-en: 77.1%, es-es: 70.48%, ar-ar: 59.67%, ko-ko: 49.47%, es-en: 18.41% |
| STS22.v2 | 35.14% | fr-pl: 84.52%, zh: 59.82%, it: 54.55%, es: 54.15%, ar: 53.12% |
| STSBenchmark | 74.05% | default: 74.05% |



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


## Supported Languages (16)

ko, en, ja, zh, es, fr, de, pt, it, ru, ar, hi, th, vi, id, pl
