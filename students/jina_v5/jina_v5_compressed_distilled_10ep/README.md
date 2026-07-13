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

# jina_v5_compressed_distilled_10ep (Distilled)

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

model = SentenceTransformer("jina_v5_compressed_distilled_10ep", trust_remote_code=True)

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

**Overall Average: 50.45%**

| Task Group | Average |
|---|---|
| Classification | 57.93% |
| Clustering | 30.61% |
| STS | 62.83% |

### Classification

| Task | Average | Details |
|---|---|---|
| AmazonCounterfactualClassification | 66.5% | en-ext: 71.34%, en: 69.6%, de: 66.21%, ja: 58.85% |
| Banking77Classification | 78.89% | default: 78.89% |
| ImdbClassification | 74.68% | default: 74.68% |
| MTOPDomainClassification | 63.7% | en: 80.76%, es: 74.54%, fr: 72.64%, de: 69.97%, hi: 52.24% |
| MassiveIntentClassification | 31.5% | en: 66.03%, fr: 60.85%, es: 60.02%, de: 59.37%, zh-CN: 51.78% |
| MassiveScenarioClassification | 36.38% | en: 72.76%, de: 69.15%, es: 68.62%, fr: 68.06%, zh-CN: 62.0% |
| ToxicConversationsClassification | 56.66% | default: 56.66% |
| TweetSentimentExtractionClassification | 55.09% | default: 55.09% |

### Clustering

| Task | Average | Details |
|---|---|---|
| ArXivHierarchicalClusteringP2P | 49.73% | default: 49.73% |
| ArXivHierarchicalClusteringS2S | 45.54% | default: 45.54% |
| BiorxivClusteringP2P.v2 | 13.84% | default: 13.84% |
| MedrxivClusteringP2P.v2 | 23.06% | default: 23.06% |
| MedrxivClusteringS2S.v2 | 21.15% | default: 21.15% |
| StackExchangeClustering.v2 | 42.35% | default: 42.35% |
| StackExchangeClusteringP2P.v2 | 32.77% | default: 32.77% |
| TwentyNewsgroupsClustering.v2 | 16.4% | default: 16.4% |

### STS

| Task | Average | Details |
|---|---|---|
| BIOSSES | 60.94% | default: 60.94% |
| SICK-R | 69.24% | default: 69.24% |
| STS12 | 65.93% | default: 65.93% |
| STS13 | 65.39% | default: 65.39% |
| STS14 | 66.74% | default: 66.74% |
| STS15 | 77.84% | default: 77.84% |
| STS17 | 23.95% | en-en: 76.26%, es-es: 73.31%, ar-ar: 59.18%, ko-ko: 49.56%, nl-en: 20.64% |
| STSBenchmark | 72.6% | default: 72.6% |



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
