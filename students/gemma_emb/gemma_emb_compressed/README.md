---
language: ["ko", "en", "ja", "zh", "es", "fr", "de", "pt", "it", "ru", "ar", "hi", "th", "vi", "id", "tr", "nl", "pl"]
tags:
- sentence-transformers
- multilingual
- model-compression
- layer-pruning
- vocab-pruning
- progressive-distillation
- embeddinggemma-300m
library_name: sentence-transformers
pipeline_tag: sentence-similarity
license: gemma
---

# gemma_emb_compressed

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
| Distilled | No |

## Quick Start

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("gemma_emb_compressed", trust_remote_code=True)

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

**Overall Average: 27.12%**

| Task Group | Average |
|---|---|
| Classification | 37.65% |
| Clustering | 27.39% |
| STS | 17.52% |

### Classification

| Task | Average | Details |
|---|---|---|
| AmazonCounterfactualClassification | 59.01% | en: 62.78%, en-ext: 61.44%, de: 57.88%, ja: 53.94% |
| Banking77Classification | 19.07% | default: 19.07% |
| ImdbClassification | 52.55% | default: 52.55% |
| MTOPDomainClassification | 38.89% | es: 44.84%, th: 43.67%, de: 41.01%, en: 39.61%, fr: 39.45% |
| MassiveIntentClassification | 22.16% | zh-CN: 31.72%, th: 28.37%, zh-TW: 28.22%, vi: 26.23%, sv: 26.17% |
| MassiveScenarioClassification | 23.12% | zh-CN: 33.22%, zh-TW: 28.98%, ko: 28.13%, th: 27.83%, km: 26.79% |
| ToxicConversationsClassification | 50.12% | default: 50.12% |
| TweetSentimentExtractionClassification | 36.28% | default: 36.28% |

### Clustering

| Task | Average | Details |
|---|---|---|
| ArXivHierarchicalClusteringP2P | 45.54% | default: 45.54% |
| ArXivHierarchicalClusteringS2S | 45.1% | default: 45.1% |
| BiorxivClusteringP2P.v2 | 8.07% | default: 8.07% |
| MedrxivClusteringP2P.v2 | 19.06% | default: 19.06% |
| MedrxivClusteringS2S.v2 | 17.57% | default: 17.57% |
| StackExchangeClustering.v2 | 41.53% | default: 41.53% |
| StackExchangeClusteringP2P.v2 | 33.43% | default: 33.43% |
| TwentyNewsgroupsClustering.v2 | 8.85% | default: 8.85% |

### STS

| Task | Average | Details |
|---|---|---|
| BIOSSES | -0.64% | default: -0.64% |
| SICK-R | 30.8% | default: 30.8% |
| STS12 | 23.59% | default: 23.59% |
| STS13 | 19.19% | default: 19.19% |
| STS14 | 11.24% | default: 11.24% |
| STS15 | 30.55% | default: 30.55% |
| STS17 | 12.2% | es-es: 44.48%, en-en: 34.75%, ko-ko: 33.59%, ar-ar: 20.35%, en-ar: 10.34% |
| STS22.v2 | 15.19% | zh: 44.99%, es: 35.77%, es-en: 29.78%, pl-en: 28.62%, ar: 26.32% |
| STSBenchmark | 15.56% | default: 15.56% |



## Training

Created via **multi-method model compression** (no additional training):

1. **Teacher**: `google/embeddinggemma-300m` (24L, 768d, 303M params)
2. **Layer pruning**: 24 → 4 layers (uniform selection)
3. **Hidden dim**: 768 → 384
4. **Vocab pruning**: 262,144 → 19,485 (90% cumulative frequency)
5. **Compression ratio**: 24x

## License

This model is a derivative of Google's Gemma. Gemma is provided under and subject to the Gemma Terms of Use found at [ai.google.dev/gemma/terms](https://ai.google.dev/gemma/terms). Use of this model must comply with the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).


## Supported Languages (18)

ko, en, ja, zh, es, fr, de, pt, it, ru, ar, hi, th, vi, id, tr, nl, pl
