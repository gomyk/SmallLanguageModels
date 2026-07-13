---
language: ["ko", "en", "ja", "zh", "es", "fr", "de", "pt", "it", "ru", "ar", "hi", "th", "vi", "id", "pl"]
tags:
- sentence-transformers
- multilingual
- model-compression
- layer-pruning
- vocab-pruning
- jina-v5-nano
library_name: sentence-transformers
pipeline_tag: sentence-similarity
license: cc-by-nc-4.0
---

# jina_v5_compressed

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
| Distilled | No |

## Quick Start

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("jina_v5_compressed", trust_remote_code=True)

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

**Overall Average: 38.07%**

| Task Group | Average |
|---|---|
| Classification | 44.38% |
| Clustering | 28.49% |
| STS | 40.96% |

### Classification

| Task | Average | Details |
|---|---|---|
| AmazonCounterfactualClassification | 59.99% | en-ext: 61.54%, en: 61.13%, ja: 59.75%, de: 57.53% |
| Banking77Classification | 45.48% | default: 45.48% |
| ImdbClassification | 53.35% | default: 53.35% |
| MTOPDomainClassification | 46.78% | es: 50.92%, en: 48.38%, de: 48.08%, hi: 46.87%, fr: 44.64% |
| MassiveIntentClassification | 28.79% | en: 35.18%, zh-CN: 35.14%, lv: 34.01%, it: 33.41%, pt: 33.3% |
| MassiveScenarioClassification | 29.45% | en: 36.67%, zh-CN: 36.26%, ms: 35.19%, sw: 34.47%, zh-TW: 33.99% |
| ToxicConversationsClassification | 52.72% | default: 52.72% |
| TweetSentimentExtractionClassification | 38.51% | default: 38.51% |

### Clustering

| Task | Average | Details |
|---|---|---|
| ArXivHierarchicalClusteringP2P | 47.49% | default: 47.49% |
| ArXivHierarchicalClusteringS2S | 46.99% | default: 46.99% |
| BiorxivClusteringP2P.v2 | 10.67% | default: 10.67% |
| MedrxivClusteringP2P.v2 | 21.78% | default: 21.78% |
| MedrxivClusteringS2S.v2 | 20.71% | default: 20.71% |
| StackExchangeClustering.v2 | 38.42% | default: 38.42% |
| StackExchangeClusteringP2P.v2 | 32.47% | default: 32.47% |
| TwentyNewsgroupsClustering.v2 | 9.41% | default: 9.41% |

### STS

| Task | Average | Details |
|---|---|---|
| BIOSSES | 32.24% | default: 32.24% |
| SICK-R | 50.98% | default: 50.98% |
| STS12 | 42.29% | default: 42.29% |
| STS13 | 47.69% | default: 47.69% |
| STS14 | 48.34% | default: 48.34% |
| STS15 | 54.33% | default: 54.33% |
| STS17 | 16.67% | es-es: 64.75%, en-en: 56.41%, ko-ko: 48.42%, ar-ar: 44.05%, es-en: 8.31% |
| STS22.v2 | 30.46% | zh: 60.89%, ar: 55.33%, fr-pl: 50.71%, es: 49.6%, it: 44.01% |
| STSBenchmark | 45.65% | default: 45.65% |



## Training

Created via **multi-method model compression** (no additional training):

1. **Teacher**: `jinaai/jina-embeddings-v5-text-nano` (12L, 768d, 212M params)
2. **Layer pruning**: 12 → 6 layers (uniform selection)
3. **Hidden dim**: 768 → 384
4. **Vocab pruning**: 128,256 → 12,796 (90% cumulative frequency)
5. **Compression ratio**: 11x

## License

This model is a derivative of Jina AI's jina-embeddings-v5-text-nano. The original model is provided under CC BY-NC 4.0 license. See [jina-embeddings-v5-text-nano](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano) for details.


## Supported Languages (16)

ko, en, ja, zh, es, fr, de, pt, it, ru, ar, hi, th, vi, id, pl
