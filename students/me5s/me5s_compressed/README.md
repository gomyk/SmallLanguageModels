---
language: ["ko", "en", "ja", "zh", "es", "fr", "de", "pt", "it", "ru", "ar", "hi", "th", "vi", "id", "tr", "nl", "pl"]
tags:
- sentence-transformers
- multilingual
- model-compression
- layer-pruning
- vocab-pruning
- me5-small
library_name: sentence-transformers
pipeline_tag: sentence-similarity
license: apache-2.0
---

# me5s_compressed

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
| Vocab size | 15,168 (from 250,037) |
| Parameters | ~13.1M |
| Model size (FP32) | 50.6MB |
| Compression | 9x |
| Distilled | No |

## Quick Start

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("me5s_compressed", trust_remote_code=True)

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

**Overall Average: 46.19%**

| Task Group | Average |
|---|---|
| Classification | 52.2% |
| Clustering | 30.4% |
| STS | 54.88% |

### Classification

| Task | Average | Details |
|---|---|---|
| AmazonCounterfactualClassification | 67.37% | en: 69.31%, en-ext: 67.39%, ja: 67.39%, de: 65.39% |
| Banking77Classification | 58.7% | default: 58.7% |
| ImdbClassification | 57.14% | default: 57.14% |
| MTOPDomainClassification | 66.84% | en: 75.99%, es: 69.48%, hi: 68.15%, fr: 63.63%, th: 63.38% |
| MassiveIntentClassification | 31.12% | en: 53.03%, zh-CN: 51.62%, it: 47.56%, pt: 47.28%, ja: 47.03% |
| MassiveScenarioClassification | 34.85% | zh-CN: 59.05%, en: 58.06%, ja: 51.79%, it: 50.05%, vi: 49.68% |
| ToxicConversationsClassification | 55.82% | default: 55.82% |
| TweetSentimentExtractionClassification | 45.74% | default: 45.74% |

### Clustering

| Task | Average | Details |
|---|---|---|
| ArXivHierarchicalClusteringP2P | 47.08% | default: 47.08% |
| ArXivHierarchicalClusteringS2S | 48.29% | default: 48.29% |
| BiorxivClusteringP2P.v2 | 17.24% | default: 17.24% |
| MedrxivClusteringP2P.v2 | 24.42% | default: 24.42% |
| MedrxivClusteringS2S.v2 | 21.55% | default: 21.55% |
| StackExchangeClustering.v2 | 39.42% | default: 39.42% |
| StackExchangeClusteringP2P.v2 | 31.85% | default: 31.85% |
| TwentyNewsgroupsClustering.v2 | 13.35% | default: 13.35% |

### STS

| Task | Average | Details |
|---|---|---|
| BIOSSES | 56.68% | default: 56.68% |
| SICK-R | 59.22% | default: 59.22% |
| STS12 | 52.11% | default: 52.11% |
| STS13 | 64.25% | default: 64.25% |
| STS14 | 60.12% | default: 60.12% |
| STS15 | 74.19% | default: 74.19% |
| STS17 | 38.71% | es-es: 74.88%, en-en: 74.6%, ar-ar: 63.98%, ko-ko: 58.54%, nl-en: 28.91% |
| STS22.v2 | 27.7% | zh: 59.69%, fr: 57.76%, es: 56.19%, en: 50.93%, it: 50.09% |
| STSBenchmark | 60.91% | default: 60.91% |



## Training

Created via **multi-method model compression** (no additional training):

1. **Teacher**: `intfloat/multilingual-e5-small` (12L, 384d, 117M params)
2. **Layer pruning**: 12 → 4 layers (uniform selection)
3. **Hidden dim**: 384 → 384
4. **Vocab pruning**: 250,037 → 15,168 (90% cumulative frequency)
5. **Compression ratio**: 9x


## Supported Languages (18)

ko, en, ja, zh, es, fr, de, pt, it, ru, ar, hi, th, vi, id, tr, nl, pl
