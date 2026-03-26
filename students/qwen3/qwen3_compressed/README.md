---
language: ["ko", "en", "ja", "zh", "es", "fr", "de", "pt", "it", "ru", "ar", "hi", "th", "vi", "id", "tr", "nl", "pl"]
tags:
- sentence-transformers
- multilingual
- model-compression
- layer-pruning
- vocab-pruning
- progressive-distillation
- qwen3-0.6b
library_name: sentence-transformers
pipeline_tag: sentence-similarity
license: apache-2.0
---

# qwen3_compressed

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
| Distilled | No |

## Quick Start

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("qwen3_compressed", trust_remote_code=True)

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

**Overall Average: 24.18%**

| Task Group | Average |
|---|---|
| Classification | 32.61% |
| Clustering | 27.02% |
| STS | 12.92% |

### Classification

| Task | Average | Details |
|---|---|---|
| AmazonCounterfactualClassification | 53.9% | en-ext: 57.41%, en: 56.0%, de: 51.45%, ja: 50.74% |
| Banking77Classification | 9.18% | default: 9.18% |
| ImdbClassification | 50.25% | default: 50.25% |
| MTOPDomainClassification | 28.9% | th: 37.34%, es: 32.47%, de: 31.03%, en: 28.63%, fr: 26.97% |
| MassiveIntentClassification | 15.91% | it: 22.06%, pl: 21.13%, th: 21.1%, sq: 21.03%, sw: 20.97% |
| MassiveScenarioClassification | 17.94% | th: 23.36%, sq: 22.48%, cy: 22.45%, sv: 22.22%, it: 22.13% |
| ToxicConversationsClassification | 50.35% | default: 50.35% |
| TweetSentimentExtractionClassification | 34.47% | default: 34.47% |

### Clustering

| Task | Average | Details |
|---|---|---|
| ArXivHierarchicalClusteringP2P | 46.3% | default: 46.3% |
| ArXivHierarchicalClusteringS2S | 46.53% | default: 46.53% |
| BiorxivClusteringP2P.v2 | 5.86% | default: 5.86% |
| MedrxivClusteringP2P.v2 | 17.63% | default: 17.63% |
| MedrxivClusteringS2S.v2 | 18.42% | default: 18.42% |
| StackExchangeClustering.v2 | 40.91% | default: 40.91% |
| StackExchangeClusteringP2P.v2 | 30.96% | default: 30.96% |
| TwentyNewsgroupsClustering.v2 | 9.52% | default: 9.52% |

### STS

| Task | Average | Details |
|---|---|---|
| BIOSSES | 1.1% | default: 1.1% |
| SICK-R | 22.32% | default: 22.32% |
| STS12 | 7.97% | default: 7.97% |
| STS13 | 18.76% | default: 18.76% |
| STS14 | 10.4% | default: 10.4% |
| STS15 | 24.3% | default: 24.3% |
| STS17 | 10.58% | en-en: 25.95%, es-es: 25.71%, ko-ko: 21.6%, ar-ar: 20.19%, it-en: 8.19% |
| STSBenchmark | 7.96% | default: 7.96% |



## Training

Created via **multi-method model compression** (no additional training):

1. **Teacher**: `Qwen/Qwen3-0.6B` (28L, 1024d, 596M params)
2. **Layer pruning**: 28 → 4 layers (uniform selection)
3. **Hidden dim**: 1024 → 448
4. **Vocab pruning**: 151,936 → 7,341 (90% cumulative frequency)
5. **Compression ratio**: 48x


## Supported Languages (18)

ko, en, ja, zh, es, fr, de, pt, it, ru, ar, hi, th, vi, id, tr, nl, pl
