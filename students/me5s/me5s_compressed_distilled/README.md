---
language: ["ko", "en", "ja", "zh", "es", "fr", "de", "pt", "it", "ru", "ar", "hi", "th", "vi", "id", "tr", "nl", "pl"]
tags:
- sentence-transformers
- multilingual
- model-compression
- layer-pruning
- vocab-pruning
- knowledge-distillation
- me5-small
library_name: sentence-transformers
pipeline_tag: sentence-similarity
license: apache-2.0
---

# me5s_compressed_distilled (Distilled)

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
| Distilled | Yes |

## Quick Start

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("me5s_compressed_distilled", trust_remote_code=True)

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

**Overall Average: 50.81%**

| Task Group | Average |
|---|---|
| Classification | 56.17% |
| Clustering | 29.64% |
| STS | 64.86% |

### Classification

| Task | Average | Details |
|---|---|---|
| AmazonCounterfactualClassification | 67.34% | de: 71.67%, en: 71.63%, en-ext: 69.81%, ja: 56.27% |
| Banking77Classification | 67.81% | default: 67.81% |
| ImdbClassification | 54.45% | default: 54.45% |
| MTOPDomainClassification | 74.88% | en: 84.01%, es: 78.04%, fr: 75.53%, hi: 72.32%, th: 72.24% |
| MassiveIntentClassification | 30.9% | en: 61.8%, ja: 56.38%, zh-CN: 56.37%, ko: 54.29%, es: 54.0% |
| MassiveScenarioClassification | 38.29% | en: 68.19%, zh-CN: 67.13%, ja: 64.95%, de: 63.29%, ko: 63.23% |
| ToxicConversationsClassification | 55.65% | default: 55.65% |
| TweetSentimentExtractionClassification | 60.03% | default: 60.03% |

### Clustering

| Task | Average | Details |
|---|---|---|
| ArXivHierarchicalClusteringP2P | 49.54% | default: 49.54% |
| ArXivHierarchicalClusteringS2S | 46.49% | default: 46.49% |
| BiorxivClusteringP2P.v2 | 13.53% | default: 13.53% |
| MedrxivClusteringP2P.v2 | 22.28% | default: 22.28% |
| MedrxivClusteringS2S.v2 | 21.37% | default: 21.37% |
| StackExchangeClustering.v2 | 38.82% | default: 38.82% |
| StackExchangeClusteringP2P.v2 | 30.07% | default: 30.07% |
| TwentyNewsgroupsClustering.v2 | 15.01% | default: 15.01% |

### STS

| Task | Average | Details |
|---|---|---|
| BIOSSES | 63.23% | default: 63.23% |
| SICK-R | 74.84% | default: 74.84% |
| STS12 | 74.24% | default: 74.24% |
| STS13 | 69.75% | default: 69.75% |
| STS14 | 70.86% | default: 70.86% |
| STS15 | 82.14% | default: 82.14% |
| STS17 | 40.06% | en-en: 82.62%, es-es: 76.34%, ar-ar: 60.47%, ko-ko: 60.46%, nl-en: 40.14% |
| STS22.v2 | 31.5% | zh: 60.47%, es: 56.06%, fr: 54.8%, it: 40.81%, en: 40.52% |
| STSBenchmark | 77.11% | default: 77.11% |


## Distillation Impact

| Task | Before | After | Delta |
|---|---|---|---|
| AmazonCounterfactualClassification | 67.37% | 67.34% | -0.03%p |
| ArXivHierarchicalClusteringP2P | 47.08% | 49.54% | +2.46%p |
| ArXivHierarchicalClusteringS2S | 48.29% | 46.49% | -1.8%p |
| BIOSSES | 56.68% | 63.23% | +6.55%p |
| Banking77Classification | 58.7% | 67.81% | +9.11%p |
| BiorxivClusteringP2P.v2 | 17.24% | 13.53% | -3.71%p |
| ImdbClassification | 57.14% | 54.45% | -2.69%p |
| MTOPDomainClassification | 66.84% | 74.88% | +8.04%p |
| MassiveIntentClassification | 31.12% | 30.9% | -0.22%p |
| MassiveScenarioClassification | 34.85% | 38.29% | +3.44%p |
| MedrxivClusteringP2P.v2 | 24.42% | 22.28% | -2.14%p |
| MedrxivClusteringS2S.v2 | 21.55% | 21.37% | -0.18%p |
| SICK-R | 59.22% | 74.84% | +15.62%p |
| STS12 | 52.11% | 74.24% | +22.13%p |
| STS13 | 64.25% | 69.75% | +5.5%p |
| STS14 | 60.12% | 70.86% | +10.74%p |
| STS15 | 74.19% | 82.14% | +7.95%p |
| STS17 | 38.71% | 40.06% | +1.35%p |
| STS22.v2 | 27.7% | 31.5% | +3.8%p |
| STSBenchmark | 60.91% | 77.11% | +16.2%p |
| StackExchangeClustering.v2 | 39.42% | 38.82% | -0.6%p |
| StackExchangeClusteringP2P.v2 | 31.85% | 30.07% | -1.78%p |
| ToxicConversationsClassification | 55.82% | 55.65% | -0.17%p |
| TweetSentimentExtractionClassification | 45.74% | 60.03% | +14.29%p |
| TwentyNewsgroupsClustering.v2 | 13.35% | 15.01% | +1.66%p |


## Training

### Stage 1: Model Compression
- **Teacher**: `intfloat/multilingual-e5-small` (12L, 384d)
- **Compression**: Layer pruning + Vocab pruning
- **Result**: 4L / 384d / 15,168 vocab

### Stage 2: Knowledge Distillation
- **Method**: MSE + Cosine Similarity loss
- **Data**: MTEB Classification/Clustering/STS task datasets
- **Optimizer**: AdamW (lr=2e-5, weight_decay=0.01)
- **Schedule**: Cosine annealing over 3 epochs


## Supported Languages (18)

ko, en, ja, zh, es, fr, de, pt, it, ru, ar, hi, th, vi, id, tr, nl, pl
