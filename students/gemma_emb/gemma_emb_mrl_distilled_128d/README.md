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

# gemma_emb_mrl_distilled_128d (Distilled)

Compact multilingual sentence encoder compressed from `google/embeddinggemma-300m` (27x compression).

## Model Details

| Property | Value |
|---|---|
| Base model | `google/embeddinggemma-300m` |
| Architecture | gemma3_text (decoder) |
| Hidden dim | 128 (from 768) |
| Layers | 6 (from 24) |
| Intermediate | 192 |
| Attention heads | 1 |
| KV heads | 1 |
| Vocab size | 81,052 (from 262,144) |
| Parameters | ~11.2M |
| Model size (FP32) | 42.8MB |
| Compression | 27x |
| Distilled | Yes (2-stage) |

## Quick Start

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("gemma_emb_mrl_distilled_128d", trust_remote_code=True)

sentences = [
    "Hello, how are you?",
    "안녕하세요, 잘 지내세요?",
    "こんにちは、元気ですか？",
    "你好，你好吗？",
]

embeddings = model.encode(sentences)
print(embeddings.shape)  # (4, 128)
```

## MTEB Evaluation Results

**Overall Average: 23.8%**

| Task Group | Average |
|---|---|
| Classification | 31.32% |
| Clustering | 27.55% |
| STS | 13.78% |

### Classification

| Task | Average | Details |
|---|---|---|
| AmazonCounterfactualClassification | 52.78% | en: 54.49%, ja: 52.73%, en-ext: 52.64%, de: 51.27% |
| Banking77Classification | 8.31% | default: 8.31% |
| ImdbClassification | 50.36% | default: 50.36% |
| MTOPDomainClassification | 27.22% | th: 36.78%, de: 29.73%, es: 29.22%, en: 26.76%, fr: 24.91% |
| MassiveIntentClassification | 13.73% | th: 19.26%, cy: 18.33%, pl: 18.0%, sq: 17.67%, it: 17.43% |
| MassiveScenarioClassification | 15.97% | cy: 20.38%, sl: 20.23%, sv: 19.6%, km: 19.52%, ka: 19.4% |
| ToxicConversationsClassification | 47.43% | default: 47.43% |
| TweetSentimentExtractionClassification | 34.78% | default: 34.78% |

### Clustering

| Task | Average | Details |
|---|---|---|
| ArXivHierarchicalClusteringP2P | 48.99% | default: 48.99% |
| ArXivHierarchicalClusteringS2S | 46.35% | default: 46.35% |
| BiorxivClusteringP2P.v2 | 6.42% | default: 6.42% |
| MedrxivClusteringP2P.v2 | 18.95% | default: 18.95% |
| MedrxivClusteringS2S.v2 | 18.15% | default: 18.15% |
| StackExchangeClustering.v2 | 40.92% | default: 40.92% |
| StackExchangeClusteringP2P.v2 | 32.65% | default: 32.65% |
| TwentyNewsgroupsClustering.v2 | 7.97% | default: 7.97% |

### STS

| Task | Average | Details |
|---|---|---|
| BIOSSES | 8.94% | default: 8.94% |
| SICK-R | 20.63% | default: 20.63% |
| STS12 | 12.23% | default: 12.23% |
| STS13 | 15.73% | default: 15.73% |
| STS14 | 8.76% | default: 8.76% |
| STS15 | 27.92% | default: 27.92% |
| STS17 | 17.1% | es-es: 34.2%, en-en: 33.26%, ar-ar: 30.05%, ko-ko: 21.98%, nl-en: 13.79% |
| STS22.v2 | 5.47% | es: 18.55%, fr: 18.16%, en: 17.61%, fr-pl: 16.9%, zh: 14.38% |
| STSBenchmark | 7.27% | default: 7.27% |



## Training

### Stage 1: Model Compression
- **Teacher**: `google/embeddinggemma-300m` (24L, 768d, 303M params)
- **Compression**: Layer pruning → Hidden dim reduction → Vocab pruning
- **Result**: 6L / 128d / 81,052 vocab

### Stage 2: Two-Stage Knowledge Distillation
Compression ratio 27x requires progressive distillation:

1. **Stage 1**: Teacher (303M) → Intermediate (~61M)
   - MSE + Cosine Similarity loss
   - MTEB task datasets (Classification/Clustering/STS)
2. **Stage 2**: Intermediate → Final Student (11.2M)
   - Same training objective
   - AdamW (lr=2e-5, weight_decay=0.01), Cosine annealing

## License

This model is a derivative of Google's Gemma. Gemma is provided under and subject to the Gemma Terms of Use found at [ai.google.dev/gemma/terms](https://ai.google.dev/gemma/terms). Use of this model must comply with the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).


## Supported Languages (18)

ko, en, ja, zh, es, fr, de, pt, it, ru, ar, hi, th, vi, id, tr, nl, pl
