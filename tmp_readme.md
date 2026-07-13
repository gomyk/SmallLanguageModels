---
language:
- en
- ko
- ja
- zh
tags:
- sentence-transformers
- multilingual
- layer-pruning
- vocab-pruning
- knowledge-distillation
- eurobert
library_name: sentence-transformers
pipeline_tag: sentence-similarity
license: apache-2.0
base_model: jinaai/jina-embeddings-v5-text-nano
---

# jina-v5-h256-distilled-conv

Lightweight multilingual sentence encoder compressed from [`jinaai/jina-embeddings-v5-text-nano`](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano) (EuroBERT-210M, 12L/768d) via layer pruning + vocabulary pruning + knowledge distillation with conversation data.

## Model Details

| Property | Value |
|---|---|
| Teacher | jinaai/jina-embeddings-v5-text-nano (239M params) |
| Architecture | EuroBERT (pruned) |
| Hidden dim | 256 |
| Layers | 6 / 12 |
| Intermediate | 1024 |
| Model size (FP32) | 64.8 MB |
| Embedding dim | 256 |
| Compression | ~3.7x (239M -> ~16M params) |

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("gomyk/jina-v5-h256-distilled-conv", trust_remote_code=True)
embeddings = model.encode(["Hello world", "How are you?"])
print(embeddings.shape)  # (2, 256)
```

## Training

### Stage 1: Architecture Compression
- **Teacher**: `jinaai/jina-embeddings-v5-text-nano` (12 layers, 768d, 128K vocab)
- **Layer pruning**: 12 -> 6 layers
- **Dimension reduction**: 768d -> 256d
- **Vocabulary pruning**: BPE merge-backtracked corpus-based filtering

### Stage 2: Knowledge Distillation
- **Method**: MSE + Cosine Similarity loss (MSE weight=1.0, Cosine weight=0.5)
- **Data**: MTEB task datasets (~1.4M) + conversation data (~19.5M) = ~20.9M texts
- **Optimizer**: AdamW (lr=2e-5, weight_decay=0.01)
- **Schedule**: Cosine annealing
- **Training**: ~2.5 epochs (1.37M global steps), best loss=0.0232
- **Projection**: Learnable linear projection 256d -> 768d for distillation

## MTEB Evaluation Results

**Overall Average: 56.29%**

| Task Group | Average |
|---|---|
| Classification | 63.86% |
| Clustering | 32.62% |
| STS | 70.60% |

### Classification

| Task | Score |
|---|---|
| AmazonCounterfactualClassification | 72.81% |
| Banking77Classification | 77.90% |
| ImdbClassification | 73.39% |
| MTOPDomainClassification | 86.91% |
| MassiveIntentClassification | 32.96% |
| MassiveScenarioClassification | 40.19% |
| ToxicConversationsClassification | 62.45% |
| TweetSentimentExtractionClassification | 64.25% |

### Clustering

| Task | Score |
|---|---|
| ArXivHierarchicalClusteringP2P | 48.36% |
| ArXivHierarchicalClusteringS2S | 46.85% |
| BiorxivClusteringP2P.v2 | 18.36% |
| MedrxivClusteringP2P.v2 | 24.59% |
| MedrxivClusteringS2S.v2 | 23.35% |
| StackExchangeClustering.v2 | 42.75% |
| StackExchangeClusteringP2P.v2 | 33.83% |
| TwentyNewsgroupsClustering.v2 | 22.86% |

### STS (Semantic Textual Similarity)

| Task | Score |
|---|---|
| BIOSSES | 63.87% |
| SICK-R | 81.44% |
| STS12 | 75.81% |
| STS13 | 76.53% |
| STS14 | 77.58% |
| STS15 | 84.28% |
| STS17 | 57.45% |
| STS22.v2 | 34.21% |
| STSBenchmark | 84.20% |
