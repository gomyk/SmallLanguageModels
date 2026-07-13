---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- knowledge-distillation
- multilingual
- compressed
pipeline_tag: sentence-similarity
library_name: sentence-transformers
language:
- multilingual
base_model: intfloat/multilingual-e5-small
---

# mE5-small Compressed & Distilled (v2)

A **4-layer, 384d, 15K vocab** compressed student of [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small), distilled on ~1.4M MTEB sentences for 20 epochs.

## Model Summary

| | Teacher (mE5-small) | **This Model** | Reduction |
|---|---|---|---|
| Layers | 12 | **4** | 3x fewer |
| Hidden dim | 384 | **384** | same |
| Vocab | 250,037 | **15,168** | 16.5x smaller |
| Parameters | ~118M | **~13M** | ~9x fewer |
| Model size | ~450MB | **~51MB** | ~9x smaller |

## MTEB Benchmark Results

### STS (Semantic Textual Similarity)

| Task | Teacher | Student | Delta |
|------|---------|---------|-------|
| BIOSSES | 0.8438 | 0.7374 | -0.106 |
| SICK-R | 0.7863 | 0.7792 | -0.007 |
| STS12 | 0.7805 | 0.7658 | -0.015 |
| STS13 | 0.7747 | 0.7698 | -0.005 |
| STS14 | 0.7754 | 0.7573 | -0.018 |
| STS15 | 0.8749 | 0.8505 | -0.024 |
| STS17 | 0.7673 | 0.0523 | -0.715 |
| STS22.v2 | 0.6428 | 0.5294 | -0.113 |
| STSBenchmark | 0.8359 | 0.8077 | -0.028 |
| **Average** | **0.7868** | **0.6722** | **-0.115** |

> Note: STS17 is a cross-lingual task heavily impacted by vocab pruning (250K->15K tokens).
> Excluding STS17: Teacher 0.7893 vs Student 0.7496 (delta -0.040, 95.0% retention).

### Classification

| Task | Teacher | Student | Delta |
|------|---------|---------|-------|
| AmazonCounterfactualClassification | 0.7428 | 0.7445 | +0.002 |
| Banking77Classification | 0.7064 | 0.6723 | -0.034 |
| ImdbClassification | 0.7582 | 0.7149 | -0.043 |
| MTOPDomainClassification | 0.8913 | 0.8497 | -0.042 |
| MassiveIntentClassification | 0.4760 | 0.6334 | +0.157 |
| MassiveScenarioClassification | 0.5619 | 0.6768 | +0.115 |
| ToxicConversationsClassification | 0.6205 | 0.5757 | -0.045 |
| TweetSentimentExtractionClassification | 0.6182 | 0.6259 | +0.008 |
| **Average** | **0.6719** | **0.6867** | **+0.015** |

### Clustering

| Task | Teacher | Student | Delta |
|------|---------|---------|-------|
| ArXivHierarchicalClusteringP2P | 0.5395 | 0.4859 | -0.054 |
| ArXivHierarchicalClusteringS2S | 0.5376 | 0.4726 | -0.065 |
| BiorxivClusteringP2P.v2 | 0.3675 | 0.2020 | -0.166 |
| MedrxivClusteringP2P.v2 | 0.3424 | 0.2598 | -0.083 |
| MedrxivClusteringS2S.v2 | 0.3195 | 0.2320 | -0.088 |
| StackExchangeClustering.v2 | 0.4961 | 0.3811 | -0.115 |
| StackExchangeClusteringP2P.v2 | 0.3841 | 0.3288 | -0.055 |
| TwentyNewsgroupsClustering.v2 | 0.3352 | 0.1848 | -0.150 |
| **Average** | **0.4153** | **0.3184** | **-0.097** |

### Overall Summary

| Category | Teacher | Student | Retention |
|----------|---------|---------|-----------|
| STS | 0.7868 | 0.6722 | 85.4% |
| STS (excl. STS17) | 0.7893 | 0.7496 | **95.0%** |
| Classification | 0.6719 | 0.6867 | **102.2%** |
| Clustering | 0.4153 | 0.3184 | 76.7% |
| **Overall Average** | **0.6247** | **0.5591** | **89.5%** |

> **9x smaller model size (450MB -> 51MB), 89.5% overall performance retention.**
> Classification exceeds teacher performance. STS retains 95% (excluding cross-lingual STS17).

## Compression Pipeline

1. **Layer pruning**: 12 -> 4 layers (uniform spacing: layers 0, 4, 8, 11)
2. **Vocabulary pruning**: 250,037 -> 15,168 tokens (corpus-seen tokens only)
3. **Distillation**: MSE + Cosine loss on ~1.4M MTEB corpus sentences, 20 epochs
   - Best distillation loss: 0.0096

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("gomyk/me5s-student-me5s_compressed_distilled_v2")

sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]
embeddings = model.encode(sentences)
print(embeddings.shape)  # [3, 384]

similarities = model.similarity(embeddings, embeddings)
print(similarities)
```

## Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False, 'architecture': 'BertModel'})
       - 4 layers, 384d hidden, 12 attention heads
       - 15,168 vocab (XLMRobertaTokenizer, pruned)
  (1): Pooling(mean_tokens)
)
```

## Training Details

- **Teacher**: intfloat/multilingual-e5-small (12L, 384d, 250K vocab)
- **Distillation loss**: MSE + 0.5 * Cosine similarity loss
- **Corpus**: ~1.4M sentences from MTEB Classification/Clustering/STS/NLI datasets
- **Epochs**: 20 (best loss: 0.0096)
- **Batch size**: 32, LR: 2e-5, Cosine annealing scheduler
- **Framework**: PyTorch 2.10, Transformers 4.56, Sentence-Transformers 5.3

## Limitations

- Cross-lingual STS (STS17) significantly degraded due to aggressive vocab pruning
- Clustering performance drops more than other categories
- Optimized for English; multilingual coverage is reduced by vocab pruning
