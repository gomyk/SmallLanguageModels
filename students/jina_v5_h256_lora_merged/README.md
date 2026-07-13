---
language:
- en
tags:
- sentence-transformers
- eurobert
- jina-embeddings
- model-compression
- lora
- classification
- fine-tuning
library_name: transformers
pipeline_tag: sentence-similarity
license: apache-2.0
base_model: gomyk/jina-v5-h256-distilled-conv
---

# Jina v5 H256 Distilled + Classification LoRA (Merged)

Compressed Jina v5 embedding model with a **universal classification LoRA merged** into the weights.
The LoRA was trained via multi-task classification on all 8 MTEB Classification tasks,
then permanently merged into the base model for zero-overhead inference.

## Model Details

| Property | Value |
|---|---|
| Base model | [`gomyk/jina-v5-h256-distilled-conv`](https://huggingface.co/gomyk/jina-v5-h256-distilled-conv) |
| Original teacher | [`jinaai/jina-embeddings-v5-text-nano`](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano) (239M) |
| Architecture | EuroBERT (encoder) |
| Hidden dim | 256 |
| Layers | 6 |
| Attention heads | 4 (head_dim=64) |
| Intermediate | 1024 (SiLU GLU) |
| Vocab size | 41,778 |
| Parameters | ~16.9M |
| Model size (FP32) | 64.8MB |

## Merged LoRA Specification

| Property | Value |
|---|---|
| LoRA rank | **8** |
| LoRA alpha | **16** |
| LoRA scaling | 2.0 (alpha/rank) |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| LoRA matrices | 24 (6 layers x 4 projections) |
| LoRA A shape | `[256, 8]` per projection |
| LoRA B shape | `[8, 256]` per projection |
| LoRA params | 98,304 (0.58% of base) |
| Merge method | `W_new = W + 2.0 * (A @ B)^T` |
| Training method | Multi-task classification (8 tasks, 112K samples) |
| Training loss | Cross-Entropy per task head (shared LoRA backbone) |

## MTEB Evaluation Results

**Overall Average (25 tasks): 50.67%**

> **Note:** This model is optimized for **classification only**. The LoRA merge significantly
> improves classification but degrades STS performance. Use the base model
> [`gomyk/jina-v5-h256-distilled-conv`](https://huggingface.co/gomyk/jina-v5-h256-distilled-conv)
> if you need STS or general-purpose embeddings.

| Task Group | Base Model | + LoRA Merged | Delta |
|---|---|---|---|
| **Classification** (8 tasks) | 73.20% | **82.61%** | **+9.41%p** |
| STS (9 tasks) | **73.14%** | 41.54% | -31.60%p |
| Clustering (8 tasks) | **33.11%** | 29.02% | -4.09%p |

### Classification (+9.41%p improvement)

| Task | Base | Merged | Delta |
|---|---|---|---|
| AmazonCounterfactualClassification | 76.93% | **90.86%** | +13.93%p |
| Banking77Classification | 77.83% | **84.69%** | +6.86%p |
| ImdbClassification | 73.03% | **78.99%** | +5.96%p |
| MTOPDomainClassification | 90.63% | **96.20%** | +5.57%p |
| MassiveIntentClassification | 67.90% | **77.81%** | +9.91%p |
| MassiveScenarioClassification | 72.97% | **85.89%** | +12.92%p |
| ToxicConversationsClassification | 61.83% | **75.95%** | +14.12%p |
| TweetSentimentExtractionClassification | 64.46% | **70.45%** | +5.99%p |

### STS (-31.60%p regression)

| Task | Base | Merged | Delta |
|---|---|---|---|
| BIOSSES | 64.04% | 30.63% | -33.41%p |
| SICK-R | 81.45% | 59.48% | -21.97%p |
| STS12 | 75.81% | 43.44% | -32.37%p |
| STS13 | 76.53% | 41.75% | -34.78%p |
| STS14 | 77.58% | 39.18% | -38.40%p |
| STS15 | 84.28% | 52.87% | -31.41%p |
| STS17 | 58.86% | 10.05% | -48.81%p |
| STS22.v2 | 55.54% | 47.22% | -8.32%p |
| STSBenchmark | 84.20% | 49.27% | -34.93%p |

### Clustering (-4.09%p regression)

| Task | Base | Merged | Delta |
|---|---|---|---|
| ArXivHierarchicalClusteringP2P | 47.94% | 47.96% | +0.02%p |
| ArXivHierarchicalClusteringS2S | 46.85% | 48.34% | +1.49%p |
| BiorxivClusteringP2P.v2 | 19.40% | 9.85% | -9.55%p |
| MedrxivClusteringP2P.v2 | 27.74% | 21.20% | -6.54%p |
| MedrxivClusteringS2S.v2 | 23.38% | 19.21% | -4.17%p |
| StackExchangeClustering.v2 | 42.76% | 42.50% | -0.26%p |
| StackExchangeClusteringP2P.v2 | 33.98% | 32.97% | -1.01%p |
| TwentyNewsgroupsClustering.v2 | 22.80% | 10.11% | -12.69%p |

## Quick Start

```python
from transformers import AutoModel, AutoTokenizer
import torch

model = AutoModel.from_pretrained(
    "gomyk/jina-v5-h256-lora-clf-merged", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(
    "gomyk/jina-v5-h256-lora-clf-merged", trust_remote_code=True)

texts = ["This movie was amazing!", "I need to transfer money"]
inputs = tokenizer(texts, padding=True, truncation=True,
                   max_length=128, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    hidden = outputs.last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    embeddings = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

print(embeddings.shape)  # [2, 256]

# Cosine similarity
from torch.nn.functional import cosine_similarity
sim = cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))
print(f"Similarity: {sim.item():.4f}")
```

## Training Pipeline

```
1. Compression (from base model repo):
   jinaai/jina-embeddings-v5-text-nano (12L/768d/128K vocab/239M)
   → Layer pruning (12→6)
   → Hidden dim PCA (768→256)
   → Vocab pruning with BPE backtracking (128K→42K)
   → Knowledge distillation (MSE + Cosine loss)
   = gomyk/jina-v5-h256-distilled-conv (6L/256d/42K vocab/16.9M)

2. Classification LoRA (this model):
   → LoRA (rank=8) on all attention projections
   → Multi-task training on 8 MTEB Classification tasks
   → Merge LoRA into base weights
   = This model (same size, better classification)
```

## License

Apache 2.0
