---
language:
- en
tags:
- lora
- classification
- sentence-transformers
- eurobert
- jina-embeddings
- model-compression
- fine-tuning
- multi-task-learning
library_name: transformers
pipeline_tag: text-classification
license: apache-2.0
base_model: gomyk/jina-v5-h256-distilled-conv
---

# Jina v5 H256 — Universal Classification LoRA

A single **LoRA adapter** (405KB) that improves classification performance across all MTEB Classification tasks, built on top of the compressed [`gomyk/jina-v5-h256-distilled-conv`](https://huggingface.co/gomyk/jina-v5-h256-distilled-conv) embedding model.

Unlike per-task adapters, this is **one universal adapter** that enhances the embedding space for classification in general — no task-specific fine-tuning needed at inference time.

## LoRA Specification

| Property | Value |
|---|---|
| Base model | [`gomyk/jina-v5-h256-distilled-conv`](https://huggingface.co/gomyk/jina-v5-h256-distilled-conv) |
| Base architecture | EuroBERT (6L / 256d / 41,778 vocab / 16.9M params) |
| LoRA rank | **8** |
| LoRA alpha | **16** |
| LoRA scaling (alpha/rank) | **2.0** |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| LoRA layers | 6 layers x 4 projections = **24 LoRA matrices** |
| LoRA A shape | `[256, 8]` per projection |
| LoRA B shape | `[8, 256]` per projection |
| Total LoRA params | **98,304** (0.58% of base model) |
| Adapter file size | **405KB** |

## MTEB Classification Results

**Average: 73.20% → 82.98% (+9.78%p)**

| Task | Classes | Baseline | + LoRA | Delta |
|---|---|---|---|---|
| AmazonCounterfactualClassification | 2 | 76.93% | **91.57%** | +14.64%p |
| Banking77Classification | 77 | 77.83% | **84.81%** | +6.98%p |
| ImdbClassification | 2 | 73.03% | **79.54%** | +6.51%p |
| MTOPDomainClassification | 11 | 90.63% | **96.18%** | +5.55%p |
| MassiveIntentClassification | 60 | 67.90% | **78.08%** | +10.18%p |
| MassiveScenarioClassification | 18 | 72.97% | **85.68%** | +12.71%p |
| ToxicConversationsClassification | 2 | 61.83% | **77.99%** | +16.16%p |
| TweetSentimentExtractionClassification | 3 | 64.46% | **70.01%** | +5.55%p |
| **Average** | | **73.20%** | **82.98%** | **+9.78%p** |

> **Baseline**: MTEB default evaluation (logistic regression on frozen embeddings)
> **+ LoRA**: Same evaluation, but with universal LoRA adapter merged into embeddings

## Training Method

**Multi-Task Classification** with shared LoRA backbone:

```
                    ┌─ Head_Amazon(256→2)   ─→ CE loss
                    ├─ Head_Banking(256→77)  ─→ CE loss
Input → [EuroBERT   ├─ Head_IMDB(256→2)     ─→ CE loss
         + LoRA]    ├─ Head_MTOP(256→11)    ─→ CE loss
         → pool  ──►├─ Head_Massive_I(256→60)─→ CE loss
                    ├─ Head_Massive_S(256→18)─→ CE loss
                    ├─ Head_Toxic(256→2)    ─→ CE loss
                    └─ Head_Tweet(256→3)    ─→ CE loss

All heads share the same LoRA backbone.
After training, heads are discarded — only the LoRA adapter is kept.
The LoRA-enhanced embeddings are universally better for classification.
```

| Parameter | Value |
|---|---|
| Training data | 112,716 samples from 8 MTEB Classification tasks |
| Total classes | 175 (across all tasks) |
| Optimizer | AdamW (lr=2e-4, weight_decay=0.01) |
| Scheduler | CosineAnnealingLR |
| Epochs | 10 |
| Batch size | 32 |
| Max sequence length | 128 |
| Gradient clipping | max_norm=1.0 |
| Loss function | Cross-Entropy (per task head) |
| Final training loss | 0.189 |
| Final training accuracy | ~93% |

## Usage

### Option 1: Merge LoRA into base model

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download

# Load base model
model = AutoModel.from_pretrained(
    "gomyk/jina-v5-h256-distilled-conv", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(
    "gomyk/jina-v5-h256-distilled-conv", trust_remote_code=True)

# Download and merge LoRA
lora_path = hf_hub_download(
    "gomyk/jina-v5-h256-lora-classification-universal",
    "lora_adapter.pt")
lora_state = torch.load(lora_path, map_location="cpu", weights_only=True)

idx = 0
for name, module in model.named_modules():
    for target in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        child = getattr(module, target, None)
        if child is not None and isinstance(child, nn.Linear):
            A = lora_state[f"lora_{idx}_A"]
            B = lora_state[f"lora_{idx}_B"]
            scaling = lora_state[f"lora_{idx}_scaling"].item()
            child.weight.data += (scaling * (A @ B)).T
            idx += 1

# Now use as normal embedding model
model.eval()
inputs = tokenizer("This movie was great!", return_tensors="pt",
                    truncation=True, max_length=128)
with torch.no_grad():
    outputs = model(**inputs)
    hidden = outputs.last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    embedding = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

print(embedding.shape)  # [1, 256]
```

### Option 2: Use with downstream classifier

```python
# After merging LoRA (see above), add your own classifier
from sklearn.linear_model import LogisticRegression

# Encode your training data
train_embeddings = []  # encode your texts with the merged model
train_labels = [...]

clf = LogisticRegression(max_iter=1000)
clf.fit(train_embeddings, train_labels)

# Predict
test_embedding = ...  # encode test text
prediction = clf.predict(test_embedding)
```

## File Structure

```
.
├── README.md              # This file
├── lora_adapter.pt        # LoRA A/B matrices (405KB)
├── meta.json              # Training metadata
└── task_info.json         # Per-task class counts
```

## How LoRA Works

```
For each of the 24 attention projections (4 per layer x 6 layers):

  Original:  y = W @ x                    W: [256, 256] — frozen
  With LoRA: y = W @ x + (16/8) * (x @ A) @ B

  A: [256, 8]  — learned down-projection
  B: [8, 256]  — learned up-projection

  At merge time: W_new = W + 2.0 * (A @ B)^T  — zero overhead at inference
```

## License

Apache 2.0
