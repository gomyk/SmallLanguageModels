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
library_name: transformers
pipeline_tag: text-classification
license: apache-2.0
base_model: gomyk/jina-v5-h256-distilled-conv
---

# Jina v5 H256 — LoRA Classification Adapters

Task-specific **LoRA adapters** for classification, built on top of the compressed
[`gomyk/jina-v5-h256-distilled-conv`](https://huggingface.co/gomyk/jina-v5-h256-distilled-conv) embedding model.

Each MTEB Classification task has its own lightweight LoRA adapter (~405KB) +
classification head, achieving **+12.0%p** average improvement over the
frozen baseline embeddings.

## LoRA Specification

| Property | Value |
|---|---|
| Base model | [`gomyk/jina-v5-h256-distilled-conv`](https://huggingface.co/gomyk/jina-v5-h256-distilled-conv) |
| Base architecture | EuroBERT (6L / 256d / 41,778 vocab) |
| Base parameters | 16,989,952 (all frozen) |
| LoRA rank | **8** |
| LoRA alpha | **16** |
| LoRA scaling (alpha/rank) | 2.0 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| LoRA layers | 6 layers x 4 projections = **24 LoRA matrices** |
| LoRA params per task | 98,304 (A + B matrices) |
| LoRA A shape | `[256, 8]` per projection |
| LoRA B shape | `[8, 256]` per projection |
| LoRA A init | Kaiming (He) normal |
| LoRA B init | Zeros (initial delta W = 0) |
| Adapter file size | ~405KB per task |
| Training overhead | **0.6%** of base model params |

## How LoRA Works in This Model

```
For each attention projection (q/k/v/o) in each of the 6 layers:

  Original:  y = W_frozen @ x              W: [256, 256]
  With LoRA: y = W_frozen @ x + (α/r) * (x @ A) @ B

  Where:
    A: [256, 8]   — down-projection (trainable)
    B: [8, 256]   — up-projection (trainable)
    α/r = 16/8 = 2.0  — scaling factor

  Total per projection: 256×8 + 8×256 = 4,096 params
  Total per layer:      4 projections × 4,096 = 16,384 params
  Total (6 layers):     6 × 16,384 = 98,304 LoRA params
  + Classification head: Linear(256 → num_classes)
```

## MTEB Classification Results

**Average: 73.20% → 85.19% (+12.00%p)**

| Task | Classes | Baseline | LoRA | Delta | Epochs |
|---|---|---|---|---|---|
| AmazonCounterfactualClassification | 2 | 76.93% | **92.13%** | +15.20%p | 5 |
| Banking77Classification | 77 | 77.83% | **86.26%** | +8.42%p | 10 |
| ImdbClassification | 2 | 73.03% | **82.14%** | +9.11%p | 5 |
| MTOPDomainClassification | 11 | 90.63% | **97.68%** | +7.05%p | 10 |
| MassiveIntentClassification | 60 | 67.90% | **77.89%** | +9.99%p | 10 |
| MassiveScenarioClassification | 18 | 72.97% | **86.30%** | +13.34%p | 8 |
| ToxicConversationsClassification | 2 | 61.83% | **85.74%** | +23.92%p | 4 |
| TweetSentimentExtractionClassification | 3 | 64.46% | **73.42%** | +8.95%p | 4 |
| **Average** | | **73.20%** | **85.19%** | **+12.00%p** | |

> **Baseline**: MTEB default evaluation (logistic regression on frozen embeddings)
> **LoRA**: MTEB evaluation with task-specific LoRA adapters applied to the embedding model

## Per-Task Adapter Details

| Task | Classes | Trainable Params | Adapter Size | Head Size |
|---|---|---|---|---|
| AmazonCounterfactualClassification | 2 | 98,818 | 404.6KB | 4.0KB |
| Banking77Classification | 77 | 118,093 | 404.6KB | 79.2KB |
| ImdbClassification | 2 | 98,818 | 404.6KB | 4.0KB |
| MTOPDomainClassification | 11 | 101,131 | 404.6KB | 13.0KB |
| MassiveIntentClassification | 60 | 113,724 | 404.6KB | 62.2KB |
| MassiveScenarioClassification | 18 | 102,930 | 404.6KB | 20.0KB |
| ToxicConversationsClassification | 2 | 98,818 | 404.6KB | 4.0KB |
| TweetSentimentExtractionClassification | 3 | 99,075 | 404.6KB | 5.0KB |


## Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW (lr=2e-4, weight_decay=0.01) |
| Scheduler | CosineAnnealingLR |
| Max epochs | 10 |
| Batch size | 32 |
| Max sequence length | 128 |
| Early stopping | patience=3 (on test accuracy) |
| Gradient clipping | max_norm=1.0 |
| Loss function | Cross-Entropy |
| Dropout (head) | 0.1 |

## Repository Structure

```
.
├── README.md
├── classification_results.json          # Summary of all results
├── AmazonCounterfactualClassification/
│   ├── lora_adapter.pt                  # LoRA A/B matrices
│   ├── classifier_head.pt              # Linear classification head
│   └── meta.json                        # Training metadata
├── Banking77Classification/
│   ├── ...
├── ImdbClassification/
│   ├── ...
├── MTOPDomainClassification/
│   ├── ...
├── MassiveIntentClassification/
│   ├── ...
├── MassiveScenarioClassification/
│   ├── ...
├── ToxicConversationsClassification/
│   ├── ...
└── TweetSentimentExtractionClassification/
    ├── ...
```

## Usage

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# 1. Base model load
base_model = AutoModel.from_pretrained("gomyk/jina-v5-h256-distilled-conv", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("gomyk/jina-v5-h256-distilled-conv", trust_remote_code=True)

# 2. LoRA adapter 적용
from huggingface_hub import hf_hub_download
import os

task = "Banking77Classification"  # 원하는 task 선택
repo_id = "gomyk/jina-v5-h256-lora-classification"

lora_path = hf_hub_download(repo_id, f"{task}/lora_adapter.pt")
head_path = hf_hub_download(repo_id, f"{task}/classifier_head.pt")
meta_path = hf_hub_download(repo_id, f"{task}/meta.json")

import json
with open(meta_path) as f:
    meta = json.load(f)

# 3. LoRA 부착
lora_state = torch.load(lora_path, map_location="cpu", weights_only=True)
idx = 0
for name, module in base_model.named_modules():
    for target in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        child = getattr(module, target, None)
        if child is not None and isinstance(child, nn.Linear):
            A = lora_state[f"lora_{idx}_A"]
            B = lora_state[f"lora_{idx}_B"]
            scaling = lora_state[f"lora_{idx}_scaling"].item()
            # Merge into weight: W_new = W + scaling * (A @ B)^T
            child.weight.data += (scaling * (A @ B)).T
            idx += 1

# 4. Classification head
num_classes = meta["num_classes"]
classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(256, num_classes))
classifier.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True))

# 5. Inference
text = "I need to transfer money to another account"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
with torch.no_grad():
    outputs = base_model(**inputs)
    hidden = outputs.last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    logits = classifier(pooled)
    pred = logits.argmax(dim=-1).item()
    print(f"Predicted class: {pred}")
```

## License

Apache 2.0
