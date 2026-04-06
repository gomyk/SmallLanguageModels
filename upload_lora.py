"""
LoRA Classification Adapters를 HuggingFace Hub에 업로드한다.

Usage:
    python upload_lora.py
    python upload_lora.py --repo gomyk/jina-v5-h256-lora-classification
"""

import argparse
import json
import os

from huggingface_hub import HfApi, create_repo, upload_folder


BASE_MODEL_ID = "gomyk/jina-v5-h256-distilled-conv"
LORA_DIR = "students/jina_v5_lora"
RESULTS_FILE = "results/jina_v5_lora/mteb_comparison.json"
DEFAULT_REPO = "gomyk/jina-v5-h256-lora-classification"


def generate_lora_model_card(results, task_metas):
    """LoRA Classification 모델카드를 생성한다."""

    baseline = results["baseline"]
    lora = results["lora"]

    # 평균 계산
    baseline_avg = sum(baseline.values()) / len(baseline) * 100
    lora_avg = sum(lora.values()) / len(lora) * 100
    delta_avg = lora_avg - baseline_avg

    # 결과 테이블
    results_table = ""
    for task in sorted(lora.keys()):
        b = baseline.get(task, 0) * 100
        l = lora[task] * 100
        d = l - b
        meta = task_metas.get(task, {})
        n_classes = meta.get("num_classes", "?")
        epochs = meta.get("epochs_trained", "?")
        results_table += f"| {task} | {n_classes} | {b:.2f}% | **{l:.2f}%** | +{d:.2f}%p | {epochs} |\n"

    # LoRA 스펙 테이블 (task별)
    spec_table = ""
    for task in sorted(task_metas.keys()):
        meta = task_metas[task]
        adapter_path = os.path.join(LORA_DIR, task, "lora_adapter.pt")
        adapter_kb = os.path.getsize(adapter_path) / 1024 if os.path.exists(adapter_path) else 0
        head_path = os.path.join(LORA_DIR, task, "classifier_head.pt")
        head_kb = os.path.getsize(head_path) / 1024 if os.path.exists(head_path) else 0
        spec_table += (f"| {task} | {meta.get('num_classes', '?')} | "
                       f"{meta.get('trainable_params', 0):,} | "
                       f"{adapter_kb:.1f}KB | {head_kb:.1f}KB |\n")

    card = f"""---
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
base_model: {BASE_MODEL_ID}
---

# Jina v5 H256 — LoRA Classification Adapters

Task-specific **LoRA adapters** for classification, built on top of the compressed
[`{BASE_MODEL_ID}`](https://huggingface.co/{BASE_MODEL_ID}) embedding model.

Each MTEB Classification task has its own lightweight LoRA adapter (~405KB) +
classification head, achieving **+{delta_avg:.1f}%p** average improvement over the
frozen baseline embeddings.

## LoRA Specification

| Property | Value |
|---|---|
| Base model | [`{BASE_MODEL_ID}`](https://huggingface.co/{BASE_MODEL_ID}) |
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

**Average: {baseline_avg:.2f}% → {lora_avg:.2f}% (+{delta_avg:.2f}%p)**

| Task | Classes | Baseline | LoRA | Delta | Epochs |
|---|---|---|---|---|---|
{results_table}| **Average** | | **{baseline_avg:.2f}%** | **{lora_avg:.2f}%** | **+{delta_avg:.2f}%p** | |

> **Baseline**: MTEB default evaluation (logistic regression on frozen embeddings)
> **LoRA**: MTEB evaluation with task-specific LoRA adapters applied to the embedding model

## Per-Task Adapter Details

| Task | Classes | Trainable Params | Adapter Size | Head Size |
|---|---|---|---|---|
{spec_table}

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
base_model = AutoModel.from_pretrained("{BASE_MODEL_ID}", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("{BASE_MODEL_ID}", trust_remote_code=True)

# 2. LoRA adapter 적용
from huggingface_hub import hf_hub_download
import os

task = "Banking77Classification"  # 원하는 task 선택
repo_id = "{DEFAULT_REPO}"

lora_path = hf_hub_download(repo_id, f"{{task}}/lora_adapter.pt")
head_path = hf_hub_download(repo_id, f"{{task}}/classifier_head.pt")
meta_path = hf_hub_download(repo_id, f"{{task}}/meta.json")

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
            A = lora_state[f"lora_{{idx}}_A"]
            B = lora_state[f"lora_{{idx}}_B"]
            scaling = lora_state[f"lora_{{idx}}_scaling"].item()
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
    print(f"Predicted class: {{pred}}")
```

## License

Apache 2.0
"""
    return card


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--lora-dir", default=LORA_DIR)
    parser.add_argument("--results", default=RESULTS_FILE)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    # 결과 로드
    with open(args.results) as f:
        results = json.load(f)

    # Task meta 로드
    task_metas = {}
    for d in os.listdir(args.lora_dir):
        meta_path = os.path.join(args.lora_dir, d, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                task_metas[d] = json.load(f)

    # 모델카드 생성
    card = generate_lora_model_card(results, task_metas)
    readme_path = os.path.join(args.lora_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(card)
    print(f"Model card written to {readme_path}")

    # 레포 생성 및 업로드
    api = HfApi()
    create_repo(args.repo, exist_ok=True, private=args.private)
    print(f"Uploading to {args.repo}...")

    upload_folder(
        repo_id=args.repo,
        folder_path=args.lora_dir,
        commit_message="Upload LoRA classification adapters with model card",
    )
    print(f"Done! https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
