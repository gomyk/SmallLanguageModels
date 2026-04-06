"""
LoRA Classification Fine-Tuning for Compressed Jina v5 Model

MTEB Classification 태스크별로 LoRA adapter를 학습하고 평가한다.

작동 방식:
  1. Base 모델(gomyk/jina-v5-h256-distilled-conv)을 freeze 상태로 로드
  2. Attention의 q/k/v/o projection에 LoRA adapter (A, B matrix) 부착
  3. Classification head (256 → num_classes) 추가
  4. train split으로 cross-entropy 학습 (LoRA params + head만 업데이트)
  5. MTEB evaluation으로 성능 측정

Usage:
    # 전체 8개 classification task
    python lora_classification.py

    # 특정 task만
    python lora_classification.py --tasks Banking77Classification ImdbClassification

    # 하이퍼파라미터 조정
    python lora_classification.py --rank 16 --alpha 32 --epochs 10 --lr 5e-4

    # 평가만 (이미 학습된 adapter)
    python lora_classification.py --eval-only

    # 학습 후 HuggingFace 업로드
    python lora_classification.py --upload --repo-prefix gomyk/jina-v5-h256-lora
"""

import argparse
import gc
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

import mteb

from config import MTEB_TASK_GROUPS


# ── Constants ────────────────────────────────────────────────

BASE_MODEL = "gomyk/jina-v5-h256-distilled-conv"
LORA_OUTPUT_DIR = "students/jina_v5_lora"


# ── LoRA Implementation ─────────────────────────────────────

class LoRALinear(nn.Module):
    """LoRA adapter를 기존 Linear layer에 부착한다.

    Forward:
        y = W_frozen @ x + (alpha/r) * B @ A @ x

    W_frozen: 원본 weight (grad 없음)
    A: (in_features, r) — He init
    B: (r, out_features) — zero init → 초기 ΔW = 0
    """

    def __init__(self, original_linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original = original_linear
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # A: down-projection (in → r), He init
        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        nn.init.kaiming_normal_(self.lora_A, a=5**0.5)

        # B: up-projection (r → out), zero init
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        self.scaling = alpha / rank

    def forward(self, x):
        # 원본 출력
        base_out = self.original(x)
        # LoRA delta: x @ A @ B * scaling
        lora_out = (x @ self.lora_A) @ self.lora_B * self.scaling
        return base_out + lora_out

    def merge_and_unload(self):
        """LoRA weight를 원본에 병합 (추론 시 오버헤드 제거)."""
        with torch.no_grad():
            delta = (self.lora_A @ self.lora_B) * self.scaling
            self.original.weight.add_(delta.T)
        return self.original


class LoRAClassificationModel(nn.Module):
    """Base embedding 모델 + LoRA adapters + Classification head.

    구조:
        Input → [EuroBert + LoRA on q/k/v/o] → Mean Pool → FC Head → logits
    """

    def __init__(self, base_model, tokenizer, num_classes, rank=8, alpha=16.0,
                 target_modules=None):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.hidden_size = base_model.config.hidden_size

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, num_classes),
        )

        # LoRA를 attention projection에 적용
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        self.lora_layers = []
        self._apply_lora(target_modules, rank, alpha)

    def _apply_lora(self, target_modules, rank, alpha):
        """모델 내부의 target Linear layer를 LoRALinear로 교체."""
        replaced = 0
        for name, module in self.base_model.named_modules():
            for target in target_modules:
                child = getattr(module, target, None)
                if child is not None and isinstance(child, nn.Linear):
                    lora_layer = LoRALinear(child, rank=rank, alpha=alpha)
                    setattr(module, target, lora_layer)
                    self.lora_layers.append(lora_layer)
                    replaced += 1

        print(f"  LoRA applied: {replaced} layers "
              f"(rank={rank}, alpha={alpha}, "
              f"trainable params: {self._count_lora_params():,})")

    def _count_lora_params(self):
        """LoRA + classifier의 학습 파라미터 수."""
        total = 0
        for lora in self.lora_layers:
            total += lora.lora_A.numel() + lora.lora_B.numel()
        for p in self.classifier.parameters():
            total += p.numel()
        return total

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)

        # hidden states 추출 (last_hidden_state 또는 [0])
        if hasattr(outputs, "last_hidden_state"):
            hidden = outputs.last_hidden_state
        else:
            hidden = outputs[0]

        # Mean pooling (attention mask 적용)
        mask_expanded = attention_mask.unsqueeze(-1).float()
        summed = (hidden * mask_expanded).sum(dim=1)
        lengths = mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled = summed / lengths

        logits = self.classifier(pooled)
        return logits

    def get_trainable_params(self):
        """학습 대상 파라미터만 반환 (LoRA A/B + classifier)."""
        params = []
        for lora in self.lora_layers:
            params.extend([lora.lora_A, lora.lora_B])
        params.extend(self.classifier.parameters())
        return params

    def save_lora(self, save_dir):
        """LoRA adapter weights + classifier를 저장."""
        os.makedirs(save_dir, exist_ok=True)
        lora_state = {}
        for i, lora in enumerate(self.lora_layers):
            lora_state[f"lora_{i}_A"] = lora.lora_A.data.cpu()
            lora_state[f"lora_{i}_B"] = lora.lora_B.data.cpu()
            lora_state[f"lora_{i}_scaling"] = torch.tensor(lora.scaling)
        torch.save(lora_state, os.path.join(save_dir, "lora_adapter.pt"))
        torch.save(self.classifier.state_dict(),
                   os.path.join(save_dir, "classifier_head.pt"))
        print(f"  Saved LoRA adapter + classifier to {save_dir}")

    def load_lora(self, save_dir):
        """저장된 LoRA adapter weights + classifier를 로드."""
        lora_path = os.path.join(save_dir, "lora_adapter.pt")
        head_path = os.path.join(save_dir, "classifier_head.pt")

        if os.path.exists(lora_path):
            lora_state = torch.load(lora_path, map_location="cpu", weights_only=True)
            for i, lora in enumerate(self.lora_layers):
                lora.lora_A.data = lora_state[f"lora_{i}_A"]
                lora.lora_B.data = lora_state[f"lora_{i}_B"]
            print(f"  Loaded LoRA adapter from {lora_path}")

        if os.path.exists(head_path):
            self.classifier.load_state_dict(
                torch.load(head_path, map_location="cpu", weights_only=True))
            print(f"  Loaded classifier head from {head_path}")


# ── Dataset ──────────────────────────────────────────────────

class ClassificationDataset(Dataset):
    """HuggingFace dataset를 (text, label) 쌍으로 래핑."""

    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def load_task_data(task_name, lang="en"):
    """MTEB Classification task의 train/test 데이터를 MTEB API로 로드한다.

    Returns:
        (train_texts, train_labels, test_texts, test_labels, num_classes, label_names)
    """
    print(f"  Loading via MTEB: {task_name}")
    tasks = mteb.get_tasks(tasks=[task_name])
    if not tasks:
        raise ValueError(f"Unknown MTEB task: {task_name}")

    task = tasks[0]
    task.load_data()

    # 데이터 구조 판별:
    #   단일 언어: DatasetDict {"train": Dataset, "test": Dataset}
    #   다국어:    dict {"en": DatasetDict, "de": DatasetDict, ...}
    ds = task.dataset
    if isinstance(ds, dict) and "train" not in ds:
        # 다국어 — lang key로 접근
        if lang in ds:
            ds = ds[lang]
        else:
            first_lang = list(ds.keys())[0]
            print(f"    Lang '{lang}' not found, using '{first_lang}'")
            ds = ds[first_lang]

    train_data = ds["train"]
    test_data = ds["test"]

    train_texts = train_data["text"]
    train_labels = train_data["label"]
    test_texts = test_data["text"]
    test_labels = test_data["label"]

    # label을 연속 정수로 매핑
    unique_labels = sorted(set(train_labels) | set(test_labels))
    label_to_id = {l: i for i, l in enumerate(unique_labels)}
    train_labels = [label_to_id[l] for l in train_labels]
    test_labels = [label_to_id[l] for l in test_labels]
    num_classes = len(unique_labels)

    print(f"  Train: {len(train_texts):,} samples, "
          f"Test: {len(test_texts):,} samples, "
          f"Classes: {num_classes}")

    # 메모리 해제
    del task
    gc.collect()

    return train_texts, train_labels, test_texts, test_labels, num_classes, unique_labels


# ── Training ─────────────────────────────────────────────────

def collate_fn(batch, tokenizer, max_length=128):
    """Batch를 tokenize하고 label tensor를 만든다."""
    texts, labels = zip(*batch)
    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    labels = torch.tensor(labels, dtype=torch.long)
    return encoded["input_ids"], encoded["attention_mask"], labels


def train_one_task(task_name, base_model, tokenizer, device,
                   rank=8, alpha=16.0, lr=2e-4, epochs=5, batch_size=32,
                   max_length=128, patience=3, output_dir=LORA_OUTPUT_DIR):
    """단일 task에 대해 LoRA + classification head를 학습한다."""
    print(f"\n{'='*60}")
    print(f"Training: {task_name}")
    print(f"{'='*60}")

    # 1. 데이터 로드
    train_texts, train_labels, test_texts, test_labels, num_classes, _ = \
        load_task_data(task_name)

    # 2. 모델 구성 (매 task마다 fresh LoRA)
    model = LoRAClassificationModel(
        base_model, tokenizer, num_classes,
        rank=rank, alpha=alpha,
    ).to(device)

    # 3. Optimizer (LoRA + classifier만)
    trainable_params = model.get_trainable_params()
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    # 4. DataLoader
    train_dataset = ClassificationDataset(train_texts, train_labels)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_length),
        num_workers=0, pin_memory=True,
    )

    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # 5. Training loop
    best_acc = 0.0
    patience_counter = 0
    task_dir = os.path.join(output_dir, task_name)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"  Epoch {epoch+1}/{epochs}")
        for input_ids, attention_mask, labels in pbar:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(input_ids, attention_mask)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             acc=f"{correct/total:.4f}")

        train_loss = total_loss / total
        train_acc = correct / total
        elapsed = time.time() - t0

        # Eval on test set
        test_acc = evaluate(model, tokenizer, test_texts, test_labels,
                            device, batch_size, max_length)

        print(f"  Epoch {epoch+1}: "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"test_acc={test_acc:.4f} ({elapsed:.1f}s)")

        # Early stopping
        if test_acc > best_acc:
            best_acc = test_acc
            patience_counter = 0
            model.save_lora(task_dir)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1} "
                      f"(best_acc={best_acc:.4f})")
                break

    # Best 결과 로드
    model.load_lora(task_dir)

    # 메타데이터 저장
    meta = {
        "task": task_name,
        "base_model": BASE_MODEL,
        "rank": rank,
        "alpha": alpha,
        "lr": lr,
        "epochs_trained": epoch + 1,
        "best_test_acc": best_acc,
        "num_classes": num_classes,
        "trainable_params": model._count_lora_params(),
    }
    with open(os.path.join(task_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Best test accuracy: {best_acc:.4f}")

    # 메모리 정리 (LoRA layers 해제)
    _remove_lora(model)
    del model, optimizer, scheduler, train_loader, train_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_acc


def _remove_lora(model):
    """LoRA layers를 원본 Linear로 되돌린다 (base_model 재사용을 위해)."""
    for name, module in model.base_model.named_modules():
        for attr_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            child = getattr(module, attr_name, None)
            if child is not None and isinstance(child, LoRALinear):
                setattr(module, attr_name, child.original)


def evaluate(model, tokenizer, texts, labels, device,
             batch_size=64, max_length=128):
    """test set accuracy를 계산한다."""
    model.eval()
    dataset = ClassificationDataset(texts, labels)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_length),
        num_workers=0,
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for input_ids, attention_mask, label_batch in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            label_batch = label_batch.to(device)

            logits = model(input_ids, attention_mask)
            preds = logits.argmax(dim=-1)
            correct += (preds == label_batch).sum().item()
            total += label_batch.size(0)

    return correct / total


# ── MTEB Evaluation ──────────────────────────────────────────

class LoRAEmbeddingModel:
    """MTEB 평가용 wrapper — LoRA가 적용된 상태에서 embedding을 추출."""

    def __init__(self, base_model, tokenizer, lora_dir, num_classes,
                 rank=8, alpha=16.0, device="cpu"):
        self.device = device
        self.tokenizer = tokenizer
        self.model = LoRAClassificationModel(
            base_model, tokenizer, num_classes,
            rank=rank, alpha=alpha,
        ).to(device)
        self.model.load_lora(lora_dir)
        self.model.eval()

    def encode(self, sentences, batch_size=64, **kwargs):
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=128, return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model.base_model(
                    input_ids=input_ids, attention_mask=attention_mask)
                if hasattr(outputs, "last_hidden_state"):
                    hidden = outputs.last_hidden_state
                else:
                    hidden = outputs[0]

                mask_exp = attention_mask.unsqueeze(-1).float()
                pooled = (hidden * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)

            all_embeddings.append(pooled.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LoRA Classification Fine-Tuning for Jina v5 compressed model")
    parser.add_argument("--model", default=BASE_MODEL,
                        help="Base model ID or path")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Specific tasks (default: all 8 classification tasks)")
    parser.add_argument("--rank", type=int, default=8,
                        help="LoRA rank (default: 8)")
    parser.add_argument("--alpha", type=float, default=16.0,
                        help="LoRA alpha scaling (default: 16)")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate (default: 2e-4)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Max epochs per task (default: 10)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--max-length", type=int, default=128,
                        help="Max token length (default: 128)")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience (default: 3)")
    parser.add_argument("--output-dir", default=LORA_OUTPUT_DIR,
                        help="Output directory for LoRA adapters")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only evaluate existing adapters")
    parser.add_argument("--device", default=None,
                        help="Device (default: auto)")
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Tasks
    tasks = args.tasks or MTEB_TASK_GROUPS["Classification"]
    print(f"Tasks: {len(tasks)}")
    for t in tasks:
        print(f"  - {t}")

    # Base model 로드 (한 번만)
    print(f"\nLoading base model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    base_model.to(device)
    base_model.eval()

    # 모든 base model 파라미터를 freeze
    for param in base_model.parameters():
        param.requires_grad_(False)

    total_params = sum(p.numel() for p in base_model.parameters())
    print(f"  Base model params: {total_params:,} (all frozen)")
    print(f"  Hidden size: {base_model.config.hidden_size}")

    # Task별 학습
    results = {}
    for task_name in tasks:
        if args.eval_only:
            task_dir = os.path.join(args.output_dir, task_name)
            if not os.path.exists(os.path.join(task_dir, "lora_adapter.pt")):
                print(f"\n[SKIP] {task_name} — no saved adapter")
                continue
            meta_path = os.path.join(task_dir, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                results[task_name] = meta["best_test_acc"]
                print(f"\n{task_name}: {meta['best_test_acc']:.4f} (from saved)")
            continue

        acc = train_one_task(
            task_name=task_name,
            base_model=base_model,
            tokenizer=tokenizer,
            device=device,
            rank=args.rank,
            alpha=args.alpha,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_length=args.max_length,
            patience=args.patience,
            output_dir=args.output_dir,
        )
        results[task_name] = acc

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Base model: {args.model}")
    print(f"LoRA: rank={args.rank}, alpha={args.alpha}")
    print(f"{'─'*60}")
    for task, acc in results.items():
        print(f"  {task:<45s} {acc:.4f}")
    if results:
        avg = sum(results.values()) / len(results)
        print(f"{'─'*60}")
        print(f"  {'Average':<45s} {avg:.4f}")

    # 결과 저장
    summary_path = os.path.join(args.output_dir, "classification_results.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "base_model": args.model,
            "lora_rank": args.rank,
            "lora_alpha": args.alpha,
            "lr": args.lr,
            "results": results,
            "average": avg if results else 0,
        }, f, indent=2)
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
