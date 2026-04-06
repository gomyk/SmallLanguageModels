"""
Universal Classification LoRA — Multi-Task Learning

모든 MTEB Classification task에서 공유 LoRA backbone + task별 classification head로
동시에 학습한다. 학습 후 head는 버리고, LoRA-enhanced embedding만 사용한다.

핵심: 여러 task의 gradient가 하나의 LoRA에 모이면서,
      classification에 범용적으로 유용한 feature를 학습한다.

Usage:
    python lora_clf_universal.py
    python lora_clf_universal.py --rank 16 --alpha 32 --epochs 10 --lr 2e-4
    python lora_clf_universal.py --eval-only
"""

import argparse
import gc
import json
import os
import random
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


BASE_MODEL = "gomyk/jina-v5-h256-distilled-conv"
OUTPUT_DIR = "students/jina_v5_lora_universal"


# ── LoRA ─────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original = original_linear
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        nn.init.kaiming_normal_(self.lora_A, a=5**0.5)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = alpha / rank

    def forward(self, x):
        base_out = self.original(x)
        lora_out = (x @ self.lora_A) @ self.lora_B * self.scaling
        return base_out + lora_out

    def merge_and_unload(self):
        with torch.no_grad():
            delta = (self.lora_A @ self.lora_B) * self.scaling
            self.original.weight.add_(delta.T)
        return self.original


def apply_lora(base_model, rank=8, alpha=16.0,
               target_modules=None):
    """base_model에 LoRA를 부착하고 LoRA layer 리스트를 반환한다."""
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    lora_layers = []
    replaced = 0
    for name, module in base_model.named_modules():
        for target in target_modules:
            child = getattr(module, target, None)
            if child is not None and isinstance(child, nn.Linear):
                lora_layer = LoRALinear(child, rank=rank, alpha=alpha)
                setattr(module, target, lora_layer)
                lora_layers.append(lora_layer)
                replaced += 1

    lora_params = sum(l.lora_A.numel() + l.lora_B.numel() for l in lora_layers)
    print(f"  LoRA applied: {replaced} layers "
          f"(rank={rank}, alpha={alpha}, params={lora_params:,})")
    return lora_layers


def remove_lora(base_model):
    """LoRA를 제거하고 원본 Linear로 복원."""
    for name, module in base_model.named_modules():
        for attr in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            child = getattr(module, attr, None)
            if child is not None and isinstance(child, LoRALinear):
                setattr(module, attr, child.original)


def save_lora(lora_layers, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    state = {}
    for i, lora in enumerate(lora_layers):
        state[f"lora_{i}_A"] = lora.lora_A.data.cpu()
        state[f"lora_{i}_B"] = lora.lora_B.data.cpu()
        state[f"lora_{i}_scaling"] = torch.tensor(lora.scaling)
    torch.save(state, os.path.join(save_dir, "lora_adapter.pt"))


def load_lora(lora_layers, save_dir):
    path = os.path.join(save_dir, "lora_adapter.pt")
    state = torch.load(path, map_location="cpu", weights_only=True)
    for i, lora in enumerate(lora_layers):
        lora.lora_A.data = state[f"lora_{i}_A"].to(lora.lora_A.device)
        lora.lora_B.data = state[f"lora_{i}_B"].to(lora.lora_B.device)
    print(f"  Loaded LoRA from {path}")


# ── Multi-Task Model ─────────────────────────────────────────

class MultiTaskClassifier(nn.Module):
    """공유 backbone(base + LoRA) + task별 classification head.

    Forward:
        backbone: input → mean_pool → embedding [B, 256]
        head[task_id]: embedding → logits [B, num_classes_for_task]
    """

    def __init__(self, base_model, task_num_classes, hidden_size=256):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = hidden_size

        # task별 head: {task_id: Linear(hidden_size, num_classes)}
        self.heads = nn.ModuleDict()
        for task_id, n_classes in task_num_classes.items():
            self.heads[task_id] = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(hidden_size, n_classes),
            )

    def encode(self, input_ids, attention_mask):
        """Mean-pooled embedding을 반환."""
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "last_hidden_state"):
            hidden = outputs.last_hidden_state
        else:
            hidden = outputs[0]
        mask_exp = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)
        return pooled

    def forward(self, input_ids, attention_mask, task_id):
        emb = self.encode(input_ids, attention_mask)
        logits = self.heads[task_id](emb)
        return logits


# ── Dataset ──────────────────────────────────────────────────

class MultiTaskDataset(Dataset):
    """(text, label, task_id) 튜플."""
    def __init__(self, samples):
        # samples: list of (text, label, task_id)
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def load_all_classification_data(max_per_task=20000, lang="en"):
    """모든 MTEB Classification task에서 train data를 로드한다.

    Returns:
        samples: [(text, label, task_id), ...]
        task_num_classes: {task_id: num_classes}
    """
    tasks = MTEB_TASK_GROUPS["Classification"]
    all_samples = []
    task_num_classes = {}

    for task_name in tasks:
        print(f"  Loading: {task_name}")
        mteb_tasks = mteb.get_tasks(tasks=[task_name])
        if not mteb_tasks:
            continue

        task = mteb_tasks[0]
        task.load_data()
        ds = task.dataset

        if isinstance(ds, dict) and "train" not in ds:
            ds = ds.get(lang, ds.get(list(ds.keys())[0]))

        train = ds["train"]
        texts = train["text"]
        labels = train["label"]

        if len(texts) > max_per_task:
            indices = random.sample(range(len(texts)), max_per_task)
            texts = [texts[i] for i in indices]
            labels = [labels[i] for i in indices]

        unique = sorted(set(labels))
        label_map = {l: i for i, l in enumerate(unique)}
        num_classes = len(unique)
        task_num_classes[task_name] = num_classes

        for text, label in zip(texts, labels):
            all_samples.append((text, label_map[label], task_name))

        print(f"    {len(texts):,} samples, {num_classes} classes")
        del task
        gc.collect()

    random.shuffle(all_samples)
    print(f"\n  Total: {len(all_samples):,} samples from {len(task_num_classes)} tasks")
    return all_samples, task_num_classes


def collate_fn(batch, tokenizer, max_length=128):
    texts, labels, task_ids = zip(*batch)
    encoded = tokenizer(
        list(texts), padding=True, truncation=True,
        max_length=max_length, return_tensors="pt",
    )
    labels = torch.tensor(labels, dtype=torch.long)
    return encoded["input_ids"], encoded["attention_mask"], labels, task_ids


# ── Training ─────────────────────────────────────────────────

def train(base_model, lora_layers, tokenizer, samples, task_num_classes,
          device, lr=2e-4, epochs=10, batch_size=32, max_length=128,
          patience=3, output_dir=OUTPUT_DIR):
    """Multi-task 학습: 공유 LoRA + task별 head."""

    model = MultiTaskClassifier(
        base_model, task_num_classes,
        hidden_size=base_model.config.hidden_size).to(device)

    # LoRA params + all heads params
    trainable = []
    for lora in lora_layers:
        trainable.extend([lora.lora_A, lora.lora_B])
    for head in model.heads.values():
        trainable.extend(head.parameters())

    n_lora = sum(l.lora_A.numel() + l.lora_B.numel() for l in lora_layers)
    n_head = sum(p.numel() for h in model.heads.values() for p in h.parameters())
    print(f"  Trainable: LoRA {n_lora:,} + heads {n_head:,} = {n_lora+n_head:,}")

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    dataset = MultiTaskDataset(samples)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_length),
        num_workers=0, pin_memory=True,
    )

    total_steps = len(loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()

        pbar = tqdm(loader, desc=f"  Epoch {epoch+1}/{epochs}")
        for input_ids, attention_mask, labels, task_ids in pbar:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # batch 내 task별로 분리해서 forward
            loss = torch.tensor(0.0, device=device)
            batch_correct = 0
            batch_total = 0

            unique_tasks = set(task_ids)
            for tid in unique_tasks:
                mask = [i for i, t in enumerate(task_ids) if t == tid]
                if not mask:
                    continue
                idx = torch.tensor(mask, device=device)
                t_ids = input_ids[idx]
                t_mask = attention_mask[idx]
                t_labels = labels[idx]

                logits = model(t_ids, t_mask, tid)
                task_loss = F.cross_entropy(logits, t_labels)
                loss = loss + task_loss * len(mask)

                preds = logits.argmax(dim=-1)
                batch_correct += (preds == t_labels).sum().item()
                batch_total += len(mask)

            loss = loss / max(batch_total, 1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * batch_total
            correct += batch_correct
            total += batch_total
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             acc=f"{correct/max(total,1):.4f}")

        avg_loss = total_loss / max(total, 1)
        avg_acc = correct / max(total, 1)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, "
              f"acc={avg_acc:.4f} ({elapsed:.1f}s)")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            save_lora(lora_layers, output_dir)
            print(f"    Saved (best_loss={best_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    load_lora(lora_layers, output_dir)
    return best_loss


# ── MTEB Evaluation ──────────────────────────────────────────

class MTEBWrapper:
    """MTEB EncoderProtocol 호환 wrapper."""

    def __init__(self, base_model, tokenizer, device="cpu"):
        self._model = base_model
        self.tokenizer = tokenizer
        self.device = device
        self._model.eval()
        self.mteb_model_meta = None

    def _encode_texts(self, sentences, batch_size=64):
        all_emb = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=128, return_tensors="pt",
            )
            ids = encoded["input_ids"].to(self.device)
            mask = encoded["attention_mask"].to(self.device)
            with torch.no_grad():
                outputs = self._model(ids, mask)
                if hasattr(outputs, "last_hidden_state"):
                    hidden = outputs.last_hidden_state
                else:
                    hidden = outputs[0]
                mask_exp = mask.unsqueeze(-1).float()
                pooled = (hidden * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)
            all_emb.append(pooled.cpu().float())
        return torch.cat(all_emb, dim=0)

    def encode(self, inputs, *, task_metadata=None, hf_split=None,
               hf_subset=None, prompt_type=None, **kwargs):
        texts = [t for batch in inputs for t in batch["text"]]
        return self._encode_texts(texts)

    def similarity(self, e1, e2):
        if isinstance(e1, np.ndarray): e1 = torch.from_numpy(e1)
        if isinstance(e2, np.ndarray): e2 = torch.from_numpy(e2)
        return F.cosine_similarity(e1.unsqueeze(1), e2.unsqueeze(0), dim=2)

    def similarity_pairwise(self, e1, e2):
        if isinstance(e1, np.ndarray): e1 = torch.from_numpy(e1)
        if isinstance(e2, np.ndarray): e2 = torch.from_numpy(e2)
        return F.cosine_similarity(e1, e2, dim=1)


def run_mteb_eval(base_model, tokenizer, device, label=""):
    """MTEB Classification 전체 평가."""
    tasks = MTEB_TASK_GROUPS["Classification"]
    wrapper = MTEBWrapper(base_model, tokenizer, device)
    results = {}

    for task_name in tasks:
        print(f"  [{task_name}]")
        eval_tasks = mteb.get_tasks(tasks=[task_name], languages=["eng"])
        if not eval_tasks:
            eval_tasks = mteb.get_tasks(tasks=[task_name])
        if not eval_tasks:
            continue
        try:
            mr = mteb.evaluate(model=wrapper, tasks=eval_tasks,
                               overwrite_strategy="always")
            score = None
            if mr and hasattr(mr, 'task_results'):
                for tr in mr.task_results:
                    if hasattr(tr, 'scores') and "test" in tr.scores:
                        for s in tr.scores["test"]:
                            if "main_score" in s:
                                score = s["main_score"]
                                break
            if score is not None:
                results[task_name] = score
                print(f"    {score:.4f}")
            del mr
        except Exception as e:
            print(f"    [FAIL] {e}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def _print_comparison(baseline, lora):
    print(f"\n{'='*70}")
    print(f"  {'Task':<45s} {'Baseline':>9s} {'LoRA':>9s} {'Delta':>9s}")
    print(f"  {'─'*45} {'─'*9} {'─'*9} {'─'*9}")
    b_all, l_all = [], []
    for task in sorted(set(list(baseline.keys()) + list(lora.keys()))):
        b = baseline.get(task)
        l = lora.get(task)
        b_str = f"{b:.4f}" if b else "  -  "
        l_str = f"{l:.4f}" if l else "  -  "
        d_str = f"{l-b:+.4f}" if b and l else "  -  "
        if b: b_all.append(b)
        if l: l_all.append(l)
        print(f"  {task:<45s} {b_str:>9s} {l_str:>9s} {d_str:>9s}")
    print(f"  {'─'*45} {'─'*9} {'─'*9} {'─'*9}")
    ba = sum(b_all)/len(b_all) if b_all else 0
    la = sum(l_all)/len(l_all) if l_all else 0
    print(f"  {'Average':<45s} {ba:>9.4f} {la:>9.4f} {la-ba:>+9.4f}")


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Universal Classification LoRA via Multi-Task Learning")
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-per-task", type=int, default=20000)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # Base model
    print(f"\nLoading: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    base_model.to(device)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)
    print(f"  Params: {sum(p.numel() for p in base_model.parameters()):,} (frozen)")

    if args.eval_only:
        lora_layers = apply_lora(base_model, rank=args.rank, alpha=args.alpha)
        # LoRA params를 device로 이동
        for lora in lora_layers:
            lora.lora_A.data = lora.lora_A.data.to(device)
            lora.lora_B.data = lora.lora_B.data.to(device)
        load_lora(lora_layers, args.output_dir)
        print("\n── MTEB (LoRA) ──")
        results = run_mteb_eval(base_model, tokenizer, device)
        _print_comparison({}, results)
        return

    # ── Baseline ──
    results_base = {}
    if not args.skip_baseline:
        print("\n── Baseline MTEB ──")
        results_base = run_mteb_eval(base_model, tokenizer, device)

    # ── LoRA 부착 + 학습 ──
    lora_layers = apply_lora(base_model, rank=args.rank, alpha=args.alpha)

    print("\n── Loading Data ──")
    samples, task_num_classes = load_all_classification_data(
        max_per_task=args.max_per_task)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "task_info.json"), "w") as f:
        json.dump(task_num_classes, f, indent=2)

    print("\n── Training ──")
    best_loss = train(
        base_model, lora_layers, tokenizer, samples, task_num_classes,
        device, lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
        max_length=args.max_length, patience=args.patience,
        output_dir=args.output_dir,
    )

    meta = {
        "base_model": args.model,
        "rank": args.rank,
        "alpha": args.alpha,
        "lr": args.lr,
        "best_loss": best_loss,
        "total_samples": len(samples),
        "lora_params": sum(l.lora_A.numel() + l.lora_B.numel() for l in lora_layers),
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    if args.skip_eval:
        print("Done (eval skipped).")
        return

    # ── LoRA MTEB ──
    print("\n── MTEB (LoRA) ──")
    results_lora = run_mteb_eval(base_model, tokenizer, device)

    _print_comparison(results_base, results_lora)

    summary = {
        "base_model": args.model,
        "lora_rank": args.rank,
        "lora_alpha": args.alpha,
        "baseline": results_base,
        "lora": results_lora,
    }
    with open(os.path.join(args.output_dir, "mteb_comparison.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {args.output_dir}/mteb_comparison.json")


if __name__ == "__main__":
    main()
