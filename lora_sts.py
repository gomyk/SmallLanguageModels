"""
STS LoRA Fine-Tuning — Semantic Textual Similarity

의미론적 유사도를 강화하는 LoRA adapter를 학습한다.
3가지 데이터 소스를 결합:
  1. MTEB STS 데이터셋 (sentence1, sentence2, score) — 직접적 유사도 regression
  2. NLI 데이터 (SNLI, MultiNLI) — entail/neutral/contradict → 1.0/0.5/0.0
  3. 개인 대화 데이터 — 같은 class의 마지막 턴끼리 positive pair 생성

학습 방식:
  [Sent A] → [Base + LoRA] → emb_a ─┐
                                      ├→ cos_sim(emb_a, emb_b) → MSE(gold_score)
  [Sent B] → [Base + LoRA] → emb_b ─┘

Usage:
    python lora_sts.py
    python lora_sts.py --model students/me5s/me5s_compressed_distilled_v2
    python lora_sts.py --rank 8 --alpha 16 --epochs 10
    python lora_sts.py --skip-nli --skip-personal   # STS 데이터만
    python lora_sts.py --eval-only
"""

import argparse
import ast
import csv
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
from datasets import load_dataset
from tqdm import tqdm
import mteb

from config import MTEB_TASK_GROUPS


# ── Constants ────────────────────────────────────────────────
BASE_MODEL = "students/me5s/me5s_compressed_distilled_v2"
OUTPUT_DIR = "students/me5s_lora_sts"
PERSONAL_DATA_PATH = "data/training.csv"

# NLI label → similarity score mapping
NLI_LABEL_TO_SCORE = {
    0: 1.0,   # entailment → high similarity
    1: 0.5,   # neutral → medium
    2: 0.0,   # contradiction → low
}


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


def apply_lora(base_model, rank=8, alpha=16.0, target_modules=None):
    if target_modules is None:
        # EuroBert/Jina: q_proj, k_proj, v_proj, o_proj
        # BERT/XLM-R/mE5: query, key, value, dense (attention output)
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "query", "key", "value"]

    lora_layers = []
    replaced = 0
    for name, module in base_model.named_modules():
        for target in target_modules:
            child = getattr(module, target, None)
            if child is not None and isinstance(child, nn.Linear):
                device = child.weight.device
                lora_layer = LoRALinear(child, rank=rank, alpha=alpha)
                lora_layer.to(device)
                setattr(module, target, lora_layer)
                lora_layers.append(lora_layer)
                replaced += 1

    lora_params = sum(l.lora_A.numel() + l.lora_B.numel() for l in lora_layers)
    print(f"  LoRA applied: {replaced} layers "
          f"(rank={rank}, alpha={alpha}, params={lora_params:,})")
    return lora_layers


def remove_lora(base_model):
    for name, module in base_model.named_modules():
        for attr in ["q_proj", "k_proj", "v_proj", "o_proj",
                      "query", "key", "value"]:
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


# ── Dataset ──────────────────────────────────────────────────

class STPairDataset(Dataset):
    """(sentence1, sentence2, similarity_score) 쌍."""
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def load_sts_pairs():
    """MTEB STS 태스크에서 (sent1, sent2, score) 쌍을 로드한다.

    score는 0~1 범위로 정규화한다 (원본은 0~5).
    """
    sts_tasks = MTEB_TASK_GROUPS["STS"]
    all_pairs = []

    for task_name in sts_tasks:
        print(f"  Loading STS: {task_name}")
        tasks = mteb.get_tasks(tasks=[task_name])
        if not tasks:
            continue

        task = tasks[0]
        task.load_data()
        ds = task.dataset

        # 다국어 태스크 처리
        if isinstance(ds, dict) and "train" not in ds and "test" not in ds:
            # 영어 우선
            for lang_key in ["en", "en-en", "eng"]:
                if lang_key in ds:
                    ds = ds[lang_key]
                    break
            else:
                ds = ds[list(ds.keys())[0]]

        count = 0
        for split_name in ["train", "validation", "test"]:
            if split_name not in ds:
                continue
            split_data = ds[split_name]
            for row in split_data:
                s1 = row.get("sentence1", "")
                s2 = row.get("sentence2", "")
                score = row.get("score", 0.0)
                if not s1 or not s2:
                    continue
                # score 정규화: 0~5 → 0~1
                if score > 1.0:
                    score = score / 5.0
                score = max(0.0, min(1.0, score))
                all_pairs.append((s1, s2, score))
                count += 1

        print(f"    {count:,} pairs")
        del task
        gc.collect()

    print(f"  Total STS pairs: {len(all_pairs):,}")
    return all_pairs


def load_nli_pairs(max_per_dataset=50000):
    """NLI 데이터에서 (premise, hypothesis, sim_score) 쌍을 생성한다.

    entailment → 1.0, neutral → 0.5, contradiction → 0.0
    label == -1 (unknown) 은 제외.
    """
    all_pairs = []

    for ds_name, hf_id in [("SNLI", "stanfordnlp/snli"),
                            ("MultiNLI", "nyu-mll/multi_nli")]:
        print(f"  Loading NLI: {ds_name}")
        try:
            ds = load_dataset(hf_id, split="train")
        except Exception as e:
            print(f"    Failed: {e}")
            continue

        pairs = []
        for row in ds:
            label = row["label"]
            if label not in NLI_LABEL_TO_SCORE:
                continue
            premise = row["premise"]
            hypothesis = row["hypothesis"]
            if not premise or not hypothesis:
                continue
            score = NLI_LABEL_TO_SCORE[label]
            pairs.append((premise, hypothesis, score))

        if len(pairs) > max_per_dataset:
            pairs = random.sample(pairs, max_per_dataset)

        all_pairs.extend(pairs)
        print(f"    {len(pairs):,} pairs")
        del ds
        gc.collect()

    print(f"  Total NLI pairs: {len(all_pairs):,}")
    return all_pairs


def load_personal_pairs(csv_path=PERSONAL_DATA_PATH, max_pairs=30000,
                        max_rows=None):
    """개인 대화 데이터에서 contrastive pair를 생성한다.

    같은 class의 마지막 턴 텍스트끼리 → positive pair (score=0.8)
    다른 class의 마지막 턴 텍스트끼리 → negative pair (score=0.1)

    완전 동일 문장이 아니므로 score를 0.8/0.1로 soft하게 설정.

    Args:
        max_pairs: 최종 pair 수 상한
        max_rows: CSV에서 읽을 최대 행 수 (None=전체)
    """
    print(f"  Loading personal data for STS pairs: {csv_path}")
    print(f"    max_pairs={max_pairs:,}, max_rows={max_rows or 'all'}")

    # 클래스별로 마지막 턴 텍스트 수집
    class_texts = {}
    read_count = 0
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            try:
                chat = ast.literal_eval(row["chat"])
                labels = ast.literal_eval(row["labels"])
            except (ValueError, SyntaxError):
                continue
            if not chat:
                continue
            last_turn = chat[-1]
            try:
                if isinstance(last_turn, dict):
                    text = str(list(last_turn.values())[0]) if last_turn else ""
                elif isinstance(last_turn, str):
                    text = last_turn
                else:
                    continue
            except Exception:
                continue
            if not text or not isinstance(text, str) or len(text) < 5:
                continue
            cls = labels.index(1) if 1 in labels else -1
            if cls < 0:
                continue
            if cls not in class_texts:
                class_texts[cls] = []
            class_texts[cls].append(text)
            read_count += 1

    for cls in sorted(class_texts.keys()):
        print(f"    Class {cls}: {len(class_texts[cls]):,} texts")
    print(f"    Total texts collected: {read_count:,}")

    # 각 클래스 내 텍스트 셔플
    for cls in class_texts:
        random.shuffle(class_texts[cls])

    pairs = []
    classes = list(class_texts.keys())

    # Positive pairs: 같은 class에서 2개씩 뽑기
    pos_per_class = max_pairs // (2 * max(len(classes), 1))
    for cls in classes:
        texts = class_texts[cls]
        for j in range(0, min(len(texts) - 1, pos_per_class * 2), 2):
            pairs.append((texts[j], texts[j + 1], 0.8))

    # Negative pairs: 다른 class에서 1개씩 뽑기 (positive와 동일 수)
    neg_target = len(pairs)
    neg_made = 0
    for _ in range(neg_target * 3):
        if neg_made >= neg_target:
            break
        c1, c2 = random.sample(classes, 2)
        if not class_texts[c1] or not class_texts[c2]:
            continue
        t1 = random.choice(class_texts[c1])
        t2 = random.choice(class_texts[c2])
        pairs.append((t1, t2, 0.1))
        neg_made += 1

    random.shuffle(pairs)
    if len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]

    pos_count = sum(1 for _, _, s in pairs if s > 0.5)
    neg_count = len(pairs) - pos_count
    print(f"    {len(pairs):,} pairs (pos={pos_count:,}, neg={neg_count:,})")
    return pairs


def collate_fn(batch, tokenizer, max_length=128):
    sent1s, sent2s, scores = zip(*batch)
    enc1 = tokenizer(list(sent1s), padding=True, truncation=True,
                     max_length=max_length, return_tensors="pt")
    enc2 = tokenizer(list(sent2s), padding=True, truncation=True,
                     max_length=max_length, return_tensors="pt")
    scores = torch.tensor(scores, dtype=torch.float)
    return enc1, enc2, scores


# ── Model ────────────────────────────────────────────────────

def mean_pool(base_model, input_ids, attention_mask):
    """Mean-pooled embedding."""
    outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
    if hasattr(outputs, "last_hidden_state"):
        hidden = outputs.last_hidden_state
    else:
        hidden = outputs[0]
    mask_exp = attention_mask.unsqueeze(-1).float()
    pooled = (hidden * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)
    return pooled


# ── Training ─────────────────────────────────────────────────

def train_sts(base_model, lora_layers, tokenizer, pairs, device,
              lr=2e-4, epochs=10, batch_size=32, max_length=128,
              patience=3, output_dir=OUTPUT_DIR):
    """STS LoRA 학습: cosine similarity regression."""

    print(f"\n{'='*60}")
    print(f"  STS LoRA Training")
    print(f"  Pairs: {len(pairs):,}")
    print(f"  Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
    print(f"{'='*60}")

    # Base model 완전 freeze 확인
    for p in base_model.parameters():
        p.requires_grad_(False)

    # LoRA params만 학습 대상으로 설정
    trainable = []
    for lora in lora_layers:
        lora.lora_A.requires_grad_(True)
        lora.lora_B.requires_grad_(True)
        trainable.extend([lora.lora_A, lora.lora_B])
    n_params = sum(p.numel() for p in trainable)

    n_frozen = sum(p.numel() for p in base_model.parameters() if not p.requires_grad)
    n_train = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    print(f"  Frozen params: {n_frozen:,}, Trainable (LoRA): {n_train:,} + {n_params:,}")

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    dataset = STPairDataset(pairs)
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
        # Base model은 eval 유지 (BatchNorm/Dropout frozen)
        # LoRA layer만 train 모드
        base_model.eval()
        for lora in lora_layers:
            lora.train()
        total_loss = 0.0
        total_samples = 0
        t0 = time.time()

        pbar = tqdm(loader, desc=f"  Epoch {epoch+1}/{epochs}")
        for enc1, enc2, scores in pbar:
            ids1 = enc1["input_ids"].to(device)
            mask1 = enc1["attention_mask"].to(device)
            ids2 = enc2["input_ids"].to(device)
            mask2 = enc2["attention_mask"].to(device)
            scores = scores.to(device)

            # 두 문장 encode
            emb1 = mean_pool(base_model, ids1, mask1)
            emb2 = mean_pool(base_model, ids2, mask2)

            # Cosine similarity → MSE with gold score
            cos_sim = F.cosine_similarity(emb1, emb2)
            loss = F.mse_loss(cos_sim, scores)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            bs = scores.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

            # 상관계수 (monitoring)
            with torch.no_grad():
                corr = torch.corrcoef(
                    torch.stack([cos_sim.detach(), scores])
                )[0, 1].item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", corr=f"{corr:.3f}")

        avg_loss = total_loss / max(total_samples, 1)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1}: avg_loss={avg_loss:.4f} ({elapsed:.1f}s)")

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
                pooled = mean_pool(self._model, ids, mask)
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


def run_mteb_eval(base_model, tokenizer, device, task_group="STS"):
    """MTEB 평가. task_group: 'STS', 'Classification', 'Clustering', or 'all'."""
    if task_group == "all":
        task_names = []
        for tasks in MTEB_TASK_GROUPS.values():
            task_names.extend(tasks)
    else:
        task_names = MTEB_TASK_GROUPS.get(task_group, [])

    wrapper = MTEBWrapper(base_model, tokenizer, device)
    results = {}

    for task_name in task_names:
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


def _print_comparison(baseline, lora, group_name="STS"):
    print(f"\n{'='*70}")
    print(f"  {group_name} Results: Baseline vs LoRA")
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
        description="STS LoRA: Semantic Textual Similarity Fine-Tuning")
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--nli-max", type=int, default=50000,
                        help="NLI 데이터셋당 최대 쌍 수")
    parser.add_argument("--personal-max", type=int, default=200000,
                        help="개인 데이터 최대 쌍 수")
    parser.add_argument("--personal-rows", type=int, default=None,
                        help="개인 데이터 CSV 최대 읽기 행 수 (None=전체)")
    parser.add_argument("--personal-csv", default=PERSONAL_DATA_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--skip-nli", action="store_true",
                        help="NLI 데이터 제외")
    parser.add_argument("--skip-personal", action="store_true",
                        help="개인 데이터 제외")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--eval-all-groups", action="store_true",
                        help="STS뿐만 아니라 Classification, Clustering도 평가")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # Base model
    print(f"\nLoading: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    base_model.to(device)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)
    total_params = sum(p.numel() for p in base_model.parameters())
    print(f"  Params: {total_params:,} (frozen)")
    print(f"  Hidden size: {base_model.config.hidden_size}")

    eval_groups = ["STS"]
    if args.eval_all_groups:
        eval_groups = ["STS", "Classification", "Clustering"]

    if args.eval_only:
        lora_layers = apply_lora(base_model, rank=args.rank, alpha=args.alpha)
        for lora in lora_layers:
            lora.lora_A.data = lora.lora_A.data.to(device)
            lora.lora_B.data = lora.lora_B.data.to(device)
        load_lora(lora_layers, args.output_dir)
        for group in eval_groups:
            print(f"\n── MTEB {group} (LoRA) ──")
            results = run_mteb_eval(base_model, tokenizer, device, group)
            _print_comparison({}, results, group)
        return

    # ── Baseline ──
    results_base = {}
    if not args.skip_baseline:
        for group in eval_groups:
            print(f"\n── Baseline MTEB {group} ──")
            results_base.update(
                run_mteb_eval(base_model, tokenizer, device, group))

    # ── LoRA 부착 ──
    lora_layers = apply_lora(base_model, rank=args.rank, alpha=args.alpha)

    # ── 데이터 로드 ──
    print("\n── Loading Data ──")
    all_pairs = load_sts_pairs()

    if not args.skip_nli:
        nli_pairs = load_nli_pairs(max_per_dataset=args.nli_max)
        all_pairs.extend(nli_pairs)

    if not args.skip_personal and os.path.exists(args.personal_csv):
        personal_pairs = load_personal_pairs(
            csv_path=args.personal_csv, max_pairs=args.personal_max,
            max_rows=args.personal_rows)
        all_pairs.extend(personal_pairs)

    random.shuffle(all_pairs)
    print(f"\n  Total training pairs: {len(all_pairs):,}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Training ──
    best_loss = train_sts(
        base_model, lora_layers, tokenizer, all_pairs, device,
        lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
        max_length=args.max_length, patience=args.patience,
        output_dir=args.output_dir,
    )

    meta = {
        "base_model": args.model,
        "rank": args.rank,
        "alpha": args.alpha,
        "lr": args.lr,
        "best_loss": best_loss,
        "total_pairs": len(all_pairs),
        "includes_nli": not args.skip_nli,
        "includes_personal": not args.skip_personal,
        "lora_params": sum(l.lora_A.numel() + l.lora_B.numel() for l in lora_layers),
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    if args.skip_eval:
        print("Done (eval skipped).")
        return

    # ── LoRA MTEB ──
    results_lora = {}
    for group in eval_groups:
        print(f"\n── MTEB {group} (LoRA) ──")
        results_lora.update(
            run_mteb_eval(base_model, tokenizer, device, group))

    for group in eval_groups:
        group_tasks = set(MTEB_TASK_GROUPS[group])
        b_group = {k: v for k, v in results_base.items() if k in group_tasks}
        l_group = {k: v for k, v in results_lora.items() if k in group_tasks}
        _print_comparison(b_group, l_group, group)

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
