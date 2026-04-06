"""
LoRA Classification MTEB Evaluation

task별 LoRA adapter를 base 모델에 merge한 뒤 MTEB 공식 평가를 실행한다.
baseline(LoRA 없음)과 LoRA 적용 결과를 비교한다.

Usage:
    # 전체 classification task 평가
    python run_lora_mteb.py

    # 특정 task만
    python run_lora_mteb.py --tasks Banking77Classification

    # baseline만 (LoRA 없이)
    python run_lora_mteb.py --baseline-only
"""

import argparse
import gc
import json
import os
import time
import copy

import numpy as np
import torch
import torch.nn as nn
import mteb
from transformers import AutoModel, AutoTokenizer

from config import MTEB_TASK_GROUPS
from lora_classification import LoRALinear, LoRAClassificationModel, _remove_lora

BASE_MODEL = "gomyk/jina-v5-h256-distilled-conv"
LORA_DIR = "students/jina_v5_lora"
RESULTS_DIR = "results/jina_v5_lora"


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class EmbeddingWrapper:
    """MTEB EncoderProtocol 호환 embedding model wrapper.

    base_model(+LoRA merged)에서 mean-pooled embedding을 추출한다.
    MTEB의 새 API(DataLoader 기반 encode)를 따른다.
    """

    def __init__(self, base_model, tokenizer, device="cpu"):
        self._model = base_model
        self.tokenizer = tokenizer
        self.device = device
        self._model.eval()
        self.mteb_model_meta = None

    def _encode_texts(self, sentences, batch_size=64):
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=128, return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self._model(input_ids=input_ids,
                                      attention_mask=attention_mask)
                if hasattr(outputs, "last_hidden_state"):
                    hidden = outputs.last_hidden_state
                else:
                    hidden = outputs[0]

                mask_exp = attention_mask.unsqueeze(-1).float()
                pooled = (hidden * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)

            all_embeddings.append(pooled.cpu().float())

        return torch.cat(all_embeddings, dim=0)

    def encode(self, inputs, *, task_metadata=None, hf_split=None,
               hf_subset=None, prompt_type=None, **kwargs):
        """MTEB EncoderProtocol.encode — DataLoader[BatchedInput]을 받는다."""
        _texts = [text for batch in inputs for text in batch["text"]]
        return self._encode_texts(_texts)

    def similarity(self, embeddings1, embeddings2):
        if isinstance(embeddings1, np.ndarray):
            embeddings1 = torch.from_numpy(embeddings1)
        if isinstance(embeddings2, np.ndarray):
            embeddings2 = torch.from_numpy(embeddings2)
        return torch.nn.functional.cosine_similarity(
            embeddings1.unsqueeze(1), embeddings2.unsqueeze(0), dim=2)

    def similarity_pairwise(self, embeddings1, embeddings2):
        if isinstance(embeddings1, np.ndarray):
            embeddings1 = torch.from_numpy(embeddings1)
        if isinstance(embeddings2, np.ndarray):
            embeddings2 = torch.from_numpy(embeddings2)
        return torch.nn.functional.cosine_similarity(embeddings1, embeddings2, dim=1)


def apply_lora(base_model, lora_dir, rank=8, alpha=16.0):
    """base_model의 attention에 LoRA를 부착하고 저장된 weight를 로드한다."""
    lora_path = os.path.join(lora_dir, "lora_adapter.pt")
    if not os.path.exists(lora_path):
        return None

    lora_state = torch.load(lora_path, map_location="cpu", weights_only=True)

    lora_layers = []
    idx = 0
    for name, module in base_model.named_modules():
        for target in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            child = getattr(module, target, None)
            if child is not None and isinstance(child, nn.Linear):
                lora_layer = LoRALinear(child, rank=rank, alpha=alpha)
                lora_layer.lora_A.data = lora_state[f"lora_{idx}_A"]
                lora_layer.lora_B.data = lora_state[f"lora_{idx}_B"]
                setattr(module, target, lora_layer)
                lora_layers.append(lora_layer)
                idx += 1

    print(f"    Applied {idx} LoRA layers from {lora_dir}")
    return lora_layers


def remove_lora(base_model):
    """LoRA를 제거하고 원본 Linear로 복원한다."""
    for name, module in base_model.named_modules():
        for attr_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            child = getattr(module, attr_name, None)
            if child is not None and isinstance(child, LoRALinear):
                setattr(module, attr_name, child.original)


def run_single_task(model_wrapper, task_name, output_dir, model_name):
    """단일 MTEB task를 평가한다."""
    save_path = os.path.join(output_dir, model_name)
    os.makedirs(save_path, exist_ok=True)

    # 이미 결과가 있는지 확인
    for root, dirs, files in os.walk(save_path):
        for f in files:
            if task_name.replace(".", "") in f.replace(".", "") and f.endswith(".json"):
                print(f"    [SKIP] {task_name} (already evaluated)")
                return load_score(save_path, task_name)

    try:
        eval_tasks = mteb.get_tasks(tasks=[task_name], languages=["eng"])
        if not eval_tasks:
            eval_tasks = mteb.get_tasks(tasks=[task_name])
        if not eval_tasks:
            print(f"    [SKIP] {task_name} (task not found)")
            return None

        model_result = mteb.evaluate(
            model=model_wrapper,
            tasks=eval_tasks,
            overwrite_strategy="always",
        )

        # ModelResult → task_results에서 score 추출
        score = None
        if model_result and hasattr(model_result, 'task_results'):
            for tr in model_result.task_results:
                # scores: {"test": [{"main_score": ..., ...}]}
                if hasattr(tr, 'scores') and "test" in tr.scores:
                    for s in tr.scores["test"]:
                        if "main_score" in s:
                            score = s["main_score"]
                            break
                # 결과 JSON 저장
                result_path = os.path.join(save_path, f"{task_name}.json")
                result_data = {
                    "task_name": tr.task_name,
                    "scores": tr.scores,
                }
                with open(result_path, "w", encoding="utf-8") as f:
                    json.dump(result_data, f, indent=2, default=str)

        print(f"    [OK] {task_name}")
        del eval_tasks, model_result
        free_memory()
        return score

    except Exception as e:
        print(f"    [FAIL] {task_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_score(result_dir, task_name):
    """평가 결과 JSON에서 main_score를 읽는다."""
    for root, dirs, files in os.walk(result_dir):
        for f in files:
            if task_name.replace(".", "") in f.replace(".", "") and f.endswith(".json"):
                with open(os.path.join(root, f)) as fh:
                    data = json.load(fh)
                # MTEB v2 format: scores → test → [0] → main_score
                try:
                    return data["scores"]["test"][0]["main_score"]
                except (KeyError, IndexError):
                    pass
                # fallback
                if "test" in data:
                    if isinstance(data["test"], dict):
                        return data["test"].get("main_score") or data["test"].get("accuracy")
    return None


def main():
    parser = argparse.ArgumentParser(description="MTEB evaluation with LoRA adapters")
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--lora-dir", default=LORA_DIR)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tasks = args.tasks or MTEB_TASK_GROUPS["Classification"]
    print(f"Tasks: {len(tasks)}")

    # 모델 로드
    print(f"\nLoading base model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    base_model.to(device)
    base_model.eval()

    results_baseline = {}
    results_lora = {}

    # ── 1. Baseline (LoRA 없이) ──
    if not args.skip_baseline:
        print(f"\n{'='*60}")
        print("BASELINE (no LoRA)")
        print(f"{'='*60}")
        wrapper = EmbeddingWrapper(base_model, tokenizer, device)
        for task_name in tasks:
            print(f"\n  [{task_name}]")
            score = run_single_task(
                wrapper, task_name, args.results_dir, "baseline")
            if score is not None:
                results_baseline[task_name] = score
                print(f"    Score: {score:.4f}")
        del wrapper
        free_memory()

    if args.baseline_only:
        _print_summary(results_baseline, {}, tasks)
        return

    # ── 2. LoRA per task ──
    print(f"\n{'='*60}")
    print("LoRA-ADAPTED (per task)")
    print(f"{'='*60}")

    for task_name in tasks:
        print(f"\n  [{task_name}]")
        task_lora_dir = os.path.join(args.lora_dir, task_name)

        if not os.path.exists(os.path.join(task_lora_dir, "lora_adapter.pt")):
            print(f"    [SKIP] No LoRA adapter found")
            continue

        # LoRA 적용
        lora_layers = apply_lora(base_model, task_lora_dir,
                                 rank=args.rank, alpha=args.alpha)
        base_model.to(device)

        wrapper = EmbeddingWrapper(base_model, tokenizer, device)
        score = run_single_task(
            wrapper, task_name, args.results_dir, f"lora_{task_name}")
        if score is not None:
            results_lora[task_name] = score
            print(f"    Score: {score:.4f}")

        # LoRA 제거 (다음 task를 위해)
        remove_lora(base_model)
        del wrapper
        free_memory()

    _print_summary(results_baseline, results_lora, tasks)

    # 결과 저장
    summary = {
        "base_model": args.model,
        "lora_rank": args.rank,
        "lora_alpha": args.alpha,
        "baseline": results_baseline,
        "lora": results_lora,
    }
    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, "mteb_comparison.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {args.results_dir}/mteb_comparison.json")


def _print_summary(baseline, lora, tasks):
    print(f"\n{'='*70}")
    print("MTEB CLASSIFICATION RESULTS")
    print(f"{'='*70}")
    print(f"  {'Task':<45s} {'Baseline':>9s} {'LoRA':>9s} {'Delta':>9s}")
    print(f"  {'─'*45} {'─'*9} {'─'*9} {'─'*9}")

    b_scores, l_scores = [], []
    for task in tasks:
        b = baseline.get(task)
        l = lora.get(task)
        b_str = f"{b:.4f}" if b else "  -  "
        l_str = f"{l:.4f}" if l else "  -  "
        if b and l:
            delta = l - b
            d_str = f"{delta:+.4f}"
            b_scores.append(b)
            l_scores.append(l)
        else:
            d_str = "  -  "
            if b: b_scores.append(b)
            if l: l_scores.append(l)
        print(f"  {task:<45s} {b_str:>9s} {l_str:>9s} {d_str:>9s}")

    print(f"  {'─'*45} {'─'*9} {'─'*9} {'─'*9}")
    b_avg = sum(b_scores) / len(b_scores) if b_scores else 0
    l_avg = sum(l_scores) / len(l_scores) if l_scores else 0
    d_avg = l_avg - b_avg if b_scores and l_scores else 0
    print(f"  {'Average':<45s} {b_avg:>9.4f} {l_avg:>9.4f} {d_avg:>+9.4f}")


if __name__ == "__main__":
    main()
