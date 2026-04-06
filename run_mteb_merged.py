"""
Merged LoRA 모델의 전체 MTEB 평가를 실행한다.
SentenceTransformer 호환 wrapper로 감싸서 평가.
"""

import gc
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import mteb
from transformers import AutoModel, AutoTokenizer

from config import MTEB_TASK_GROUPS

MODEL_PATH = "students/jina_v5_h256_lora_merged"
RESULTS_DIR = "results/jina_v5_lora_merged"


class EmbeddingModel:
    """MTEB EncoderProtocol wrapper."""

    def __init__(self, model_path, device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.mteb_model_meta = None

    def _encode_texts(self, sentences, batch_size=64):
        all_emb = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                 max_length=128, return_tensors="pt")
            ids = enc["input_ids"].to(self.device)
            mask = enc["attention_mask"].to(self.device)
            with torch.no_grad():
                out = self.model(input_ids=ids, attention_mask=mask)
                hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
                m = mask.unsqueeze(-1).float()
                pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)
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


def run_task(wrapper, task_name, results_dir):
    save_path = os.path.join(results_dir, task_name + ".json")
    if os.path.exists(save_path):
        with open(save_path) as f:
            data = json.load(f)
        score = None
        for s in data.get("scores", {}).get("test", []):
            if "main_score" in s:
                score = s["main_score"]
                break
        print(f"  [SKIP] {task_name} → {score:.4f}" if score else f"  [SKIP] {task_name}")
        return score

    eval_tasks = mteb.get_tasks(tasks=[task_name], languages=["eng"])
    if not eval_tasks:
        eval_tasks = mteb.get_tasks(tasks=[task_name])
    if not eval_tasks:
        print(f"  [SKIP] {task_name} (not found)")
        return None

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
                    result_data = {"task_name": tr.task_name, "scores": tr.scores}
                    with open(save_path, "w") as f:
                        json.dump(result_data, f, indent=2, default=str)
        print(f"  [OK] {task_name} → {score:.4f}" if score else f"  [OK] {task_name}")
        del mr
    except Exception as e:
        print(f"  [FAIL] {task_name}: {e}")
        score = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return score


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {MODEL_PATH}")

    wrapper = EmbeddingModel(MODEL_PATH, device)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = {}
    for group, tasks in MTEB_TASK_GROUPS.items():
        print(f"\n── {group} ({len(tasks)} tasks) ──")
        for task_name in tasks:
            score = run_task(wrapper, task_name, RESULTS_DIR)
            if score is not None:
                all_results[task_name] = {"score": score, "group": group}

    # Summary
    print(f"\n{'='*70}")
    print("MTEB RESULTS - LoRA-Merged Model")
    print(f"{'='*70}")

    group_scores = {}
    for group in ["Classification", "STS", "Clustering"]:
        scores = [v["score"] for v in all_results.values() if v["group"] == group]
        if scores:
            avg = sum(scores) / len(scores)
            group_scores[group] = avg

    for group in ["Classification", "STS", "Clustering"]:
        tasks = MTEB_TASK_GROUPS[group]
        avg = group_scores.get(group, 0)
        print(f"\n  {group} (avg: {avg*100:.2f}%)")
        print(f"  {'─'*50}")
        for t in tasks:
            if t in all_results:
                s = all_results[t]["score"]
                print(f"    {t:<45s} {s*100:.2f}%")

    overall = [v["score"] for v in all_results.values()]
    overall_avg = sum(overall) / len(overall) if overall else 0
    print(f"\n  {'='*50}")
    print(f"  Overall Average ({len(overall)} tasks): {overall_avg*100:.2f}%")

    summary = {
        "model": MODEL_PATH,
        "group_averages": {g: round(s*100, 2) for g, s in group_scores.items()},
        "overall_average": round(overall_avg*100, 2),
        "scores": {k: round(v["score"]*100, 2) for k, v in all_results.items()},
    }
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR}/summary.json")


if __name__ == "__main__":
    main()
