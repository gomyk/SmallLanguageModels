"""PCA vs Slice hidden dim reduction 비교 실험.

vocab pruning 없이 hidden dim만 768 → 384로 줄여서
PCA와 단순 슬라이싱의 차이를 비교한다.
"""

import os
import sys
import torch

from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

from config import TEACHERS, make_uniform_indices
from arch_utils import (
    create_pruned_student,
    reduce_hidden_dim,
    reduce_hidden_dim_pca,
    save_as_sentence_transformer,
)
from create_students import load_distill_corpus


def build_model(teacher_key, method, corpus_texts=None):
    """Layer pruning + hidden dim reduction (no vocab pruning).

    method: 'slice' or 'pca'
    """
    t = TEACHERS[teacher_key]
    target_layers = 6
    target_hidden = t["hidden_dim"] // 2  # 384
    target_inter = t["intermediate_size"] // 2  # 1536
    layer_indices = make_uniform_indices(t["num_layers"], target_layers)

    save_name = f"{teacher_key}_{method}_novocab"
    save_dir = os.path.join("students", teacher_key, save_name)

    if os.path.exists(save_dir) and os.path.exists(os.path.join(save_dir, "config.json")):
        print(f"  [SKIP] {save_name} already exists")
        return save_dir

    print(f"\n{'='*60}")
    print(f"Building: {save_name}")
    print(f"  Method: {method}")
    print(f"  Layers: {t['num_layers']} → {target_layers} ({layer_indices})")
    print(f"  Hidden: {t['hidden_dim']} → {target_hidden}")
    print(f"  Intermediate: {t['intermediate_size']} → {target_inter}")
    print(f"  Vocab: {t['vocab_size']} (no pruning)")
    print(f"{'='*60}")

    # Load teacher + layer pruning
    model, tokenizer = create_pruned_student(
        t["model_id"], layer_indices,
        layer_accessor=t["layer_accessor"],
        trust_remote_code=t["trust_remote_code"],
    )
    print(f"  Layers pruned: {layer_indices}")

    # Hidden dim reduction
    if method == "pca":
        assert corpus_texts is not None, "PCA requires corpus_texts"
        model = reduce_hidden_dim_pca(
            model, tokenizer, target_hidden, corpus_texts,
            new_intermediate_size=target_inter,
            trust_remote_code=t["trust_remote_code"],
        )
    else:  # slice
        model = reduce_hidden_dim(
            model, target_hidden, target_inter,
            trust_remote_code=t["trust_remote_code"],
        )

    # Save as SentenceTransformer (no vocab pruning)
    os.makedirs(save_dir, exist_ok=True)
    save_as_sentence_transformer(model, tokenizer, save_dir)

    # Size check
    safetensors = os.path.join(save_dir, "model.safetensors")
    if os.path.exists(safetensors):
        size_mb = os.path.getsize(safetensors) / (1024 ** 2)
        print(f"  Saved: {save_dir} ({size_mb:.1f}MB)")

    return save_dir


def evaluate_sts(model_path, model_name, trust_remote_code=True):
    """STSBenchmark로 빠르게 품질 비교."""
    import mteb

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.5)

    model = SentenceTransformer(model_path, device=device,
                                trust_remote_code=trust_remote_code)

    results_dir = os.path.join("results", "jina_v5", model_name)
    os.makedirs(results_dir, exist_ok=True)

    tasks = ["STSBenchmark", "STS12", "STS13", "STS14", "STS15", "SICK-R"]
    for task_name in tasks:
        # Skip if already done
        skip = False
        for root, dirs, files in os.walk(results_dir):
            for f in files:
                if task_name.replace(".", "") in f.replace(".", "") and f.endswith(".json"):
                    print(f"    [SKIP] {task_name}")
                    skip = True
                    break
            if skip:
                break
        if skip:
            continue

        eval_tasks = mteb.get_tasks(tasks=[task_name])
        evaluation = mteb.MTEB(tasks=eval_tasks)
        evaluation.run(model, output_folder=results_dir, eval_splits=["test"])
        print(f"    [OK] {task_name}")

    del model
    torch.cuda.empty_cache()


def main():
    teacher_key = "jina_v5"

    # Load corpus for PCA
    print("Loading corpus...")
    corpus_texts = load_distill_corpus()

    # Build both models
    slice_path = build_model(teacher_key, "slice")
    pca_path = build_model(teacher_key, "pca", corpus_texts=corpus_texts)

    # Evaluate both
    print("\n" + "=" * 60)
    print("Evaluating models...")
    print("=" * 60)

    for name, path in [
        ("jina_v5_slice_novocab", slice_path),
        ("jina_v5_pca_novocab", pca_path),
    ]:
        print(f"\n--- {name} ---")
        evaluate_sts(path, name)

    # Print comparison
    print("\n" + "=" * 60)
    print("Run: python compare_results.py --teacher jina_v5")
    print("=" * 60)


if __name__ == "__main__":
    main()
