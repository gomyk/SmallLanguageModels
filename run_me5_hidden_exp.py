"""
ME5 Hidden Dimension Reduction Experiments

ME5-base (768d/12L/250K vocab)에서 hidden_dim만 축소하고 vocab은 유지한 채로
어떤 hidden_size가 가장 효과적인지 비교 실험한다.

Pipeline per hidden_dim:
  1. Create student (PCA-based hidden reduction, keep vocab)
  2. Eval before distillation
  3. Distill (3 epochs)
  4. Eval after distillation
  5. Compare all results
  6. Upload to HuggingFace

Usage:
    python run_me5_hidden_exp.py
    python run_me5_hidden_exp.py --hidden-dims 512 384
    python run_me5_hidden_exp.py --skip-create --skip-teacher-eval
    python run_me5_hidden_exp.py --upload --repo-prefix gomyk/me5-hidden
"""

import argparse
import gc
import json
import os
import time

import torch
from sentence_transformers import SentenceTransformer

from config import (
    TEACHERS, MTEB_TASK_GROUPS,
    get_teacher_students_dir, get_teacher_results_dir,
    _estimate_for_teacher, make_uniform_indices,
)
from arch_utils import (
    create_pruned_student,
    save_as_sentence_transformer,
    reduce_hidden_dim_pca,
)
from distill import distill_student, load_mteb_task_texts
from create_students import load_distill_corpus
from upload_to_hub import (
    load_mteb_scores, get_model_size_mb, read_model_config,
    generate_compressed_model_card,
)


TEACHER_KEY = "me5"
HIDDEN_DIMS = [512, 384, 256]
DISTILL_EPOCHS = 3


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Step 1: Create Student ──────────────────────────────────

def create_me5_hidden_student(hidden_dim, corpus_texts):
    """ME5-base에서 hidden_dim만 PCA로 축소한 student를 생성한다 (vocab 유지)."""
    t = TEACHERS[TEACHER_KEY]
    students_dir = get_teacher_students_dir(TEACHER_KEY)
    name = f"me5_h{hidden_dim}"
    save_path = os.path.join(students_dir, name)

    if os.path.exists(save_path):
        print(f"\n  [SKIP] Already exists: {save_path}")
        return save_path

    print(f"\n{'='*60}")
    print(f"Creating ME5 student: hidden_dim={hidden_dim} (PCA)")
    print(f"  Layers: {t['num_layers']} (all), Vocab: {t['vocab_size']:,} (kept)")
    print(f"{'='*60}")

    # All 12 layers (no layer pruning)
    layer_indices = list(range(t["num_layers"]))
    student_hf, tokenizer = create_pruned_student(
        t["model_id"], layer_indices,
        layer_accessor=t["layer_accessor"],
        trust_remote_code=t["trust_remote_code"],
    )

    # Hidden dim reduction via PCA
    ratio = hidden_dim / t["hidden_dim"]
    target_inter = max(64, (int(t["intermediate_size"] * ratio) // 64) * 64)

    print(f"  Hidden: {t['hidden_dim']} -> {hidden_dim}")
    print(f"  Intermediate: {t['intermediate_size']} -> {target_inter}")

    student_hf = reduce_hidden_dim_pca(
        student_hf, tokenizer, hidden_dim, corpus_texts,
        new_intermediate_size=target_inter,
        trust_remote_code=t["trust_remote_code"],
    )

    # Save as SentenceTransformer (no vocab pruning)
    save_as_sentence_transformer(student_hf, tokenizer, save_path)

    # Sanity check
    try:
        st = SentenceTransformer(save_path, trust_remote_code=True)
        emb = st.encode(["Hello world", "안녕하세요", "こんにちは"])
        print(f"  Output shape: {emb.shape}")
        del st
    except Exception as e:
        print(f"  Sanity check failed: {e}")

    # File size
    size_mb = get_model_size_mb(save_path)
    if size_mb:
        print(f"  Model size: {size_mb:.1f}MB")

    # Param estimate
    est = _estimate_for_teacher(
        TEACHER_KEY, layer_indices, t["vocab_size"],
        hidden_dim=hidden_dim, intermediate_size=target_inter,
    )
    print(f"  Estimated params: {est['total_params']:,}")
    print(f"  Saved: {save_path}")

    del student_hf
    free_memory()
    return save_path


# ── Step 2: Evaluate ────────────────────────────────────────

def evaluate_model(model_name, model_path, results_dir, task_groups):
    """단일 모델 MTEB 평가."""
    import mteb

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'─'*60}")
    print(f"Evaluating: {model_name} on {device}")
    print(f"{'─'*60}")

    model = SentenceTransformer(model_path, trust_remote_code=True, device=device)

    save_path = os.path.join(results_dir, model_name)
    os.makedirs(save_path, exist_ok=True)

    for group_name in task_groups:
        tasks = MTEB_TASK_GROUPS.get(group_name, [])
        print(f"  {group_name} ({len(tasks)} tasks)")
        for task_name in tasks:
            # Skip if already done
            already_done = False
            for root, dirs, files in os.walk(save_path):
                for f in files:
                    if (task_name.replace(".", "") in f.replace(".", "")
                            and f.endswith(".json")):
                        already_done = True
                        break
                if already_done:
                    break
            if already_done:
                print(f"    [SKIP] {task_name}")
                continue

            try:
                eval_tasks = mteb.get_tasks(tasks=[task_name])
                if not eval_tasks:
                    print(f"    [SKIP] {task_name} (not found)")
                    continue
                evaluation = mteb.MTEB(tasks=eval_tasks)
                evaluation.run(model, output_folder=save_path, eval_splits=["test"])
                print(f"    [OK] {task_name}")
                del eval_tasks, evaluation
                free_memory()
            except Exception as e:
                print(f"    [FAIL] {task_name}: {e}")

    del model
    free_memory()


# ── Step 3: Compare Results ─────────────────────────────────

def collect_scores(model_name, results_dir):
    """모델의 MTEB 점수를 그룹별 평균으로 수집."""
    scores = load_mteb_scores(model_name, results_dir)
    if not scores:
        return None

    group_avgs = {}
    all_scores = []
    for task, data in scores.items():
        group = data.get("group", "Other")
        avg = data["average"]
        all_scores.append(avg)
        group_avgs.setdefault(group, []).append(avg)

    result = {
        "overall": round(sum(all_scores) / len(all_scores), 2) if all_scores else 0,
    }
    for g, vals in group_avgs.items():
        result[g] = round(sum(vals) / len(vals), 2)

    return result


def print_comparison_table(models_info, results_dir):
    """모든 모델의 비교 테이블을 출력."""
    print(f"\n{'='*80}")
    print("RESULTS COMPARISON")
    print(f"{'='*80}")

    header = f"  {'Model':<35} {'Overall':>8} {'Cls':>8} {'STS':>8} {'Clust':>8}"
    print(header)
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for name, label in models_info:
        scores = collect_scores(name, results_dir)
        if scores is None:
            print(f"  {label:<35} {'N/A':>8}")
            continue
        overall = scores.get("overall", 0)
        cls = scores.get("Classification", 0)
        sts = scores.get("STS", 0)
        clust = scores.get("Clustering", 0)
        print(f"  {label:<35} {overall:>7.2f}% {cls:>7.2f}% {sts:>7.2f}% {clust:>7.2f}%")

    print(f"{'='*80}")


# ── Step 4: Upload to HuggingFace ──────────────────────────

def upload_models(hidden_dims, results_dir, students_dir, repo_prefix, dry_run=False):
    """distilled 모델들을 HuggingFace에 업로드."""
    from huggingface_hub import HfApi, create_repo, upload_folder

    api = HfApi()
    t = TEACHERS[TEACHER_KEY]

    for h in hidden_dims:
        name = f"me5_h{h}_distilled"
        model_path = os.path.join(students_dir, name)
        if not os.path.exists(model_path):
            print(f"  [SKIP] {name} not found")
            continue

        repo_id = f"{repo_prefix}-{name}"
        print(f"\n{'='*60}")
        print(f"{'[DRY RUN] ' if dry_run else ''}Uploading: {name} -> {repo_id}")
        print(f"{'='*60}")

        # MTEB scores
        mteb_scores = load_mteb_scores(name, results_dir)
        base_scores = load_mteb_scores(f"me5_h{h}", results_dir)
        model_size = get_model_size_mb(model_path)
        model_cfg = read_model_config(model_path)

        card = generate_compressed_model_card(
            name=name,
            teacher_key=TEACHER_KEY,
            mteb_scores=mteb_scores,
            is_distilled=True,
            base_mteb_scores=base_scores,
            model_size_mb=model_size,
            model_config=model_cfg,
            two_stage=False,
        )

        readme_path = os.path.join(model_path, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(card)
        print(f"  Model card: {readme_path}")

        if dry_run:
            print(f"  [DRY RUN] Would upload to {repo_id}")
            continue

        try:
            create_repo(repo_id, exist_ok=True)
            upload_folder(
                repo_id=repo_id,
                folder_path=model_path,
                commit_message=f"Upload {name} (ME5-base h{h} PCA + 3ep distilled)",
            )
            print(f"  [OK] https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"  [FAIL] Upload error: {e}")


# ── Main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ME5 Hidden Dimension Reduction Experiments"
    )
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=HIDDEN_DIMS,
                        help="Hidden dimensions to test (default: 512 384 256)")
    parser.add_argument("--epochs", type=int, default=DISTILL_EPOCHS,
                        help="Distillation epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--task-groups", nargs="+",
                        default=["Classification", "STS", "Clustering"],
                        help="MTEB task groups to evaluate")

    # Skip flags
    parser.add_argument("--skip-create", action="store_true")
    parser.add_argument("--skip-teacher-eval", action="store_true")
    parser.add_argument("--skip-eval-before", action="store_true")
    parser.add_argument("--skip-distill", action="store_true")
    parser.add_argument("--skip-eval-after", action="store_true")

    # Upload
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--repo-prefix", type=str, default=None,
                        help="HuggingFace repo prefix (e.g., gomyk/me5-hidden)")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    t = TEACHERS[TEACHER_KEY]
    students_dir = get_teacher_students_dir(TEACHER_KEY)
    results_dir = get_teacher_results_dir(TEACHER_KEY)
    os.makedirs(students_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"VRAM: {vram_gb:.1f}GB")
        # 24GB limit
        if vram_gb > 24:
            torch.cuda.set_per_process_memory_fraction(24.0 / vram_gb)
            print(f"VRAM limited to 24GB")

    print(f"\nTeacher: {t['model_id']} ({t['hidden_dim']}d / {t['num_layers']}L / {t['vocab_size']:,} vocab)")
    print(f"Hidden dims to test: {args.hidden_dims}")
    print(f"Distill epochs: {args.epochs}")
    print(f"Task groups: {args.task_groups}")

    # ── Load corpus once ──
    print("\nLoading corpus for PCA and distillation...")
    corpus_texts = load_distill_corpus()
    distill_texts = load_mteb_task_texts()
    print(f"  PCA corpus: {len(corpus_texts):,} sentences")
    print(f"  Distill corpus: {len(distill_texts):,} sentences")

    total_start = time.time()

    # ══════════════════════════════════════════════════════════
    # Step 1: Create students
    # ══════════════════════════════════════════════════════════
    if not args.skip_create:
        print(f"\n{'#'*60}")
        print(f"  STEP 1: Creating ME5 hidden-dim students")
        print(f"{'#'*60}")
        for h in args.hidden_dims:
            create_me5_hidden_student(h, corpus_texts)

    # ══════════════════════════════════════════════════════════
    # Step 2: Evaluate teacher baseline
    # ══════════════════════════════════════════════════════════
    if not args.skip_teacher_eval:
        print(f"\n{'#'*60}")
        print(f"  STEP 2: Evaluating teacher (ME5-base)")
        print(f"{'#'*60}")
        evaluate_model("me5_teacher", t["model_id"], results_dir, args.task_groups)

    # ══════════════════════════════════════════════════════════
    # Step 3-5: For each hidden_dim: eval → distill → eval
    # ══════════════════════════════════════════════════════════
    for h in args.hidden_dims:
        name = f"me5_h{h}"
        student_path = os.path.join(students_dir, name)
        distilled_path = student_path + "_distilled"

        if not os.path.exists(student_path):
            print(f"\n  [SKIP] {name}: student not found at {student_path}")
            continue

        print(f"\n{'#'*60}")
        print(f"  Processing: {name} (hidden_dim={h})")
        print(f"{'#'*60}")

        # ── Eval before distillation ──
        if not args.skip_eval_before:
            print(f"\n  --- Eval BEFORE distillation ---")
            evaluate_model(name, student_path, results_dir, args.task_groups)

        # ── Distillation (3 epochs, no early stopping) ──
        if not args.skip_distill:
            print(f"\n  --- Distillation ({args.epochs} epochs) ---")
            start = time.time()
            distill_student(
                teacher_name=t["model_id"],
                student_path=student_path,
                texts=distill_texts,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=device,
                patience=args.epochs,  # no early stopping
                trust_remote_code=t["trust_remote_code"],
            )
            elapsed = time.time() - start
            print(f"  Distillation time: {elapsed/60:.1f} min")

        # ── Eval after distillation ──
        if not args.skip_eval_after:
            if os.path.exists(distilled_path):
                print(f"\n  --- Eval AFTER distillation ---")
                evaluate_model(f"{name}_distilled", distilled_path,
                             results_dir, args.task_groups)
            else:
                print(f"\n  [SKIP] {name}_distilled not found")

        free_memory()

    # ══════════════════════════════════════════════════════════
    # Results comparison
    # ══════════════════════════════════════════════════════════
    print(f"\n{'#'*60}")
    print(f"  FINAL RESULTS")
    print(f"{'#'*60}")

    models_info = [("me5_teacher", "Teacher (768d/12L)")]
    for h in args.hidden_dims:
        models_info.append((f"me5_h{h}", f"h{h} (before distill)"))
        models_info.append((f"me5_h{h}_distilled", f"h{h} (after 3ep distill)"))

    print_comparison_table(models_info, results_dir)

    total_elapsed = time.time() - total_start
    print(f"\nTotal time: {total_elapsed/60:.1f} min ({total_elapsed/3600:.1f} hours)")

    # ══════════════════════════════════════════════════════════
    # Upload to HuggingFace
    # ══════════════════════════════════════════════════════════
    if args.upload:
        if not args.repo_prefix:
            print("\n  --repo-prefix required for upload")
        else:
            print(f"\n{'#'*60}")
            print(f"  UPLOADING TO HUGGINGFACE")
            print(f"{'#'*60}")
            upload_models(args.hidden_dims, results_dir, students_dir,
                         args.repo_prefix, args.dry_run)

    print("\nDone!")


if __name__ == "__main__":
    main()
