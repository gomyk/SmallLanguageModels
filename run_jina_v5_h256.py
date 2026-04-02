"""
Jina v5 nano 압축 파이프라인

BPE 토크나이저의 merge rule을 역추적하여 vocab pruning 품질을 보장하면서
layer pruning + hidden dim PCA + vocab pruning + knowledge distillation을 수행한다.

Pipeline:
  1. Create: layer pruning → hidden dim PCA → vocab pruning (BPE merge backtrack)
  2. Evaluate teacher / before distillation
  3. Distill (early stopping, projection layer 저장/복원)
  4. Evaluate after distillation
  5. Compare & Upload

Usage:
    # 기본: 6L/256d/42K vocab
    python run_jina_v5_h256.py

    # 커스텀 설정
    python run_jina_v5_h256.py --hidden-dim 384 --num-layers 6 --target-vocab 30000
    python run_jina_v5_h256.py --hidden-dim 256 --num-layers 4 --target-vocab 20000

    # 단계별 실행
    python run_jina_v5_h256.py --skip-create --skip-teacher-eval
    python run_jina_v5_h256.py --skip-create --skip-eval-before --skip-distill  # eval after만

    # Distillation만 이어서
    python run_jina_v5_h256.py --skip-create --skip-teacher-eval --skip-eval-before --skip-eval-after

    # 업로드
    python run_jina_v5_h256.py --skip-create --skip-distill --upload --repo-prefix gomyk/jina-v5
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
    collect_corpus_tokens,
    prune_tokenizer_and_embeddings,
)
from distill import distill_student, load_mteb_task_texts
from create_students import load_distill_corpus
from upload_to_hub import (
    load_mteb_scores, get_model_size_mb, read_model_config,
    generate_compressed_model_card,
)


TEACHER_KEY = "jina_v5"


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_model_name(hidden_dim, num_layers, target_vocab=None):
    """실험 설정에서 모델 이름을 생성한다."""
    name = f"jina_v5_h{hidden_dim}_L{num_layers}"
    if target_vocab:
        name += f"_v{target_vocab // 1000}k"
    return name


# ── Step 1: Create ──────────────────────────────────────────

def create_model(hidden_dim, num_layers, target_vocab, corpus_texts):
    """Jina v5 nano → compressed model 생성.

    Args:
        hidden_dim: 목표 hidden dimension (예: 256, 384)
        num_layers: 목표 레이어 수 (예: 4, 6)
        target_vocab: 목표 vocab 크기. None이면 코퍼스 전체 토큰 유지.
                      BPE merge backtracking이 자동 적용되므로 실제 vocab은
                      target보다 클 수 있다.
        corpus_texts: PCA + vocab pruning에 사용할 코퍼스
    """
    from transformers import AutoTokenizer
    import shutil

    t = TEACHERS[TEACHER_KEY]
    students_dir = get_teacher_students_dir(TEACHER_KEY)
    model_name = make_model_name(hidden_dim, num_layers, target_vocab)
    save_path = os.path.join(students_dir, model_name)

    if os.path.exists(save_path):
        print(f"\n  [SKIP] Already exists: {save_path}")
        return save_path, model_name

    print(f"\n{'='*60}")
    print(f"Creating: {model_name}")
    print(f"  Teacher: {t['model_id']} ({t['hidden_dim']}d/{t['num_layers']}L/{t['vocab_size']:,} vocab)")
    print(f"  Target:  {hidden_dim}d / {num_layers}L / {f'{target_vocab:,} vocab' if target_vocab else 'corpus vocab'}")
    print(f"{'='*60}")

    # Step 1: Layer pruning
    layer_indices = make_uniform_indices(t["num_layers"], num_layers)
    print(f"\n[1] Layer pruning: {t['num_layers']} → {num_layers} (indices: {layer_indices})")

    student_hf, tokenizer = create_pruned_student(
        t["model_id"], layer_indices,
        layer_accessor=t["layer_accessor"],
        trust_remote_code=t["trust_remote_code"],
    )

    # Step 2: Hidden dim reduction via PCA
    needs_hidden_reduction = hidden_dim < t["hidden_dim"]
    if needs_hidden_reduction:
        ratio = hidden_dim / t["hidden_dim"]
        target_inter = max(64, (int(t["intermediate_size"] * ratio) // 64) * 64)
        print(f"\n[2] Hidden dim PCA: {t['hidden_dim']} → {hidden_dim}")
        print(f"  Intermediate: {t['intermediate_size']} → {target_inter}")

        student_hf = reduce_hidden_dim_pca(
            student_hf, tokenizer, hidden_dim, corpus_texts,
            new_intermediate_size=target_inter,
            trust_remote_code=t["trust_remote_code"],
        )
    else:
        print(f"\n[2] Hidden dim: {t['hidden_dim']} (no reduction)")

    # Step 3: Vocab pruning with BPE merge backtracking
    print(f"\n[3] Vocab pruning (BPE merge backtracked)...")
    if target_vocab:
        keep_ids = collect_corpus_tokens(tokenizer, texts=corpus_texts,
                                          max_vocab=target_vocab)
    else:
        keep_ids = collect_corpus_tokens(tokenizer, texts=corpus_texts)

    hf_tmp = os.path.join(save_path, "_hf_pruned")
    os.makedirs(save_path, exist_ok=True)
    student_hf = prune_tokenizer_and_embeddings(
        student_hf, tokenizer, keep_ids, hf_tmp
    )
    print(f"  Vocab: {t['vocab_size']:,} → {student_hf.config.vocab_size:,}")

    tokenizer = AutoTokenizer.from_pretrained(
        hf_tmp, trust_remote_code=t["trust_remote_code"]
    )

    save_as_sentence_transformer(student_hf, tokenizer, save_path)
    shutil.rmtree(hf_tmp, ignore_errors=True)

    # Sanity check
    try:
        st = SentenceTransformer(save_path, trust_remote_code=True)
        emb = st.encode(["Hello world", "안녕하세요", "こんにちは"])
        print(f"  Output shape: {emb.shape}")
        del st
    except Exception as e:
        print(f"  Sanity check failed: {e}")

    size_mb = get_model_size_mb(save_path)
    if size_mb:
        print(f"  Model size: {size_mb:.1f}MB")

    print(f"  Saved: {save_path}")
    del student_hf
    free_memory()
    return save_path, model_name


# ── Step 2: Evaluate ────────────────────────────────────────

def evaluate_model(model_name, model_path, results_dir, task_groups):
    """단일 모델 MTEB 평가."""
    import mteb

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'─'*60}")
    print(f"Evaluating: {model_name} on {device}")
    print(f"{'─'*60}")

    t = TEACHERS[TEACHER_KEY]
    is_teacher = (model_path == t["model_id"])
    model_kwargs = t.get("model_kwargs", {}) if is_teacher else {}
    try:
        model = SentenceTransformer(model_path, trust_remote_code=True,
                                     device=device, model_kwargs=model_kwargs)
    except Exception:
        from distill import load_teacher
        model = load_teacher(model_path, device=device,
                           trust_remote_code=True,
                           model_kwargs=model_kwargs)

    save_path = os.path.join(results_dir, model_name)
    os.makedirs(save_path, exist_ok=True)

    for group_name in task_groups:
        tasks = MTEB_TASK_GROUPS.get(group_name, [])
        print(f"  {group_name} ({len(tasks)} tasks)")
        for task_name in tasks:
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


# ── Results ─────────────────────────────────────────────────

def collect_scores(model_name, results_dir):
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
    result = {"overall": round(sum(all_scores) / len(all_scores), 2) if all_scores else 0}
    for g, vals in group_avgs.items():
        result[g] = round(sum(vals) / len(vals), 2)
    return result


def print_comparison(models_info, results_dir):
    print(f"\n{'='*80}")
    print("RESULTS COMPARISON")
    print(f"{'='*80}")
    print(f"  {'Model':<40} {'Overall':>8} {'Cls':>8} {'STS':>8} {'Clust':>8}")
    print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for name, label in models_info:
        scores = collect_scores(name, results_dir)
        if scores is None:
            print(f"  {label:<40} {'N/A':>8}")
            continue
        print(f"  {label:<40} {scores.get('overall',0):>7.2f}% "
              f"{scores.get('Classification',0):>7.2f}% "
              f"{scores.get('STS',0):>7.2f}% "
              f"{scores.get('Clustering',0):>7.2f}%")
    print(f"{'='*80}")


# ── Main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Jina v5 nano compression pipeline "
                    "(BPE merge-backtracked vocab pruning)"
    )
    # Model architecture
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="Target hidden dimension (default: 256)")
    parser.add_argument("--num-layers", type=int, default=6,
                        help="Target number of layers (default: 6)")
    parser.add_argument("--target-vocab", type=int, default=None,
                        help="Target vocab size. BPE merge backtracking이 적용되어 "
                             "실제 vocab은 이보다 클 수 있다. "
                             "None이면 코퍼스 전체 토큰 유지 (~42K).")

    # Training
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-vram-gb", type=float, default=None,
                        help="GPU VRAM 사용량 제한 (GB). None이면 전체 사용.")

    # Evaluation
    parser.add_argument("--task-groups", nargs="+",
                        default=["Classification", "STS", "Clustering"])

    # Skip flags
    parser.add_argument("--skip-create", action="store_true")
    parser.add_argument("--skip-teacher-eval", action="store_true")
    parser.add_argument("--skip-eval-before", action="store_true")
    parser.add_argument("--skip-distill", action="store_true")
    parser.add_argument("--skip-eval-after", action="store_true")

    # Upload
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--repo-prefix", type=str, default=None,
                        help="HuggingFace repo prefix (e.g., gomyk/jina-v5)")

    args = parser.parse_args()

    t = TEACHERS[TEACHER_KEY]
    students_dir = get_teacher_students_dir(TEACHER_KEY)
    results_dir = get_teacher_results_dir(TEACHER_KEY)
    os.makedirs(students_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    model_name = make_model_name(args.hidden_dim, args.num_layers, args.target_vocab)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"VRAM: {vram_gb:.1f}GB")
        if args.max_vram_gb and args.max_vram_gb < vram_gb:
            torch.cuda.set_per_process_memory_fraction(args.max_vram_gb / vram_gb)
            print(f"VRAM limited to {args.max_vram_gb:.0f}GB")

    print(f"\nTeacher: {t['model_id']}")
    print(f"Target: {model_name}")

    # Load corpus
    print("\nLoading corpus...")
    corpus_texts = load_distill_corpus()
    distill_texts = load_mteb_task_texts()

    total_start = time.time()

    # ── Step 1: Create ──
    if not args.skip_create:
        print(f"\n{'#'*60}")
        print(f"  STEP 1: Create compressed model")
        print(f"{'#'*60}")
        create_model(args.hidden_dim, args.num_layers, args.target_vocab, corpus_texts)

    student_path = os.path.join(students_dir, model_name)
    distilled_path = student_path + "_distilled"

    # ── Step 2: Evaluate teacher ──
    if not args.skip_teacher_eval:
        print(f"\n{'#'*60}")
        print(f"  STEP 2: Evaluate teacher")
        print(f"{'#'*60}")
        evaluate_model("jina_v5_teacher", t["model_id"], results_dir, args.task_groups)

    # ── Step 3: Evaluate before ──
    if not args.skip_eval_before:
        print(f"\n{'#'*60}")
        print(f"  STEP 3: Evaluate before distillation")
        print(f"{'#'*60}")
        evaluate_model(model_name, student_path, results_dir, args.task_groups)

    # ── Step 4: Distillation ──
    if not args.skip_distill:
        print(f"\n{'#'*60}")
        print(f"  STEP 4: Distillation (max {args.max_epochs} epochs, patience={args.patience})")
        print(f"{'#'*60}")
        start = time.time()
        distill_student(
            teacher_name=t["model_id"],
            student_path=student_path,
            texts=distill_texts,
            epochs=args.max_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            patience=args.patience,
            trust_remote_code=t["trust_remote_code"],
        )
        print(f"  Distillation time: {(time.time() - start)/60:.1f} min")

    # ── Step 5: Evaluate after ──
    if not args.skip_eval_after:
        if os.path.exists(distilled_path):
            print(f"\n{'#'*60}")
            print(f"  STEP 5: Evaluate after distillation")
            print(f"{'#'*60}")
            evaluate_model(f"{model_name}_distilled", distilled_path,
                         results_dir, args.task_groups)

    # ── Results ──
    print(f"\n{'#'*60}")
    print(f"  FINAL RESULTS")
    print(f"{'#'*60}")
    models_info = [
        ("jina_v5_teacher", f"Teacher ({t['hidden_dim']}d/{t['num_layers']}L)"),
        (model_name, f"{model_name} (before distill)"),
        (f"{model_name}_distilled", f"{model_name} (after distill)"),
    ]
    print_comparison(models_info, results_dir)

    total_elapsed = time.time() - total_start
    print(f"\nTotal time: {total_elapsed/60:.1f} min")

    # ── Upload ──
    if args.upload and args.repo_prefix:
        from huggingface_hub import create_repo, upload_folder
        for suffix, is_distilled in [("", False), ("_distilled", True)]:
            name = f"{model_name}{suffix}"
            path = os.path.join(students_dir, name)
            if not os.path.exists(path):
                continue
            repo_id = f"{args.repo_prefix}-{name}"
            print(f"\nUploading {name} → {repo_id}")
            mteb_scores = load_mteb_scores(name, results_dir)
            base_scores = load_mteb_scores(model_name, results_dir) if is_distilled else None
            card = generate_compressed_model_card(
                name=name, teacher_key=TEACHER_KEY,
                mteb_scores=mteb_scores, is_distilled=is_distilled,
                base_mteb_scores=base_scores,
                model_size_mb=get_model_size_mb(path),
                model_config=read_model_config(path),
            )
            with open(os.path.join(path, "README.md"), "w", encoding="utf-8") as f:
                f.write(card)
            create_repo(repo_id, exist_ok=True)
            upload_folder(repo_id=repo_id, folder_path=path,
                         commit_message=f"Upload {name}")
            print(f"  [OK] https://huggingface.co/{repo_id}")

    print("\nDone!")


if __name__ == "__main__":
    main()
