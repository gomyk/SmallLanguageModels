"""4개 모델 순차 파이프라인: eval → distill(10ep+early stop) → eval → upload"""
import subprocess, sys, time, gc

PY = sys.executable
TEACHERS = [
    ("gemma_emb", "gomyk/gemma-student"),
    ("me5s", "gomyk/me5s-student"),
    ("gte", "gomyk/gte-student"),
    ("qwen3", "gomyk/qwen3-student"),
]

def run(cmd, label):
    print(f"\n{'#'*60}")
    print(f"# {label}")
    print(f"# CMD: {cmd}")
    print(f"{'#'*60}\n")
    start = time.time()
    r = subprocess.run(cmd, shell=True)
    elapsed = time.time() - start
    status = "OK" if r.returncode == 0 else "FAIL"
    print(f"[{status}] {label} ({elapsed/60:.1f}min)")
    gc.collect()
    return r.returncode

for teacher, prefix in TEACHERS:
    print(f"\n{'='*60}")
    print(f"  PIPELINE: {teacher}")
    print(f"{'='*60}")

    # Step 1: MTEB eval (teacher baseline + compressed)
    run(f'{PY} run_mteb.py --teacher {teacher} --only {teacher}_compressed --include-teacher',
        f'{teacher}: MTEB eval (teacher + compressed)')

    # Step 2: Distillation (10 epochs, early stop patience=3)
    run(f'{PY} distill.py --teacher {teacher} --student {teacher}_compressed '
        f'--epochs 10 --patience 3 --batch-size 32',
        f'{teacher}: Distillation (10ep, early stop)')

    # Step 3: MTEB eval (distilled)
    run(f'{PY} run_mteb.py --teacher {teacher} --only {teacher}_compressed_distilled',
        f'{teacher}: MTEB eval (distilled)')

    # Step 4: Compare results
    run(f'{PY} compare_results.py --teacher {teacher}',
        f'{teacher}: Compare results')

    # Step 5: Upload to HuggingFace (update existing repos)
    run(f'{PY} upload_to_hub.py --teacher {teacher} --repo-prefix {prefix} '
        f'--only {teacher}_compressed {teacher}_compressed_distilled',
        f'{teacher}: Upload to HuggingFace')

print(f"\n{'='*60}")
print("  ALL PIPELINES COMPLETE!")
print(f"{'='*60}")
