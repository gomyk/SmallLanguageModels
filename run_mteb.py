"""
Student 모델들의 MTEB 벤치마크 평가를 실행한다.

Usage:
    python run_mteb.py                          # 모든 student, quick eval
    python run_mteb.py --full                   # 전체 언어 평가
    python run_mteb.py --only L6_uniform L4_uniform  # 특정 모델만
    python run_mteb.py --include-teacher        # teacher도 포함 (베이스라인)
"""

import argparse
import json
import os
import time

import mteb
from sentence_transformers import SentenceTransformer

from config import (
    EXPERIMENTS,
    MTEB_TASKS,
    QUICK_EVAL_LANGS,
    RESULTS_DIR,
    STUDENTS_DIR,
    TEACHER_MODEL,
)


def run_evaluation(model_path, model_name, output_dir, tasks, languages=None):
    """단일 모델에 대해 MTEB 평가를 실행한다."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"  Tasks: {tasks}")
    if languages:
        print(f"  Languages: {languages}")
    print(f"{'='*60}")

    model = SentenceTransformer(model_path)

    # MTEB 태스크 로드
    eval_tasks = mteb.get_tasks(tasks=tasks, languages=languages)

    if not eval_tasks:
        print(f"  No matching tasks found. Trying without language filter...")
        eval_tasks = mteb.get_tasks(tasks=tasks)

    evaluation = mteb.MTEB(tasks=eval_tasks)

    save_path = os.path.join(output_dir, model_name)
    os.makedirs(save_path, exist_ok=True)

    start = time.time()
    results = evaluation.run(model, output_folder=save_path, eval_splits=["test"])
    elapsed = time.time() - start

    print(f"  Completed in {elapsed:.1f}s")
    print(f"  Results saved to {save_path}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", help="특정 모델만 평가")
    parser.add_argument("--full", action="store_true", help="전체 언어 평가 (느림)")
    parser.add_argument("--include-teacher", action="store_true", help="Teacher도 평가 (베이스라인)")
    parser.add_argument("--tasks", nargs="+", default=MTEB_TASKS, help="평가할 MTEB 태스크")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    languages = None if args.full else QUICK_EVAL_LANGS

    # 평가 대상 모델 목록
    models_to_eval = []

    if args.include_teacher:
        models_to_eval.append({
            "name": "teacher_baseline",
            "path": TEACHER_MODEL,
        })

    experiments = EXPERIMENTS
    if args.only:
        experiments = [e for e in experiments if e["name"] in args.only]

    for exp in experiments:
        model_path = os.path.join(STUDENTS_DIR, exp["name"])
        if not os.path.exists(model_path):
            print(f"⚠ Student model not found: {model_path} (run create_students.py first)")
            continue
        models_to_eval.append({
            "name": exp["name"],
            "path": model_path,
        })

    if not models_to_eval:
        print("No models to evaluate. Run create_students.py first.")
        return

    print(f"Models to evaluate: {len(models_to_eval)}")
    print(f"Tasks: {args.tasks}")
    print(f"Mode: {'full' if args.full else 'quick'}")

    # 순차 평가
    all_results = {}
    for model_info in models_to_eval:
        results = run_evaluation(
            model_path=model_info["path"],
            model_name=model_info["name"],
            output_dir=RESULTS_DIR,
            tasks=args.tasks,
            languages=languages,
        )
        all_results[model_info["name"]] = results

    # 결과 요약 저장
    summary_path = os.path.join(RESULTS_DIR, "evaluation_summary.json")
    summary = {
        "mode": "full" if args.full else "quick",
        "tasks": args.tasks,
        "languages": languages,
        "models_evaluated": list(all_results.keys()),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nAll evaluations complete!")
    print(f"Next step: python compare_results.py")


if __name__ == "__main__":
    main()
