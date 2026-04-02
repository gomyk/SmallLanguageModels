"""
Multi-Teacher MTEB Benchmark Evaluation (Memory-Optimized)

모델과 태스크를 하나씩 로드/평가/해제하여 메모리를 절약한다.

Usage:
    # 특정 teacher의 student 평가
    python run_mteb.py --teacher modernbert
    python run_mteb.py --teacher gte

    # 특정 모델만
    python run_mteb.py --teacher modernbert --only modernbert_L6_uniform

    # 특정 태스크 그룹만
    python run_mteb.py --teacher gte --task-groups Classification STS

    # Teacher 베이스라인 포함
    python run_mteb.py --teacher modernbert --include-teacher

    # 기존 동작 (하위 호환)
    python run_mteb.py
"""

import argparse
import gc
import json
import os
import time

import torch
import mteb
from sentence_transformers import SentenceTransformer

from config import (
    TEACHERS, EXPERIMENTS, MTEB_TASKS, MTEB_TASK_GROUPS,
    QUICK_EVAL_LANGS, RESULTS_DIR, STUDENTS_DIR,
    TEACHER_MODEL,
    generate_experiments, generate_me5_experiments,
    get_teacher_students_dir, get_teacher_results_dir,
    get_mteb_task_groups,
)


def free_memory():
    """GPU/CPU 메모리를 적극적으로 해제한다."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_single_task(model, task_name, model_name, output_dir, languages=None):
    """단일 태스크를 평가하고 결과를 저장한다. 이미 결과가 있으면 스킵."""
    save_path = os.path.join(output_dir, model_name)
    os.makedirs(save_path, exist_ok=True)

    # 이미 결과가 있는지 확인 (재시작 시 스킵)
    for root, dirs, files in os.walk(save_path):
        for f in files:
            if task_name.replace(".", "") in f.replace(".", "") and f.endswith(".json"):
                print(f"    [SKIP] {task_name} (already evaluated)")
                return True

    try:
        eval_tasks = mteb.get_tasks(tasks=[task_name], languages=languages)
        if not eval_tasks:
            eval_tasks = mteb.get_tasks(tasks=[task_name])
        if not eval_tasks:
            print(f"    [SKIP] {task_name} (task not found)")
            return False

        evaluation = mteb.MTEB(tasks=eval_tasks)
        evaluation.run(model, output_folder=save_path, eval_splits=["test"])
        print(f"    [OK] {task_name}")

        # 태스크 데이터 메모리 해제
        del eval_tasks, evaluation
        free_memory()
        return True

    except Exception as e:
        print(f"    [FAIL] {task_name}: {e}")
        return False


def evaluate_model(model_info, results_dir, task_groups, languages=None,
                   trust_remote_code=False):
    """단일 모델을 로드하고 태스크를 하나씩 평가한 뒤 모델을 해제한다."""
    model_name = model_info["name"]
    model_path = model_info["path"]

    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"  Path: {model_path}")
    print(f"{'='*60}")

    # 모델 로드 (GPU 자동 감지)
    start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_path, trust_remote_code=trust_remote_code,
                                 device=device)
    print(f"  Model loaded on {device} ({time.time() - start:.1f}s)")

    # 태스크 그룹별로 순차 평가
    for group_name, task_list in task_groups.items():
        print(f"\n  --- {group_name} ({len(task_list)} tasks) ---")
        for task_name in task_list:
            run_single_task(model, task_name, model_name, results_dir, languages)

    # 모델 메모리 해제
    del model
    free_memory()
    elapsed = time.time() - start
    print(f"\n  {model_name} done ({elapsed:.1f}s). Memory freed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=str, default=None,
                        choices=list(TEACHERS.keys()),
                        help="Teacher 모델 키")
    parser.add_argument("--only", nargs="+", help="특정 모델만 평가")
    parser.add_argument("--task-groups", nargs="+",
                        choices=list(MTEB_TASK_GROUPS.keys()),
                        default=list(MTEB_TASK_GROUPS.keys()),
                        help="평가할 태스크 그룹 (Classification, Clustering, STS)")
    parser.add_argument("--exclude-tasks", nargs="+", default=None,
                        help="제외할 MTEB 태스크 이름")
    parser.add_argument("--include-teacher", action="store_true",
                        help="Teacher도 평가 (베이스라인)")
    parser.add_argument("--languages", nargs="+", default=None,
                        help="특정 언어만 평가 (ISO 639-3)")
    parser.add_argument("--max-vram-frac", type=float, default=None,
                        help="GPU VRAM 사용 비율 제한 (0.0~1.0)")
    args = parser.parse_args()

    # 평가할 태스크 그룹 수집 (exclude 적용)
    base_groups = {g: MTEB_TASK_GROUPS[g] for g in args.task_groups}
    selected_groups = get_mteb_task_groups(exclude=args.exclude_tasks) if args.exclude_tasks else base_groups
    # task_groups 필터도 적용
    selected_groups = {g: selected_groups[g] for g in args.task_groups if g in selected_groups}
    total_tasks = sum(len(v) for v in selected_groups.values())

    print(f"Task groups: {list(selected_groups.keys())}")
    print(f"Total tasks: {total_tasks}")

    # Teacher 결정
    teacher_key = args.teacher or "minilm"
    t = TEACHERS[teacher_key]
    results_dir = get_teacher_results_dir(teacher_key)
    students_dir = get_teacher_students_dir(teacher_key)
    os.makedirs(results_dir, exist_ok=True)

    languages = args.languages

    # 평가 대상 모델 목록
    models_to_eval = []

    if args.include_teacher:
        models_to_eval.append({
            "name": f"{teacher_key}_teacher",
            "path": t["model_id"],
        })

    # Student 실험 목록
    if teacher_key == "minilm" and not args.teacher:
        experiments = EXPERIMENTS
    elif teacher_key == "me5":
        experiments = generate_me5_experiments()
    else:
        experiments = generate_experiments(teacher_key)

    if args.only:
        experiments = [e for e in experiments if e["name"] in args.only]

    for exp in experiments:
        model_path = os.path.join(students_dir, exp["name"])
        if not os.path.exists(model_path):
            model_path = os.path.join(STUDENTS_DIR, exp["name"])
        if not os.path.exists(model_path):
            print(f"  Student not found: {exp['name']} (run create_students.py first)")
            continue

        models_to_eval.append({
            "name": exp["name"],
            "path": model_path,
        })

        distilled_path = model_path + "_distilled"
        if os.path.exists(distilled_path):
            models_to_eval.append({
                "name": exp["name"] + "_distilled",
                "path": distilled_path,
            })

    # --only에 지정되었지만 실험 목록에 없는 모델도 디렉토리에서 직접 탐색
    if args.only:
        found_names = {m["name"] for m in models_to_eval}
        for name in args.only:
            if name in found_names:
                continue
            # 디렉토리 직접 탐색
            for base_dir in [students_dir, STUDENTS_DIR]:
                model_path = os.path.join(base_dir, name)
                if os.path.exists(model_path):
                    models_to_eval.append({"name": name, "path": model_path})
                    found_names.add(name)
                    # _distilled도 확인
                    dp = model_path + "_distilled"
                    if os.path.exists(dp):
                        models_to_eval.append({"name": name + "_distilled", "path": dp})
                        found_names.add(name + "_distilled")
                    break

    if not models_to_eval:
        print("No models to evaluate. Run create_students.py first.")
        return

    print(f"\nModels to evaluate: {len(models_to_eval)}")
    for m in models_to_eval:
        print(f"  - {m['name']}")

    # VRAM 제한 (다른 작업과 공유 시)
    if torch.cuda.is_available() and args.max_vram_frac:
        torch.cuda.set_per_process_memory_fraction(args.max_vram_frac)
        print(f"VRAM limit: {args.max_vram_frac*100:.0f}%")

    # 모델별 순차 평가 (하나 로드 → 전체 태스크 평가 → 해제)
    for model_info in models_to_eval:
        evaluate_model(
            model_info=model_info,
            results_dir=results_dir,
            task_groups=selected_groups,
            languages=languages,
            trust_remote_code=t["trust_remote_code"],
        )

    # 결과 요약 저장
    summary_path = os.path.join(results_dir, "evaluation_summary.json")
    summary = {
        "teacher": teacher_key,
        "teacher_model": t["model_id"],
        "task_groups": args.task_groups,
        "tasks": [t for tasks in selected_groups.values() for t in tasks],
        "languages": languages,
        "models_evaluated": [m["name"] for m in models_to_eval],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nAll evaluations complete!")
    print(f"Results directory: {results_dir}")
    print(f"Next step: python compare_results.py --teacher {teacher_key}")


if __name__ == "__main__":
    main()
