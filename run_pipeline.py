"""
Full Pipeline Runner (Sequential, Memory-Optimized)

각 Teacher별로 순차 실행:
  1. Teacher 베이스라인 MTEB 평가
  2. Student 모델 MTEB 평가
  3. Knowledge Distillation
  4. Distilled 모델 MTEB 평가
  5. 결과 비교
  6. HuggingFace 업로드

Usage:
    python run_pipeline.py --teacher modernbert --repo-prefix gomyk/modernbert-student
    python run_pipeline.py --teacher gte --repo-prefix gomyk/gte-student
    python run_pipeline.py --teacher modernbert --skip-upload
    python run_pipeline.py --teacher modernbert --start-from distill
"""

import argparse
import gc
import os
import subprocess
import sys
import time


def run_step(cmd, step_name):
    """단일 스텝을 실행한다."""
    print(f"\n{'#'*60}")
    print(f"# STEP: {step_name}")
    print(f"# CMD:  {cmd}")
    print(f"{'#'*60}\n")

    start = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed = time.time() - start

    status = "OK" if result.returncode == 0 else "WARN"
    print(f"\n[{status}] {step_name} ({elapsed:.0f}s)")
    gc.collect()
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Full pipeline runner")
    parser.add_argument("--teacher", required=True, choices=["modernbert", "gte", "minilm", "me5"])
    parser.add_argument("--repo-prefix", default=None,
                        help="HuggingFace repo prefix")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--skip-distill", action="store_true")
    parser.add_argument("--start-from", default="teacher-eval",
                        choices=["teacher-eval", "student-eval", "distill",
                                 "distill-eval", "compare", "upload"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    teacher = args.teacher
    py = sys.executable

    steps = ["teacher-eval", "student-eval", "distill", "distill-eval", "compare", "upload"]
    start_idx = steps.index(args.start_from)

    # Student 이름 미리 확인
    from config import (
        generate_experiments, generate_me5_experiments,
        get_teacher_students_dir,
    )
    if teacher == "me5":
        exps = generate_me5_experiments()
    else:
        exps = generate_experiments(teacher)
    student_names = [e["name"] for e in exps]
    students_dir = get_teacher_students_dir(teacher)

    # 태스크 제외 없음 (모든 태스크 평가)
    exclude_tasks_flag = ""

    print(f"{'='*60}")
    print(f"Pipeline: {teacher}")
    print(f"Students: {student_names}")
    print(f"Start from: {args.start_from}")
    print(f"{'='*60}")

    # 1. Teacher 베이스라인 평가
    if start_idx <= 0:
        # --only _NONE_ 은 매치 안 되므로 teacher만 평가됨
        run_step(
            f'{py} run_mteb.py --teacher {teacher} --include-teacher --only _NONE_{exclude_tasks_flag}',
            "1/6 Teacher baseline MTEB"
        )

    # 2. Student 평가
    if start_idx <= 1:
        only_str = " ".join(student_names)
        run_step(
            f'{py} run_mteb.py --teacher {teacher} --only {only_str}{exclude_tasks_flag}',
            f"2/6 Student MTEB ({len(student_names)} models)"
        )

    # 3. Distillation
    if start_idx <= 2 and not args.skip_distill:
        names_str = " ".join(student_names)
        run_step(
            f'{py} distill.py --teacher {teacher} --student {names_str} '
            f'--epochs {args.epochs} --batch-size {args.batch_size}',
            "3/6 Knowledge Distillation"
        )

    # 4. Distilled 모델 평가
    if start_idx <= 3 and not args.skip_distill:
        distilled = []
        for name in student_names:
            dp = os.path.join(students_dir, name + "_distilled")
            if os.path.exists(dp):
                distilled.append(name + "_distilled")

        if distilled:
            only_str = " ".join(distilled)
            run_step(
                f'{py} run_mteb.py --teacher {teacher} --only {only_str}{exclude_tasks_flag}',
                "4/6 Distilled model MTEB"
            )
        else:
            print("\n[SKIP] 4/6 No distilled models found")

    # 5. 결과 비교
    if start_idx <= 4:
        run_step(
            f'{py} compare_results.py --teacher {teacher}',
            "5/6 Results comparison"
        )

    # 6. HuggingFace 업로드
    if start_idx <= 5 and not args.skip_upload:
        if args.repo_prefix:
            run_step(
                f'{py} upload_to_hub.py --teacher {teacher} '
                f'--repo-prefix {args.repo_prefix}',
                "6/6 Upload to HuggingFace"
            )
        else:
            print("\n[SKIP] 6/6 No --repo-prefix provided")

    print(f"\n{'='*60}")
    print(f"Pipeline for {teacher} complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
