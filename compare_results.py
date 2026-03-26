"""
Multi-Teacher MTEB 결과 비교 및 최적 student 모델 선정

Usage:
    python compare_results.py --teacher modernbert
    python compare_results.py --teacher gte --detailed
    python compare_results.py --all-teachers          # 모든 teacher 비교
"""

import argparse
import json
import os

import pandas as pd

from config import (
    TEACHERS, EXPERIMENTS, RESULTS_DIR, STUDENTS_DIR,
    MTEB_TASK_GROUPS,
    generate_experiments, generate_me5_experiments,
    get_teacher_results_dir, get_teacher_students_dir,
    estimate_size,
)


def load_mteb_results(results_dir):
    """MTEB 결과 디렉토리에서 모든 결과를 로드한다."""
    all_results = {}

    if not os.path.isdir(results_dir):
        return all_results

    for model_name in os.listdir(results_dir):
        model_dir = os.path.join(results_dir, model_name)
        if not os.path.isdir(model_dir) or model_name.startswith("."):
            continue

        model_results = {}
        for root, dirs, files in os.walk(model_dir):
            for fname in files:
                if not fname.endswith(".json") or fname == "model_meta.json":
                    continue
                fpath = os.path.join(root, fname)
                with open(fpath) as f:
                    data = json.load(f)
                task_name = fname.replace(".json", "")
                if isinstance(data, list) and len(data) > 0:
                    data = data[0]
                model_results[task_name] = data

        if model_results:
            all_results[model_name] = model_results

    return all_results


def extract_scores(results):
    """결과에서 주요 점수를 추출한다."""
    rows = []
    for model_name, tasks in results.items():
        for task_name, task_data in tasks.items():
            scores = task_data.get("scores", {}).get("test", [])
            if not scores:
                continue

            lang_scores = {}
            for entry in scores:
                for score_item in entry.get("scores", [entry]):
                    lang = score_item.get("hf_subset", score_item.get("language", "unknown"))
                    main_score = score_item.get("main_score", 0)
                    lang_scores[lang] = main_score

            if lang_scores:
                avg_score = sum(lang_scores.values()) / len(lang_scores)

                # 태스크 그룹 판별
                task_group = "Other"
                for group, group_tasks in MTEB_TASK_GROUPS.items():
                    if task_name in group_tasks:
                        task_group = group
                        break

                rows.append({
                    "model": model_name,
                    "task": task_name,
                    "task_group": task_group,
                    "avg_score": round(avg_score * 100, 2),
                    "num_langs": len(lang_scores),
                    "lang_scores": lang_scores,
                })

    return rows


def get_experiment_info(name, teacher_key=None):
    """실험 설정 정보를 가져온다."""
    # Teacher-scoped experiments
    if teacher_key:
        if teacher_key == "me5":
            for exp in generate_me5_experiments():
                if exp["name"] == name:
                    return exp, teacher_key
        else:
            for exp in generate_experiments(teacher_key):
                if exp["name"] == name:
                    return exp, teacher_key

    # 기존 experiments
    for exp in EXPERIMENTS:
        if exp["name"] == name:
            return exp, "minilm"

    # me5 실험 탐색
    for exp in generate_me5_experiments():
        if exp["name"] == name:
            return exp, "me5"

    # 모든 teacher에서 탐색
    for tk in TEACHERS:
        if tk == "me5":
            continue  # 이미 위에서 처리
        for exp in generate_experiments(tk):
            if exp["name"] == name:
                return exp, tk

    return None, None


def print_results_for_teacher(teacher_key, detailed=False):
    """특정 teacher의 결과를 출력한다."""
    t = TEACHERS[teacher_key]
    results_dir = get_teacher_results_dir(teacher_key)

    results = load_mteb_results(results_dir)
    if not results:
        # fallback: 기존 경로
        results = load_mteb_results(RESULTS_DIR)
    if not results:
        print(f"No results found for {teacher_key}.")
        return

    rows = extract_scores(results)
    if not rows:
        print("No scores extracted.")
        return

    df = pd.DataFrame(rows)

    print(f"\n{'='*70}")
    print(f"Results for Teacher: {t['short_name']} ({t['model_id']})")
    print(f"{'='*70}")

    # 태스크 그룹별 결과
    for group in ["Classification", "Clustering", "STS"]:
        group_df = df[df["task_group"] == group]
        if group_df.empty:
            continue

        print(f"\n--- {group} ---")
        for task in group_df["task"].unique():
            task_df = group_df[group_df["task"] == task].sort_values("avg_score", ascending=False)
            print(f"\n  {task}:")
            print(f"    {'Model':<30} {'Avg Score':>10} {'#Langs':>8}")
            print(f"    {'-'*50}")
            for _, row in task_df.iterrows():
                marker = " *" if row["avg_score"] == task_df["avg_score"].max() else ""
                print(f"    {row['model']:<30} {row['avg_score']:>9.2f}% {row['num_langs']:>7}{marker}")

    # ── 태스크 그룹별 평균 요약 ──
    print(f"\n{'='*70}")
    print("Task Group Averages")
    print(f"{'='*70}")

    group_avgs = df.groupby(["model", "task_group"])["avg_score"].mean().unstack(fill_value=0)
    overall = df.groupby("model")["avg_score"].mean().sort_values(ascending=False)

    group_cols = [g for g in ["Classification", "Clustering", "STS"] if g in group_avgs.columns]

    print(f"\n  {'Model':<30} {'Overall':>8}", end="")
    for g in group_cols:
        print(f"  {g:>15}", end="")
    print()
    print(f"  {'-'*30} {'-'*8}", end="")
    for g in group_cols:
        print(f"  {'-'*15}", end="")
    print()

    for model in overall.index:
        print(f"  {model:<30} {overall[model]:>7.2f}%", end="")
        for g in group_cols:
            val = group_avgs.loc[model, g] if model in group_avgs.index else 0
            print(f"  {val:>14.2f}%", end="")
        print()

    # 종합 순위 (사이즈 포함)
    print(f"\n{'='*70}")
    print("Overall Ranking (with size)")
    print(f"{'='*70}")

    print(f"\n  {'Rank':<6} {'Model':<30} {'Overall':>8} {'Size (FP32)':>12}")
    print(f"  {'-'*60}")

    for rank, (model, score) in enumerate(overall.items(), 1):
        exp, tk = get_experiment_info(model, teacher_key)
        if exp:
            tconf = TEACHERS[tk]
            size = estimate_size(exp["layer_indices"], tconf["hidden_dim"],
                                  tconf["vocab_size"], tconf["intermediate_size"])
            size_str = f"{size['fp32_mb']}MB"
        else:
            size_str = "N/A"

        print(f"  #{rank:<5} {model:<30} {score:>7.2f}% {size_str:>12}")

    # 언어별 상세
    if detailed:
        print(f"\n{'='*70}")
        print("Detailed Scores by Language")
        print(f"{'='*70}")
        for _, row in df.iterrows():
            print(f"\n{row['model']} / {row['task']} ({row['task_group']}):")
            for lang, score in sorted(row["lang_scores"].items()):
                print(f"  {lang:<10} {score*100:>7.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=str, default=None,
                        choices=list(TEACHERS.keys()))
    parser.add_argument("--all-teachers", action="store_true",
                        help="모든 teacher 결과 비교")
    parser.add_argument("--detailed", action="store_true")
    args = parser.parse_args()

    if args.all_teachers:
        for tk in TEACHERS:
            print_results_for_teacher(tk, args.detailed)
    elif args.teacher:
        print_results_for_teacher(args.teacher, args.detailed)
    else:
        # 기존 동작
        print_results_for_teacher("minilm", args.detailed)


if __name__ == "__main__":
    main()
