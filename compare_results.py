"""
MTEB 평가 결과를 비교하여 최적 student 모델을 선정한다.

Usage:
    python compare_results.py
    python compare_results.py --detailed   # 언어별 상세 결과
"""

import argparse
import json
import os

import pandas as pd

from config import EXPERIMENTS, RESULTS_DIR, STUDENTS_DIR
from create_students import estimate_size


def load_mteb_results(results_dir):
    """MTEB 결과 디렉토리에서 모든 결과를 로드한다."""
    all_results = {}

    for model_name in os.listdir(results_dir):
        model_dir = os.path.join(results_dir, model_name)
        if not os.path.isdir(model_dir) or model_name.startswith("."):
            continue

        model_results = {}
        # 재귀적으로 JSON 파일 탐색 (MTEB가 깊은 하위 디렉토리에 저장)
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
            # MTEB stores scores in test split
            scores = task_data.get("scores", {}).get("test", [])
            if not scores:
                continue

            # 각 언어/서브셋의 main_score 수집
            lang_scores = {}
            for entry in scores:
                for score_item in entry.get("scores", [entry]):
                    lang = score_item.get("hf_subset", score_item.get("language", "unknown"))
                    main_score = score_item.get("main_score", 0)
                    lang_scores[lang] = main_score

            # 평균 점수
            if lang_scores:
                avg_score = sum(lang_scores.values()) / len(lang_scores)
                rows.append({
                    "model": model_name,
                    "task": task_name,
                    "avg_score": round(avg_score * 100, 2),
                    "num_langs": len(lang_scores),
                    "lang_scores": lang_scores,
                })

    return rows


def get_experiment_info(name):
    """실험 설정 정보를 가져온다."""
    for exp in EXPERIMENTS:
        if exp["name"] == name:
            return exp
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detailed", action="store_true", help="언어별 상세 결과")
    args = parser.parse_args()

    results = load_mteb_results(RESULTS_DIR)
    if not results:
        print("No results found. Run run_mteb.py first.")
        return

    rows = extract_scores(results)
    if not rows:
        print("No scores extracted. Check result format.")
        return

    df = pd.DataFrame(rows)

    # ── 태스크별 모델 비교 ──
    print("\n" + "=" * 70)
    print("MTEB Results by Task")
    print("=" * 70)

    for task in df["task"].unique():
        task_df = df[df["task"] == task].sort_values("avg_score", ascending=False)
        print(f"\n📋 {task}:")
        print(f"  {'Model':<20} {'Avg Score':>10} {'#Langs':>8}")
        print(f"  {'-'*40}")
        for _, row in task_df.iterrows():
            marker = " ★" if row["avg_score"] == task_df["avg_score"].max() else ""
            print(f"  {row['model']:<20} {row['avg_score']:>9.2f}% {row['num_langs']:>7}{marker}")

    # ── 종합 순위 ──
    print("\n" + "=" * 70)
    print("Overall Ranking (Average across all tasks)")
    print("=" * 70)

    overall = df.groupby("model")["avg_score"].mean().sort_values(ascending=False)

    print(f"\n  {'Rank':<6} {'Model':<20} {'Avg MTEB':>10} {'Est. Size (FP32)':>24}")
    print(f"  {'-'*62}")

    for rank, (model, score) in enumerate(overall.items(), 1):
        exp = get_experiment_info(model)
        if exp:
            size = estimate_size(exp["layer_indices"])
            size_str = f"{size['fp32_mb']}MB"
            fits = "✓" if size["fp32_mb"] <= 100 else "✗"
        else:
            size_str = "N/A"
            fits = ""

        marker = " ← BEST" if rank == 1 else ""
        print(f"  #{rank:<5} {model:<20} {score:>9.2f}% {size_str:>14} {fits}{marker}")

    # ── 최종 추천 ──
    # 50MB 이하 모델 중 가장 높은 점수
    print("\n" + "=" * 70)
    print("Recommendation (best score under 100MB)")
    print("=" * 70)

    candidates = []
    for model, score in overall.items():
        exp = get_experiment_info(model)
        if exp:
            size = estimate_size(exp["layer_indices"])
            if size["fp32_mb"] <= 100:
                candidates.append((model, score, size["fp32_mb"]))

    if candidates:
        best = max(candidates, key=lambda x: x[1])
        print(f"\n  ★ Best model: {best[0]}")
        print(f"    MTEB avg score: {best[1]:.2f}%")
        print(f"    Estimated FP32 size: {best[2]}MB")
        exp = get_experiment_info(best[0])
        print(f"    Layers: {exp['layer_indices']}")
        print(f"    Description: {exp['description']}")
        print(f"\n  Next step: python prune_and_export.py --model {best[0]}")
    else:
        print("  No model meets the 100MB constraint.")

    # ── 언어별 상세 (옵션) ──
    if args.detailed:
        print("\n" + "=" * 70)
        print("Detailed Scores by Language")
        print("=" * 70)
        for _, row in df.iterrows():
            print(f"\n{row['model']} / {row['task']}:")
            for lang, score in sorted(row["lang_scores"].items()):
                print(f"  {lang:<10} {score*100:>7.2f}%")


if __name__ == "__main__":
    main()
