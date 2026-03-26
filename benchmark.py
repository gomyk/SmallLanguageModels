"""
Multi-Teacher Student 모델 크기/속도/파라미터 비교 벤치마크

Usage:
    python benchmark.py --teacher modernbert
    python benchmark.py --teacher gte
    python benchmark.py --all-teachers
"""

import argparse
import os
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from config import (
    TEACHERS, EXPERIMENTS, STUDENTS_DIR,
    generate_experiments, get_teacher_students_dir,
    estimate_size,
)


TEST_SENTENCES = {
    "ko": ["예약 좀 해줘", "지난번 주문 뭐였지?", "안녕하세요 반갑습니다"],
    "en": ["Book a table for me", "What did I order last time?", "Hello how are you"],
    "ja": ["予約をお願いします", "前回の注文は何でしたか", "こんにちは元気ですか"],
    "zh": ["帮我预约一下", "上次我点了什么", "你好你好吗"],
    "es": ["Reserva una mesa", "Qué pedí la última vez", "Hola cómo estás"],
}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def measure_model_disk_size(path):
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith((".bin", ".safetensors", ".onnx", ".npy")):
                total += os.path.getsize(os.path.join(root, f))
    return total / (1024 ** 2)


def benchmark_model(model, n_runs=50):
    all_sentences = []
    for lang_sents in TEST_SENTENCES.values():
        all_sentences.extend(lang_sents)

    model.encode(all_sentences[:3])  # warmup

    single_times = []
    for i in range(n_runs):
        sent = all_sentences[i % len(all_sentences)]
        start = time.perf_counter()
        model.encode([sent])
        elapsed = (time.perf_counter() - start) * 1000
        single_times.append(elapsed)

    batch_times = []
    for _ in range(n_runs // 5):
        start = time.perf_counter()
        model.encode(all_sentences)
        elapsed = (time.perf_counter() - start) * 1000
        batch_times.append(elapsed)

    return {
        "single_mean_ms": round(np.mean(single_times), 2),
        "single_median_ms": round(np.median(single_times), 2),
        "single_p95_ms": round(np.percentile(single_times, 95), 2),
        "batch_mean_ms": round(np.mean(batch_times), 2) if batch_times else 0,
        "batch_size": len(all_sentences),
    }


def benchmark_teacher(teacher_key, runs=50):
    """특정 teacher의 student 모델들을 벤치마크한다."""
    t = TEACHERS[teacher_key]
    students_dir = get_teacher_students_dir(teacher_key)

    if teacher_key == "minilm":
        experiments = EXPERIMENTS
    else:
        experiments = generate_experiments(teacher_key)

    results = []

    for exp in experiments:
        name = exp["name"]
        path = os.path.join(students_dir, name)
        if not os.path.exists(path):
            path = os.path.join(STUDENTS_DIR, name)
        if not os.path.exists(path):
            print(f"  {name}: not found, skipping")
            continue

        print(f"  Benchmarking {name}...")
        try:
            model = SentenceTransformer(path, trust_remote_code=True)
            model.to("cpu")

            params = count_parameters(model)
            disk_size = measure_model_disk_size(path)
            est = estimate_size(exp["layer_indices"], t["hidden_dim"],
                                t["vocab_size"], t["intermediate_size"])
            timings = benchmark_model(model, runs)

            results.append({
                "name": name,
                "teacher": teacher_key,
                "layers": len(exp["layer_indices"]),
                "layer_indices": exp["layer_indices"],
                "params": params,
                "disk_mb": round(disk_size, 1),
                "est_fp32_mb": est["fp32_mb"],
                **timings,
            })

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {name}: error - {e}")

    return results


def print_benchmark_results(results, teacher_key):
    """벤치마크 결과를 출력한다."""
    t = TEACHERS[teacher_key]

    print(f"\n{'='*100}")
    print(f"Benchmark: {t['short_name']} ({t['model_id']})")
    print(f"{'='*100}")

    header = (f"{'Model':<30} {'Layers':>6} {'Params':>12} {'Disk':>8} "
              f"{'Single':>8} {'Median':>8} {'P95':>8} {'Batch':>8}")
    print(header)
    print("-" * 100)

    for r in sorted(results, key=lambda x: x["single_median_ms"]):
        print(
            f"{r['name']:<30} {r['layers']:>6} {r['params']:>12,} "
            f"{r['disk_mb']:>7.1f}M "
            f"{r['single_mean_ms']:>7.2f}ms {r['single_median_ms']:>7.2f}ms "
            f"{r['single_p95_ms']:>7.2f}ms {r['batch_mean_ms']:>7.2f}ms"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=str, default=None,
                        choices=list(TEACHERS.keys()))
    parser.add_argument("--all-teachers", action="store_true")
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()

    if args.all_teachers:
        teacher_keys = list(TEACHERS.keys())
    elif args.teacher:
        teacher_keys = [args.teacher]
    else:
        teacher_keys = ["minilm"]

    for tk in teacher_keys:
        results = benchmark_teacher(tk, args.runs)
        if results:
            print_benchmark_results(results, tk)


if __name__ == "__main__":
    main()
