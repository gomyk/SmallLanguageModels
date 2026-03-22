"""
Student 모델들의 크기, 속도, 파라미터 수를 종합적으로 비교한다.

Usage:
    python benchmark.py                    # 모든 student 벤치마크
    python benchmark.py --only L6_uniform  # 특정 모델만
"""

import argparse
import os
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from config import EXPERIMENTS, STUDENTS_DIR
from create_students import estimate_size


TEST_SENTENCES = {
    "ko": ["예약 좀 해줘", "지난번 주문 뭐였지?", "안녕하세요 반갑습니다"],
    "en": ["Book a table for me", "What did I order last time?", "Hello how are you"],
    "ja": ["予約をお願いします", "前回の注文は何でしたか", "こんにちは元気ですか"],
    "zh": ["帮我预约一下", "上次我点了什么", "你好你好吗"],
    "es": ["Reserva una mesa", "Qué pedí la última vez", "Hola cómo estás"],
}


def count_parameters(model):
    """모델의 전체 파라미터 수를 센다."""
    return sum(p.numel() for p in model.parameters())


def measure_model_disk_size(path):
    """디스크 상의 모델 크기 (MB)."""
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith((".bin", ".safetensors", ".onnx", ".npy")):
                total += os.path.getsize(os.path.join(root, f))
    return total / (1024 ** 2)


def benchmark_model(model, n_runs=50):
    """PyTorch 모델의 CPU 추론 속도를 측정한다."""
    all_sentences = []
    for lang_sents in TEST_SENTENCES.values():
        all_sentences.extend(lang_sents)

    # Warmup
    model.encode(all_sentences[:3])

    # Single sentence latency
    single_times = []
    for _ in range(n_runs):
        sent = all_sentences[_ % len(all_sentences)]
        start = time.perf_counter()
        model.encode([sent])
        elapsed = (time.perf_counter() - start) * 1000
        single_times.append(elapsed)

    # Batch latency
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
        "batch_mean_ms": round(np.mean(batch_times), 2),
        "batch_size": len(all_sentences),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", help="특정 모델만")
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()

    experiments = EXPERIMENTS
    if args.only:
        experiments = [e for e in experiments if e["name"] in args.only]

    results = []

    for exp in experiments:
        name = exp["name"]
        path = os.path.join(STUDENTS_DIR, name)
        if not os.path.exists(path):
            print(f"⚠ {name}: not found, skipping")
            continue

        print(f"Benchmarking {name}...")
        model = SentenceTransformer(path)
        model.to("cpu")

        params = count_parameters(model)
        disk_size = measure_model_disk_size(path)
        est = estimate_size(exp["layer_indices"])
        timings = benchmark_model(model, args.runs)

        results.append({
            "name": name,
            "layers": len(exp["layer_indices"]),
            "layer_indices": exp["layer_indices"],
            "params": params,
            "disk_mb": round(disk_size, 1),
            "est_int8_pruned_mb": est["int8_pruned_mb"],
            **timings,
        })

        # 메모리 해제
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── 결과 출력 ──
    print("\n" + "=" * 100)
    print("Benchmark Results")
    print("=" * 100)

    header = (f"{'Model':<16} {'Layers':>6} {'Params':>12} {'Disk':>8} "
              f"{'~INT8+P':>8} {'Single':>8} {'Median':>8} {'P95':>8} "
              f"{'Batch':>8} {'Size OK':>8} {'Speed OK':>9}")
    print(header)
    print("-" * 100)

    for r in sorted(results, key=lambda x: x["single_median_ms"]):
        size_ok = "✓" if r["est_int8_pruned_mb"] <= 50 else "✗"
        speed_ok = "✓" if r["single_median_ms"] < 10 else "✗"
        print(
            f"{r['name']:<16} {r['layers']:>6} {r['params']:>12,} "
            f"{r['disk_mb']:>7.1f}M {r['est_int8_pruned_mb']:>7.1f}M "
            f"{r['single_mean_ms']:>7.2f}ms {r['single_median_ms']:>7.2f}ms "
            f"{r['single_p95_ms']:>7.2f}ms {r['batch_mean_ms']:>7.2f}ms "
            f"{size_ok:>8} {speed_ok:>9}"
        )

    print("\nSize OK: estimated INT8 + vocab pruned ≤ 50MB")
    print("Speed OK: single sentence median < 10ms on CPU")

    # 최적 후보 추천
    valid = [r for r in results if r["est_int8_pruned_mb"] <= 50]
    if valid:
        best = min(valid, key=lambda x: x["single_median_ms"])
        print(f"\n★ Fastest under 50MB: {best['name']} "
              f"({best['single_median_ms']}ms, ~{best['est_int8_pruned_mb']}MB)")


if __name__ == "__main__":
    main()
