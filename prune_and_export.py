"""
최적 student 모델에 vocabulary pruning + ONNX INT8 양자화를 적용한다.

1. 타겟 18개 언어에서 사용되는 토큰만 유지 (250K → ~55K)
2. ONNX 변환
3. INT8 dynamic quantization
4. 크기 및 속도 검증

Usage:
    python prune_and_export.py --model L6_uniform
    python prune_and_export.py --model L6_uniform --skip-pruning  # 프루닝 생략
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from optimum.onnxruntime import ORTModelForFeatureExtraction
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.exporters.onnx import main_export
import onnxruntime as ort

from config import EXPERIMENTS, EXPORT_DIR, STUDENTS_DIR, TARGET_LANGUAGES


# ── Vocabulary Pruning (Unicode Script 기반) ──────────────────

import unicodedata

# 타겟 18개 언어가 사용하는 Unicode script 목록
# 각 토큰의 문자들이 이 script에 속하면 해당 토큰을 유지한다.
TARGET_SCRIPTS = {
    # Latin 계열: en, fr, de, pt, it, es, nl, pl, vi, id, tr
    "LATIN",
    # 한국어
    "HANGUL",
    # 일본어
    "HIRAGANA", "KATAKANA",
    # 중국어 + 일본어 한자
    "CJK",  # CJK Unified Ideographs (별도 처리)
    # 러시아어
    "CYRILLIC",
    # 아랍어
    "ARABIC",
    # 힌디어
    "DEVANAGARI",
    # 태국어
    "THAI",
    # 공통
    "COMMON", "INHERITED",
}

# CJK Unified Ideographs 범위 (unicodedata.script()가 없는 Python용)
CJK_RANGES = [
    (0x4E00, 0x9FFF),     # CJK Unified Ideographs
    (0x3400, 0x4DBF),     # CJK Extension A
    (0x20000, 0x2A6DF),   # CJK Extension B
    (0x2A700, 0x2B73F),   # CJK Extension C
    (0x2B740, 0x2B81F),   # CJK Extension D
    (0xF900, 0xFAFF),     # CJK Compatibility Ideographs
    (0x2F800, 0x2FA1F),   # CJK Compatibility Supplement
]

HANGUL_RANGES = [
    (0xAC00, 0xD7AF),     # Hangul Syllables
    (0x1100, 0x11FF),     # Hangul Jamo
    (0x3130, 0x318F),     # Hangul Compatibility Jamo
    (0xA960, 0xA97F),     # Hangul Jamo Extended-A
    (0xD7B0, 0xD7FF),     # Hangul Jamo Extended-B
]

HIRAGANA_RANGE = (0x3040, 0x309F)
KATAKANA_RANGES = [(0x30A0, 0x30FF), (0x31F0, 0x31FF), (0xFF65, 0xFF9F)]
CYRILLIC_RANGES = [(0x0400, 0x04FF), (0x0500, 0x052F)]
ARABIC_RANGES = [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)]
DEVANAGARI_RANGE = (0x0900, 0x097F)
THAI_RANGE = (0x0E00, 0x0E7F)
LATIN_RANGES = [
    (0x0000, 0x007F),     # Basic Latin (ASCII)
    (0x0080, 0x00FF),     # Latin-1 Supplement (àáâ, ñ, ü 등)
    (0x0100, 0x024F),     # Latin Extended-A/B (ą, ć, ş, ơ 등 — pl, tr, vi)
    (0x0250, 0x02AF),     # IPA Extensions
    (0x1E00, 0x1EFF),     # Latin Extended Additional (Vietnamese: ậ, ề, ồ 등)
    (0x2C60, 0x2C7F),     # Latin Extended-C
    (0xA720, 0xA7FF),     # Latin Extended-D
]


def _in_ranges(cp, ranges):
    """코드포인트가 주어진 범위 목록 안에 있는지 확인."""
    for start, end in ranges:
        if start <= cp <= end:
            return True
    return False


def _is_target_char(ch):
    """문자가 타겟 언어의 Unicode 범위에 속하는지 판별."""
    cp = ord(ch)

    # 공통: 숫자, 구두점, 기호, 공백, 제어문자
    cat = unicodedata.category(ch)
    if cat.startswith(("N", "P", "S", "Z", "C")):  # Number, Punct, Symbol, Separator, Control
        return True

    # 각 문자 체계 검사
    if _in_ranges(cp, LATIN_RANGES):
        return True
    if _in_ranges(cp, HANGUL_RANGES):
        return True
    if _in_ranges(cp, CJK_RANGES):
        return True
    if HIRAGANA_RANGE[0] <= cp <= HIRAGANA_RANGE[1]:
        return True
    if _in_ranges(cp, KATAKANA_RANGES):
        return True
    if _in_ranges(cp, CYRILLIC_RANGES):
        return True
    if _in_ranges(cp, ARABIC_RANGES):
        return True
    if DEVANAGARI_RANGE[0] <= cp <= DEVANAGARI_RANGE[1]:
        return True
    if THAI_RANGE[0] <= cp <= THAI_RANGE[1]:
        return True

    return False


def _is_target_token(token_str):
    """토큰 문자열의 모든 실질 문자가 타겟 언어에 속하는지 확인."""
    # sentencepiece 접두사 '▁' 제거
    clean = token_str.replace("▁", "").strip()
    if not clean:
        return True  # 공백/접두사만 있는 토큰은 유지

    return all(_is_target_char(ch) for ch in clean)


def collect_used_tokens(tokenizer, extra_texts=None):
    """Unicode script 기반으로 타겟 18개 언어에 필요한 토큰을 수집한다.

    전략:
    1. 특수 토큰 (CLS, SEP, PAD, UNK 등) → 무조건 유지
    2. 각 vocab 토큰을 디코딩하여 문자 단위로 Unicode script 검사
    3. 모든 문자가 타겟 언어 script에 속하는 토큰만 유지
    4. sentencepiece '▁' 접두사는 무시하고 실질 문자만 검사
    """
    keep_ids = set()

    # 1) 특수 토큰
    keep_ids.update(tokenizer.all_special_ids)

    # 2) 전체 vocab 스캔
    vocab = tokenizer.get_vocab()  # {token_str: token_id}
    total = len(vocab)

    for token_str, token_id in vocab.items():
        if token_id in keep_ids:
            continue
        if _is_target_token(token_str):
            keep_ids.add(token_id)

    # 3) 추가 텍스트에서 나온 토큰도 포함 (safety net)
    if extra_texts:
        for text in extra_texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            keep_ids.update(ids)

    kept = sorted(keep_ids)
    print(f"  Vocab scan: {len(kept):,} / {total:,} tokens kept "
          f"({len(kept)/total*100:.1f}%)")

    # 스크립트별 통계
    script_counts = {
        "Latin": 0, "Hangul": 0, "CJK": 0, "Hiragana": 0,
        "Katakana": 0, "Cyrillic": 0, "Arabic": 0,
        "Devanagari": 0, "Thai": 0, "Other": 0,
    }
    for tid in kept:
        tok = tokenizer.convert_ids_to_tokens(tid)
        if tok is None:
            continue
        clean = tok.replace("▁", "").strip()
        if not clean:
            continue
        first_ch = clean[0]
        cp = ord(first_ch)
        if _in_ranges(cp, LATIN_RANGES):
            script_counts["Latin"] += 1
        elif _in_ranges(cp, HANGUL_RANGES):
            script_counts["Hangul"] += 1
        elif _in_ranges(cp, CJK_RANGES):
            script_counts["CJK"] += 1
        elif HIRAGANA_RANGE[0] <= cp <= HIRAGANA_RANGE[1]:
            script_counts["Hiragana"] += 1
        elif _in_ranges(cp, KATAKANA_RANGES):
            script_counts["Katakana"] += 1
        elif _in_ranges(cp, CYRILLIC_RANGES):
            script_counts["Cyrillic"] += 1
        elif _in_ranges(cp, ARABIC_RANGES):
            script_counts["Arabic"] += 1
        elif DEVANAGARI_RANGE[0] <= cp <= DEVANAGARI_RANGE[1]:
            script_counts["Devanagari"] += 1
        elif THAI_RANGE[0] <= cp <= THAI_RANGE[1]:
            script_counts["Thai"] += 1
        else:
            script_counts["Other"] += 1

    for script, count in sorted(script_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"    {script:<12}: {count:>6} tokens")

    return kept


def prune_model_vocab(model, tokenizer, keep_ids):
    """모델의 임베딩 레이어를 pruning한다."""
    old_embeddings = model.get_input_embeddings()
    old_weight = old_embeddings.weight.data

    new_vocab_size = len(keep_ids)
    new_embeddings = nn.Embedding(new_vocab_size, old_weight.shape[1])

    for new_id, old_id in enumerate(keep_ids):
        new_embeddings.weight.data[new_id] = old_weight[old_id]

    model.set_input_embeddings(new_embeddings)

    # Config 업데이트
    model.config.vocab_size = new_vocab_size

    # ID 매핑 테이블 생성
    id_map = {old_id: new_id for new_id, old_id in enumerate(keep_ids)}

    return model, id_map


class PrunedTokenizerWrapper:
    """원본 토크나이저 + ID 매핑을 래핑하는 추론용 토크나이저."""

    def __init__(self, original_tokenizer, id_map, max_length=128):
        self.tokenizer = original_tokenizer
        self.id_map = id_map
        self.max_length = max_length
        # unknown 토큰 매핑 (pruned vocab에 없는 토큰 처리)
        self.unk_new_id = id_map.get(original_tokenizer.unk_token_id, 0)

    def __call__(self, texts, **kwargs):
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
            **kwargs,
        )
        # ID 매핑 적용
        mapped_ids = np.vectorize(
            lambda x: self.id_map.get(x, self.unk_new_id)
        )(encoded["input_ids"])
        encoded["input_ids"] = mapped_ids
        return encoded

    def save(self, path):
        self.tokenizer.save_pretrained(path)
        with open(os.path.join(path, "id_map.json"), "w") as f:
            json.dump({str(k): v for k, v in self.id_map.items()}, f)

    @classmethod
    def load(cls, path):
        tokenizer = AutoTokenizer.from_pretrained(path)
        with open(os.path.join(path, "id_map.json")) as f:
            id_map = {int(k): v for k, v in json.load(f).items()}
        return cls(tokenizer, id_map)


# ── ONNX Export & Quantization ─────────────────────────────────

def export_to_onnx(model_path, output_path):
    """HuggingFace 모델을 ONNX로 변환한다."""
    main_export(
        model_name_or_path=model_path,
        output=output_path,
        task="feature-extraction",
        opset=17,
    )


def quantize_onnx(onnx_path, output_path):
    """ONNX 모델에 INT8 dynamic quantization을 적용한다."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    model_path = os.path.join(onnx_path, "model.onnx")
    quantized_path = os.path.join(output_path, "model_quantized.onnx")
    os.makedirs(output_path, exist_ok=True)

    quantize_dynamic(
        model_input=model_path,
        model_output=quantized_path,
        weight_type=QuantType.QInt8,
    )

    return quantized_path


def measure_model_size(path):
    """모델 파일 크기를 MB 단위로 반환한다."""
    total = 0
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 ** 2)
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith((".onnx", ".bin", ".safetensors", ".npy")):
                total += os.path.getsize(os.path.join(root, f))
    return total / (1024 ** 2)


def benchmark_inference(onnx_path, tokenizer_wrapper, n_runs=100):
    """ONNX 모델의 추론 속도를 측정한다."""
    sess = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )

    test_texts = [
        "예약 좀 해줘",
        "What did I order last time?",
        "今日はいい天気ですね",
    ]

    # Warmup
    for text in test_texts:
        encoded = tokenizer_wrapper([text])
        feeds = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }
        sess.run(None, feeds)

    # Benchmark
    times = []
    for _ in range(n_runs):
        text = test_texts[_ % len(test_texts)]
        encoded = tokenizer_wrapper([text])
        feeds = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }
        start = time.perf_counter()
        sess.run(None, feeds)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    return {
        "mean_ms": round(np.mean(times), 2),
        "median_ms": round(np.median(times), 2),
        "p95_ms": round(np.percentile(times, 95), 2),
        "min_ms": round(np.min(times), 2),
    }


# ── Main ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Student 모델 이름 (e.g., L6_uniform)")
    parser.add_argument("--skip-pruning", action="store_true", help="Vocab pruning 생략")
    parser.add_argument("--benchmark-runs", type=int, default=100)
    args = parser.parse_args()

    student_path = os.path.join(STUDENTS_DIR, args.model)
    export_path = os.path.join(EXPORT_DIR, args.model)
    os.makedirs(export_path, exist_ok=True)

    # Student 모델의 HF transformer 경로
    hf_model_path = os.path.join(student_path, "0_Transformer")
    if not os.path.exists(hf_model_path):
        # sentence-transformers 구조가 아닌 경우
        hf_model_path = student_path

    print(f"Loading student model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
    model = AutoModel.from_pretrained(hf_model_path)

    orig_vocab = model.config.vocab_size
    print(f"  Original vocab size: {orig_vocab:,}")
    print(f"  Layers: {model.config.num_hidden_layers}")
    print(f"  Hidden dim: {model.config.hidden_size}")

    # ── Vocab Pruning ──
    id_map = None
    if not args.skip_pruning:
        print("\nPruning vocabulary...")
        keep_ids = collect_used_tokens(tokenizer)
        print(f"  Tokens to keep: {len(keep_ids):,} / {orig_vocab:,} "
              f"({len(keep_ids)/orig_vocab*100:.1f}%)")

        model, id_map = prune_model_vocab(model, tokenizer, keep_ids)
        print(f"  New vocab size: {model.config.vocab_size:,}")

        # pruned 모델 저장
        pruned_path = os.path.join(export_path, "pruned_hf")
        model.save_pretrained(pruned_path)
        tokenizer.save_pretrained(pruned_path)

        # ID 매핑 저장
        wrapper = PrunedTokenizerWrapper(tokenizer, id_map)
        wrapper.save(pruned_path)
        hf_model_path = pruned_path
    else:
        pruned_path = hf_model_path

    # ── ONNX Export ──
    print("\nExporting to ONNX...")
    onnx_path = os.path.join(export_path, "onnx")
    export_to_onnx(pruned_path if not args.skip_pruning else hf_model_path, onnx_path)

    onnx_size = measure_model_size(onnx_path)
    print(f"  ONNX FP32 size: {onnx_size:.1f}MB")

    # ── INT8 Quantization ──
    print("\nQuantizing to INT8...")
    quant_path = os.path.join(export_path, "onnx_int8")
    quantized_model_path = quantize_onnx(onnx_path, quant_path)

    quant_size = os.path.getsize(quantized_model_path) / (1024 ** 2)
    print(f"  INT8 quantized size: {quant_size:.1f}MB")

    if quant_size <= 50:
        print(f"  ✓ Size constraint met: {quant_size:.1f}MB ≤ 50MB")
    else:
        print(f"  ✗ Size constraint NOT met: {quant_size:.1f}MB > 50MB")

    # ── Benchmark ──
    print(f"\nBenchmarking inference ({args.benchmark_runs} runs)...")
    if id_map:
        tok_wrapper = PrunedTokenizerWrapper(tokenizer, id_map)
    else:
        tok_wrapper = lambda texts: tokenizer(
            texts, padding=True, truncation=True,
            max_length=128, return_tensors="np"
        )

    timings = benchmark_inference(quantized_model_path, tok_wrapper, args.benchmark_runs)
    print(f"  Mean:   {timings['mean_ms']:.2f}ms")
    print(f"  Median: {timings['median_ms']:.2f}ms")
    print(f"  P95:    {timings['p95_ms']:.2f}ms")
    print(f"  Min:    {timings['min_ms']:.2f}ms")

    # ── Summary ──
    summary = {
        "model": args.model,
        "original_vocab": orig_vocab,
        "pruned_vocab": model.config.vocab_size if not args.skip_pruning else orig_vocab,
        "num_layers": model.config.num_hidden_layers,
        "hidden_dim": model.config.hidden_size,
        "onnx_fp32_mb": round(onnx_size, 1),
        "onnx_int8_mb": round(quant_size, 1),
        "inference_ms": timings,
        "meets_size_constraint": quant_size <= 50,
        "meets_speed_constraint": timings["median_ms"] < 10,
    }

    summary_path = os.path.join(export_path, "export_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
