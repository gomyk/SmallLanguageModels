"""
Multi-Teacher Student Model Creation via Layer Pruning + Vocab Pruning

다양한 teacher 모델(MiniLM, ModernBERT, GTE 등)에서 layer pruning + vocab pruning으로
경량 student 모델을 생성한다.

Usage:
    # 특정 teacher의 student 생성
    python create_students.py --teacher modernbert
    python create_students.py --teacher gte

    # 기존 MiniLM 실험 (하위 호환)
    python create_students.py

    # 특정 실험만
    python create_students.py --teacher modernbert --only modernbert_L6_uniform

    # vocab pruning 없이
    python create_students.py --teacher gte --no-prune

    # 커스텀 vocab 크기
    python create_students.py --teacher modernbert --max-vocab 15000
"""

import argparse
import os
import shutil

import torch
from sentence_transformers import SentenceTransformer

from config import (
    TEACHERS, EXPERIMENTS, STUDENTS_DIR,
    TEACHER_MODEL, TARGET_LANGUAGES,
    generate_experiments, generate_me5_experiments,
    get_teacher_students_dir,
    estimate_size, calculate_target_vocab,
    find_optimal_config, _estimate_for_teacher,
)
from arch_utils import (
    create_pruned_student,
    collect_corpus_tokens,
    prune_tokenizer_and_embeddings,
    save_as_sentence_transformer,
    detect_tokenizer_type,
    reduce_hidden_dim,
    reduce_hidden_dim_pca,
    reduce_hidden_dim_activation,
)


# ── Distillation Corpus Loading ───────────────────────────────

def load_distill_corpus(max_per_lang=5000, cache_dir="data/distill_corpus"):
    """MTEB 태스크 데이터셋에서 텍스트를 수집한다."""
    from datasets import load_dataset

    cache_file = os.path.join(cache_dir, f"distill_texts_{max_per_lang}.txt")

    if os.path.exists(cache_file):
        print(f"  Loading cached corpus: {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"  Corpus: {len(texts):,} sentences")
        return texts

    os.makedirs(cache_dir, exist_ok=True)
    texts = []

    # MASSIVE (다국어 - 가장 큰 소스)
    MASSIVE_LANGS = {
        "ko": "ko-KR", "en": "en-US", "ja": "ja-JP", "zh": "zh-CN",
        "es": "es-ES", "fr": "fr-FR", "de": "de-DE", "pt": "pt-PT",
        "it": "it-IT", "ru": "ru-RU", "ar": "ar-SA", "hi": "hi-IN",
        "th": "th-TH", "vi": "vi-VN", "id": "id-ID", "pl": "pl-PL",
    }

    print("  Loading MASSIVE dataset...")
    for lang, subset in MASSIVE_LANGS.items():
        try:
            # MASSIVE의 실제 config 이름은 locale 형태
            try:
                ds = load_dataset("mteb/amazon_massive_intent", subset, split="train")
            except Exception:
                # fallback: 단순 언어 코드
                ds = load_dataset("mteb/amazon_massive_intent", lang, split="train")
            lang_texts = [row["text"] for row in ds if row.get("text")][:max_per_lang]
            texts.extend(lang_texts)
            print(f"    {lang}: {len(lang_texts)} sentences")
        except Exception as e:
            print(f"    {lang}: failed - {e}")

    # STS benchmark (영어 sentence pairs)
    print("  Loading STS benchmark...")
    try:
        ds = load_dataset("mteb/stsbenchmark-sts", split="train")
        for row in ds:
            for field in ["sentence1", "sentence2"]:
                if row.get(field):
                    texts.append(row[field])
        print(f"    STSBenchmark: {len(ds)*2} sentences")
    except Exception as e:
        print(f"    STSBenchmark: failed - {e}")

    # Banking77 (영어 classification)
    print("  Loading Banking77...")
    try:
        ds = load_dataset("mteb/banking77", split="train")
        b77_texts = [row["text"] for row in ds if row.get("text")]
        texts.extend(b77_texts)
        print(f"    Banking77: {len(b77_texts)} sentences")
    except Exception as e:
        print(f"    Banking77: failed - {e}")

    # 캐시 저장
    with open(cache_file, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t.strip() + "\n")

    print(f"  Total corpus: {len(texts):,} sentences")
    return texts


# ── Student 생성 (Architecture-Agnostic) ──────────────────────

def create_student_for_teacher(teacher_key, experiment, max_vocab=None,
                                no_prune=False, corpus_texts=None):
    """특정 teacher에 대해 student 모델을 생성한다."""
    t = TEACHERS[teacher_key]
    name = experiment["name"]
    layers = experiment["layer_indices"]
    vocab_keep_ratio = experiment.get("vocab_keep_ratio", None)

    students_dir = get_teacher_students_dir(teacher_key)
    save_path = os.path.join(students_dir, name)

    # 사이즈 예측
    if no_prune:
        vocab_for_est = t["vocab_size"]
    elif max_vocab:
        vocab_for_est = max_vocab
    else:
        # vocab pruning은 코퍼스 미등장 토큰만 제거 → 사이즈 예측은 원본 vocab 기준
        vocab_for_est = t["vocab_size"]

    size = _estimate_for_teacher(teacher_key, layers, vocab_for_est)
    print(f"\nCreating {name}: layers={layers}")
    print(f"  {experiment['description']}")
    print(f"  Teacher: {t['model_id']}")
    print(f"  Estimated FP32 size: {size['fp32_mb']}MB ({size['total_params']:,} params)")
    if not no_prune:
        if vocab_keep_ratio:
            print(f"  Vocab keep ratio: {vocab_keep_ratio} ({int(vocab_keep_ratio*100)}%)")
        else:
            print(f"  Target vocab: {vocab_for_est:,}")

    # Teacher에서 student 생성 (layer pruning)
    student_hf, tokenizer = create_pruned_student(
        t["model_id"], layers,
        layer_accessor=t["layer_accessor"],
        trust_remote_code=t["trust_remote_code"],
    )
    print(f"  Layer pruning: {t['num_layers']} → {len(layers)} layers")

    # Vocab pruning
    if not no_prune:
        print("  Collecting corpus tokens for vocab pruning...")
        if corpus_texts is None:
            corpus_texts = load_distill_corpus()

        # vocab_keep_ratio가 실험에 지정되어 있으면 사용, 아니면 max_vocab 또는 기본 동작
        keep_ids = collect_corpus_tokens(tokenizer, texts=corpus_texts,
                                          max_vocab=max_vocab,
                                          vocab_keep_ratio=vocab_keep_ratio)

        hf_tmp = os.path.join(save_path, "_hf_pruned")
        student_hf = prune_tokenizer_and_embeddings(
            student_hf, tokenizer, keep_ids, hf_tmp
        )
        print(f"  Vocab pruned: {t['vocab_size']:,} → {student_hf.config.vocab_size:,}")

        # Pruned tokenizer 다시 로드
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            hf_tmp, trust_remote_code=t["trust_remote_code"]
        )

        # SentenceTransformer로 저장
        save_as_sentence_transformer(student_hf, tokenizer, save_path)
        shutil.rmtree(hf_tmp, ignore_errors=True)
    else:
        save_as_sentence_transformer(student_hf, tokenizer, save_path)

    # Sanity check
    try:
        st = SentenceTransformer(save_path, trust_remote_code=True)
        test_sentences = ["Hello world", "안녕하세요", "こんにちは"]
        embeddings = st.encode(test_sentences)
        print(f"  Sanity check: output shape = {embeddings.shape}")
        del st
    except Exception as e:
        print(f"  Sanity check failed: {e}")

    # 실제 safetensors 크기 확인
    for fname in ["model.safetensors", "pytorch_model.bin"]:
        st_path = os.path.join(save_path, fname)
        if not os.path.exists(st_path):
            # sentence_transformers 구조: 0_Transformer/ 하위
            st_path = os.path.join(save_path, "0_Transformer", fname)
        if os.path.exists(st_path):
            size_mb = os.path.getsize(st_path) / (1024 ** 2)
            print(f"  {fname}: {size_mb:.1f}MB")
            break

    print(f"  Saved to {save_path}")
    return save_path


# ── Compressed Model Creation ─────────────────────────────────

def _build_and_save_model(teacher_key, layer_indices, target_hidden, target_inter,
                          keep_ids, save_name, needs_hidden_reduction,
                          use_pca=False, use_activation=False, corpus_texts=None):
    """Teacher에서 압축 모델을 생성하여 저장한다 (내부 헬퍼).

    Layer pruning → Hidden dim 축소 (PCA 또는 슬라이싱) → Vocab pruning → 저장.

    Returns:
        (save_path, actual_vocab_size)
    """
    from transformers import AutoTokenizer

    t = TEACHERS[teacher_key]
    students_dir = get_teacher_students_dir(teacher_key)
    save_path = os.path.join(students_dir, save_name)
    os.makedirs(save_path, exist_ok=True)

    # Step 1: Layer pruning
    student_hf, tokenizer = create_pruned_student(
        t["model_id"], layer_indices,
        layer_accessor=t["layer_accessor"],
        trust_remote_code=t["trust_remote_code"],
    )
    print(f"  Layer pruning: {t['num_layers']} → {len(layer_indices)} layers")

    # Step 2: Hidden dim 축소 (필요한 경우)
    if needs_hidden_reduction:
        if use_activation and corpus_texts:
            student_hf = reduce_hidden_dim_activation(
                student_hf, tokenizer, target_hidden, corpus_texts,
                new_intermediate_size=target_inter,
                trust_remote_code=t["trust_remote_code"],
            )
        elif use_pca and corpus_texts:
            student_hf = reduce_hidden_dim_pca(
                student_hf, tokenizer, target_hidden, corpus_texts,
                new_intermediate_size=target_inter,
                trust_remote_code=t["trust_remote_code"],
            )
        else:
            student_hf = reduce_hidden_dim(
                student_hf, target_hidden, target_inter,
                trust_remote_code=t["trust_remote_code"],
            )

    # Step 3: Vocab pruning
    hf_tmp = os.path.join(save_path, "_hf_pruned")
    student_hf = prune_tokenizer_and_embeddings(
        student_hf, tokenizer, keep_ids, hf_tmp
    )
    print(f"  Vocab pruned: {t['vocab_size']:,} → {student_hf.config.vocab_size:,}")

    tokenizer = AutoTokenizer.from_pretrained(
        hf_tmp, trust_remote_code=t["trust_remote_code"]
    )

    save_as_sentence_transformer(student_hf, tokenizer, save_path)
    shutil.rmtree(hf_tmp, ignore_errors=True)

    # Sanity check
    try:
        st = SentenceTransformer(save_path, trust_remote_code=True)
        test_sentences = ["Hello world", "안녕하세요", "こんにちは"]
        embeddings = st.encode(test_sentences)
        print(f"  Sanity check: output shape = {embeddings.shape}")
        del st
    except Exception as e:
        print(f"  Sanity check failed: {e}")

    # 실제 safetensors 크기 확인
    for fname in ["model.safetensors", "pytorch_model.bin"]:
        st_path = os.path.join(save_path, fname)
        if not os.path.exists(st_path):
            st_path = os.path.join(save_path, "0_Transformer", fname)
        if os.path.exists(st_path):
            size_mb = os.path.getsize(st_path) / (1024 ** 2)
            print(f"  {fname}: {size_mb:.1f}MB")
            break

    print(f"  Saved to {save_path}")
    return save_path, student_hf.config.vocab_size


def create_compressed_model(teacher_key, max_params=20_000_000, max_fp32_mb=50.0,
                            min_layers=4, vocab_percentile=0.95,
                            corpus_texts=None, use_pca=False, use_activation=False,
                            min_vocab=None):
    """크기 제약을 만족하는 압축 모델을 생성한다.

    Teacher/Student 파라미터 비율이 10x를 초과하면 중간 모델(~1/5 teacher)을
    추가 생성하여 2단계 증류를 지원한다.

    절차:
      1. 코퍼스 기반 vocab 분석 (누적 빈도 percentile)
      2. 제약 내 최적 설정 탐색 (레이어 → hidden dim 순)
      3. Teacher/Student 비율 확인 → 10x 초과 시 중간 모델도 생성
      4. Layer pruning → Hidden dim 축소 (PCA 또는 슬라이싱) → Vocab pruning → 저장

    Returns:
        dict with keys: final_path, intermediate_path (or None), needs_two_stage
    """
    from transformers import AutoTokenizer

    t = TEACHERS[teacher_key]
    print(f"\n{'='*60}")
    print(f"Compressed Model Creation: {t['model_id']}")
    print(f"  Constraints: max {max_params/1e6:.0f}M params, {max_fp32_mb}MB FP32")
    print(f"  Min layers: {min_layers}, Vocab percentile: {int(vocab_percentile*100)}%")
    print(f"{'='*60}")

    # ── Phase 1: Vocab 분석 ──
    print("\n[Phase 1] Vocab analysis...")
    tokenizer = AutoTokenizer.from_pretrained(
        t["model_id"], trust_remote_code=t["trust_remote_code"]
    )
    if corpus_texts is None:
        corpus_texts = load_distill_corpus()

    # 코퍼스 전체 토큰 수집 (빈도순, vocab_percentile은 상한으로만 사용)
    keep_ids_all = collect_corpus_tokens(
        tokenizer, texts=corpus_texts, vocab_keep_ratio=vocab_percentile
    )
    corpus_vocab_size = len(keep_ids_all)

    # min_vocab floor 적용: percentile 결과보다 floor이 크면 확장
    if min_vocab and corpus_vocab_size < min_vocab:
        print(f"  Corpus vocab ({int(vocab_percentile*100)}% cumulative): {corpus_vocab_size:,} "
              f"< min_vocab ({min_vocab:,}), expanding...")
        keep_ids_all = collect_corpus_tokens(
            tokenizer, texts=corpus_texts, max_vocab=min_vocab
        )
        corpus_vocab_size = len(keep_ids_all)

    print(f"  Corpus vocab: {t['vocab_size']:,} → {corpus_vocab_size:,}"
          f"{f' (min_vocab={min_vocab:,})' if min_vocab else f' ({int(vocab_percentile*100)}% cumulative)'}")

    # ── Phase 2: Layer + Hidden + Vocab joint 최적화 ──
    print("\n[Phase 2] Joint optimization (layers + hidden_dim + vocab)...")
    opt = find_optimal_config(
        teacher_key, max_params, max_fp32_mb, min_layers,
        corpus_vocab_size=corpus_vocab_size,
    )

    layer_indices = opt["layer_indices"]
    target_hidden = opt["hidden_dim"]
    target_inter = opt["intermediate_size"]
    target_vocab = opt["target_vocab"]

    # Vocab을 target_vocab까지만 유지 (빈도순 상위)
    if target_vocab < corpus_vocab_size:
        keep_ids = collect_corpus_tokens(
            tokenizer, texts=corpus_texts, max_vocab=target_vocab
        )
    else:
        keep_ids = keep_ids_all
    actual_vocab = len(keep_ids)

    # Vocab이 budget 초과 시 (특수토큰/byte토큰 강제 포함) → vocab 고정, 나머지 재조정
    if actual_vocab > target_vocab:
        print(f"    Vocab floor ({actual_vocab:,}) > budget ({target_vocab:,}), "
              f"re-optimizing with fixed vocab...")
        reopt = find_optimal_config(
            teacher_key, max_params, max_fp32_mb, min_layers,
            estimated_vocab_size=actual_vocab,
        )
        layer_indices = reopt["layer_indices"]
        target_hidden = reopt["hidden_dim"]
        target_inter = reopt["intermediate_size"]

    est = _estimate_for_teacher(teacher_key, layer_indices, actual_vocab,
                                hidden_dim=target_hidden, intermediate_size=target_inter)
    print(f"  Optimized config:")
    print(f"    Layers: {len(layer_indices)} (indices: {layer_indices})")
    print(f"    Hidden dim: {t['hidden_dim']} → {target_hidden}")
    print(f"    Intermediate: {t['intermediate_size']} → {target_inter}")
    print(f"    Vocab: {t['vocab_size']:,} → {actual_vocab:,} "
          f"(budget: {target_vocab:,}, corpus: {corpus_vocab_size:,})")
    print(f"    Estimated: {est['total_params']:,} params, {est['fp32_mb']}MB FP32")

    meets_constraints = (est['total_params'] <= max_params
                         and est['fp32_mb'] <= max_fp32_mb)
    if not meets_constraints:
        print(f"    (cannot fully meet constraints - maximally reduced version)")

    # ── Phase 2.5: Teacher/Student 비율 확인 → 2단계 증류 필요 여부 ──
    teacher_all_layers = list(range(t["num_layers"]))
    teacher_est = _estimate_for_teacher(teacher_key, teacher_all_layers, t["vocab_size"])
    ratio = teacher_est["total_params"] / max(est["total_params"], 1)
    needs_two_stage = ratio > 10

    intermediate_path = None
    if needs_two_stage:
        print(f"\n  Teacher/Student ratio: {ratio:.0f}x (>10x)")
        print(f"  → 2-stage distillation required, creating intermediate model (~1/5 teacher)")

        # 중간 모델: teacher 파라미터의 ~1/5
        mid_target_params = teacher_est["total_params"] // 5
        mid_target_mb = mid_target_params * 4 / (1024 ** 2)

        mid_opt = find_optimal_config(
            teacher_key, mid_target_params, mid_target_mb,
            min_layers=min_layers, corpus_vocab_size=actual_vocab,
        )
        mid_vocab = mid_opt.get("target_vocab", actual_vocab)
        mid_est = _estimate_for_teacher(
            teacher_key, mid_opt["layer_indices"], mid_vocab,
            hidden_dim=mid_opt["hidden_dim"],
            intermediate_size=mid_opt["intermediate_size"],
        )
        print(f"\n  Intermediate model config:")
        print(f"    Layers: {len(mid_opt['layer_indices'])} (indices: {mid_opt['layer_indices']})")
        print(f"    Hidden dim: {t['hidden_dim']} → {mid_opt['hidden_dim']}")
        print(f"    Intermediate: {t['intermediate_size']} → {mid_opt['intermediate_size']}")
        print(f"    Estimated: {mid_est['total_params']:,} params, {mid_est['fp32_mb']}MB FP32")

        # 중간 모델 생성 (중간 모델은 PCA 불필요 — 크기가 충분히 큼)
        print(f"\n[Phase 3a] Building intermediate model...")
        intermediate_path, _ = _build_and_save_model(
            teacher_key,
            mid_opt["layer_indices"],
            mid_opt["hidden_dim"],
            mid_opt["intermediate_size"],
            keep_ids,
            f"{teacher_key}_intermediate",
            mid_opt["needs_hidden_reduction"],
        )

    # ── Phase 3: 최종 모델 생성 ──
    phase_label = "3b" if needs_two_stage else "3"
    method = " (Activation)" if use_activation and opt["needs_hidden_reduction"] else \
             " (PCA)" if use_pca and opt["needs_hidden_reduction"] else ""
    print(f"\n[Phase {phase_label}] Building final model{method}...")
    final_path, final_vocab = _build_and_save_model(
        teacher_key,
        layer_indices,
        target_hidden,
        target_inter,
        keep_ids,
        f"{teacher_key}_compressed",
        opt["needs_hidden_reduction"],
        use_pca=use_pca,
        use_activation=use_activation,
        corpus_texts=corpus_texts,
    )

    # ── 결과 요약 ──
    print(f"\n{'='*60}")
    print(f"  COMPRESSION SUMMARY: {t['model_id']}")
    print(f"{'='*60}")
    if needs_two_stage:
        print(f"  Intermediate: {intermediate_path}")
        mid_est_str = f"{mid_est['total_params']/1e6:.1f}M params, {mid_est['fp32_mb']}MB"
        print(f"    Config: {len(mid_opt['layer_indices'])}L / {mid_opt['hidden_dim']}d → {mid_est_str}")
    print(f"  Final: {final_path}")
    print(f"    Config: {len(layer_indices)}L / {target_hidden}d / {target_inter}i")
    print(f"    Vocab: {final_vocab:,}, Est: {est['total_params']:,} params, {est['fp32_mb']}MB")
    print(f"    Meets constraints: {'Yes' if meets_constraints else 'No (maximally reduced)'}")
    print(f"{'─'*60}")

    if needs_two_stage:
        print(f"\n  Next steps (2-stage distillation):")
        print(f"    python distill.py --teacher {teacher_key} "
              f"--student {teacher_key}_compressed --two-stage")
    else:
        print(f"\n  Next step:")
        print(f"    python distill.py --teacher {teacher_key} "
              f"--student {teacher_key}_compressed")
    print(f"{'='*60}")

    return {
        "final_path": final_path,
        "intermediate_path": intermediate_path,
        "needs_two_stage": needs_two_stage,
    }


def create_explicit_compressed_model(teacher_key, hidden_dim=None, num_layers=None,
                                     vocab_percentile=0.95, corpus_texts=None,
                                     use_pca=False, use_activation=False,
                                     min_vocab=None):
    """명시적으로 지정된 hidden_dim/num_layers로 압축 모델을 생성한다.

    find_optimal_config를 건너뛰고 사용자가 지정한 값을 직접 사용한다.
    Vocab은 코퍼스 기반 pruning (미등장 토큰 제거).

    Args:
        teacher_key: TEACHERS dict의 키
        hidden_dim: 목표 hidden dimension. None이면 원본 유지.
        num_layers: 목표 레이어 수. None이면 원본 유지.
        vocab_percentile: vocab 누적 빈도 유지 비율
        corpus_texts: 코퍼스 텍스트 리스트
        use_pca: PCA 기반 hidden dim 축소 사용 여부
        min_vocab: vocab 최소 하한
    """
    from transformers import AutoTokenizer
    from config import make_uniform_indices

    t = TEACHERS[teacher_key]
    target_hidden = hidden_dim if hidden_dim is not None else t["hidden_dim"]
    target_layers = num_layers if num_layers is not None else t["num_layers"]
    needs_hidden_reduction = target_hidden < t["hidden_dim"]

    # Intermediate size 비례 축소
    if needs_hidden_reduction:
        ratio = target_hidden / t["hidden_dim"]
        target_inter = max(64, (int(t["intermediate_size"] * ratio) // 64) * 64)
    else:
        target_inter = t["intermediate_size"]

    layer_indices = make_uniform_indices(t["num_layers"], target_layers)

    print(f"\n{'='*60}")
    print(f"Explicit Compressed Model: {t['model_id']}")
    print(f"  Layers: {t['num_layers']} → {target_layers} (indices: {layer_indices})")
    print(f"  Hidden: {t['hidden_dim']} → {target_hidden}")
    print(f"  Intermediate: {t['intermediate_size']} → {target_inter}")
    print(f"{'='*60}")

    # ── Vocab 분석 ──
    print("\n[Phase 1] Vocab analysis...")
    tokenizer = AutoTokenizer.from_pretrained(
        t["model_id"], trust_remote_code=t["trust_remote_code"]
    )
    if corpus_texts is None:
        corpus_texts = load_distill_corpus()

    keep_ids = collect_corpus_tokens(
        tokenizer, texts=corpus_texts, vocab_keep_ratio=vocab_percentile
    )

    if min_vocab and len(keep_ids) < min_vocab:
        print(f"  Expanding vocab to min_vocab={min_vocab:,}...")
        keep_ids = collect_corpus_tokens(
            tokenizer, texts=corpus_texts, max_vocab=min_vocab
        )

    actual_vocab = len(keep_ids)
    print(f"  Vocab: {t['vocab_size']:,} → {actual_vocab:,}")

    # ── Size 예측 ──
    est = _estimate_for_teacher(teacher_key, layer_indices, actual_vocab,
                                hidden_dim=target_hidden, intermediate_size=target_inter)
    print(f"  Estimated: {est['total_params']:,} params, {est['fp32_mb']}MB FP32")

    # ── 최종 모델 생성 ──
    method = " (Activation)" if use_activation and needs_hidden_reduction else \
             " (PCA)" if use_pca and needs_hidden_reduction else ""
    print(f"\n[Phase 2] Building final model{method}...")
    final_path, final_vocab = _build_and_save_model(
        teacher_key, layer_indices, target_hidden, target_inter,
        keep_ids, f"{teacher_key}_compressed",
        needs_hidden_reduction, use_pca=use_pca, use_activation=use_activation,
        corpus_texts=corpus_texts,
    )

    # ── 요약 ──
    print(f"\n{'='*60}")
    print(f"  COMPRESSION SUMMARY")
    print(f"{'='*60}")
    print(f"  Final: {final_path}")
    print(f"    {target_layers}L / {target_hidden}d / {target_inter}i / {final_vocab:,} vocab")
    print(f"    {est['total_params']:,} params, {est['fp32_mb']}MB FP32")
    print(f"{'─'*60}")
    print(f"\n  Next: python distill.py --teacher {teacher_key} "
          f"--student {teacher_key}_compressed")
    print(f"{'='*60}")

    return {
        "final_path": final_path,
        "intermediate_path": None,
        "needs_two_stage": False,
    }


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-teacher student model creation via layer pruning + vocab pruning"
    )
    parser.add_argument("--teacher", type=str, default=None,
                        choices=sorted(TEACHERS.keys()),
                        help=f"Teacher 모델 키 ({', '.join(sorted(TEACHERS.keys()))})")

    # 기존 모드
    parser.add_argument("--only", nargs="+", help="특정 실험만 실행")
    parser.add_argument("--no-prune", action="store_true",
                        help="Vocab pruning 생략")
    parser.add_argument("--max-vocab", type=int, default=None,
                        help="최대 vocab 크기 (빈도 기반 필터링)")

    # Compress 모드: 단일 최적 모델 생성
    parser.add_argument("--compress", action="store_true",
                        help="크기 제약 내 단일 최적 모델 생성 모드")
    parser.add_argument("--max-params", type=int, default=20_000_000,
                        help="최대 파라미터 수 (기본: 20M)")
    parser.add_argument("--max-mb", type=float, default=50.0,
                        help="최대 FP32 모델 크기 MB (기본: 50)")
    parser.add_argument("--min-layers", type=int, default=4,
                        help="최소 레이어 수 (기본: 4)")
    parser.add_argument("--vocab-percentile", type=float, default=0.95,
                        help="Vocab 누적 빈도 유지 비율 (기본: 0.95 = 95%%)")
    parser.add_argument("--min-vocab", type=int, default=None,
                        help="Vocab 최소 하한 (예: 90000). 95%% 기준보다 이 값이 "
                             "크면 vocab을 이 값까지 유지. 미지정 시 제한 없음.")
    parser.add_argument("--pca", action="store_true",
                        help="Hidden dim 축소 시 PCA 기반 projection 사용 "
                             "(코퍼스 hidden states에서 최적 방향 계산)")
    parser.add_argument("--activation", action="store_true",
                        help="Hidden dim 축소 시 활성화 기반 차원 선별 사용 "
                             "(코퍼스에서 중요도가 높은 차원만 유지)")
    parser.add_argument("--hidden-dim", type=int, default=None,
                        help="명시적 hidden dimension (예: 384). "
                             "지정 시 자동 최적화 대신 이 값을 사용")
    parser.add_argument("--num-layers", type=int, default=None,
                        help="명시적 레이어 수 (예: 6). "
                             "지정 시 자동 최적화 대신 이 값을 사용")

    args = parser.parse_args()

    # ── Compress 모드 ──
    if args.compress:
        if not args.teacher:
            parser.error("--compress requires --teacher")

        print("Loading distillation corpus for vocab pruning...")
        corpus_texts = load_distill_corpus()
        print()

        # --hidden-dim / --num-layers 명시 시: 자동 최적화 건너뛰고 직접 빌드
        if args.hidden_dim is not None or args.num_layers is not None:
            create_explicit_compressed_model(
                args.teacher,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                vocab_percentile=args.vocab_percentile,
                corpus_texts=corpus_texts,
                use_pca=args.pca,
                use_activation=args.activation,
                min_vocab=args.min_vocab,
            )
        else:
            create_compressed_model(
                args.teacher,
                max_params=args.max_params,
                max_fp32_mb=args.max_mb,
                min_layers=args.min_layers,
                vocab_percentile=args.vocab_percentile,
                corpus_texts=corpus_texts,
                use_pca=args.pca,
                use_activation=args.activation,
                min_vocab=args.min_vocab,
            )
        print("\nDone!")
        print("Next step: python distill.py --teacher <teacher_key> --student <name>_compressed")
        return

    # ── 기존 모드 (다중 실험) ──
    # Teacher 선택
    if args.teacher:
        teacher_keys = [args.teacher]
    else:
        teacher_keys = ["minilm"]

    # 코퍼스 한 번만 로드
    corpus_texts = None
    if not args.no_prune:
        print("Loading distillation corpus for vocab pruning...")
        corpus_texts = load_distill_corpus()
        print()

    for teacher_key in teacher_keys:
        t = TEACHERS[teacher_key]
        print(f"\n{'='*60}")
        print(f"Teacher: {t['model_id']}")
        print(f"  Layers: {t['num_layers']}, Hidden: {t['hidden_dim']}")
        print(f"  Tokenizer: {t['tokenizer_type']}, Vocab: {t['vocab_size']:,}")
        print(f"{'='*60}")

        # 실험 목록
        if teacher_key == "minilm" and not args.teacher:
            experiments = EXPERIMENTS
        elif teacher_key == "me5":
            experiments = generate_me5_experiments()
        else:
            experiments = generate_experiments(teacher_key)

        if args.only:
            experiments = [e for e in experiments if e["name"] in args.only]

        if not experiments:
            print(f"  No experiments to run for {teacher_key}")
            continue

        students_dir = get_teacher_students_dir(teacher_key)
        os.makedirs(students_dir, exist_ok=True)

        for exp in experiments:
            try:
                create_student_for_teacher(
                    teacher_key, exp,
                    max_vocab=args.max_vocab,
                    no_prune=args.no_prune,
                    corpus_texts=corpus_texts,
                )
            except Exception as e:
                print(f"  ERROR creating {exp['name']}: {e}")
                import traceback
                traceback.print_exc()

    print("\nDone! All student models created.")
    print("Next step: python run_mteb.py --teacher <teacher_key>")


if __name__ == "__main__":
    main()
