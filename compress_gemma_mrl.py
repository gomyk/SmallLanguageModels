"""
Gemma Embedding 300M → 128d MRL Compression Pipeline

1. Layer pruning: 24L → 6L
2. Language-based vocab pruning: 262K → ~65K (16개 언어 코퍼스 기반)
3. MRL (Matryoshka) distillation: 20 epochs, dims=[128, 256, 384, 768]
4. 물리적 128d slice: 학습 완료 후 모델을 128d로 잘라냄

Usage:
    python compress_gemma_mrl.py
    python compress_gemma_mrl.py --epochs 20 --batch-size 64
    python compress_gemma_mrl.py --skip-create   # student가 이미 있으면 distill만
    python compress_gemma_mrl.py --skip-distill   # student 생성만
"""

import argparse
import copy
import json
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import (
    TEACHERS, STUDENTS_DIR, TARGET_LANGUAGES,
    make_uniform_indices, _estimate_for_teacher,
    get_teacher_students_dir,
)
from arch_utils import (
    create_pruned_student,
    prune_tokenizer_and_embeddings,
    save_as_sentence_transformer,
    detect_tokenizer_type,
    reduce_hidden_dim,
)


TEACHER_KEY = "gemma_emb"
TARGET_LAYERS = 6
TARGET_HIDDEN = 128
MRL_DIMS = [128, 256, 384, 768]

# 16개 대상 언어 (MASSIVE dataset locale codes)
TARGET_LANG_LOCALES = {
    "ko": "Korean", "en": "English", "ja": "Japanese", "zh-CN": "Chinese",
    "hi": "Hindi", "pl": "Polish", "pt": "Portuguese", "de": "German",
    "it": "Italian", "th": "Thai", "vi": "Vietnamese", "fr": "French",
    "es": "Spanish", "ar": "Arabic", "id": "Indonesian", "ru": "Russian",
}


# ── Language-based Vocab Pruning ─────────────────────────────

def collect_language_tokens(tokenizer, max_per_lang=10000):
    """16개 대상 언어 코퍼스에서 등장하는 토큰만 수집한다.

    기존 코퍼스 빈도 기반과 달리, 언어별 데이터셋을 사용하여
    대상 언어에서 실제 사용되는 토큰만 유지한다.

    Returns:
        유지할 토큰 ID 리스트 (정렬됨)
    """
    from datasets import load_dataset

    keep_ids = set(tokenizer.all_special_ids)

    # 기본 문자/구두점
    basic_chars = list("0123456789.,!?;:'\"-()[]{}/@#$%^&*+=<>~_ \t\n")
    for ch in basic_chars:
        ids = tokenizer.encode(ch, add_special_tokens=False)
        keep_ids.update(ids)

    # BPE byte-level fallback 토큰 유지
    tok_type = detect_tokenizer_type(tokenizer)
    if tok_type == "BPE":
        tok_json = json.loads(tokenizer.backend_tokenizer.to_str())
        vocab = tok_json["model"]["vocab"]
        for token, tid in vocab.items():
            if tid < 256 or len(token) <= 1:
                keep_ids.add(tid)

    print(f"  Base keep (special + basic + byte): {len(keep_ids):,}")

    # MASSIVE dataset에서 16개 언어 토큰 수집
    print(f"  Loading MASSIVE dataset for {len(TARGET_LANG_LOCALES)} target languages...")
    for locale, lang_name in TARGET_LANG_LOCALES.items():
        try:
            ds = load_dataset("mteb/amazon_massive_intent", locale, split="train")
            texts = [row["text"] for row in ds if row.get("text")][:max_per_lang]

            lang_ids = set()
            batch_size = 500
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                encoded = tokenizer(batch, add_special_tokens=False,
                                    truncation=True, max_length=128)
                for ids in encoded["input_ids"]:
                    lang_ids.update(ids)

            before = len(keep_ids)
            keep_ids.update(lang_ids)
            new_tokens = len(keep_ids) - before
            print(f"    {lang_name:15s} ({locale:5s}): {len(lang_ids):>6,} "
                  f"unique (+{new_tokens:,} new), total: {len(keep_ids):>6,}")
        except Exception as e:
            print(f"    {lang_name:15s} ({locale:5s}): FAILED - {e}")

    # STS benchmark (영어 sentence pairs - 평가 데이터 커버리지)
    print("  Loading STS benchmark...")
    try:
        ds = load_dataset("mteb/stsbenchmark-sts", split="train")
        texts = []
        for row in ds:
            for f in ["sentence1", "sentence2"]:
                if row.get(f):
                    texts.append(row[f])
        encoded = tokenizer(texts[:20000], add_special_tokens=False,
                            truncation=True, max_length=128)
        for ids in encoded["input_ids"]:
            keep_ids.update(ids)
        print(f"    STS: total {len(keep_ids):,}")
    except Exception as e:
        print(f"    STS: FAILED - {e}")

    original = getattr(tokenizer, "vocab_size", len(tokenizer.get_vocab()))
    print(f"\n  Language-based vocab pruning: {original:,} → {len(keep_ids):,} "
          f"({(original - len(keep_ids))/original*100:.1f}% removed)")

    return sorted(keep_ids)


# ── Student Creation ─────────────────────────────────────────

def create_gemma_student(use_pca=False, corpus_texts=None):
    """Layer pruning + language-based vocab pruning으로 student 생성.

    hidden_dim은 768 유지 (MRL distillation 후 128d로 slice).
    """
    t = TEACHERS[TEACHER_KEY]
    layer_indices = make_uniform_indices(t["num_layers"], TARGET_LAYERS)
    students_dir = get_teacher_students_dir(TEACHER_KEY)
    save_name = f"{TEACHER_KEY}_mrl"
    save_path = os.path.join(students_dir, save_name)

    print(f"\n{'='*60}")
    print(f"Creating Gemma MRL Student")
    print(f"  Teacher: {t['model_id']}")
    print(f"  Layers: {t['num_layers']} → {TARGET_LAYERS} (indices: {layer_indices})")
    print(f"  Hidden: 768 (MRL → {TARGET_HIDDEN}d after training)")
    print(f"{'='*60}")

    # Step 1: Layer pruning
    print("\n[Step 1] Layer pruning...")
    student_hf, tokenizer = create_pruned_student(
        t["model_id"], layer_indices,
        layer_accessor=t["layer_accessor"],
        trust_remote_code=t["trust_remote_code"],
    )
    print(f"  {t['num_layers']} → {len(layer_indices)} layers")

    # Step 2: Language-based vocab pruning
    print("\n[Step 2] Language-based vocab pruning...")
    keep_ids = collect_language_tokens(tokenizer)

    hf_tmp = os.path.join(save_path, "_hf_pruned")
    student_hf = prune_tokenizer_and_embeddings(
        student_hf, tokenizer, keep_ids, hf_tmp
    )
    print(f"  Vocab: {t['vocab_size']:,} → {student_hf.config.vocab_size:,}")

    # Pruned tokenizer 재로드
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        hf_tmp, trust_remote_code=t["trust_remote_code"]
    )

    # SentenceTransformer로 저장
    save_as_sentence_transformer(student_hf, tokenizer, save_path)
    shutil.rmtree(hf_tmp, ignore_errors=True)

    # Sanity check
    from sentence_transformers import SentenceTransformer
    try:
        st = SentenceTransformer(save_path, trust_remote_code=True)
        test_sents = ["Hello world", "안녕하세요", "こんにちは", "Bonjour"]
        embs = st.encode(test_sents)
        print(f"  Sanity check: output shape = {embs.shape}")
        del st
    except Exception as e:
        print(f"  Sanity check failed: {e}")

    # 실제 파일 크기
    for fname in ["model.safetensors", "pytorch_model.bin"]:
        for prefix in ["", "0_Transformer/"]:
            fpath = os.path.join(save_path, prefix, fname)
            if os.path.exists(fpath):
                size_mb = os.path.getsize(fpath) / (1024 ** 2)
                print(f"  {fname}: {size_mb:.1f}MB")
                break

    print(f"  Saved to {save_path}")
    return save_path


# ── MRL Distillation ─────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx]


def load_distill_texts(max_per_dataset=10000):
    """Distillation 학습용 텍스트 로드 (기존 캐시 활용)."""
    cache_dir = "data/distill_corpus"
    cache_file = os.path.join(cache_dir, f"mteb_distill_{max_per_dataset}.txt")

    if os.path.exists(cache_file):
        print(f"  Loading cached corpus: {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"  Loaded {len(texts):,} sentences")
        return texts

    # 캐시 없으면 distill.py의 로더 사용
    from distill import load_mteb_task_texts
    return load_mteb_task_texts(max_per_dataset=max_per_dataset)


def distill_mrl(
    student_path,
    epochs=20,
    batch_size=32,
    lr=2e-5,
    max_length=64,
    device="cpu",
    patience=5,
    mrl_dims=None,
):
    """MRL (Matryoshka Representation Learning) Distillation.

    Teacher의 embedding을 student가 재현하되, 여러 차원 레벨에서
    동시에 loss를 계산하여 첫 N차원에 정보가 집중되도록 학습한다.

    Args:
        student_path: student 모델 경로
        epochs: 학습 epoch 수
        mrl_dims: 마트료시카 차원 리스트 [128, 256, 384, 768]
    """
    from sentence_transformers import SentenceTransformer

    if mrl_dims is None:
        mrl_dims = MRL_DIMS

    t = TEACHERS[TEACHER_KEY]
    teacher_name = t["model_id"]

    print(f"\n{'='*60}")
    print(f"MRL Distillation")
    print(f"  Teacher: {teacher_name}")
    print(f"  Student: {os.path.basename(student_path)}")
    print(f"  MRL dims: {mrl_dims}")
    print(f"  Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
    print(f"  Patience: {patience}")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    # Teacher 로드 (frozen)
    teacher = SentenceTransformer(teacher_name, device=device,
                                   trust_remote_code=t["trust_remote_code"])
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    teacher_dim = teacher.get_sentence_embedding_dimension()
    print(f"  Teacher dim: {teacher_dim}")

    # Student 로드
    student = SentenceTransformer(student_path, device=device,
                                   trust_remote_code=True)
    student.train()
    student_dim = student.get_sentence_embedding_dimension()
    print(f"  Student dim: {student_dim}")

    # Teacher와 student dim이 다를 경우 대비 projection
    # (vocab pruning 후에도 hidden_dim=768 유지이므로 보통 같음)
    proj = None
    if student_dim != teacher_dim:
        proj = nn.Linear(student_dim, teacher_dim).to(device)
        print(f"  Projection: {student_dim}d → {teacher_dim}d")

    # MRL dim별 projection (teacher_dim != mrl_dim인 경우)
    # teacher가 768d이고 mrl_dims도 768까지이므로 truncation으로 충분
    valid_mrl_dims = [d for d in mrl_dims if d <= min(teacher_dim, student_dim)]
    print(f"  Valid MRL dims: {valid_mrl_dims}")

    # Optimizer
    params = list(student.parameters())
    if proj:
        params += list(proj.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

    # 학습 데이터
    texts = load_distill_texts()
    dataset = TextDataset(texts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            drop_last=True)

    total_steps = len(dataloader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps
    )

    # Training loop
    tokenizer = student.tokenizer
    student_transformer = student[0].auto_model
    best_loss = float("inf")
    no_improve_count = 0
    distilled_path = student_path + "_distilled"

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_mrl_losses = {d: 0.0 for d in valid_mrl_dims}
        n_batches = 0
        student_transformer.train()
        if proj:
            proj.train()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_texts in pbar:
            try:
                # Teacher forward
                with torch.no_grad():
                    teacher_embs = teacher.encode(
                        list(batch_texts),
                        convert_to_tensor=True,
                        show_progress_bar=False,
                        device=device,
                    ).clone()

                # Student forward
                encoded = tokenizer(
                    list(batch_texts),
                    padding=True, truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(device)
            except Exception:
                continue

            model_output = student_transformer(**encoded)
            token_emb = model_output[0]
            mask = encoded["attention_mask"].unsqueeze(-1).expand(token_emb.size()).float()
            student_embs = torch.sum(token_emb * mask, 1) / torch.clamp(
                mask.sum(1), min=1e-9
            )

            if proj:
                student_embs = proj(student_embs)

            # MRL Loss: 각 차원 레벨에서 MSE + Cosine loss 계산
            total_loss = torch.tensor(0.0, device=device)
            n_dims = len(valid_mrl_dims)

            for dim in valid_mrl_dims:
                s_trunc = student_embs[:, :dim]
                t_trunc = teacher_embs[:, :dim]

                # L2 normalize
                s_norm = F.normalize(s_trunc, p=2, dim=1)
                t_norm = F.normalize(t_trunc, p=2, dim=1)

                mse_loss = F.mse_loss(s_norm, t_norm)
                cos_loss = 1.0 - F.cosine_similarity(s_trunc, t_trunc).mean()

                dim_loss = mse_loss + 0.5 * cos_loss
                total_loss = total_loss + dim_loss / n_dims
                epoch_mrl_losses[dim] += dim_loss.item()

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += total_loss.item()
            n_batches += 1

            # 가장 작은 dim (128) loss를 표시
            smallest_dim = valid_mrl_dims[0]
            pbar.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                f"d{smallest_dim}": f"{epoch_mrl_losses[smallest_dim]/max(n_batches,1):.4f}",
            })

        avg_loss = epoch_loss / max(n_batches, 1)
        dim_report = ", ".join(
            f"d{d}={epoch_mrl_losses[d]/max(n_batches,1):.4f}"
            for d in valid_mrl_dims
        )
        print(f"  Epoch {epoch+1}: avg_loss={avg_loss:.4f} [{dim_report}]")

        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve_count = 0
            # 최적 모델 저장
            os.makedirs(distilled_path, exist_ok=True)
            student_transformer.save_pretrained(distilled_path)
            tokenizer.save_pretrained(distilled_path)
            # SentenceTransformer 포맷으로 저장
            from sentence_transformers import models as st_models
            word_model = st_models.Transformer(
                distilled_path,
                config_args={"trust_remote_code": True},
                model_args={"trust_remote_code": True},
                tokenizer_args={"trust_remote_code": True},
            )
            pool_model = st_models.Pooling(
                word_model.get_word_embedding_dimension(),
                pooling_mode_mean_tokens=True,
            )
            st_model = SentenceTransformer(modules=[word_model, pool_model])
            st_model.save(distilled_path)
            print(f"  Saved best model (loss={best_loss:.4f})")
        else:
            no_improve_count += 1
            print(f"  No improvement ({no_improve_count}/{patience})")
            if no_improve_count >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print(f"\nMRL Distillation complete. Best loss: {best_loss:.4f}")
    return distilled_path


# ── Physical Dimension Slice ─────────────────────────────────

def slice_to_target_dim(model_path, target_dim=TARGET_HIDDEN):
    """MRL 학습 완료된 모델을 물리적으로 target_dim으로 잘라낸다.

    첫 target_dim 차원만 유지하여 모델 크기를 실제로 줄인다.
    """
    from transformers import AutoModel, AutoTokenizer

    print(f"\n{'='*60}")
    print(f"Physical dimension slice: 768 → {target_dim}")
    print(f"{'='*60}")

    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    sliced_model = reduce_hidden_dim(
        model, target_dim,
        trust_remote_code=True,
    )

    # 저장
    output_path = model_path + f"_{target_dim}d"
    os.makedirs(output_path, exist_ok=True)
    save_as_sentence_transformer(sliced_model, tokenizer, output_path)

    # Sanity check
    from sentence_transformers import SentenceTransformer
    try:
        st = SentenceTransformer(output_path, trust_remote_code=True)
        test_sents = ["Hello world", "안녕하세요", "こんにちは", "Bonjour"]
        embs = st.encode(test_sents)
        print(f"  Output shape: {embs.shape}")
        del st
    except Exception as e:
        print(f"  Sanity check failed: {e}")

    # 파일 크기
    for fname in ["model.safetensors", "pytorch_model.bin"]:
        for prefix in ["", "0_Transformer/"]:
            fpath = os.path.join(output_path, prefix, fname)
            if os.path.exists(fpath):
                size_mb = os.path.getsize(fpath) / (1024 ** 2)
                print(f"  {fname}: {size_mb:.1f}MB")
                break

    print(f"  Saved to {output_path}")
    return output_path


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Gemma Embedding MRL Compression Pipeline"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--target-dim", type=int, default=TARGET_HIDDEN,
                        help=f"최종 embedding 차원 (기본: {TARGET_HIDDEN})")
    parser.add_argument("--skip-create", action="store_true",
                        help="Student 생성 건너뛰기 (이미 존재할 때)")
    parser.add_argument("--skip-distill", action="store_true",
                        help="Distillation 건너뛰기 (student 생성만)")
    parser.add_argument("--skip-slice", action="store_true",
                        help="Dimension slice 건너뛰기")
    args = parser.parse_args()

    # GPU 자동 감지
    device = args.device
    if device == "cpu" and torch.cuda.is_available():
        device = "cuda"
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")

    students_dir = get_teacher_students_dir(TEACHER_KEY)
    student_path = os.path.join(students_dir, f"{TEACHER_KEY}_mrl")

    total_start = time.time()

    # ── Phase 1: Student 생성 (L6 + vocab pruning) ──
    if not args.skip_create:
        print("\n" + "#" * 60)
        print("  PHASE 1: Create Student (L6 + Language Vocab Pruning)")
        print("#" * 60)
        student_path = create_gemma_student()
    else:
        print(f"\nSkipping student creation, using: {student_path}")

    # ── Phase 2: MRL Distillation ──
    distilled_path = student_path + "_distilled"
    if not args.skip_distill:
        print("\n" + "#" * 60)
        print("  PHASE 2: MRL Distillation")
        print("#" * 60)
        distilled_path = distill_mrl(
            student_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            patience=args.patience,
        )
    else:
        print(f"\nSkipping distillation, using: {distilled_path}")

    # ── Phase 3: Physical 128d Slice ──
    if not args.skip_slice:
        print("\n" + "#" * 60)
        print(f"  PHASE 3: Physical Slice → {args.target_dim}d")
        print("#" * 60)
        final_path = slice_to_target_dim(distilled_path, args.target_dim)
    else:
        final_path = distilled_path
        print(f"\nSkipping slice, final model: {final_path}")

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Total time: {total_time/60:.1f} min")
    print(f"  Student (768d): {student_path}")
    print(f"  Distilled (768d): {distilled_path}")
    print(f"  Final ({args.target_dim}d): {final_path}")
    print(f"{'='*60}")
    print(f"\nNext: python run_mteb.py (evaluate)")


if __name__ == "__main__":
    main()
