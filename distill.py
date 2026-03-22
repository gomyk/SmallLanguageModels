"""
Teacher → Student Knowledge Distillation

Teacher의 sentence embedding을 student가 재현하도록 학습한다.
MSE loss로 teacher/student의 출력 임베딩을 정렬(align)한다.

Strategy:
  1. Embedding-level KD: teacher와 student의 최종 sentence embedding을 MSE로 정렬
  2. (선택) Layer-level KD: teacher의 중간 레이어 출력을 student 레이어에 매칭

학습 데이터: HuggingFace의 다국어 문장 데이터셋 (tatoeba, wikimedia 등)

Usage:
    python distill.py --student L6_uniform
    python distill.py --student L6_uniform --epochs 5 --batch-size 64
    python distill.py --student L6_uniform L4_uniform L3_uniform  # 여러 모델
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

from config import EXPERIMENTS, STUDENTS_DIR, TEACHER_MODEL, TARGET_LANGUAGES


# ── 다국어 학습 데이터 ─────────────────────────────────────────

# ISO 639-1 → MTEB/tatoeba 언어 코드 매핑
LANG_CODES_FOR_DATA = {
    "ko": "kor", "en": "eng", "ja": "jpn", "zh": "cmn",
    "es": "spa", "fr": "fra", "de": "deu", "pt": "por",
    "it": "ita", "ru": "rus", "ar": "ara", "hi": "hin",
    "th": "tha", "vi": "vie", "id": "ind", "tr": "tur",
    "nl": "nld", "pl": "pol",
}


class TextDataset(Dataset):
    """단순 텍스트 리스트를 Dataset으로 래핑."""
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def load_multilingual_texts(max_per_lang=5000, cache_dir="data/distill_corpus"):
    """다국어 문장 데이터를 로드한다.

    tatoeba 데이터셋에서 타겟 언어별 문장을 수집한다.
    없는 언어는 wikimedia/wikipedia로 대체한다.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"texts_{max_per_lang}.txt")

    if os.path.exists(cache_file):
        print(f"Loading cached corpus from {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"  Loaded {len(texts):,} sentences")
        return texts

    print("Loading multilingual corpus...")
    all_texts = []

    # MASSIVE dataset: 언어별 서브셋으로 로드 (이미 MTEB에서 캐시됨)
    # MASSIVE config 이름은 단순 ISO 코드 (ko, en, ja 등)
    MASSIVE_LANG_MAP = {
        "ko": "ko", "en": "en", "ja": "ja", "zh": "zh-CN",
        "es": "es", "fr": "fr", "de": "de", "pt": "pt",
        "it": "it", "ru": "ru", "ar": "ar", "hi": "hi",
        "th": "th", "vi": "vi", "id": "id", "tr": "tr",
        "nl": "nl", "pl": "pl",
    }

    for lang, subset in MASSIVE_LANG_MAP.items():
        try:
            ds = load_dataset("mteb/amazon_massive_intent", subset, split="train")
            lang_texts = [row["text"] for row in ds if row.get("text") and len(row["text"]) > 5]
            lang_texts = lang_texts[:max_per_lang]
            all_texts.extend(lang_texts)
            print(f"    {lang} ({subset}): {len(lang_texts)} sentences")
        except Exception as e:
            print(f"    {lang}: failed - {e}")

    # Fallback: tatoeba parallel sentences
    if len(all_texts) < 1000:
        print("  Loading tatoeba fallback...")
        for config in ["en-ko", "en-ja", "en-de", "en-fr", "en-es", "en-zh", "en-ru", "en-ar"]:
            try:
                ds = load_dataset("sentence-transformers/parallel-sentences-tatoeba",
                                  config, split="train")
                for row in ds:
                    for key in ["sentence1", "sentence2", "source_sentence", "target_sentence"]:
                        text = row.get(key, "")
                        if text and len(text) > 5:
                            all_texts.append(text)
                    if len(all_texts) >= max_per_lang * len(TARGET_LANGUAGES):
                        break
            except Exception:
                pass

    # 최소한의 데이터 보장
    if len(all_texts) < 100:
        print("  WARNING: Very few training texts found. Using built-in samples.")
        samples = [
            # 최소한의 다국어 샘플
            "예약 좀 해줘", "지난번 주문 뭐였지?", "안녕하세요 반갑습니다",
            "오늘 날씨 어때?", "메뉴 추천해줘", "결제하고 싶어요",
            "Book a table for me", "What did I order last time?", "Hello how are you",
            "Cancel my reservation", "Show me my order history", "Thank you goodbye",
            "予約をお願いします", "前回の注文は何でしたか", "こんにちは元気ですか",
            "帮我预约一下", "上次我点了什么", "你好你好吗",
            "Reserva una mesa", "Qué pedí la última vez", "Hola cómo estás",
            "Réservez une table", "Qu'est-ce que j'ai commandé", "Bonjour comment allez-vous",
            "Reservieren Sie einen Tisch", "Was habe ich bestellt", "Hallo wie geht es",
            "Prenota un tavolo", "Cosa ho ordinato", "Ciao come stai",
            "Забронируйте столик", "Что я заказывал", "Привет как дела",
            "احجز لي طاولة", "ماذا طلبت", "مرحبا كيف حالك",
            "एक टेबल बुक करें", "मैंने क्या ऑर्डर किया", "नमस्ते कैसे हैं",
            "จองโต๊ะให้หน่อย", "สั่งอะไรไป", "สวัสดีครับ",
            "Đặt bàn cho tôi", "Tôi đã gọi gì", "Xin chào",
            "Pesan meja", "Apa yang saya pesan", "Halo apa kabar",
            "Bir masa ayırtın", "Ne sipariş etmiştim", "Merhaba nasılsınız",
            "Reserveer een tafel", "Wat had ik besteld", "Hallo hoe gaat het",
            "Zarezerwuj stolik", "Co zamówiłem", "Cześć jak się masz",
        ]
        all_texts.extend(samples * 20)  # 반복해서 최소 학습 가능하게

    # 캐시 저장
    with open(cache_file, "w", encoding="utf-8") as f:
        for text in all_texts:
            f.write(text.strip() + "\n")

    print(f"  Total: {len(all_texts):,} sentences")
    return all_texts


# ── Distillation Training ──────────────────────────────────────

def mean_pooling(model_output, attention_mask):
    """Mean pooling over token embeddings."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def distill_student(
    teacher_name,
    student_path,
    texts,
    epochs=3,
    batch_size=32,
    lr=2e-5,
    max_length=64,
    device="cpu",
    cos_weight=0.5,
    mse_weight=1.0,
    suffix="_distilled",
):
    """Teacher의 embedding을 student가 재현하도록 distillation 학습.

    Args:
        cos_weight: Cosine similarity loss 가중치 (높을수록 방향 정렬 중심)
        mse_weight: MSE loss 가중치 (높을수록 크기+방향 모두 강제)
        suffix: 저장 경로 접미사
    """

    print(f"\n{'='*60}")
    print(f"Distilling: {os.path.basename(student_path)}")
    print(f"  Teacher: {teacher_name}")
    print(f"  Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
    print(f"  Loss weights: MSE={mse_weight}, Cosine={cos_weight}")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    # Teacher 로드 (frozen)
    teacher = SentenceTransformer(teacher_name, device=device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # Student 로드
    student = SentenceTransformer(student_path, device=device)
    student.train()

    # Optimizer
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)

    # DataLoader
    dataset = TextDataset(texts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Cosine LR scheduler
    total_steps = len(dataloader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Training loop
    best_loss = float("inf")
    loss_fn = nn.MSELoss()

    # Tokenizer와 내부 모델에 직접 접근 (encode()는 gradient를 끊음)
    tokenizer = student.tokenizer
    student_transformer = student[0].auto_model  # HF transformer
    student_pooling = student[1]  # Pooling layer

    for epoch in range(epochs):
        epoch_loss = 0
        n_batches = 0
        student_transformer.train()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_texts in pbar:
            # Teacher forward (no grad, encode() OK)
            with torch.no_grad():
                teacher_embeddings = teacher.encode(
                    list(batch_texts),
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    device=device,
                ).clone()  # inference mode에서 벗어나기 위해 clone

            # Student forward (직접 tokenize → forward → pooling)
            encoded = tokenizer(
                list(batch_texts),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            model_output = student_transformer(**encoded)
            # Mean pooling
            token_emb = model_output[0]  # (batch, seq_len, hidden)
            mask = encoded["attention_mask"].unsqueeze(-1).expand(token_emb.size()).float()
            student_embeddings = torch.sum(token_emb * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

            # MSE loss between embeddings
            loss = loss_fn(student_embeddings, teacher_embeddings)

            # Cosine similarity loss (방향 정렬)
            cos_loss = 1 - F.cosine_similarity(student_embeddings, teacher_embeddings).mean()
            total_loss = mse_weight * loss + cos_weight * cos_loss

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += total_loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{total_loss.item():.4f}", "cos": f"{cos_loss.item():.4f}"})

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"  Epoch {epoch+1}: avg_loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            # 최적 모델을 별도 경로에 저장 (파일 잠금 방지)
            distilled_path = student_path + suffix
            os.makedirs(distilled_path, exist_ok=True)
            student_transformer.save_pretrained(distilled_path)
            tokenizer.save_pretrained(distilled_path)
            # sentence-transformers 포맷으로도 저장
            from sentence_transformers import models as st_models
            word_model = st_models.Transformer(distilled_path)
            pool_model = st_models.Pooling(word_model.get_word_embedding_dimension(),
                                           pooling_mode_mean_tokens=True)
            st_model = SentenceTransformer(modules=[word_model, pool_model])
            st_model.save(distilled_path)
            print(f"  Saved best model to {distilled_path} (loss={best_loss:.4f})")

    print(f"Distillation complete. Best loss: {best_loss:.4f}")
    return student


# ── Main ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", nargs="+", required=True,
                        help="Student 모델 이름 (e.g., L6_uniform)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--cos-weight", type=float, default=0.5,
                        help="Cosine loss 가중치")
    parser.add_argument("--mse-weight", type=float, default=1.0,
                        help="MSE loss 가중치")
    parser.add_argument("--suffix", default="_distilled",
                        help="저장 경로 접미사 (e.g., _distilled_v2)")
    parser.add_argument("--max-per-lang", type=int, default=5000,
                        help="언어당 최대 학습 문장 수")
    parser.add_argument("--device", default="cpu",
                        help="학습 장치 (cpu/cuda)")
    args = parser.parse_args()

    # GPU 자동 감지
    device = args.device
    if device == "cpu" and torch.cuda.is_available():
        device = "cuda"
        print(f"CUDA available, using GPU: {torch.cuda.get_device_name(0)}")

    # 학습 데이터 로드
    texts = load_multilingual_texts(max_per_lang=args.max_per_lang)

    # 각 student에 대해 distillation 실행
    for student_name in args.student:
        student_path = os.path.join(STUDENTS_DIR, student_name)
        if not os.path.exists(student_path):
            print(f"Student not found: {student_path}")
            continue

        start = time.time()
        distill_student(
            teacher_name=TEACHER_MODEL,
            student_path=student_path,
            texts=texts,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            cos_weight=args.cos_weight,
            mse_weight=args.mse_weight,
            suffix=args.suffix,
            device=device,
        )
        elapsed = time.time() - start
        print(f"  Time: {elapsed/60:.1f} min")

    print("\nAll distillation complete!")
    print("Next: python run_mteb.py  (re-evaluate distilled models)")


if __name__ == "__main__":
    main()
