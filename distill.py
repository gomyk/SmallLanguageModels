"""
Multi-Teacher Knowledge Distillation

Teacher의 sentence embedding을 student가 재현하도록 학습한다.
MSE + Cosine Similarity loss로 teacher/student의 출력 임베딩을 정렬한다.

학습 데이터: MTEB Classification/Clustering/STS 태스크 데이터셋

Usage:
    # 새 teacher의 student distillation
    python distill.py --teacher modernbert --student modernbert_L6_uniform
    python distill.py --teacher gte --student gte_L4_uniform gte_L6_uniform

    # 기존 MiniLM (하위 호환)
    python distill.py --student L6_uniform

    # 커스텀 설정
    python distill.py --teacher modernbert --student modernbert_L6_uniform --epochs 5 --batch-size 64
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from tqdm import tqdm

from config import (
    TEACHERS, EXPERIMENTS, STUDENTS_DIR, TEACHER_MODEL, TARGET_LANGUAGES,
    DISTILL_DATASETS, MTEB_TASK_GROUPS,
    generate_experiments, get_teacher_students_dir,
    _estimate_for_teacher,
)


# ── AutoModel Teacher Wrapper ────────────────────────────────

class AutoModelTeacher:
    """AutoModel 기반 teacher wrapper (SentenceTransformer 비호환 모델용).

    jina-v5 등 PEFT/custom 모델은 SentenceTransformer로 직접 로드가 어려울 수 있다.
    이 경우 AutoModel로 로드하고 모델 내장 encode()를 사용한다.
    """

    def __init__(self, model, device="cpu"):
        self.model = model
        self._device = device
        self._dim = getattr(model.config, 'hidden_size', 768)

    def encode(self, texts, convert_to_tensor=True, show_progress_bar=False,
               device=None, **kwargs):
        target_device = device or self._device
        with torch.no_grad():
            emb = self.model.encode(texts)
        if isinstance(emb, np.ndarray):
            emb = torch.from_numpy(emb)
        if not isinstance(emb, torch.Tensor):
            emb = torch.tensor(emb)
        if convert_to_tensor:
            return emb.to(target_device)
        return emb

    def eval(self):
        self.model.eval()
        return self

    def parameters(self):
        return self.model.parameters()

    def get_sentence_embedding_dimension(self):
        return self._dim


def load_teacher(teacher_name, device="cpu", trust_remote_code=False,
                 model_kwargs=None):
    """Teacher 모델을 로드한다. SentenceTransformer 우선, AutoModel fallback."""
    # 1차: SentenceTransformer로 로드 시도
    try:
        teacher = SentenceTransformer(teacher_name, device=device,
                                       trust_remote_code=trust_remote_code,
                                       model_kwargs=model_kwargs or {})
        teacher.eval()
        print(f"  Teacher loaded as SentenceTransformer")
        return teacher
    except Exception as e:
        print(f"  SentenceTransformer load failed: {e}")

    # 2차: AutoModel로 로드 (PEFT/custom 모델용)
    print(f"  Falling back to AutoModel (custom model)...")
    from transformers import AutoModel
    model = AutoModel.from_pretrained(teacher_name, trust_remote_code=True)
    model.to(device)
    model.eval()
    teacher = AutoModelTeacher(model, device=device)
    dim = teacher.get_sentence_embedding_dimension()
    print(f"  Teacher loaded as AutoModel (dim={dim})")
    return teacher


# ── 학습 데이터 ───────────────────────────────────────────────

class TextDataset(Dataset):
    """단순 텍스트 리스트를 Dataset으로 래핑."""
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]




def load_conversation_texts(cache_dir="data/distill_corpus"):
    """대화 데이터에서 추출한 텍스트를 로드한다 (개별 발화 + 전체 대화)."""
    conv_file = os.path.join(cache_dir, "conversation_distill.txt")
    if not os.path.exists(conv_file):
        return []
    print(f"Loading conversation distillation corpus from {conv_file}")
    with open(conv_file, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    print(f"  Loaded {len(texts):,} conversation texts")
    return texts


def load_mteb_task_texts(max_per_dataset=10000, cache_dir="data/distill_corpus",
                         include_conversations=True):
    """MTEB Classification/Clustering/STS 태스크 데이터셋에서 텍스트를 수집한다.

    include_conversations=True이면 대화 데이터도 함께 로드한다.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"mteb_distill_{max_per_dataset}.txt")

    if os.path.exists(cache_file):
        print(f"Loading cached distillation corpus from {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"  Loaded {len(texts):,} sentences")
        if include_conversations:
            conv_texts = load_conversation_texts(cache_dir)
            texts.extend(conv_texts)
            print(f"  Total distillation corpus: {len(texts):,} texts")
        return texts

    print("Loading MTEB task datasets for distillation...")
    all_texts = []

    for ds_name, ds_config in DISTILL_DATASETS.items():
        hf_id = ds_config["hf_id"]
        text_fields = ds_config["text_fields"]
        splits = ds_config["splits"]
        subsets = ds_config.get("subsets", [None])

        for subset in subsets:
            for split in splits:
                try:
                    if subset:
                        ds = load_dataset(hf_id, subset, split=split)
                    else:
                        ds = load_dataset(hf_id, split=split)

                    count = 0
                    for row in ds:
                        for field in text_fields:
                            text = row.get(field, "")
                            if text and len(str(text)) > 5:
                                all_texts.append(str(text))
                                count += 1
                                if count >= max_per_dataset:
                                    break
                        if count >= max_per_dataset:
                            break

                    subset_label = f"/{subset}" if subset else ""
                    print(f"    {ds_name}{subset_label}/{split}: {count} texts")
                except Exception as e:
                    subset_label = f"/{subset}" if subset else ""
                    print(f"    {ds_name}{subset_label}/{split}: failed - {e}")

    # 중복 제거
    unique_texts = list(set(all_texts))
    print(f"  Total unique texts: {len(unique_texts):,} (from {len(all_texts):,})")

    # 캐시 저장
    with open(cache_file, "w", encoding="utf-8") as f:
        for text in unique_texts:
            f.write(text.strip() + "\n")

    if include_conversations:
        conv_texts = load_conversation_texts(cache_dir)
        unique_texts.extend(conv_texts)
        print(f"  Total distillation corpus: {len(unique_texts):,} texts")

    return unique_texts


def load_multilingual_texts(max_per_lang=5000, cache_dir="data/distill_corpus"):
    """하위 호환: 기존 다국어 문장 데이터를 로드한다."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"texts_{max_per_lang}.txt")

    if os.path.exists(cache_file):
        print(f"Loading cached corpus from {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"  Loaded {len(texts):,} sentences")
        return texts

    # MTEB 태스크 데이터로 대체
    return load_mteb_task_texts(max_per_dataset=max_per_lang, cache_dir=cache_dir)


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
    trust_remote_code=False,
    patience=3,
):
    """Teacher의 embedding을 student가 재현하도록 distillation 학습.

    Early stopping: patience epoch 동안 loss가 개선되지 않으면 조기 종료.
    """

    print(f"\n{'='*60}")
    print(f"Distilling: {os.path.basename(student_path)}")
    print(f"  Teacher: {teacher_name}")
    print(f"  Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
    print(f"  Loss weights: MSE={mse_weight}, Cosine={cos_weight}")
    print(f"  Early stopping: patience={patience}")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    # Teacher 로드 (frozen)
    # TEACHERS config에서 model_kwargs 가져오기 (task 지정 등)
    _teacher_kwargs = {}
    for _tk, _tv in TEACHERS.items():
        if _tv["model_id"] == teacher_name:
            _teacher_kwargs = _tv.get("model_kwargs", {})
            break
    teacher = load_teacher(teacher_name, device=device,
                           trust_remote_code=trust_remote_code,
                           model_kwargs=_teacher_kwargs)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # ── 체크포인트 경로 ──
    ckpt_dir = student_path + suffix
    ckpt_state_path = os.path.join(ckpt_dir, "training_state.pt")

    # Student 로드 (체크포인트가 있으면 이어서 학습)
    _student_kwargs = {"attn_implementation": "eager"}
    if os.path.exists(ckpt_dir) and os.path.exists(os.path.join(ckpt_dir, "config.json")):
        print(f"  Resuming model from checkpoint: {ckpt_dir}")
        student = SentenceTransformer(ckpt_dir, device=device,
                                       trust_remote_code=True,
                                       model_kwargs=_student_kwargs)
    else:
        student = SentenceTransformer(student_path, device=device,
                                       trust_remote_code=True,
                                       model_kwargs=_student_kwargs)
    student.train()

    tokenizer = student.tokenizer
    student_transformer = student[0].auto_model
    student_pooling = student[1]

    # Teacher/Student 차원이 다를 때 학습 가능한 projection layer 추가
    teacher_dim = teacher.get_sentence_embedding_dimension()
    student_dim = student.get_sentence_embedding_dimension()
    if teacher_dim is None:
        with torch.no_grad():
            sample = teacher.encode(["test"], convert_to_tensor=True, device=device)
            teacher_dim = sample.shape[-1]
            print(f"  Teacher dim inferred from encode: {teacher_dim}")
    if student_dim is None:
        with torch.no_grad():
            sample = student.encode(["test"], convert_to_tensor=True, device=device)
            student_dim = sample.shape[-1]
            print(f"  Student dim inferred from encode: {student_dim}")
    proj = None
    if student_dim != teacher_dim:
        proj = nn.Linear(student_dim, teacher_dim).to(device)
        proj_path = os.path.join(ckpt_dir, "proj.pt")
        if os.path.exists(proj_path):
            proj.load_state_dict(torch.load(proj_path, map_location=device, weights_only=True))
            print(f"  Projection: {student_dim}d → {teacher_dim}d (resumed)")
        else:
            print(f"  Projection: {student_dim}d → {teacher_dim}d (new)")

    # Optimizer
    all_params = list(student.parameters()) + (list(proj.parameters()) if proj else [])
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=0.01)

    # DataLoader (고정 seed로 shuffle 순서 재현)
    dataset = TextDataset(texts)
    g = torch.Generator()
    g.manual_seed(42)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            drop_last=True, generator=g)

    # Cosine LR scheduler
    total_steps = len(dataloader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Training state
    best_loss = float("inf")
    no_improve_count = 0
    start_epoch = 0
    start_step = 0
    global_step = 0
    loss_fn = nn.MSELoss()

    # ── 체크포인트에서 학습 상태 복원 ──
    if os.path.exists(ckpt_state_path):
        print(f"  Restoring training state from {ckpt_state_path}")
        state = torch.load(ckpt_state_path, map_location=device, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        best_loss = state["best_loss"]
        no_improve_count = state["no_improve_count"]
        start_epoch = state["epoch"]
        start_step = state["step"]
        global_step = state["global_step"]
        rng = state["rng_state"].cpu()
        g.set_state(rng)
        print(f"  Resumed: epoch={start_epoch+1}, step={start_step}, "
              f"global_step={global_step}, best_loss={best_loss:.4f}")

    def save_checkpoint(epoch, step, global_step, epoch_loss, n_batches):
        """모델 + 전체 학습 상태를 체크포인트로 저장."""
        import gc
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        os.makedirs(ckpt_dir, exist_ok=True)
        student_transformer.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        if proj:
            torch.save(proj.state_dict(), os.path.join(ckpt_dir, "proj.pt"))
        torch.save({
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_loss": best_loss,
            "no_improve_count": no_improve_count,
            "epoch": epoch,
            "step": step,
            "global_step": global_step,
            "rng_state": g.get_state(),
        }, ckpt_state_path)
        avg = epoch_loss / max(n_batches, 1)
        print(f"\n  Checkpoint @ epoch {epoch+1} step {step} "
              f"(global {global_step}): avg_loss={avg:.4f}", flush=True)

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0
        n_batches = 0
        student_transformer.train()
        if proj:
            proj.train()

        skip_steps = start_step if epoch == start_epoch else 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}",
                    disable=os.environ.get("TQDM_DISABLE", "") == "1")
        for step_in_epoch, batch_texts in enumerate(pbar):
            # 이전 체크포인트까지의 step은 건너뛰기
            if step_in_epoch < skip_steps:
                if step_in_epoch % 10000 == 0 and step_in_epoch > 0:
                    pbar.set_postfix({"skipping": f"{step_in_epoch}/{skip_steps}"})
                continue

            try:
                truncated_texts = [t[:max_length * 6] for t in list(batch_texts)]

                with torch.no_grad():
                    teacher_embeddings = teacher.encode(
                        truncated_texts,
                        convert_to_tensor=True,
                        show_progress_bar=False,
                        device=device,
                    ).clone()

                encoded = tokenizer(
                    truncated_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(device)
            except BaseException:
                global_step += 1
                continue

            model_output = student_transformer(**encoded)
            token_emb = model_output[0]
            mask = encoded["attention_mask"].unsqueeze(-1).expand(token_emb.size()).float()
            student_embeddings = torch.sum(token_emb * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

            projected = proj(student_embeddings) if proj else student_embeddings

            loss = loss_fn(projected, teacher_embeddings)
            cos_loss = 1 - F.cosine_similarity(projected, teacher_embeddings).mean()
            total_loss = mse_weight * loss + cos_weight * cos_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += total_loss.item()
            n_batches += 1
            global_step += 1
            pbar.set_postfix({"loss": f"{total_loss.item():.4f}", "cos": f"{cos_loss.item():.4f}"})

            # 주기적 체크포인트 (1000 step마다)
            if n_batches % 1000 == 0:
                save_checkpoint(epoch, step_in_epoch + 1, global_step,
                                epoch_loss, n_batches)

        # epoch 종료 후 체크포인트 저장
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"  Epoch {epoch+1}: avg_loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve_count = 0
            save_checkpoint(epoch + 1, 0, global_step, epoch_loss, n_batches)
            # sentence-transformers 포맷으로도 저장
            from sentence_transformers import models as st_models
            word_model = st_models.Transformer(
                ckpt_dir,
                config_args={"trust_remote_code": True},
                model_args={"trust_remote_code": True},
                tokenizer_args={"trust_remote_code": True},
            )
            pool_model = st_models.Pooling(word_model.get_word_embedding_dimension(),
                                           pooling_mode_mean_tokens=True)
            st_model = SentenceTransformer(modules=[word_model, pool_model])
            st_model.save(ckpt_dir)
            if proj:
                torch.save(proj.state_dict(), os.path.join(ckpt_dir, "proj.pt"))
            print(f"  Saved best model (loss={best_loss:.4f})")
        else:
            no_improve_count += 1
            print(f"  No improvement ({no_improve_count}/{patience})")
            save_checkpoint(epoch + 1, 0, global_step, epoch_loss, n_batches)
            if no_improve_count >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print(f"Distillation complete. Best loss: {best_loss:.4f} "
          f"(stopped at epoch {epoch+1}/{epochs})")
    return student


# ── Two-Stage Distillation ─────────────────────────────────────

def distill_two_stage(
    teacher_key,
    student_name,
    texts,
    epochs=3,
    batch_size=32,
    lr=2e-5,
    cos_weight=0.5,
    mse_weight=1.0,
    device="cpu",
    patience=3,
):
    """2단계 증류: Teacher → Intermediate → Final Student.

    Teacher/Student 파라미터 비율이 10x를 초과할 때 사용한다.
    중간 모델({teacher_key}_intermediate)을 경유하여 지식을 단계적으로 전달한다.

    Stage 1: Teacher → Intermediate (suffix: _distilled)
    Stage 2: Intermediate_distilled → Final Student (suffix: _distilled)
    """
    t = TEACHERS[teacher_key]
    teacher_name = t["model_id"]
    students_dir = get_teacher_students_dir(teacher_key)

    # 중간 모델 경로
    intermediate_name = f"{teacher_key}_intermediate"
    intermediate_path = os.path.join(students_dir, intermediate_name)
    if not os.path.exists(intermediate_path):
        print(f"ERROR: Intermediate model not found: {intermediate_path}")
        print(f"  Run: python create_students.py --teacher {teacher_key} --compress")
        return

    # 최종 모델 경로
    student_path = os.path.join(students_dir, student_name)
    if not os.path.exists(student_path):
        student_path = os.path.join(STUDENTS_DIR, student_name)
    if not os.path.exists(student_path):
        print(f"ERROR: Student model not found: {student_path}")
        return

    print(f"\n{'#'*60}")
    print(f"  2-STAGE DISTILLATION")
    print(f"  Teacher: {teacher_name}")
    print(f"  Intermediate: {intermediate_name}")
    print(f"  Final Student: {student_name}")
    print(f"{'#'*60}")

    # ── Stage 1: Teacher → Intermediate ──
    print(f"\n{'='*60}")
    print(f"  STAGE 1/2: Teacher → Intermediate")
    print(f"{'='*60}")
    start = time.time()
    distill_student(
        teacher_name=teacher_name,
        student_path=intermediate_path,
        texts=texts,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        cos_weight=cos_weight,
        mse_weight=mse_weight,
        suffix="_distilled",
        device=device,
        trust_remote_code=t["trust_remote_code"],
        patience=patience,
    )
    stage1_time = time.time() - start
    print(f"  Stage 1 time: {stage1_time/60:.1f} min")

    # ── Stage 2: Intermediate_distilled → Final Student ──
    intermediate_distilled = intermediate_path + "_distilled"
    if not os.path.exists(intermediate_distilled):
        print(f"ERROR: Stage 1 output not found: {intermediate_distilled}")
        return

    print(f"\n{'='*60}")
    print(f"  STAGE 2/2: Intermediate → Final Student")
    print(f"{'='*60}")
    start = time.time()
    distill_student(
        teacher_name=intermediate_distilled,
        student_path=student_path,
        texts=texts,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        cos_weight=cos_weight,
        mse_weight=mse_weight,
        suffix="_distilled",
        device=device,
        trust_remote_code=True,
        patience=patience,
    )
    stage2_time = time.time() - start
    print(f"  Stage 2 time: {stage2_time/60:.1f} min")

    total_time = stage1_time + stage2_time
    print(f"\n{'#'*60}")
    print(f"  2-Stage distillation complete!")
    print(f"  Total time: {total_time/60:.1f} min")
    print(f"  Final model: {student_path}_distilled")
    print(f"{'#'*60}")


# ── Main ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=str, default=None,
                        choices=sorted(TEACHERS.keys()),
                        help=f"Teacher 모델 키 ({', '.join(sorted(TEACHERS.keys()))})")
    parser.add_argument("--student", nargs="+", required=True,
                        help="Student 모델 이름 (e.g., modernbert_L6_uniform)")
    parser.add_argument("--two-stage", action="store_true",
                        help="2단계 증류: Teacher → Intermediate → Student "
                             "(teacher 대비 10x+ 압축 시 사용)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="최대 epoch 수 (기본: 10, early stopping과 함께 사용)")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience (기본: 3)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--cos-weight", type=float, default=0.5)
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--suffix", default="_distilled")
    parser.add_argument("--max-per-dataset", type=int, default=999999999,
                        help="데이터셋당 최대 텍스트 수 (기본: 전체)")
    parser.add_argument("--no-conversations", action="store_true",
                        help="대화 데이터 제외 (MTEB 데이터만 사용)")
    parser.add_argument("--student-hf", type=str, default=None,
                        help="HuggingFace에서 student 모델 로드 (e.g., gomyk/jina-v5-h256-distilled)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-vram-frac", type=float, default=None,
                        help="GPU VRAM 사용 비율 제한 (0.0~1.0, 예: 0.5 = 50%%)")
    args = parser.parse_args()

    # Teacher 결정
    teacher_key = args.teacher or "minilm"
    t = TEACHERS[teacher_key]
    teacher_name = t["model_id"]
    trust_remote_code = t["trust_remote_code"]

    # GPU 자동 감지
    device = args.device
    if device == "cpu" and torch.cuda.is_available():
        device = "cuda"
        # VRAM 사용량 제한 (기본: 전체의 50%)
        if args.max_vram_frac:
            torch.cuda.set_per_process_memory_fraction(args.max_vram_frac)
            print(f"CUDA VRAM limit: {args.max_vram_frac*100:.0f}%")
        print(f"CUDA available, using GPU: {torch.cuda.get_device_name(0)}")

    # 학습 데이터 로드 (MTEB 태스크 + 대화 데이터)
    include_conv = not args.no_conversations
    texts = load_mteb_task_texts(
        max_per_dataset=args.max_per_dataset,
        include_conversations=include_conv,
    )

    # Student 디렉토리
    students_dir = get_teacher_students_dir(teacher_key)

    # 각 student에 대해 distillation 실행
    for student_name in args.student:

        # ── 2단계 증류 모드 (명시적 또는 자동 감지) ──
        use_two_stage = args.two_stage
        if not use_two_stage:
            # 자동 감지: intermediate 모델이 존재하면 2-stage 제안
            intermediate_path = os.path.join(
                students_dir, f"{teacher_key}_intermediate"
            )
            if os.path.exists(intermediate_path):
                # Teacher/Student 비율 추정
                teacher_layers = list(range(t["num_layers"]))
                teacher_est = _estimate_for_teacher(teacher_key, teacher_layers)
                # Student 크기를 모델 파일에서 추정
                for fname in ["model.safetensors", "pytorch_model.bin"]:
                    sp = os.path.join(students_dir, student_name, fname)
                    if not os.path.exists(sp):
                        sp = os.path.join(students_dir, student_name,
                                          "0_Transformer", fname)
                    if os.path.exists(sp):
                        student_file_mb = os.path.getsize(sp) / (1024 ** 2)
                        student_est_params = int(student_file_mb * 1024 * 1024 / 4)
                        ratio = teacher_est["total_params"] / max(student_est_params, 1)
                        if ratio > 10:
                            print(f"Auto-detected: Teacher/Student ratio ~{ratio:.0f}x "
                                  f"(>10x), enabling 2-stage distillation")
                            use_two_stage = True
                        break

        start = time.time()

        if use_two_stage:
            distill_two_stage(
                teacher_key=teacher_key,
                student_name=student_name,
                texts=texts,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                cos_weight=args.cos_weight,
                mse_weight=args.mse_weight,
                device=device,
                patience=args.patience,
            )
        else:
            # HF 모델을 student로 사용하는 경우: 로컬에 다운로드
            if args.student_hf:
                from huggingface_hub import snapshot_download
                hf_local = os.path.join(students_dir, student_name)
                if not os.path.exists(hf_local):
                    print(f"Downloading {args.student_hf} → {hf_local}")
                    snapshot_download(args.student_hf, local_dir=hf_local)
                student_path = hf_local
            else:
                student_path = os.path.join(students_dir, student_name)
            if not os.path.exists(student_path):
                student_path = os.path.join(STUDENTS_DIR, student_name)
            if not os.path.exists(student_path):
                print(f"Student not found: {student_path}")
                continue

            distill_student(
                teacher_name=teacher_name,
                student_path=student_path,
                texts=texts,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                cos_weight=args.cos_weight,
                mse_weight=args.mse_weight,
                suffix=args.suffix,
                device=device,
                trust_remote_code=trust_remote_code,
                patience=args.patience,
            )

        elapsed = time.time() - start
        print(f"  Total time: {elapsed/60:.1f} min")

    print("\nAll distillation complete!")
    print("Next: python run_mteb.py --teacher <teacher_key>")


if __name__ == "__main__":
    main()
