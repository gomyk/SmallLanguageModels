# SmallModel - Jina v5 nano Compression Pipeline

Jina v5 nano (`jinaai/jina-embeddings-v5-text-nano`, 768d/12L/128K vocab) sentence embedding 모델을
layer pruning + PCA hidden dim reduction + BPE vocab pruning + knowledge distillation으로 압축한다.

> ## ⚠️ 학습 데이터는 별도로 넣어야 함 (training.csv)
>
> **개인 대화 데이터셋 `data/training.csv`(1.8GB)와 이를 가공한 `data/distill_corpus/conversation_distill.txt`(2.3GB)는
> git에 포함되어 있지 않다.** (GitHub 용량 한도 초과 + 개인 데이터)
>
> 다른 환경에서 동일하게 학습하려면 **`training.csv`를 직접 받아 `data/`에 넣은 뒤**
> `python build_conversation_corpus.py`로 대화 코퍼스를 재생성해야 한다.
> 공개 MTEB 코퍼스(`mteb_distill_10000.txt`)는 git에 포함되어 clone 시 함께 온다.
> → 상세: **[docs/DATA_SOURCES.md](docs/DATA_SOURCES.md)**, [docs/HANDOFF.md](docs/HANDOFF.md)

## Quick Start

```bash
# 기본: 6L/256d/~42K vocab, 전체 파이프라인
python run_jina_v5_h256.py

# 커스텀 설정
python run_jina_v5_h256.py --hidden-dim 384 --num-layers 6 --target-vocab 30000
python run_jina_v5_h256.py --hidden-dim 256 --num-layers 4 --target-vocab 20000
```

## Pipeline

```
Teacher (768d/12L/128K) → Layer Pruning → PCA Hidden Dim → Vocab Pruning → Distillation → MTEB Eval
```

### 1. Create: 모델 압축

```bash
# 기본 (6L/256d, corpus 전체 vocab ~42K)
python run_jina_v5_h256.py --skip-distill --skip-eval-before --skip-eval-after --skip-teacher-eval

# target vocab 지정 (BPE merge backtracking 자동 적용)
python run_jina_v5_h256.py --target-vocab 20000 --skip-distill --skip-eval-before --skip-eval-after
```

- **Layer pruning**: 12L → 지정 레이어 수 (균등 간격 선택)
- **PCA hidden dim**: 768d → 지정 차원 (코퍼스 hidden state 기반 SVD)
- **Vocab pruning**: 128K → target vocab (BPE merge rule 역추적으로 subword 보존)

### 2. Distill: Knowledge Distillation

```bash
# 모델 생성 후 distillation만
python run_jina_v5_h256.py --skip-create --skip-teacher-eval --skip-eval-before --skip-eval-after

# VRAM 제한
python run_jina_v5_h256.py --skip-create --max-vram-gb 16

# tqdm 끄기 (로그 리다이렉트 시 필수)
TQDM_DISABLE=1 python run_jina_v5_h256.py --skip-create
```

- MSE + Cosine Similarity loss
- Projection layer (student_dim → teacher_dim) 자동 생성, `proj.pt`로 저장/복원
- Early stopping (patience=3)

### 3. Eval: MTEB Benchmark

```bash
# distilled 모델만 평가
python run_jina_v5_h256.py --skip-create --skip-distill --skip-teacher-eval --skip-eval-before

# Classification + STS만 (Clustering 제외, 빠르게)
python run_jina_v5_h256.py --task-groups Classification STS
```

25개 MTEB 태스크: Classification(8) + STS(9) + Clustering(8)

### 4. Upload: HuggingFace Hub

```bash
python run_jina_v5_h256.py --skip-create --skip-distill --upload --repo-prefix gomyk/jina-v5
```

## BPE Merge Backtracking

BPE vocab pruning 시 단순히 빈도 기반으로 토큰을 제거하면 토크나이저가 일부 단어를
조립할 수 없어 byte-level fallback으로 과도하게 분해된다.

이 파이프라인은 유지할 토큰의 merge rule을 역추적하여 필요한 중간 subword를 자동으로 보존한다:

```
코퍼스 토큰 36,402개
  + 특수 토큰 256개
  + byte 토큰 94개
  + merge 역추적 +5,275개  ← 중간 subword
  = 총 41,959개 (실제 vocab)
```

`--target-vocab 30000` 지정 시 빈도 상위 30K 토큰을 선택한 뒤 merge backtracking을 적용하므로
실제 vocab은 30K보다 클 수 있다.

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--hidden-dim` | 256 | Target hidden dimension |
| `--num-layers` | 6 | Target layer count |
| `--target-vocab` | None | Target vocab size (None = corpus 전체) |
| `--max-epochs` | 20 | Max distillation epochs |
| `--patience` | 3 | Early stopping patience |
| `--batch-size` | 32 | Distillation batch size |
| `--lr` | 2e-5 | Learning rate |
| `--max-vram-gb` | None | GPU VRAM limit (GB) |
| `--task-groups` | Cls STS Clust | MTEB task groups |
| `--skip-create` | - | Skip model creation |
| `--skip-teacher-eval` | - | Skip teacher evaluation |
| `--skip-eval-before` | - | Skip pre-distillation eval |
| `--skip-distill` | - | Skip distillation |
| `--skip-eval-after` | - | Skip post-distillation eval |
| `--upload` | - | Upload to HuggingFace |
| `--repo-prefix` | - | HF repo prefix |

## File Structure

```
arch_utils.py         # Layer/vocab/hidden pruning, BPE merge backtracking
config.py             # Teacher model configs, MTEB tasks, size estimation
create_students.py    # Student model creation (multi-teacher)
distill.py            # Knowledge distillation (projection persistence)
run_jina_v5_h256.py   # Jina v5 nano end-to-end pipeline
run_mteb.py           # MTEB evaluation runner
compare_results.py    # Results comparison
upload_to_hub.py      # HuggingFace upload with model card
```

## 16-Language Corpus

Distillation/vocab pruning 코퍼스 (101K sentences):

| Source | Languages | Count |
|--------|-----------|-------|
| MASSIVE (amazon_massive_intent) | ko, en, ja, zh, es, fr, de, pt, it, ru, ar, hi, th, vi, id, pl | ~5K each |
| STSBenchmark | en | ~11.5K |
| Banking77 | en | ~10K |
