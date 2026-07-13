# PIPELINE — 압축 파이프라인 상세 + 스크립트 레퍼런스

## 전체 흐름

```
create_students.py  →  distill.py  →  run_mteb.py  →  compare_results.py  →  upload_to_hub.py
  (압축: 3단계)         (distill)      (평가)          (비교/요약)            (HF 업로드)
```

teacher 하나에 대해 위 순서로 돌린다. 모든 스크립트가 공통으로 `--teacher <key>`를 받는다
(`key`는 `config.py`의 `TEACHERS` 딕셔너리 키).

---

## 1단계 — 압축 (`create_students.py`)

Teacher를 layer/hidden/vocab 축소해 student를 만든다. 핵심 로직은 `arch_utils.py`.

```bash
# teacher별 기본 실험(L6/L4 uniform) 생성
python create_students.py --teacher modernbert
python create_students.py --teacher gte

# 특정 실험만
python create_students.py --teacher modernbert --only modernbert_L6_uniform

# vocab pruning 없이
python create_students.py --teacher gte --no-prune

# 크기 예산 기반 자동 압축 (layer/hidden/vocab joint 최적화)
python create_students.py --teacher qwen3 --compress --max-mb 50 --min-layers 4
```

주요 옵션:
| 옵션 | 의미 |
|------|------|
| `--teacher` | teacher key (필수, `--compress`엔 특히) |
| `--only` | 특정 실험명만 실행 |
| `--no-prune` | vocab pruning 스킵 |
| `--max-vocab` | vocab 상한 (기본 None = 코퍼스 전체 토큰) |
| `--compress` | 크기 예산 기반 joint 최적화 모드 |
| `--max-mb` / `--max-params` | 크기/파라미터 예산 |
| `--min-layers` | 최소 레이어 수 |
| `--pca` / `--activation` | hidden dim 축소 방식 |
| `--hidden-dim` / `--num-layers` | 직접 지정 |

산출물: `students/<teacher_key>/<name>_compressed/` (config.json, tokenizer, model.safetensors 등).

> **Vocab pruning 원칙:** 코퍼스에 등장하지 않는 토큰만 제거한다. 크기 타겟에 맞춰 고빈도
> 토큰을 자르지 않는다. BPE는 merge rule을 역추적해 필요한 중간 subword를 보존한다
> (README의 "BPE Merge Backtracking" 참고).

## 2단계 — Distillation (`distill.py`)

Teacher의 문장 임베딩을 student가 재현하도록 학습. **MSE(1.0) + Cosine(0.5)** loss,
early stopping(patience 기본 3), 차원이 다르면 학습 가능한 projection layer(student→teacher)
자동 생성 후 `proj.pt`로 저장/복원.

```bash
python distill.py --teacher me5s --student me5s_compressed
```

- 코퍼스는 `load_mteb_task_texts(include_conversations=...)`로 로드
  (`data/distill_corpus/mteb_distill_10000.txt` + 선택적 `conversation_distill.txt`).
- **1000 step마다 체크포인트 저장**, 재시작 시 자동 resume (config.json 존재하면 이어서).
- DataLoader는 seed 42 고정 → shuffle 순서 재현 가능.
- crash가 잦으면 `distill_resilient.py` 래퍼가 자동 재시작.

주요 파라미터(함수 `distill_student`): `epochs`, `batch_size`(기본 32, 대규모는 512),
`lr`(2e-5), `max_length`(64), `cos_weight`(0.5), `mse_weight`(1.0), `patience`(3).

산출물: `students/<teacher_key>/<name>_compressed_distilled/`.

## 3단계 — MTEB 평가 (`run_mteb.py`)

```bash
python run_mteb.py --teacher me5s
python run_mteb.py --teacher gte --task-groups Classification STS
python run_mteb.py --teacher modernbert --include-teacher      # teacher baseline 포함
python run_mteb.py --teacher me5s --max-vram-frac 0.5
```

- 태스크 정의: `config.py`의 `MTEB_TASK_GROUPS` — Classification(8) + Clustering(8) + STS(9~10).
- 결과: `results/<teacher_key>/<model_name>/.../<Task>.json` + `evaluation_summary.json`.
- `--exclude-tasks`로 오래 걸리는 태스크(예: Massive) 제외 가능.

## 4단계 — 비교/요약 (`compare_results.py`)

```bash
python compare_results.py --teacher me5s
```

teacher별 결과 JSON을 모아 표로 비교한다 (compressed vs distilled vs teacher baseline).

## 5단계 — 업로드 (`upload_to_hub.py`)

```bash
HF_TOKEN=... python upload_to_hub.py --teacher me5s --repo-prefix gomyk/...
```

model card 자동 생성, license notice(gemma/jina 등) 반영.

---

## 특수 실험 스크립트 (teacher별 end-to-end 러너)

| 스크립트 | 용도 |
|----------|------|
| `run_jina_v5_h256.py` | jina-v5-nano end-to-end (압축+distill+eval+upload), README에 CLI 상세 |
| `run_me5_hidden_exp.py` | mE5-base hidden dim 실험 (h256/h384/h512) |
| `run_me5s_v3_distill.py` / `run_me5s_v4_distill.py` | me5s v3/v4 대규모 distill 러너 |
| `prepare_me5s_v3.py` / `prepare_me5s_v4.py` | me5s student vocab 준비 (v4는 teacher-anchor 임베딩 초기화) |
| `run_v4_vs_baseline_1ep.py` | v4 vs baseline 1 epoch 비교 |
| `compress_gemma_mrl.py` | embeddinggemma MRL 압축 |
| `run_pipeline.py` / `_run_all_pipeline.py` | 전체 파이프라인 배치 러너 |
| `benchmark.py` | 추론 속도/크기 벤치마크 |
| `prune_and_export.py` | ONNX export |
| `umap_viz_me5s*.py` | 임베딩 UMAP 시각화 |

## LoRA 파이프라인 (별도 트랙)

압축과 별개로, teacher/student에 LoRA를 붙여 태스크별 분류/STS를 실험한 트랙이 있다:
`lora_classification.py`, `lora_clf_universal.py`, `lora_clf_with_personal.py`, `lora_sts.py`,
`run_lora_mteb.py`, `run_mteb_merged.py`, `upload_lora.py`. (jina_v5 / me5s에 적용됨)

## Android on-device tokenizer (`android/`)

student tokenizer를 Kotlin으로 포팅해 검증. `BpeTokenizer.kt`(BPE), `UnigramTokenizer.kt`(Unigram/SentencePiece),
각 테스트(`*Test.kt`), `RunTokenizerTests.kt`. golden ID는 `golden_bpe.json` / `golden_unigram.json`
(생성기 `generate_golden_ids.py`), Python 측 검증은 `test_tokenizers.py` (16개 언어, 227 TC).
