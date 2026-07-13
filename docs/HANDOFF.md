# SmallLanguageModels — 인수인계 문서 (HANDOFF)

> 이 문서는 다른 환경(다른 PC/서버)에서 distillation 작업을 **그대로 이어받아** 진행하기 위한
> 최상위 가이드다. 지금까지의 작업 히스토리, 현재 상태, 그리고 "언어 추가 / base 모델 추가"를
> 어떻게 이어가는지를 정리한다.

- **작성 기준일:** 2026-07-13
- **원본 작업 디렉토리:** `C:/Users/Moon/finetuning-workshop/SmallModel`
- **GitHub:** https://github.com/gomyk/SmallLanguageModels.git

> ## ⚠️ 반드시: `training.csv`는 git에 없음 — 따로 넣어야 함
> 개인 대화 데이터셋 **`data/training.csv`**(1.8GB) 및 파생물 **`conversation_distill.txt`**(2.3GB)는
> git에 포함되지 않는다. 동일 학습을 하려면 이 파일을 **직접 받아 `data/`에 넣고**
> `python build_conversation_corpus.py`로 대화 코퍼스를 재생성해야 한다. → [DATA_SOURCES.md](DATA_SOURCES.md)

---

## 1. 이 프로젝트가 하는 일

큰 multilingual sentence-embedding teacher 모델을 **경량 student 모델로 압축**한다.
압축은 4단계 파이프라인으로 이루어진다:

```
Teacher (예: 768d/12L/250K vocab)
   │
   ├─ 1. Layer Pruning       레이어 수 축소 (균등 간격 선택, 예: 12L → 6L)
   ├─ 2. PCA Hidden Reduction hidden dim 축소 (코퍼스 hidden state SVD, 예: 768d → 384d)
   ├─ 3. Vocab Pruning        코퍼스에 등장하지 않는 토큰 제거 (BPE는 merge 역추적으로 subword 보존)
   └─ 4. Knowledge Distillation  teacher embedding을 student가 재현하도록 학습 (MSE + Cosine)
   │
   ▼
Student (예: 384d/6L/축소 vocab) → MTEB 25~26개 태스크로 평가
```

핵심 목표는 **다국어 품질을 유지하면서 크기/속도를 줄이는 것**이다. Android on-device
추론까지 염두에 두고 tokenizer를 Kotlin으로 포팅해 검증까지 해 두었다 (`android/`).

## 2. 지금 어디까지 되어 있나 (현재 상태)

### 지원되는 Teacher 모델 (`config.py`의 `TEACHERS`)
아래 9개 teacher가 config에 등록되어 있고, 파이프라인이 대응한다. 상세는
[TEACHERS.md](TEACHERS.md) 참고.

| key | 모델 | dim/layers/vocab | tokenizer | 상태 |
|-----|------|------------------|-----------|------|
| `minilm` | paraphrase-multilingual-MiniLM-L12-v2 | 384/12/250K | unigram | ✅ 완료 (초기 실험 다수) |
| `me5s` | multilingual-e5-small | 384/12/250K | unigram | ✅ **가장 최근 주력 (v4 완료)** |
| `me5` | multilingual-e5-base | 768/12/250K | unigram | ✅ hidden 실험 (h256/384/512) |
| `gte` | gte-multilingual-base | 768/12/250K | unigram | ✅ 완료 (trust_remote_code) |
| `modernbert` | ModernBERT-base | 768/22/50K | bpe | ✅ 완료 |
| `jina_v5` | jina-embeddings-v5-text-nano | 768/12/128K | bpe | ✅ 완료 (PEFT 이슈 해결) |
| `qwen3` | Qwen3-0.6B | 1024/28/152K | bpe | ⚠️ 압축본 존재, 실험 부분적 |
| `gemma_emb` | embeddinggemma-300m | 768/24/262K | unigram | ⚠️ config 등록, 실험 초기 |
| `mmbert` | mmBERT-small | 384/22/256K | bpe | ⚠️ config 등록, 실험 초기 |

### 언어 (현재 16개)
`config.py`의 `TARGET_LANGUAGES` — ko, en, ja, zh, es, fr, de, pt, it, ru, ar, hi, th, vi, id, pl.
언어 추가 방법은 [CORPUS.md](CORPUS.md) 참고.

### 최근 주력 실험 결과 요약
- **me5s_compressed_v4_distilled** (multilingual-e5-small 기반, 384d/12L, teacher-anchor 임베딩 초기화 + 19.88M 코퍼스 20ep distill)
  - STSBenchmark **0.771**, MTEB 25개 태스크 평가 완료. 모델 크기 ~166MB(FP32, distilled 디렉토리 전체).
- **jina-v5-nano** 압축 (6L/384d/13K vocab, 19.3M params, ~12x 압축): STSBenchmark distill 후 **0.682** (3ep).
- 자세한 수치/히스토리는 [RESULTS.md](RESULTS.md).

## 3. 다른 환경에서 이어받는 절차 (요약)

1. **클론 + 환경 세팅** → [SETUP.md](SETUP.md)
   ```bash
   git clone https://github.com/gomyk/SmallLanguageModels.git
   cd SmallLanguageModels
   pip install -r requirements.txt
   ```
2. **학습 데이터 맞추기 (동일 재현 시 필수)** → [CORPUS.md](CORPUS.md#️-학습-데이터-정확-재현-중요)
   - `mteb_distill_10000.txt`(364,617줄, 40MB)는 **git에 포함**되어 있어 clone하면 동일하게 딸려온다.
     (dedup이 `set()` 기반이라 재생성하면 순서가 달라져 학습이 달라진다 → 그래서 파일을 커밋해 둠.)
   - ⚠️ `conversation_distill.txt`(19,520,517줄, 2.3GB, md5 `7c45f097…`)는 GitHub 용량 한도 초과라
     git에 없다. **외장/클라우드/scp로 직접 복사** 후 `md5sum`으로 동일성 검증한다.
   - 두 파일이 다 있으면 v4와 **동일한 19.88M 코퍼스**로 재학습된다. 없으면
     `include_conversations=False`(mteb 360K만)로도 학습 가능하나 결과가 달라진다.
3. **작업 이어가기**
   - 언어 추가: [CORPUS.md](CORPUS.md#언어-추가하기)
   - base 모델(teacher) 추가: [TEACHERS.md](TEACHERS.md#새-teacher-모델-추가하기)

## 4. 저장소에 무엇이 있고 무엇이 없나

`.gitignore`로 **제외**되는 것 (용량/재생성 가능):
- 모델 가중치: `*.safetensors`, `*.bin`, `*.pt`, `*.pth` → student/teacher 가중치는 push되지 않음
- 대용량 코퍼스: `data/distill_corpus/*.txt` (conversation 2.3GB 포함)
- 학습 로그: `*.log` (일부 40MB), 시각화 `*.png`, `*.stackdump`

**포함**되는 것:
- 모든 코드 (`*.py`), config, 문서(`docs/`), MTEB 결과 JSON(`results/**`, 작은 파일),
  student 디렉토리의 메타(`config.json`, `tokenizer.json`, `README.md` 등 — 가중치 제외)

→ 즉, 다른 환경에서는 **코드+설정+결과기록**을 받고, **모델과 코퍼스는 재생성/재학습**하는 구조다.

## 5. 문서 인덱스
- [SETUP.md](SETUP.md) — 환경 세팅, 의존성, GPU/VRAM 참고
- [DATA_SOURCES.md](DATA_SOURCES.md) — **데이터셋 출처 + 다운로드 방법** (공개 MTEB 15종 + 개인 대화 코퍼스)
- [PIPELINE.md](PIPELINE.md) — 4단계 파이프라인 상세 + 스크립트 레퍼런스
- [TEACHERS.md](TEACHERS.md) — teacher별 설정/주의사항 + **새 모델 추가 방법**
- [CORPUS.md](CORPUS.md) — 코퍼스 구성/언어 + **새 언어 추가 방법**
- [RESULTS.md](RESULTS.md) — 실험 결과 및 히스토리
- 루트 [README.md](../README.md) — jina-v5-nano 파이프라인 CLI 사용법

## 6. 알아두면 좋은 함정 (Gotchas) — 빠른 참조
- **heavy ML 프로세스는 기본적으로 순차 실행.** 메모리 여유가 있을 때만 병렬. (모델 로딩/MTEB/학습)
- **Vocab pruning은 "코퍼스에 없는 토큰만" 제거.** 크기 타겟(예: 50MB)에 맞춰 고빈도 토큰을 자르지 말 것.
- **GTE / mmBERT는 `trust_remote_code=True`.** Windows에서 `HF_HUB_TRUST_REMOTE_CODE=1` 필요.
- **jina_v5는 PeftMixedModel** → AutoModel 로드 후 base EuroBertModel 추출 필요. `model_kwargs={"default_task":"text-matching","attn_implementation":"eager"}` 필수.
- **로그 리다이렉트 시 `TQDM_DISABLE=1`** (진행바가 로그를 오염시킴).
- **distill.py는 1000 step마다 체크포인트 저장 + 자동 resume** 지원. crash 재시작은 `distill_resilient.py`.
