# DATA_SOURCES — 데이터셋 출처 및 다운로드 가이드

> ## ⚠️ `training.csv`(개인 대화 데이터셋)는 git에 없음 — 반드시 따로 넣을 것
> **`data/training.csv`(1.8GB, 비공개)와 그 파생물 `data/distill_corpus/conversation_distill.txt`(2.3GB)는
> 저장소에 포함되어 있지 않다.** 공개 소스에서 다운로드할 수 없고, 원 소유자로부터 **직접 받아
> `data/`에 넣어야** 한다. 넣은 뒤 `python build_conversation_corpus.py`로 대화 코퍼스를 재생성하고
> md5(`7c45f097b12b9f5c69af3109f06b28f3`)로 검증한다. (아래 **B 섹션** 참고)

다른 환경/다른 에이전트가 **동일한 학습 데이터**를 확보하기 위한 출처·다운로드 가이드다.
학습 코퍼스는 두 갈래다:

| 갈래 | 파일 | 출처 | 확보 방법 |
|------|------|------|-----------|
| **A. 공개 MTEB 코퍼스** | `mteb_distill_10000.txt` (364,617줄) | HuggingFace Hub 공개 데이터셋 15종 | git에 포함 + 아래로 재다운로드 가능 |
| **B. 개인 대화 코퍼스** | `conversation_distill.txt` (19,520,517줄, 2.3GB) | 개인 데이터셋 `training.csv`에서 파생 (비공개) | 원본 `training.csv` 확보 후 스크립트로 재생성 |

정확한 재현/검증 md5는 [CORPUS.md](CORPUS.md#️-학습-데이터-정확-재현-중요) 참고.

---

## A. 공개 MTEB 데이터셋 (HuggingFace)

`config.py`의 `DISTILL_DATASETS`에 정의된 공개 데이터셋들이다. 전부 HuggingFace Hub에서
`datasets.load_dataset()`으로 내려받는다. `distill.py`의 `load_mteb_task_texts()`가 이들을
순회하며 텍스트를 뽑아 `mteb_distill_10000.txt`를 만든다.

### 데이터셋 목록 (hf_id / 서브셋 / split / 텍스트 필드)

| 이름 | HF id | 서브셋 | split | 텍스트 필드 |
|------|-------|--------|-------|-------------|
| amazon_counterfactual | `mteb/amazon_counterfactual` | — | train | text |
| banking77 | `mteb/banking77` | — | train | text |
| imdb | `mteb/imdb` | — | train | text |
| mtop_domain | `mteb/mtop_domain` | — | train | text |
| massive_intent | `mteb/amazon_massive_intent` | en, ko, ja, zh-CN, es, fr, de, pt, it, ru, ar, hi, th, vi, id, pl | train | text |
| massive_scenario | `mteb/amazon_massive_scenario` | (위와 동일 16개) | train | text |
| toxic_conversations | `mteb/toxic_conversations_50k` | — | train | text |
| tweet_sentiment | `mteb/tweet_sentiment_extraction` | — | train | text |
| stsb | `mteb/stsbenchmark-sts` | — | train | sentence1, sentence2 |
| snli | `stanfordnlp/snli` | — | train | premise, hypothesis |
| multi_nli | `nyu-mll/multi_nli` | — | train | premise, hypothesis |
| xnli | `facebook/xnli` | en, fr, es, de, ru, zh, ar, hi, vi, th | train | premise, hypothesis |
| sts17_cl | `mteb/sts17-crosslingual-sts` | ar-ar, ar-en, de-de, de-en, en-en, es-es, es-en, fr-en, it-en, ko-ko | test | sentence1, sentence2 |
| tatoeba | `mteb/tatoeba-bitext-mining` | kor/jpn/cmn/spa/fra/deu/por/ita/rus/ara/hin/tha/vie/ind/pol-eng | test | sentence1, sentence2 |
| miracl_corpus | `miracl/miracl-corpus` | ar, en, es, fa, fr, hi, id, ja, ko, ru, th, zh | train | title, text |

> `max_per_dataset`(기본 10000)로 데이터셋(서브셋)당 상한. 각 서브셋에서 위 필드 텍스트를
> 뽑아(길이 5 초과) 모으고 `set()`로 중복 제거해 캐시 파일에 저장한다.
> **서브셋/필드명은 config.py `DISTILL_DATASETS`가 단일 진실원본(source of truth)이다.**

### 다운로드 방법 1 — 한 번에 (권장)

`mteb_distill_10000.txt`가 이미 git에 포함돼 있으므로 **정확 재현이 목적이면 재다운로드 불필요**
(clone하면 딸려옴). 캐시가 없거나 내용을 새로 갱신할 때만 실행한다:

```bash
# 위 15개 데이터셋을 모두 내려받아 mteb_distill_10000.txt 생성 (공개 데이터만)
python -c "from distill import load_mteb_task_texts; load_mteb_task_texts(include_conversations=False)"
```
⚠️ 재생성본은 `set()` dedup 순서 차이로 기존과 **바이트 동일하지 않다**. 동일 학습이 목적이면
git에 포함된 파일을 그대로 쓴다 ([CORPUS.md](CORPUS.md#️-학습-데이터-정확-재현-중요)).

### 다운로드 방법 2 — 개별 데이터셋 확인

특정 데이터셋만 점검/디버깅할 때:
```python
from datasets import load_dataset

ds = load_dataset("mteb/banking77", split="train")                 # 서브셋 없는 것
ds = load_dataset("mteb/amazon_massive_intent", "ko", split="train")  # 서브셋 지정
ds = load_dataset("facebook/xnli", "de", split="train")
print(ds[0])
```

### HuggingFace 인증/캐시 참고
- 대부분 공개라 토큰 없이 받아진다. 일부는 접근 동의가 필요할 수 있다:
  - `stanfordnlp/snli`, `facebook/xnli`, `miracl/miracl-corpus` 등은 데이터셋 카드에서
    라이선스/동의 확인 후 `huggingface-cli login`(또는 `HF_TOKEN`)이 필요할 수 있음.
- 캐시 위치: `HF_HOME` / `HF_DATASETS_CACHE` 환경변수로 조정. 기본은 `~/.cache/huggingface`.
- 오프라인/사내망이면 미리 받아 캐시를 옮기거나 `HF_DATASETS_OFFLINE=1` 사용.
- 로딩이 실패하면 `load_mteb_task_texts`는 조용히 건너뛰고 로그에 `failed`로 남긴다
  (서브셋 이름/필드명이 데이터셋 카드와 맞는지 먼저 확인).

---

## B. 개인 대화 코퍼스 (conversation_distill.txt)

### 출처
- **공개 데이터가 아니다.** `data/training.csv`(1.8GB, 개인 대화 데이터셋)에서 파생됐다.
- `training.csv` 컬럼: `chat`, `labels`, `lang`.
  - `chat`: `"[{'Other': '...'}, {'Self': '...'}, ...]"` (대화 turn 리스트, 파이썬 리터럴 문자열)
  - `labels`: `"[0, 0, 0, 1]"` (turn별 라벨 — LoRA 분류 트랙에서 사용)
  - `lang`: 언어 코드 (de, ko, hi, ...)
- 이 CSV는 원 소유자(사용자)로부터 **직접 받아야** 한다. 공개 Hub에 없다.

### training.csv → conversation_distill.txt 재생성
추출 로직은 리포의 `build_conversation_corpus.py`에 재현돼 있다 (실제 파일 앞부분 40/40 일치 검증).
row 하나당: ① 각 turn 텍스트를 한 줄씩 ② 전체 turn을 ` [SEP] `로 이은 한 줄.

```bash
python build_conversation_corpus.py \
    --csv data/training.csv \
    --out data/distill_corpus/conversation_distill.txt
```

### 동일성 검증 (필수)
```bash
wc -l  data/distill_corpus/conversation_distill.txt   # 19520517 이어야 함
md5sum data/distill_corpus/conversation_distill.txt   # 7c45f097b12b9f5c69af3109f06b28f3
```
- md5가 맞으면 v4와 **완전히 동일한** 대화 코퍼스다.
- md5가 다르면 원본 `training.csv`가 다르거나(버전/인코딩) 줄바꿈·BOM 처리 차이다.
  스크립트는 `utf-8-sig`(BOM 제거) 입력, `\n` 출력으로 원본과 맞춰 두었다.

### 대안 — 대화 데이터 없이 진행
`training.csv`를 못 구하면 대화 코퍼스 없이 학습할 수 있다(결과는 v4와 달라짐):
```python
# run_me5s_v4_distill.py 등에서
texts = load_mteb_task_texts(include_conversations=False)   # mteb 360K만
```

---

## 최종 체크리스트 (다른 환경에서 데이터 맞추기)

- [ ] `git clone` → `data/distill_corpus/mteb_distill_10000.txt` 존재 확인 (md5 `7acf7640…`)
- [ ] (동일 학습 시) `training.csv` 확보 → `build_conversation_corpus.py` 실행
- [ ] `conversation_distill.txt` md5 `7c45f097…` 검증
- [ ] 공개 데이터 접근 동의가 필요한 경우 `huggingface-cli login`
- [ ] `run_me5s_v4_distill.py` 실행 시 총 코퍼스가 **19,885,134줄(≈19.88M)** 로 로드되는지 로그 확인
