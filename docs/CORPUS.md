# CORPUS — distillation 코퍼스 + 언어 추가

## 현재 언어 (16개)

`config.py`의 `TARGET_LANGUAGES`:
```
ko, en, ja, zh, es, fr, de, pt, it, ru, ar, hi, th, vi, id, pl
```
ISO2 → ISO3 매핑은 `LANG_TO_ISO3` (평가 시 언어 필터에 사용).

## 코퍼스 구성

distillation/vocab pruning에 쓰이는 텍스트는 `config.py`의 `DISTILL_DATASETS`에 정의된
HuggingFace 데이터셋들에서 추출한다 (`distill.py`의 `load_mteb_task_texts`).

| 소스 | HF id | 언어 | 비고 |
|------|-------|------|------|
| MASSIVE intent | mteb/amazon_massive_intent | 16개 서브셋 | 태스크 train만 |
| MASSIVE scenario | mteb/amazon_massive_scenario | 16개 서브셋 | |
| Banking77 | mteb/banking77 | en | |
| Amazon counterfactual | mteb/amazon_counterfactual | en | |
| IMDB | mteb/imdb | en | |
| MTOP domain | mteb/mtop_domain | en | |
| Toxic conversations | mteb/toxic_conversations_50k | en | |
| Tweet sentiment | mteb/tweet_sentiment_extraction | en | |
| STSBenchmark | mteb/stsbenchmark-sts | en | |
| SNLI | stanfordnlp/snli | en | 대규모 NLI |
| MultiNLI | nyu-mll/multi_nli | en | |
| XNLI | facebook/xnli | 10개 서브셋 | 다국어 NLI |
| STS17 crosslingual | mteb/sts17-crosslingual-sts | 10쌍 | 병렬/정렬쌍 (임베딩 정렬용) |
| Tatoeba bitext | mteb/tatoeba-bitext-mining | 15쌍 | 병렬 문장 |
| MIRACL corpus | miracl/miracl-corpus | 12개 서브셋 | 대규모 다국어 retrieval passage |

- `max_per_dataset`(기본 10000)로 데이터셋(서브셋)당 상한. 중복 제거 후 캐시.
- 캐시 파일: `data/distill_corpus/mteb_distill_10000.txt` (한 번 만들면 재사용).

### conversation 코퍼스 (별도 대용량 자산)
- `data/distill_corpus/conversation_distill.txt` — 2.3GB, 약 19.5M 문장(개별 발화 + 전체 대화).
- `load_conversation_texts()`가 로드, `include_conversations=True`면 위 코퍼스에 합쳐짐.
- **git에 포함되지 않는다.** 없으면 `include_conversations=False`로 학습하거나, 자체 대화
  데이터로 이 파일을 재생성한다(한 줄에 한 텍스트).
- me5s v4는 (mteb 360K + conversation 19.5M) = **19.88M** 코퍼스로 학습했다.

## ⚠️ 학습 데이터 정확 재현 (중요)

다른 환경에서 **동일한 학습 데이터**로 재학습하려면 코퍼스 파일 자체가 바이트 단위로 같아야 한다.
**재생성(regenerate)으로는 동일해지지 않는다** — `load_mteb_task_texts`의 dedup이
`list(set(all_texts))`라서 파이썬 실행마다 순서가 달라지고(PYTHONHASHSEED), 이후
`DataLoader(shuffle=True, seed=42)`가 "다른 순서"에 적용돼 배치 구성이 달라진다.

그래서 실제로 쓴 코퍼스 파일을 그대로 옮기는 것이 원칙이다.

### v4(주력) distillation에 실제로 쓴 코퍼스 = 아래 2개 파일

| 파일 | 라인수 | 크기 | md5 | 전송 방법 |
|------|--------|------|-----|-----------|
| `data/distill_corpus/mteb_distill_10000.txt` | 364,617 | 40MB | `7acf7640347111d1734f08b48caf87a5` | **git에 포함됨** ✅ |
| `data/distill_corpus/conversation_distill.txt` | 19,520,517 | 2.3GB | `7c45f097b12b9f5c69af3109f06b28f3` | **수동 전송 필요** ⚠️ |
| (합계) | **19,885,134 (≈19.88M)** | | | |

- `mteb_distill_10000.txt`는 재현 불가 문제 때문에 **일부러 git에 커밋**해 두었다 → clone하면 동일.
- `conversation_distill.txt`(2.3GB)는 GitHub 100MB/파일 한도를 넘어 git에 못 올린다.
  외장드라이브/클라우드/`scp`·`rsync` 등으로 **직접 복사**해서 `data/distill_corpus/`에 둔다.
  복사 후 아래로 동일성 검증:
  ```bash
  wc -l data/distill_corpus/conversation_distill.txt   # 19520517 이어야 함
  md5sum data/distill_corpus/conversation_distill.txt   # 7c45f097b12b9f5c69af3109f06b28f3
  ```
- 두 파일이 모두 있으면 `run_me5s_v4_distill.py`(내부에서 `include_conversations=True`)가
  19.88M 코퍼스로 v4와 **동일하게** 학습한다.
- conversation 파일 없이 진행하려면 `include_conversations=False` → mteb 360K만으로 학습(결과 달라짐).

> 참고: `distill_texts_5000.txt`(101,492줄, md5 `8425d064a7c7363ae7ee1bea766314b4`)는 레거시
> 다국어 코퍼스(`load_multilingual_texts`)로 v4에는 쓰이지 않지만, 과거 실험 재현용으로 git에 포함해 둠.
> `training.csv`(1.8GB)는 **LoRA 트랙 전용**(개인 데이터 포함)이라 임베딩 distillation과 무관하고 git 제외.

## 코퍼스 (재)생성 — 참고용

코퍼스 파일이 아예 없을 때 MTEB 소스에서 새로 만들 수 있다. 단, 위에서 설명했듯 기존 학습과
**바이트 동일하지는 않다**(순서 차이). 새 언어를 추가할 때 등 "내용을 갱신"하는 용도로 쓴다.

```bash
python -c "from distill import load_mteb_task_texts; load_mteb_task_texts(include_conversations=False)"
```
캐시 파일이 이미 있으면 그대로 로드하므로, 새로 만들려면 기존 `mteb_distill_10000.txt`를 먼저 지운다.

---

## 언어 추가하기

목표: 새 언어 N개를 코퍼스/vocab/평가에 반영한다.

### 1. `config.py`에 언어 등록
```python
TARGET_LANGUAGES = [..., "<new_iso2>"]          # 예: "nl", "tr", "fa"
LANG_TO_ISO3 = {..., "<iso2>": "<iso3>"}         # 예: "nl": "nld"
```

### 2. `DISTILL_DATASETS`의 다국어 소스 서브셋에 언어 추가
새 언어의 텍스트가 코퍼스에 들어가도록 다국어 데이터셋 서브셋을 확장한다.
- `massive_intent` / `massive_scenario`의 `subsets` — MASSIVE가 지원하는 51개 로케일 중 선택
  (형식 예: `zh-CN`, `pt`, `fa` 등 — 데이터셋 config 이름을 그대로).
- `xnli`의 `subsets` — XNLI 15개 언어 중.
- `miracl_corpus`의 `subsets` — MIRACL 18개 언어 중.
- `tatoeba`의 `subsets` — `<iso3>-eng` 형식 (예: `nld-eng`).
- 필요하면 새 다국어 데이터셋 엔트리를 추가한다 (아래 형식).

```python
"<name>": {
    "hf_id": "<huggingface/dataset>",
    "text_fields": ["<field>", ...],   # 텍스트가 담긴 컬럼명
    "splits": ["train"],
    "subsets": ["<lang1>", "<lang2>", ...],   # 없으면 생략
},
```
> 서브셋/필드명이 맞지 않으면 로딩이 조용히 실패하고 로그에 `failed`로 남는다. 데이터셋
> 카드에서 config 이름과 컬럼명을 먼저 확인하라.

### 3. 코퍼스 재생성
```bash
rm data/distill_corpus/mteb_distill_10000.txt   # 기존 캐시 삭제
python -c "from distill import load_mteb_task_texts; load_mteb_task_texts(include_conversations=False)"
```
로그에서 새 언어 서브셋이 `N texts`로 실제 로드됐는지 확인한다.

### 4. vocab 재점검 (중요)
새 언어의 문자가 student vocab에 없으면 토크나이저가 byte-fallback으로 과분해된다.
- **Unigram(me5s 등):** `prepare_me5s_v4.py` 방식으로 새 문자 anchor를 teacher 임베딩에서
  초기화해 추가한다. `build_anchor_set()`에 새 언어의 유니코드 범위를 넣는다
  (예: 그리스어 `0x0370–0x0400`, 히브리어 `0x0590–0x0600` 등).
- **BPE:** 코퍼스에 새 언어가 포함되면 vocab pruning이 자동으로 해당 subword를 보존한다
  (merge 역추적). 별도 anchor 작업은 보통 불필요.

### 5. 재학습(distill) + 평가
```bash
python distill.py --teacher <key> --student <name>_compressed
python run_mteb.py --teacher <key> --task-groups Classification STS
```
- 다국어 평가는 `MassiveIntentClassification` 등이 언어별 점수를 내며, STS는
  `STS17`(crosslingual), `STSBenchmarkMultilingualSTS`가 다국어를 커버한다.
- `run_mteb.py --languages <iso3> ...`로 특정 언어만 평가할 수도 있다.

### 언어 추가 체크리스트
- [ ] `TARGET_LANGUAGES` + `LANG_TO_ISO3`
- [ ] `DISTILL_DATASETS` 다국어 서브셋 확장 (또는 새 데이터셋)
- [ ] 코퍼스 캐시 삭제 후 재생성, 로그로 로드 확인
- [ ] 새 문자에 대한 vocab anchor(Unigram) 또는 코퍼스 포함 확인(BPE)
- [ ] 재distill + 다국어 MTEB 평가
