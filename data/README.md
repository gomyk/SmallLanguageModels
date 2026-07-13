# data/ — 데이터 배치 안내

> ## ⚠️ 여기에 `training.csv`를 따로 넣어야 합니다
> 개인 대화 데이터셋 `training.csv`는 git에 포함되지 않습니다(용량/개인정보).
> 원본을 받아 **`data/training.csv`** 경로에 두세요.

## 이 폴더에 있어야 하는 파일

| 파일 | git 포함? | 확보 방법 |
|------|-----------|-----------|
| `distill_corpus/mteb_distill_10000.txt` | ✅ 포함 | clone하면 있음 (공개 MTEB 코퍼스, 364,617줄) |
| `distill_corpus/distill_texts_5000.txt` | ✅ 포함 | clone하면 있음 (레거시 다국어 코퍼스) |
| **`training.csv`** | ❌ **미포함** | **원 소유자에게서 직접 받아 여기 배치** (1.8GB, chat/labels/lang) |
| `distill_corpus/conversation_distill.txt` | ❌ 미포함 | `training.csv` 배치 후 `python build_conversation_corpus.py`로 생성 (2.3GB) |

## training.csv 넣은 뒤

```bash
# 대화 코퍼스 재생성
python build_conversation_corpus.py --csv data/training.csv \
    --out data/distill_corpus/conversation_distill.txt

# 동일성 검증
wc -l  data/distill_corpus/conversation_distill.txt   # 19520517
md5sum data/distill_corpus/conversation_distill.txt   # 7c45f097b12b9f5c69af3109f06b28f3
```

자세한 출처/다운로드는 [../docs/DATA_SOURCES.md](../docs/DATA_SOURCES.md).
