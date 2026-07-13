# SETUP — 새 환경 세팅

## 1. 클론

```bash
git clone https://github.com/gomyk/SmallLanguageModels.git
cd SmallLanguageModels
```

## 2. Python 환경

- Python 3.10+ 권장.
- GPU(CUDA) 환경 강력 권장. distillation은 CPU로도 돌아가지만 매우 느리다.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   /   Linux/mac: source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` 주요 의존성:
```
torch>=2.0.0
transformers>=4.40.0
sentence-transformers>=3.0.0
mteb>=1.14.0
onnxruntime / onnx / optimum[onnxruntime]   # export용
numpy, pandas, tqdm, sentencepiece, protobuf
```

추가로 필요할 수 있는 것:
- `datasets` (HuggingFace, 코퍼스 다운로드용) — 보통 transformers와 함께 설치되지만 없으면 `pip install datasets`.
- `peft` — jina_v5 teacher 사용 시.
- `scikit-learn`, `umap-learn`, `matplotlib` — UMAP 시각화(`umap_viz_*.py`) 사용 시.

## 3. 환경 변수

| 변수 | 용도 |
|------|------|
| `TQDM_DISABLE=1` | 로그 파일로 리다이렉트할 때 진행바 오염 방지 |
| `HF_HUB_TRUST_REMOTE_CODE=1` | GTE / mmBERT / jina_v5 등 custom code 모델 (Windows 필수) |
| `HF_TOKEN` | HuggingFace Hub 업로드(`upload_to_hub.py`) 시 |
| `CUDA_VISIBLE_DEVICES` | 특정 GPU 지정 |

## 4. 코퍼스 준비

git에는 대용량 코퍼스 파일이 포함되지 않는다 (`.gitignore`). 아래로 재생성한다:

```bash
# MTEB 태스크 데이터셋에서 distillation 코퍼스 생성 (HF에서 다운로드, ~360K 문장)
python -c "from distill import load_mteb_task_texts; load_mteb_task_texts(include_conversations=False)"
```

결과: `data/distill_corpus/mteb_distill_10000.txt`

> `conversation_distill.txt`(2.3GB, 19.5M 문장)는 별도 자산이라 git에 없다.
> 이 파일이 없으면 학습 시 `include_conversations=False`를 쓰거나, 코퍼스만으로 학습한다.
> 자세한 내용은 [CORPUS.md](CORPUS.md).

## 5. 동작 확인 (smoke test)

```bash
# teacher 로딩 + config 확인
python -c "from config import TEACHERS; print(list(TEACHERS))"

# 작은 규모로 me5s 압축만 (distill/eval 스킵) — 스크립트별 CLI는 PIPELINE.md 참고
```

## 6. 하드웨어 참고

- 원본 작업 환경: Windows 11, 단일 GPU(≈32GB VRAM 언급). PowerShell 기본.
- **heavy 프로세스(모델 로딩/MTEB/학습)는 기본 순차 실행.** 메모리 여유 있을 때만 병렬.
- 큰 teacher(jina_v5 등)는 `--max-vram-frac 0.5` 등으로 VRAM 절반 권장.
- 학습은 1000 step마다 체크포인트를 저장하므로 crash 후 resume 가능
  (`distill_resilient.py`가 자동 재시작 래퍼).
