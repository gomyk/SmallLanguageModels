# RESULTS — 실험 결과 및 히스토리

> 상세 수치는 `results/<teacher_key>/**/*.json`과 각종 `*.log`에 있다 (로그는 git 미포함).
> 아래는 주요 마일스톤 요약이다.

## me5s (multilingual-e5-small) — 주력 트랙

student 버전 히스토리 (`students/me5s/`):
- `me5s_compressed` → `me5s_compressed_distilled` (초기)
- `me5s_compressed_distilled_v2` (2ep 계열)
- `me5s_compressed_v3` → `me5s_compressed_v3_distilled` (byte_fallback=True, 256 byte 토큰)
- `me5s_compressed_v3_baseline_1ep` (+_distilled)
- **`me5s_compressed_v4` → `me5s_compressed_v4_distilled` (최신)**

**v4 구성:** v3 기반 + CJK/아랍/데바나가리/타이/키릴 문자 anchor를 teacher 임베딩으로 초기화
추가(`prepare_me5s_v4.py`). 384d/12L. 코퍼스 19.88M(mteb 360K + conversation 19.5M),
20 epoch, batch 512, lr 2e-5, MSE 1.0 + Cos 0.5, patience 3.

### v4_distilled MTEB 점수 (main_score, 발췌)
| 태스크 | 점수 |
|--------|------|
| STSBenchmark | 0.771 |
| SICK-R | 0.741 |
| STS12 | 0.737 |
| BIOSSES | 0.715 |
| STS17 (crosslingual) | 0.709 |
| Banking77Classification | 0.665 |
| ArXivHierarchicalClusteringS2S | 0.451 |
| MassiveIntentClassification | 0.343 |

> 전체 25~26개 태스크 결과: `results/me5s/me5s_compressed_v4_distilled/.../*.json`.
> ep별 백업: `results/me5s/me5s_compressed_v3_distilled_ep13_backup`, `_ep17_backup` 등.
> 관련 로그: `v4_distill_r*.log`, `v4_mteb_final.log`, `v4ti_r*.log`, `me5s_v3_distill_r*.log`.

## jina-v5-nano (jina-embeddings-v5-text-nano)

**압축 스펙:** 6L / 384d / 1536 intermediate / 13,357 vocab = 19.3M params,
73.6MB FP32 (~12x 압축). teacher가 PeftMixedModel이라 base EuroBertModel 추출 등 이슈 해결
([TEACHERS.md](TEACHERS.md#jina_v5-jina-embeddings-v5-text-nano) 참고).

**STSBenchmark:**
| 단계 | 점수 |
|------|------|
| distill 전 | 0.420 |
| 1 epoch | 0.587 |
| 3 epochs (v2) | 0.682 |

관련 로그: `jina_v5_h256_L6_b*.log`(batch 실험 128/256/384/512), `jina_v5_h256_v2.log`,
`jina_v5_h256_eval.log`. LoRA 트랙 결과: `results/jina_v5_lora*`, `students/jina_v5_lora*`.

## 기타 teacher

- **me5-base:** hidden dim 실험 h256/h384/h512 (`run_me5_hidden_exp.py`, `students/me5/me5_h*`).
- **gte / modernbert / minilm:** compressed + distilled 완료, `results/<key>/evaluation_summary.json`,
  `compare_results.py --teacher <key>`로 비교표 확인.
- **qwen3:** `students/qwen3/qwen3_compressed`, `qwen3_intermediate` 존재. 실험 부분적.
- **gemma_emb / mmbert:** config 등록 + 초기 실험. 이어서 진행 대상.

## 초기 MiniLM layer-pruning 실험 (baseline 탐색)
`config.py`의 `EXPERIMENTS` — L6/L4/L3 uniform, top/bottom half, L2 ends 등 레이어 선택
전략별 비교 (`results/L6_uniform`, `L4_top`, `L6_bottom`, ...). "균등 간격(uniform)"이
일반 목적에 무난하다는 결론으로 이후 teacher들에 uniform 기본 적용.

## 재현 참고
- distill DataLoader는 seed 42 고정 → 같은 코퍼스/설정이면 shuffle 순서 재현.
- 1000 step마다 체크포인트 저장, resume 가능. crash 잦으면 `distill_resilient.py` 래퍼 사용.
- MTEB 결과 JSON은 태스크별로 저장되므로 중단/재개해도 완료된 태스크는 건너뜀.
