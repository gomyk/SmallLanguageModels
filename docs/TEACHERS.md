# TEACHERS — teacher 모델별 설정/주의사항 + 새 모델 추가

모든 teacher는 `config.py`의 `TEACHERS` 딕셔너리에 등록된다. 파이프라인 스크립트는
`--teacher <key>`로 이 설정을 참조한다.

## 등록된 teacher

| key | model_id | dim | layers | inter | vocab | layer_accessor | tokenizer | trust_remote | 특이사항 |
|-----|----------|-----|--------|-------|-------|----------------|-----------|--------------|----------|
| `minilm` | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 384 | 12 | 1536 | 250002 | `encoder.layer` | unigram | No | 초기 baseline |
| `me5` | intfloat/multilingual-e5-base | 768 | 12 | 3072 | 250002 | `encoder.layer` | unigram | No | hidden 실험(h256/384/512) |
| `me5s` | intfloat/multilingual-e5-small | 384 | 12 | 1536 | 250037 | `encoder.layer` | unigram | No | **주력, v4까지 진행** |
| `gte` | alibaba-NLP/gte-multilingual-base | 768 | 12 | 3072 | 250048 | `encoder.layer` | unigram | **Yes** | custom code, `model_type="new"` |
| `modernbert` | answerdotai/ModernBERT-base | 768 | 22 | 1152 | 50368 | `layers` | bpe | No | BPE vocab pruning |
| `jina_v5` | jinaai/jina-embeddings-v5-text-nano | 768 | 12 | 3072 | 128256 | `layers` | bpe | **Yes** | **PEFT**, decoder(RoPE), GLU |
| `qwen3` | Qwen/Qwen3-0.6B | 1024 | 28 | 3072 | 151936 | `layers` | bpe | No | decoder, GQA(16/8 heads, head_dim 128), GLU |
| `gemma_emb` | google/embeddinggemma-300m | 768 | 24 | 1152 | 262144 | `layers` | unigram | No | decoder, GQA(3/1), head_dim 256, GLU, gemma license |
| `mmbert` | jhu-clsp/mmBERT-small | 384 | 22 | 1152 | 256000 | `layers` | bpe | **Yes** | encoder |

## teacher config 필드 의미

```python
"me5s": {
    "model_id": "intfloat/multilingual-e5-small",  # HF repo id
    "short_name": "mE5-small",                       # 표시용 이름
    "hidden_dim": 384,                               # hidden dimension
    "num_layers": 12,                                # 전체 레이어 수
    "intermediate_size": 1536,                       # FFN intermediate dim
    "vocab_size": 250037,                            # 원본 vocab 크기
    "layer_accessor": "encoder.layer",               # 레이어 리스트 접근 경로 (아래 참고)
    "tokenizer_type": "unigram",                     # "unigram" | "bpe"
    "trust_remote_code": False,                      # custom modeling code 필요 여부
}
```

**decoder/GQA/GLU 모델**은 추가 필드가 필요하다 (크기 추정 `estimate_size`가 사용):
```python
"num_attention_heads": 16,   # Q head 수
"num_kv_heads": 8,           # K/V head 수 (GQA)
"head_dim": 128,             # head dimension (hidden//heads와 다르면 명시, 예: Qwen3)
"has_glu": True,             # SwiGLU 등 gated FFN (gate+up+down = 3배)
"is_decoder": True,          # position/token_type embedding 없음 (RoPE 등)
```

license가 있는 모델(gemma, jina)은 `"license"`, `"license_notice"`를 넣어 model card에 반영한다.

## `layer_accessor` — 레이어 리스트 접근 경로

모델 아키텍처마다 transformer 레이어 리스트의 위치가 다르다. layer pruning이 이 경로로 접근한다:
- BERT/RoBERTa/XLM-R 계열: `encoder.layer` (minilm, me5, me5s, gte)
- ModernBERT / decoder LM 계열: `layers` (modernbert, jina_v5, qwen3, gemma_emb, mmbert)

새 모델 추가 시 `python -c "from transformers import AutoModel; m=AutoModel.from_pretrained('<id>'); print(m)"`
로 구조를 출력해 레이어 리스트 attribute 경로를 확인하라.

---

## 새 teacher 모델 추가하기

### 1. 모델 구조 파악
```bash
python -c "from transformers import AutoModel, AutoConfig; \
c=AutoConfig.from_pretrained('<repo_id>', trust_remote_code=True); print(c)"
```
확인할 것: `hidden_size`, `num_hidden_layers`, `intermediate_size`, `vocab_size`,
attention heads/kv_heads/head_dim, GLU 여부, decoder 여부, 레이어 accessor 경로, tokenizer 종류.

### 2. `config.py`의 `TEACHERS`에 항목 추가
위 필드 규칙대로 딕셔너리 엔트리를 추가한다. GQA/GLU/decoder면 추가 필드도 채운다.
custom code 모델이면 `"trust_remote_code": True`.

### 3. tokenizer 종류 확인
`tokenizer.json`의 `model.type`이 `"BPE"`면 `"bpe"`, `"Unigram"`이면 `"unigram"`.
- **BPE**: vocab pruning 시 merge 역추적 필요 (`arch_utils.py`가 처리). tokenizer.json의
  vocab+merges+added_tokens, tokenizer_config.json의 added_tokens_decoder, post_processor
  special token id를 함께 갱신해야 한다.
- **Unigram(SentencePiece)**: `model.vocab` 배열에서 토큰 제거. 새 토큰 추가 시 score 지정
  (`prepare_me5s_v4.py`의 anchor score 계산 방식 참고 — 기존 single-char 토큰 score의 median).

### 4. 크기 추정 검증
```bash
python -c "from config import _estimate_for_teacher, make_uniform_indices, TEACHERS; \
t=TEACHERS['<key>']; idx=make_uniform_indices(t['num_layers'],6); \
print(_estimate_for_teacher('<key>', idx))"
```
`estimate_size`가 GQA/GLU/decoder를 반영하는지 확인 (실제 압축 후 크기와 크게 다르면 필드 점검).

### 5. 파이프라인 실행
```bash
python create_students.py --teacher <key>                 # 압축
python distill.py --teacher <key> --student <name>_compressed   # distill
python run_mteb.py --teacher <key>                        # 평가
python compare_results.py --teacher <key>                 # 비교
```

### 6. teacher별 함정 체크리스트
- [ ] custom code 모델 → `trust_remote_code=True` + Windows `HF_HUB_TRUST_REMOTE_CODE=1`
- [ ] PEFT/LoRA로 배포된 모델(jina_v5) → base backbone 추출 필요, `model_kwargs`에 task 지정
- [ ] decoder LM을 embedding으로 쓸 때 pooling(mean/last) 및 prompt 규약 확인
- [ ] GQA면 hidden 축소 시 head 수/kv_head 수/head_dim 정합성 (config.py `_estimate_for_teacher`가 비례 조정)
- [ ] license 있는 모델 → `license`/`license_notice` 필드로 파생물 라이선스 고지

---

## teacher별 알려진 이슈 (히스토리)

### `gte` (GTE-multilingual)
- custom modeling 코드 필요. `auto_map`을 config에 유지, HF 캐시에서 `.py` 파일 복사.
- `model_type`을 `"new"`로 유지해야 로딩됨. Windows는 `HF_HUB_TRUST_REMOTE_CODE=1`.

### `jina_v5` (jina-embeddings-v5-text-nano)
- **PeftMixedModel(LoRA) 구조** → `AutoModel` 로드 후 base `EuroBertModel` 추출 필요.
- config.json 패치 필요: `model_type`, `auto_map`, `architectures`.
- post_processor ID 재매핑(Sequence 중첩) 이슈 해결됨.
- `model_kwargs={"default_task": "text-matching", "attn_implementation": "eager"}` 필수.
- decoder(RoPE) 계열이라 position/token_type embedding 없음. CC BY-NC 4.0 (비상업).

### `me5s` (multilingual-e5-small) — 주력
- v3 → v4 진행. v4는 CJK/아랍/데바나가리/타이/키릴 문자 **anchor 토큰을 teacher 임베딩으로
  초기화**해 추가 (`prepare_me5s_v4.py`). teacher에 직접 매칭되는(`c` 또는 `▁c`) 토큰만 추가.
- byte-fallback 토큰 score를 낮춰(-20) anchor가 우선 선택되게 함.
- 대규모 코퍼스(19.88M) 20 epoch distill. 결과는 [RESULTS.md](RESULTS.md).

### `modernbert`
- BPE 22레이어. vocab pruning 시 merges/added_tokens/post_processor id 정합성 주의.

### `qwen3` / `gemma_emb` / `mmbert`
- config 등록 + 압축본 일부 존재하나 실험이 부분적. decoder(qwen3/gemma) pooling 규약,
  gemma license 고지에 유의하며 이어서 진행.
