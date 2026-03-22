"""
Student 모델들을 모델 카드와 함께 HuggingFace Hub에 업로드한다.
Distilled 모델과 non-distilled 모델 모두 지원.

Usage:
    python upload_to_hub.py --repo-prefix gomyk/intent-student
    python upload_to_hub.py --repo-prefix gomyk/intent-student --only L6_uniform L6_bottom
"""

import argparse
import glob as _glob
import json
import os

from huggingface_hub import HfApi, create_repo, upload_folder

from config import EXPERIMENTS, EXPORT_DIR, RESULTS_DIR, STUDENTS_DIR, TARGET_LANGUAGES
from create_students import estimate_size


def load_mteb_scores(model_name):
    """MTEB 결과에서 모델의 점수를 로드한다 (재귀 탐색)."""
    model_dir = os.path.join(RESULTS_DIR, model_name)
    scores = {}
    if not os.path.isdir(model_dir):
        return scores

    for fpath in _glob.glob(os.path.join(model_dir, "**", "Massive*.json"), recursive=True):
        fname = os.path.basename(fpath)
        task_name = fname.replace(".json", "")

        with open(fpath) as f:
            data = json.load(f)

        if isinstance(data, list) and data:
            data = data[0]

        test_scores = data.get("scores", {}).get("test", [])
        lang_scores = {}
        for entry in test_scores:
            lang = entry.get("hf_subset", entry.get("language", "unknown"))
            main_score = entry.get("main_score", 0)
            lang_scores[lang] = round(main_score * 100, 2)

        if lang_scores:
            avg = round(sum(lang_scores.values()) / len(lang_scores), 2)
            scores[task_name] = {"average": avg, "by_language": lang_scores}

    return scores


def generate_model_card(name, exp, mteb_scores, is_distilled=False,
                        base_mteb_scores=None, model_size_mb=None):
    """모델 카드 (README.md) 내용을 생성한다."""
    layers = exp["layer_indices"]
    size_info = estimate_size(layers)

    # 전체 평균 점수 계산
    overall_avg = None
    if mteb_scores:
        all_avgs = [s["average"] for s in mteb_scores.values()]
        overall_avg = round(sum(all_avgs) / len(all_avgs), 2) if all_avgs else None

    # MTEB 결과 테이블
    mteb_table = ""
    if mteb_scores:
        mteb_table = "## MTEB Evaluation Results\n\n"
        if overall_avg:
            mteb_table += f"**Overall Average: {overall_avg}%**\n\n"

        for task, scores in mteb_scores.items():
            mteb_table += f"### {task}\n\n"
            mteb_table += f"**Average: {scores['average']}%**\n\n"
            mteb_table += "| Language | Score |\n|----------|-------|\n"
            for lang, score in sorted(scores["by_language"].items()):
                mteb_table += f"| {lang} | {score}% |\n"
            mteb_table += "\n"

    # Distillation 비교 섹션
    distill_comparison = ""
    if is_distilled and base_mteb_scores:
        distill_comparison = "## Distillation Impact\n\n"
        distill_comparison += "| Task | Before Distillation | After Distillation | Delta |\n"
        distill_comparison += "|------|--------------------|--------------------|-------|\n"
        for task in mteb_scores:
            if task in base_mteb_scores:
                before = base_mteb_scores[task]["average"]
                after = mteb_scores[task]["average"]
                delta = round(after - before, 2)
                sign = "+" if delta > 0 else ""
                distill_comparison += f"| {task} | {before}% | {after}% | {sign}{delta}%p |\n"
        distill_comparison += "\n"

    # 학습 방법 섹션
    base_name = name.replace("_distilled", "")
    if is_distilled:
        training_section = f"""## Training

This model was created in two stages:

### Stage 1: Layer Pruning
1. Teacher model: `paraphrase-multilingual-MiniLM-L12-v2` (12 layers, 384 hidden dim)
2. Selected layers: `{layers}` ({exp['description']})
3. Vocabulary pruning: 250K -> ~38K tokens (corpus-based, 18 target languages)

### Stage 2: Knowledge Distillation
- **Method**: MSE + Cosine Similarity loss between teacher and student embeddings
- **Training data**: MASSIVE dataset (90K multilingual sentences, 18 languages)
- **Optimizer**: AdamW (lr=2e-5, weight_decay=0.01)
- **Schedule**: Cosine annealing over 3 epochs
- **Batch size**: 64
- **Base model**: `{base_name}` (layer-pruned only)
"""
    else:
        training_section = f"""## Training

This model was created via **layer pruning + vocabulary pruning**:

1. **Teacher**: `paraphrase-multilingual-MiniLM-L12-v2` (12 layers, 384 hidden dim)
2. **Layer selection**: `{layers}` - {exp['description']}
3. **Vocab pruning**: 250K -> ~38K tokens (corpus-based filtering for 18 target languages)
4. **No additional training** - weights are directly copied from the teacher

A distilled version of this model is also available with improved performance.
"""

    # 실제 파일 크기
    size_line = f"| Safetensors size | {model_size_mb:.1f}MB |" if model_size_mb else ""

    distill_tag = "- knowledge-distillation\n" if is_distilled else ""
    title_suffix = " (Distilled)" if is_distilled else ""

    card = f"""---
language: {json.dumps(TARGET_LANGUAGES)}
tags:
- sentence-transformers
- intent-classification
- multilingual
- layer-pruning
- vocab-pruning
{distill_tag}library_name: sentence-transformers
pipeline_tag: sentence-similarity
license: apache-2.0
---

# {name}{title_suffix}

Lightweight multilingual sentence encoder optimized for intent classification.
Created from `paraphrase-multilingual-MiniLM-L12-v2` via layer pruning + corpus-based vocabulary pruning{' + knowledge distillation' if is_distilled else ''}.

## Model Details

| Property | Value |
|----------|-------|
| Teacher | paraphrase-multilingual-MiniLM-L12-v2 |
| Architecture | XLM-RoBERTa (pruned) |
| Hidden dim | 384 |
| Layers | {len(layers)} / 12 |
| Layer indices | {layers} |
| Strategy | {exp['description']} |
| Vocab size | ~38,330 (pruned from 250K) |
| Parameters | {size_info['total_params']:,} |
{size_line}
| Distilled | {'Yes' if is_distilled else 'No'} |

## Supported Languages ({len(TARGET_LANGUAGES)})

{', '.join(TARGET_LANGUAGES)}

## Quick Start

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("{name}")

sentences = [
    "예약 좀 해줘",           # Korean
    "What did I order?",     # English
    "今日はいい天気ですね",    # Japanese
    "Reserva una mesa",      # Spanish
]

embeddings = model.encode(sentences)
print(embeddings.shape)  # (4, 384)
```

{mteb_table}
{distill_comparison}
{training_section}

## Compression Summary

| Stage | Vocab | Layers | Size |
|-------|-------|--------|------|
| Teacher (original) | 250,002 | 12 | ~480MB |
| + Layer pruning | 250,002 | {len(layers)} | ~{round(250002*384*4/1024**2 + len(layers)*6.75, 0):.0f}MB |
| + Vocab pruning | ~38,330 | {len(layers)} | ~{model_size_mb:.0f}MB |

## Limitations

- Vocabulary pruning restricts the model to the 18 target languages
- Designed for short dialogue utterances, not long documents
- Layer pruning may reduce performance on complex semantic tasks
"""
    return card


def get_model_size_mb(model_path):
    """safetensors 파일 크기를 MB로 반환."""
    st_path = os.path.join(model_path, "model.safetensors")
    if os.path.exists(st_path):
        return os.path.getsize(st_path) / (1024 ** 2)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-prefix", required=True,
                        help="HuggingFace repo prefix (e.g., gomyk/intent-student)")
    parser.add_argument("--only", nargs="+", help="특정 base 모델만 업로드")
    parser.add_argument("--private", action="store_true", help="비공개 레포로 생성")
    args = parser.parse_args()

    api = HfApi()

    experiments = EXPERIMENTS
    if args.only:
        experiments = [e for e in experiments if e["name"] in args.only]

    for exp in experiments:
        base_name = exp["name"]

        # 업로드 대상: base 모델 + distilled 모델
        upload_targets = []

        # Base model
        base_path = os.path.join(STUDENTS_DIR, base_name)
        if os.path.exists(base_path):
            upload_targets.append({
                "name": base_name,
                "path": base_path,
                "is_distilled": False,
            })

        # Distilled model
        distilled_name = f"{base_name}_distilled"
        distilled_path = os.path.join(STUDENTS_DIR, distilled_name)
        if os.path.exists(distilled_path):
            upload_targets.append({
                "name": distilled_name,
                "path": distilled_path,
                "is_distilled": True,
            })

        for target in upload_targets:
            name = target["name"]
            model_path = target["path"]
            is_distilled = target["is_distilled"]

            repo_id = f"{args.repo_prefix}-{name}"
            print(f"\n{'='*60}")
            print(f"Uploading: {name} -> {repo_id}")
            print(f"{'='*60}")

            # MTEB 점수 로드
            mteb_scores = load_mteb_scores(name)
            base_mteb_scores = load_mteb_scores(base_name) if is_distilled else None

            # 모델 크기
            model_size = get_model_size_mb(model_path)

            # 모델 카드 생성
            card = generate_model_card(
                name=name,
                exp=exp,
                mteb_scores=mteb_scores,
                is_distilled=is_distilled,
                base_mteb_scores=base_mteb_scores,
                model_size_mb=model_size,
            )

            readme_path = os.path.join(model_path, "README.md")
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(card)
            print(f"  Model card written")

            # 레포 생성
            try:
                create_repo(repo_id, private=args.private, exist_ok=True)
                print(f"  Repo: {repo_id}")
            except Exception as e:
                print(f"  Repo error: {e}")

            # 업로드
            upload_folder(
                repo_id=repo_id,
                folder_path=model_path,
                commit_message=f"Upload {name} {'(distilled) ' if is_distilled else ''}with MTEB results",
            )
            print(f"  [OK] https://huggingface.co/{repo_id}")

    print("\n[OK] All uploads complete!")


if __name__ == "__main__":
    main()
