"""
Multi-Teacher Student 모델을 아키텍처 시각화 포함 모델카드와 함께
HuggingFace Hub에 업로드한다.

Usage:
    python upload_to_hub.py --teacher modernbert --repo-prefix gomyk/modernbert-student
    python upload_to_hub.py --teacher gte --repo-prefix gomyk/gte-student
    python upload_to_hub.py --teacher minilm --repo-prefix gomyk/intent-student --only L6_uniform
"""

import argparse
import glob as _glob
import json
import os

from huggingface_hub import HfApi, create_repo, upload_folder

from config import (
    TEACHERS, EXPERIMENTS, EXPORT_DIR, RESULTS_DIR, STUDENTS_DIR,
    TARGET_LANGUAGES, MTEB_TASK_GROUPS,
    generate_experiments, generate_me5_experiments,
    get_teacher_students_dir, get_teacher_results_dir,
    estimate_size, _estimate_for_teacher,
)
from arch_utils import generate_architecture_diagram


def load_mteb_scores(model_name, results_dir):
    """MTEB 결과에서 모델의 점수를 로드한다 (모든 태스크 지원).

    model_name과 정확히 일치하는 디렉토리가 없으면,
    results_dir 하위에서 model_name을 포함하는 디렉토리를 찾는다.
    (예: "L6_uniform" → "minilm_L6_uniform" fallback)
    """
    model_dir = os.path.join(results_dir, model_name)
    scores = {}
    if not os.path.isdir(model_dir):
        # Fallback: teacher prefix가 붙은 이름으로 검색
        parent = results_dir
        if os.path.isdir(parent):
            for d in os.listdir(parent):
                if d.endswith(model_name) and os.path.isdir(os.path.join(parent, d)):
                    model_dir = os.path.join(parent, d)
                    break
            else:
                return scores
        else:
            return scores

    for fpath in _glob.glob(os.path.join(model_dir, "**", "*.json"), recursive=True):
        fname = os.path.basename(fpath)
        if fname == "model_meta.json":
            continue
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

            # 태스크 그룹 판별
            task_group = "Other"
            for group, group_tasks in MTEB_TASK_GROUPS.items():
                if task_name in group_tasks:
                    task_group = group
                    break

            scores[task_name] = {
                "average": avg,
                "group": task_group,
                "by_language": lang_scores,
            }

    return scores


def generate_model_card(name, exp, teacher_key, mteb_scores,
                        is_distilled=False, base_mteb_scores=None,
                        model_size_mb=None):
    """아키텍처 시각화를 포함한 모델 카드를 생성한다."""
    t = TEACHERS[teacher_key]
    layers = exp["layer_indices"]

    # 사이즈 계산
    size_info = estimate_size(layers, t["hidden_dim"], t["vocab_size"],
                               t["intermediate_size"])

    # 전체 평균 점수
    overall_avg = None
    group_avgs = {}
    if mteb_scores:
        all_avgs = [s["average"] for s in mteb_scores.values()]
        overall_avg = round(sum(all_avgs) / len(all_avgs), 2) if all_avgs else None

        for group in ["Classification", "Clustering", "STS"]:
            group_scores = [s["average"] for s in mteb_scores.values() if s.get("group") == group]
            if group_scores:
                group_avgs[group] = round(sum(group_scores) / len(group_scores), 2)

    # 아키텍처 시각화
    pruned_vocab = None
    if model_size_mb and model_size_mb < size_info["fp32_mb"]:
        # vocab pruning이 적용된 경우 역산
        layer_params = len(layers) * (
            3 * t["hidden_dim"] * t["hidden_dim"]
            + t["hidden_dim"] * t["hidden_dim"]
            + 2 * t["hidden_dim"] * t["intermediate_size"]
            + 4 * t["hidden_dim"]
        )
        overhead = t["hidden_dim"] + 514 * t["hidden_dim"] + 2 * t["hidden_dim"]
        total_bytes = model_size_mb * 1024 * 1024
        pruned_vocab = int((total_bytes / 4 - overhead - layer_params) / t["hidden_dim"])
        pruned_vocab = max(pruned_vocab, 1000)

    arch_diagram = generate_architecture_diagram(t, layers, t["vocab_size"], pruned_vocab)

    # MTEB 결과 테이블 (그룹별)
    mteb_table = ""
    if mteb_scores:
        mteb_table = "## MTEB Evaluation Results\n\n"
        if overall_avg:
            mteb_table += f"**Overall Average: {overall_avg}%**\n\n"

        # 그룹별 요약
        if group_avgs:
            mteb_table += "| Task Group | Average |\n|---|---|\n"
            for g, avg in group_avgs.items():
                mteb_table += f"| {g} | {avg}% |\n"
            mteb_table += "\n"

        # 태스크별 상세
        for group in ["Classification", "Clustering", "STS"]:
            group_tasks = {k: v for k, v in mteb_scores.items() if v.get("group") == group}
            if not group_tasks:
                continue

            mteb_table += f"### {group}\n\n"
            mteb_table += "| Task | Average | Details |\n|---|---|---|\n"
            for task, scores in sorted(group_tasks.items()):
                n_langs = len(scores["by_language"])
                top_langs = sorted(scores["by_language"].items(), key=lambda x: -x[1])[:3]
                top_str = ", ".join(f"{l}: {s}%" for l, s in top_langs)
                mteb_table += f"| {task} | {scores['average']}% | {top_str} |\n"
            mteb_table += "\n"

    # Distillation 비교
    distill_comparison = ""
    if is_distilled and base_mteb_scores:
        distill_comparison = "## Distillation Impact\n\n"
        distill_comparison += "| Task | Before | After | Delta |\n"
        distill_comparison += "|---|---|---|---|\n"
        for task in mteb_scores:
            if task in base_mteb_scores:
                before = base_mteb_scores[task]["average"]
                after = mteb_scores[task]["average"]
                delta = round(after - before, 2)
                sign = "+" if delta > 0 else ""
                distill_comparison += f"| {task} | {before}% | {after}% | {sign}{delta}%p |\n"
        distill_comparison += "\n"

    # 학습 방법
    base_name = name.replace("_distilled", "")
    if is_distilled:
        training_section = f"""## Training

### Stage 1: Layer Pruning
- Teacher: `{t['model_id']}` ({t['num_layers']} layers, {t['hidden_dim']}d)
- Selected layers: `{layers}` ({exp['description']})
- Vocabulary pruning applied

### Stage 2: Knowledge Distillation
- **Method**: MSE + Cosine Similarity loss
- **Data**: MTEB Classification/Clustering/STS task datasets
- **Optimizer**: AdamW (lr=2e-5, weight_decay=0.01)
- **Schedule**: Cosine annealing over 3 epochs
"""
    else:
        training_section = f"""## Training

Created via **layer pruning + vocabulary pruning** (no additional training):

1. **Teacher**: `{t['model_id']}` ({t['num_layers']} layers, {t['hidden_dim']}d)
2. **Layer selection**: `{layers}` - {exp['description']}
3. **Vocab pruning**: Corpus-based filtering for target languages
"""

    # 실제 파일 크기
    size_line = f"| Model size (FP32) | {model_size_mb:.1f}MB |" if model_size_mb else ""

    distill_tag = "- knowledge-distillation\n" if is_distilled else ""
    title_suffix = " (Distilled)" if is_distilled else ""

    card = f"""---
language: {json.dumps(TARGET_LANGUAGES)}
tags:
- sentence-transformers
- multilingual
- layer-pruning
- vocab-pruning
{distill_tag}- {t['short_name'].lower().replace(' ', '-')}
library_name: sentence-transformers
pipeline_tag: sentence-similarity
license: apache-2.0
---

# {name}{title_suffix}

Lightweight sentence encoder created from `{t['model_id']}` via layer pruning + vocabulary pruning{' + knowledge distillation' if is_distilled else ''}.

## Model Details

| Property | Value |
|---|---|
| Teacher | {t['model_id']} |
| Architecture | {t['short_name']} (pruned) |
| Hidden dim | {t['hidden_dim']} |
| Layers | {len(layers)} / {t['num_layers']} |
| Layer indices | {layers} |
| Strategy | {exp['description']} |
| Parameters | {size_info['total_params']:,} |
{size_line}
| Distilled | {'Yes' if is_distilled else 'No'} |

## Architecture

{arch_diagram}

## Quick Start

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("{name}", trust_remote_code=True)

sentences = [
    "Hello, how are you?",
    "안녕하세요",
    "Bonjour, comment allez-vous?",
]

embeddings = model.encode(sentences)
print(embeddings.shape)  # (3, {t['hidden_dim']})
```

{mteb_table}
{distill_comparison}
{training_section}

## Supported Languages ({len(TARGET_LANGUAGES)})

{', '.join(TARGET_LANGUAGES)}
"""
    return card


def read_model_config(model_path):
    """모델 디렉토리에서 config.json을 읽어 아키텍처 정보를 반환한다."""
    for subdir in ["", "0_Transformer"]:
        config_path = os.path.join(model_path, subdir, "config.json") if subdir else \
                      os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                return json.load(f)
    return {}


def generate_compressed_model_card(name, teacher_key, mteb_scores,
                                    is_distilled=False, base_mteb_scores=None,
                                    model_size_mb=None, model_config=None,
                                    two_stage=False):
    """압축 모델용 상세 모델 카드를 생성한다."""
    t = TEACHERS[teacher_key]
    cfg = model_config or {}

    # 모델 아키텍처 정보 (config.json에서)
    hidden_dim = cfg.get("hidden_size", t["hidden_dim"])
    num_layers = cfg.get("num_hidden_layers", 4)
    inter_size = cfg.get("intermediate_size", t["intermediate_size"])
    vocab_size = cfg.get("vocab_size", t["vocab_size"])
    n_heads = cfg.get("num_attention_heads", "?")
    n_kv_heads = cfg.get("num_key_value_heads", "")
    model_type = cfg.get("model_type", "unknown")

    # 추정 파라미터
    layer_indices = list(range(num_layers))
    est = _estimate_for_teacher(teacher_key, layer_indices, vocab_size,
                                 hidden_dim=hidden_dim, intermediate_size=inter_size)

    # Teacher 추정
    teacher_layers = list(range(t["num_layers"]))
    teacher_est = _estimate_for_teacher(teacher_key, teacher_layers)
    compression_ratio = teacher_est["total_params"] / max(est["total_params"], 1)

    # MTEB 결과
    overall_avg = None
    group_avgs = {}
    if mteb_scores:
        all_avgs = [s["average"] for s in mteb_scores.values()]
        overall_avg = round(sum(all_avgs) / len(all_avgs), 2) if all_avgs else None
        for group in ["Classification", "Clustering", "STS"]:
            group_scores = [s["average"] for s in mteb_scores.values()
                          if s.get("group") == group]
            if group_scores:
                group_avgs[group] = round(sum(group_scores) / len(group_scores), 2)

    # MTEB 결과 테이블
    mteb_table = ""
    if mteb_scores:
        mteb_table = "## MTEB Evaluation Results\n\n"
        if overall_avg:
            mteb_table += f"**Overall Average: {overall_avg}%**\n\n"

        if group_avgs:
            mteb_table += "| Task Group | Average |\n|---|---|\n"
            for g, avg in group_avgs.items():
                mteb_table += f"| {g} | {avg}% |\n"
            mteb_table += "\n"

        for group in ["Classification", "Clustering", "STS"]:
            group_tasks = {k: v for k, v in mteb_scores.items()
                         if v.get("group") == group}
            if not group_tasks:
                continue
            mteb_table += f"### {group}\n\n"
            mteb_table += "| Task | Average | Details |\n|---|---|---|\n"
            for task, scores in sorted(group_tasks.items()):
                top_langs = sorted(scores["by_language"].items(),
                                  key=lambda x: -x[1])[:5]
                top_str = ", ".join(f"{l}: {s}%" for l, s in top_langs)
                mteb_table += f"| {task} | {scores['average']}% | {top_str} |\n"
            mteb_table += "\n"

    # Distillation 비교
    distill_comparison = ""
    if is_distilled and base_mteb_scores:
        distill_comparison = "## Distillation Impact\n\n"
        distill_comparison += "| Task | Before | After | Delta |\n"
        distill_comparison += "|---|---|---|---|\n"
        for task in sorted(mteb_scores):
            if task in base_mteb_scores:
                before = base_mteb_scores[task]["average"]
                after = mteb_scores[task]["average"]
                delta = round(after - before, 2)
                sign = "+" if delta > 0 else ""
                distill_comparison += (f"| {task} | {before}% | {after}% "
                                      f"| {sign}{delta}%p |\n")
        distill_comparison += "\n"

    # 학습 방법
    if is_distilled and two_stage:
        training_section = f"""## Training

### Stage 1: Model Compression
- **Teacher**: `{t['model_id']}` ({t['num_layers']}L, {t['hidden_dim']}d, {teacher_est['total_params']/1e6:.0f}M params)
- **Compression**: Layer pruning → Hidden dim reduction → Vocab pruning
- **Result**: {num_layers}L / {hidden_dim}d / {vocab_size:,} vocab

### Stage 2: Two-Stage Knowledge Distillation
Compression ratio {compression_ratio:.0f}x requires progressive distillation:

1. **Stage 1**: Teacher ({teacher_est['total_params']/1e6:.0f}M) → Intermediate (~{teacher_est['total_params']/5e6:.0f}M)
   - MSE + Cosine Similarity loss
   - MTEB task datasets (Classification/Clustering/STS)
2. **Stage 2**: Intermediate → Final Student ({est['total_params']/1e6:.1f}M)
   - Same training objective
   - AdamW (lr=2e-5, weight_decay=0.01), Cosine annealing
"""
    elif is_distilled:
        training_section = f"""## Training

### Stage 1: Model Compression
- **Teacher**: `{t['model_id']}` ({t['num_layers']}L, {t['hidden_dim']}d)
- **Compression**: Layer pruning + Vocab pruning
- **Result**: {num_layers}L / {hidden_dim}d / {vocab_size:,} vocab

### Stage 2: Knowledge Distillation
- **Method**: MSE + Cosine Similarity loss
- **Data**: MTEB Classification/Clustering/STS task datasets
- **Optimizer**: AdamW (lr=2e-5, weight_decay=0.01)
- **Schedule**: Cosine annealing over 3 epochs
"""
    else:
        training_section = f"""## Training

Created via **multi-method model compression** (no additional training):

1. **Teacher**: `{t['model_id']}` ({t['num_layers']}L, {t['hidden_dim']}d, {teacher_est['total_params']/1e6:.0f}M params)
2. **Layer pruning**: {t['num_layers']} → {num_layers} layers (uniform selection)
3. **Hidden dim**: {t['hidden_dim']} → {hidden_dim}
4. **Vocab pruning**: {t['vocab_size']:,} → {vocab_size:,} (90% cumulative frequency)
5. **Compression ratio**: {compression_ratio:.0f}x
"""

    kv_heads_info = f"\n| KV heads | {n_kv_heads} |" if n_kv_heads else ""
    decoder_tag = "decoder" if t.get("is_decoder") else "encoder"
    size_line = f"\n| Model size (FP32) | {model_size_mb:.1f}MB |" if model_size_mb else ""
    distill_tag = "- knowledge-distillation\n" if is_distilled else ""
    two_stage_tag = "- progressive-distillation\n" if two_stage else ""
    title_suffix = " (Distilled)" if is_distilled else ""

    # 라이센스 처리 (Gemma 등 특수 라이센스 지원)
    license_id = t.get("license", "apache-2.0")
    license_notice = t.get("license_notice", "")
    license_section = ""
    if license_notice:
        license_section = f"""## License

{license_notice}

"""

    card = f"""---
language: {json.dumps(TARGET_LANGUAGES)}
tags:
- sentence-transformers
- multilingual
- model-compression
- layer-pruning
- vocab-pruning
{distill_tag}{two_stage_tag}- {t['short_name'].lower().replace(' ', '-')}
library_name: sentence-transformers
pipeline_tag: sentence-similarity
license: {license_id}
---

# {name}{title_suffix}

Compact multilingual sentence encoder compressed from `{t['model_id']}` ({compression_ratio:.0f}x compression).

## Model Details

| Property | Value |
|---|---|
| Base model | `{t['model_id']}` |
| Architecture | {model_type} ({decoder_tag}) |
| Hidden dim | {hidden_dim} (from {t['hidden_dim']}) |
| Layers | {num_layers} (from {t['num_layers']}) |
| Intermediate | {inter_size} |
| Attention heads | {n_heads} |{kv_heads_info}
| Vocab size | {vocab_size:,} (from {t['vocab_size']:,}) |
| Parameters | ~{est['total_params']/1e6:.1f}M |{size_line}
| Compression | {compression_ratio:.0f}x |
| Distilled | {'Yes (2-stage)' if is_distilled and two_stage else 'Yes' if is_distilled else 'No'} |

## Quick Start

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("{name}", trust_remote_code=True)

sentences = [
    "Hello, how are you?",
    "안녕하세요, 잘 지내세요?",
    "こんにちは、元気ですか？",
    "你好，你好吗？",
]

embeddings = model.encode(sentences)
print(embeddings.shape)  # (4, {hidden_dim})
```

{mteb_table}
{distill_comparison}
{training_section}
{license_section}
## Supported Languages ({len(TARGET_LANGUAGES)})

{', '.join(TARGET_LANGUAGES)}
"""
    return card


def get_model_size_mb(model_path):
    """모델 파일 크기를 MB로 반환."""
    for fname in ["model.safetensors", "pytorch_model.bin"]:
        st_path = os.path.join(model_path, fname)
        if os.path.exists(st_path):
            return os.path.getsize(st_path) / (1024 ** 2)
        # sentence-transformers 구조
        st_path = os.path.join(model_path, "0_Transformer", fname)
        if os.path.exists(st_path):
            return os.path.getsize(st_path) / (1024 ** 2)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=str, required=True,
                        choices=sorted(TEACHERS.keys()))
    parser.add_argument("--repo-prefix", required=True,
                        help="HuggingFace repo prefix (e.g., gomyk/modernbert-student)")
    parser.add_argument("--only", nargs="+", help="특정 모델만 업로드")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="모델카드만 생성, 업로드 안 함")
    args = parser.parse_args()

    api = HfApi()
    teacher_key = args.teacher
    t = TEACHERS[teacher_key]
    students_dir = get_teacher_students_dir(teacher_key)
    results_dir = get_teacher_results_dir(teacher_key)

    if not os.path.isdir(results_dir):
        results_dir = RESULTS_DIR

    # ── 업로드 대상 수집 ──
    upload_targets = []

    # 기존 실험 기반 모델
    if teacher_key == "minilm":
        experiments = EXPERIMENTS
    elif teacher_key == "me5":
        experiments = generate_me5_experiments()
    else:
        experiments = generate_experiments(teacher_key)

    if args.only:
        experiments = [e for e in experiments if e["name"] in args.only]

    def _find_student_path(name):
        """Student 경로를 찾는다. teacher prefix 붙은 이름도 검색."""
        for candidate in [
            os.path.join(students_dir, name),
            os.path.join(STUDENTS_DIR, name),
            os.path.join(students_dir, f"{teacher_key}_{name}"),
        ]:
            if os.path.exists(candidate):
                return candidate
        return None

    for exp in experiments:
        base_name = exp["name"]
        base_path = _find_student_path(base_name)
        if base_path:
            upload_targets.append({
                "name": base_name, "path": base_path,
                "is_distilled": False, "exp": exp, "compressed": False,
            })
        distilled_path = base_path + "_distilled"
        if os.path.exists(distilled_path):
            upload_targets.append({
                "name": base_name + "_distilled", "path": distilled_path,
                "is_distilled": True, "exp": exp, "compressed": False,
            })

    # Compressed 모델 (--only에 지정되었거나 디렉토리 존재 시)
    compressed_suffixes = ["_compressed", "_intermediate"]
    for suffix in compressed_suffixes:
        name = f"{teacher_key}{suffix}"
        if args.only and name not in args.only:
            continue
        for base_dir in [students_dir, STUDENTS_DIR]:
            path = os.path.join(base_dir, name)
            if os.path.exists(path):
                upload_targets.append({
                    "name": name, "path": path,
                    "is_distilled": False, "exp": None, "compressed": True,
                })
                # _distilled 버전도 확인
                dp = path + "_distilled"
                if os.path.exists(dp):
                    upload_targets.append({
                        "name": name + "_distilled", "path": dp,
                        "is_distilled": True, "exp": None, "compressed": True,
                    })
                break

    # --only 필터 적용
    if args.only:
        upload_targets = [t for t in upload_targets if t["name"] in args.only]

    # --only에 지정되었지만 목록에 없는 모델은 디렉토리에서 직접 탐색
    if args.only:
        found_names = {t["name"] for t in upload_targets}
        for name in args.only:
            if name in found_names:
                continue
            for base_dir in [students_dir, STUDENTS_DIR]:
                path = os.path.join(base_dir, name)
                if os.path.exists(path):
                    is_distilled = "_distilled" in name
                    upload_targets.append({
                        "name": name, "path": path,
                        "is_distilled": is_distilled, "exp": None,
                        "compressed": True,
                    })
                    break

    if not upload_targets:
        print("No models to upload.")
        return

    print(f"Models to upload: {[t['name'] for t in upload_targets]}")

    # ── 업로드 실행 ──
    for target in upload_targets:
        name = target["name"]
        model_path = target["path"]
        is_distilled = target["is_distilled"]

        repo_id = f"{args.repo_prefix}-{name}"
        print(f"\n{'='*60}")
        print(f"{'[DRY RUN] ' if args.dry_run else ''}Processing: {name} -> {repo_id}")
        print(f"{'='*60}")

        # MTEB 점수
        mteb_scores = load_mteb_scores(name, results_dir)
        base_name = name.replace("_distilled", "")
        base_mteb_scores = load_mteb_scores(base_name, results_dir) if is_distilled else None

        model_size = get_model_size_mb(model_path)

        # 모델 카드 생성
        if target["compressed"]:
            model_cfg = read_model_config(model_path)
            # 2-stage 여부: intermediate 모델이 존재하면 2-stage
            two_stage = os.path.exists(
                os.path.join(students_dir, f"{teacher_key}_intermediate")
            )
            card = generate_compressed_model_card(
                name=name,
                teacher_key=teacher_key,
                mteb_scores=mteb_scores,
                is_distilled=is_distilled,
                base_mteb_scores=base_mteb_scores,
                model_size_mb=model_size,
                model_config=model_cfg,
                two_stage=two_stage,
            )
        else:
            card = generate_model_card(
                name=name, exp=target["exp"],
                teacher_key=teacher_key,
                mteb_scores=mteb_scores,
                is_distilled=is_distilled,
                base_mteb_scores=base_mteb_scores,
                model_size_mb=model_size,
            )

        readme_path = os.path.join(model_path, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(card)
        print(f"  Model card written to {readme_path}")

        if args.dry_run:
            print(f"  [DRY RUN] Would upload to {repo_id}")
            continue

        try:
            create_repo(repo_id, private=args.private, exist_ok=True)
            print(f"  Repo: {repo_id}")
        except Exception as e:
            print(f"  Repo error: {e}")

        upload_folder(
            repo_id=repo_id,
            folder_path=model_path,
            commit_message=f"Upload {name} ({'distilled' if is_distilled else 'compressed'}) "
                           f"from {t['short_name']}",
        )
        print(f"  [OK] https://huggingface.co/{repo_id}")

    print("\n[OK] All uploads complete!")


if __name__ == "__main__":
    main()
