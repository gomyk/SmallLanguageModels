"""UMAP visualization: Teacher vs v3_distilled on 16 languages × 50 parallel sentences."""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Malgun Gothic']
matplotlib.rcParams['axes.unicode_minus'] = False

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import umap

LANGS = ["en", "ko", "ja", "zh-CN", "es", "fr", "de", "pt",
         "it", "ru", "ar", "hi", "vi", "tr", "th", "id"]
LANG_LABELS = [l.split("-")[0] for l in LANGS]
N_SAMPLES = 50

TEACHER = "intfloat/multilingual-e5-small"
STUDENT_V3 = "C:/Users/Moon/finetuning-workshop/SmallModel/students/me5s/me5s_compressed_v3_distilled"

print("Loading MASSIVE parallel data...")
first_ids = None
data = {}
for lang in LANGS:
    ds = load_dataset("mteb/amazon_massive_intent", lang, split="train")
    if first_ids is None:
        first_ids = [row["id"] for row in ds][:N_SAMPLES]
        first_ids_set = set(first_ids)
    id_to_text = {row["id"]: row["text"] for row in ds if row["id"] in first_ids_set}
    texts = [id_to_text[i] for i in first_ids if i in id_to_text]
    data[lang] = texts
    print(f"  {lang}: {len(texts)} sentences")

n = min(len(v) for v in data.values())
all_texts, all_langs, all_sent_ids = [], [], []
for lang in LANGS:
    for sid, t in enumerate(data[lang][:n]):
        all_texts.append(t)
        all_langs.append(lang.split("-")[0])
        all_sent_ids.append(sid)

print(f"\nTotal: {len(all_texts)} sentences ({n} parallel x {len(LANGS)} langs)")

def embed(model_path, name):
    print(f"\nEmbedding with {name}...")
    m = SentenceTransformer(model_path, trust_remote_code=True)
    emb = m.encode(all_texts, batch_size=64, show_progress_bar=True,
                   convert_to_numpy=True, normalize_embeddings=True)
    del m
    import torch, gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return emb

emb_teacher = embed(TEACHER, "Teacher (me5s)")
emb_student = embed(STUDENT_V3, "v3_distilled")

print("\nRunning UMAP...")
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                    metric="cosine", random_state=42)
xy_t = reducer.fit_transform(emb_teacher)
reducer2 = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                     metric="cosine", random_state=42)
xy_s = reducer2.fit_transform(emb_student)

cmap_lang = plt.get_cmap("tab20")
lang_to_color = {l: cmap_lang(i / len(LANG_LABELS)) for i, l in enumerate(LANG_LABELS)}
cmap_sid = plt.get_cmap("tab20")
sid_to_color = {s: cmap_sid((s % 20) / 20) for s in range(n)}

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

for row, color_by in enumerate(["lang", "sent_id"]):
    for col, (xy, title, sz) in enumerate([
        (xy_t, "Teacher (me5s) — 384d, ~448MB", emb_teacher.shape[1]),
        (xy_s, "v3_distilled (byte_fallback) — 384d, ~60MB", emb_student.shape[1]),
    ]):
        ax = axes[row, col]
        if color_by == "lang":
            for lang in LANG_LABELS:
                mask = np.array([al == lang for al in all_langs])
                ax.scatter(xy[mask, 0], xy[mask, 1], c=[lang_to_color[lang]],
                           label=lang, s=30, alpha=0.7, edgecolors="w", linewidths=0.3)
            if col == 0:
                ax.legend(loc="center left", bbox_to_anchor=(-0.22, 0.5),
                          ncol=1, fontsize=8, frameon=True, title="Language")
            subtitle = "Colored by language (mixed = good)"
        else:
            n_show = 20
            for sid in range(n_show):
                mask = np.array([s == sid for s in all_sent_ids])
                ax.scatter(xy[mask, 0], xy[mask, 1], c=[sid_to_color[sid]],
                           s=30, alpha=0.8, edgecolors="w", linewidths=0.3)
            subtitle = f"Colored by sentence id (first {n_show}, tight cluster = good)"
        ax.set_title(f"{title}\n{subtitle}", fontsize=11)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.grid(True, alpha=0.2)

fig.suptitle(f"UMAP: {len(LANGS)} languages × {n} parallel sentences (MASSIVE)",
             fontsize=13)
plt.tight_layout()
out = "C:/Users/Moon/finetuning-workshop/SmallModel/umap_me5s_v3.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out}")
