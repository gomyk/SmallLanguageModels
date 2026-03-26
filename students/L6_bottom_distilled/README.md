---
language: ["ko", "en", "ja", "zh", "es", "fr", "de", "pt", "it", "ru", "ar", "hi", "th", "vi", "id", "tr", "nl", "pl"]
tags:
- sentence-transformers
- multilingual
- layer-pruning
- vocab-pruning
- knowledge-distillation
- minilm-l12
library_name: sentence-transformers
pipeline_tag: sentence-similarity
license: apache-2.0
---

# L6_bottom_distilled (Distilled)

Lightweight sentence encoder created from `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` via layer pruning + vocabulary pruning + knowledge distillation.

## Model Details

| Property | Value |
|---|---|
| Teacher | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 |
| Architecture | MiniLM-L12 (pruned) |
| Hidden dim | 384 |
| Layers | 6 / 12 |
| Layer indices | [0, 1, 2, 3, 4, 5] |
| Strategy | 6 layers, bottom half (syntactic-focused) |
| Parameters | 106,825,344 |
| Model size (FP32) | 98.1MB |
| Distilled | Yes |

## Architecture

```
==============================================================
  TEACHER: MiniLM-L12  →  STUDENT: 6L / 38,775 vocab
==============================================================

            TEACHER                        STUDENT          
  ───────────────────────────    ───────────────────────────

  ┌─────────────────────────┐    ┌─────────────────────────┐
  │   Input Tokens          │    │   Input Tokens          │
  └────────────┬────────────┘    └────────────┬────────────┘
               │                              │
  ┌────────────┴────────────┐    ┌────────────┴────────────┐
  │  Embeddings             │    │  Embeddings (pruned)    │
  │  vocab: 250,002         │    │  vocab:  38,775         │
  │  dim:  384              │    │  dim:  384              │
  └────────────┬────────────┘    └────────────┬────────────┘
               │                              │
  ┌─────────────────────────┐    ┌─────────────────────────┐
  │  Layer  0               │ ──►  │  Layer  0 ← L0         │
  ├─────────────────────────┤    ├─────────────────────────┤
  │  Layer  1               │ ──►  │  Layer  1 ← L1         │
  ├─────────────────────────┤    ├─────────────────────────┤
  │  Layer  2               │ ──►  │  Layer  2 ← L2         │
  ├─────────────────────────┤    ├─────────────────────────┤
  │  Layer  3               │ ──►  │  Layer  3 ← L3         │
  ├─────────────────────────┤    ├─────────────────────────┤
  │  Layer  4               │ ──►  │  Layer  4 ← L4         │
  ├─────────────────────────┤    ├─────────────────────────┤
  │  Layer  5               │ ──►  │  Layer  5 ← L5         │
  ├─────────────────────────┤    ├─────────────────────────┤
  │  Layer  6               │  ╳   │                         │
  ├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤    │                         │
  │  Layer  7               │  ╳   │                         │
  ├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤    │                         │
  │  Layer  8               │  ╳   │                         │
  ├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤    │                         │
  │  Layer  9               │  ╳   │                         │
  ├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤    │                         │
  │  Layer 10               │  ╳   │                         │
  ├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤    │                         │
  │  Layer 11               │  ╳   │                         │
  └────────────┬────────────┘    └────────────┬────────────┘
               │                              │
  ┌────────────┴────────────┐    ┌────────────┴────────────┐
  │  Mean Pooling           │    │  Mean Pooling           │
  │  → 384d embedding       │    │  → 384d embedding       │
  └─────────────────────────┘    └─────────────────────────┘

  Size: 448.0MB (FP32)           →  98.1MB (FP32)
  Params: 117,451,392        →  25,714,176
  Reduction: 78.1%
==============================================================
```

## Quick Start

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("L6_bottom_distilled", trust_remote_code=True)

sentences = [
    "Hello, how are you?",
    "안녕하세요",
    "Bonjour, comment allez-vous?",
]

embeddings = model.encode(sentences)
print(embeddings.shape)  # (3, 384)
```



## Training

### Stage 1: Layer Pruning
- Teacher: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (12 layers, 384d)
- Selected layers: `[0, 1, 2, 3, 4, 5]` (6 layers, bottom half (syntactic-focused))
- Vocabulary pruning applied

### Stage 2: Knowledge Distillation
- **Method**: MSE + Cosine Similarity loss
- **Data**: MTEB Classification/Clustering/STS task datasets
- **Optimizer**: AdamW (lr=2e-5, weight_decay=0.01)
- **Schedule**: Cosine annealing over 3 epochs


## Supported Languages (18)

ko, en, ja, zh, es, fr, de, pt, it, ru, ar, hi, th, vi, id, tr, nl, pl
