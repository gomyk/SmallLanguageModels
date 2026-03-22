---
language: ["ko", "en", "ja", "zh", "es", "fr", "de", "pt", "it", "ru", "ar", "hi", "th", "vi", "id", "tr", "nl", "pl"]
tags:
- sentence-transformers
- intent-classification
- multilingual
- layer-pruning
- vocab-pruning
- knowledge-distillation
library_name: sentence-transformers
pipeline_tag: sentence-similarity
license: apache-2.0
---

# L6_uniform_distilled (Distilled)

Lightweight multilingual sentence encoder optimized for intent classification.
Created from `paraphrase-multilingual-MiniLM-L12-v2` via layer pruning + corpus-based vocabulary pruning + knowledge distillation.

## Model Details

| Property | Value |
|----------|-------|
| Teacher | paraphrase-multilingual-MiniLM-L12-v2 |
| Architecture | XLM-RoBERTa (pruned) |
| Hidden dim | 384 |
| Layers | 6 / 12 |
| Layer indices | [0, 2, 4, 7, 9, 11] |
| Strategy | 6 layers, evenly spaced (general-purpose) |
| Vocab size | ~38,330 (pruned from 250K) |
| Parameters | 26,184,576 |
| Safetensors size | 98.1MB |
| Distilled | Yes |

## Supported Languages (18)

ko, en, ja, zh, es, fr, de, pt, it, ru, ar, hi, th, vi, id, tr, nl, pl

## Quick Start

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("L6_uniform_distilled")

sentences = [
    "예약 좀 해줘",           # Korean
    "What did I order?",     # English
    "今日はいい天気ですね",    # Japanese
    "Reserva una mesa",      # Spanish
]

embeddings = model.encode(sentences)
print(embeddings.shape)  # (4, 384)
```

## MTEB Evaluation Results

**Overall Average: 56.3%**

### MassiveIntentClassification

**Average: 52.83%**

| Language | Score |
|----------|-------|
| ar | 42.91% |
| en | 63.86% |
| es | 56.57% |
| ko | 47.97% |

### MassiveScenarioClassification

**Average: 59.77%**

| Language | Score |
|----------|-------|
| ar | 48.72% |
| en | 71.38% |
| es | 63.4% |
| ko | 55.56% |


## Distillation Impact

| Task | Before Distillation | After Distillation | Delta |
|------|--------------------|--------------------|-------|
| MassiveIntentClassification | 52.9% | 52.83% | -0.07%p |
| MassiveScenarioClassification | 58.2% | 59.77% | +1.57%p |


## Training

This model was created in two stages:

### Stage 1: Layer Pruning
1. Teacher model: `paraphrase-multilingual-MiniLM-L12-v2` (12 layers, 384 hidden dim)
2. Selected layers: `[0, 2, 4, 7, 9, 11]` (6 layers, evenly spaced (general-purpose))
3. Vocabulary pruning: 250K -> ~38K tokens (corpus-based, 18 target languages)

### Stage 2: Knowledge Distillation
- **Method**: MSE + Cosine Similarity loss between teacher and student embeddings
- **Training data**: MASSIVE dataset (90K multilingual sentences, 18 languages)
- **Optimizer**: AdamW (lr=2e-5, weight_decay=0.01)
- **Schedule**: Cosine annealing over 3 epochs
- **Batch size**: 64
- **Base model**: `L6_uniform` (layer-pruned only)


## Compression Summary

| Stage | Vocab | Layers | Size |
|-------|-------|--------|------|
| Teacher (original) | 250,002 | 12 | ~480MB |
| + Layer pruning | 250,002 | 6 | ~407MB |
| + Vocab pruning | ~38,330 | 6 | ~98MB |

## Limitations

- Vocabulary pruning restricts the model to the 18 target languages
- Designed for short dialogue utterances, not long documents
- Layer pruning may reduce performance on complex semantic tasks
