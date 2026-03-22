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

# L6_bottom_distilled (Distilled)

Lightweight multilingual sentence encoder optimized for intent classification.
Created from `paraphrase-multilingual-MiniLM-L12-v2` via layer pruning + corpus-based vocabulary pruning + knowledge distillation.

## Model Details

| Property | Value |
|----------|-------|
| Teacher | paraphrase-multilingual-MiniLM-L12-v2 |
| Architecture | XLM-RoBERTa (pruned) |
| Hidden dim | 384 |
| Layers | 6 / 12 |
| Layer indices | [0, 1, 2, 3, 4, 5] |
| Strategy | 6 layers, bottom half (syntactic-focused) |
| Vocab size | ~38,330 (pruned from 250K) |
| Parameters | 26,184,576 |
| Safetensors size | 98.1MB |
| Distilled | Yes |

## Supported Languages (18)

ko, en, ja, zh, es, fr, de, pt, it, ru, ar, hi, th, vi, id, tr, nl, pl

## Quick Start

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("L6_bottom_distilled")

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

**Overall Average: 55.33%**

### MassiveIntentClassification

**Average: 51.63%**

| Language | Score |
|----------|-------|
| ar | 41.69% |
| en | 60.83% |
| es | 54.89% |
| ko | 49.1% |

### MassiveScenarioClassification

**Average: 59.03%**

| Language | Score |
|----------|-------|
| ar | 48.39% |
| en | 68.91% |
| es | 61.63% |
| ko | 57.2% |


## Distillation Impact

| Task | Before Distillation | After Distillation | Delta |
|------|--------------------|--------------------|-------|
| MassiveIntentClassification | 54.7% | 51.63% | -3.07%p |
| MassiveScenarioClassification | 59.39% | 59.03% | -0.36%p |


## Training

This model was created in two stages:

### Stage 1: Layer Pruning
1. Teacher model: `paraphrase-multilingual-MiniLM-L12-v2` (12 layers, 384 hidden dim)
2. Selected layers: `[0, 1, 2, 3, 4, 5]` (6 layers, bottom half (syntactic-focused))
3. Vocabulary pruning: 250K -> ~38K tokens (corpus-based, 18 target languages)

### Stage 2: Knowledge Distillation
- **Method**: MSE + Cosine Similarity loss between teacher and student embeddings
- **Training data**: MASSIVE dataset (90K multilingual sentences, 18 languages)
- **Optimizer**: AdamW (lr=2e-5, weight_decay=0.01)
- **Schedule**: Cosine annealing over 3 epochs
- **Batch size**: 64
- **Base model**: `L6_bottom` (layer-pruned only)


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
