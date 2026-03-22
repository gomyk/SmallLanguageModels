---
language: ["ko", "en", "ja", "zh", "es", "fr", "de", "pt", "it", "ru", "ar", "hi", "th", "vi", "id", "tr", "nl", "pl"]
tags:
- sentence-transformers
- intent-classification
- multilingual
- layer-pruning
- vocab-pruning
library_name: sentence-transformers
pipeline_tag: sentence-similarity
license: apache-2.0
---

# L6_uniform

Lightweight multilingual sentence encoder optimized for intent classification.
Created from `paraphrase-multilingual-MiniLM-L12-v2` via layer pruning + corpus-based vocabulary pruning.

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
| Distilled | No |

## Supported Languages (18)

ko, en, ja, zh, es, fr, de, pt, it, ru, ar, hi, th, vi, id, tr, nl, pl

## Quick Start

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("L6_uniform")

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

**Overall Average: 55.55%**

### MassiveIntentClassification

**Average: 52.9%**

| Language | Score |
|----------|-------|
| ar | 42.79% |
| en | 61.83% |
| es | 52.89% |
| ko | 54.08% |

### MassiveScenarioClassification

**Average: 58.2%**

| Language | Score |
|----------|-------|
| ar | 46.87% |
| en | 67.91% |
| es | 59.42% |
| ko | 58.62% |



## Training

This model was created via **layer pruning + vocabulary pruning**:

1. **Teacher**: `paraphrase-multilingual-MiniLM-L12-v2` (12 layers, 384 hidden dim)
2. **Layer selection**: `[0, 2, 4, 7, 9, 11]` - 6 layers, evenly spaced (general-purpose)
3. **Vocab pruning**: 250K -> ~38K tokens (corpus-based filtering for 18 target languages)
4. **No additional training** - weights are directly copied from the teacher

A distilled version of this model is also available with improved performance.


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
