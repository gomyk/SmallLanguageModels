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

# L6_bottom_distilled_v2

Lightweight multilingual sentence encoder optimized for intent classification.
Created from `paraphrase-multilingual-MiniLM-L12-v2` via layer pruning + vocab pruning + **improved knowledge distillation**.

## What's Different (v2)

This model uses an improved distillation strategy compared to v1:

| Hyperparameter | v1 | v2 |
|---------------|----|----|
| Learning rate | 2e-5 | **5e-6** (gentler) |
| Epochs | 3 | **10** (longer training) |
| MSE weight | 1.0 | **0.3** (less magnitude forcing) |
| Cosine weight | 0.5 | **2.0** (focus on direction alignment) |

**Key insight**: The original L6_bottom model already has strong syntactic representations from the teacher's bottom layers. Aggressive MSE loss destroys these. The v2 strategy uses cosine-dominant loss to align direction without forcing magnitude, preserving the model's existing strengths.

## Results Comparison

| Model | MassiveIntent | MassiveScenario | Average |
|-------|--------------|-----------------|---------|
| Teacher (12L, 480MB) | 55.52% | 61.01% | **58.27%** |
| L6_bottom (no distill) | 54.70% | 59.39% | **57.05%** |
| L6_bottom_distilled v1 | 51.63% | 59.03% | 55.33% |
| **L6_bottom_distilled v2** | 53.06% | **59.86%** | **56.46%** |

v2 surpasses the original on MassiveScenarioClassification (59.86% > 59.39%).

### Detailed Scores (v2)

| Language | MassiveIntent | MassiveScenario |
|----------|--------------|-----------------|
| ar | 42.95% | 48.90% |
| en | 61.44% | 69.26% |
| es | 56.33% | 62.55% |
| ko | 51.53% | 58.74% |

## Model Details

| Property | Value |
|----------|-------|
| Teacher | paraphrase-multilingual-MiniLM-L12-v2 |
| Architecture | XLM-RoBERTa (pruned) |
| Hidden dim | 384 |
| Layers | 6 / 12 |
| Layer indices | [0, 1, 2, 3, 4, 5] |
| Strategy | Bottom half (syntactic-focused) |
| Vocab size | ~38,330 (pruned from 250K) |
| Safetensors | 98.1MB |

## Quick Start



## Training

### Stage 1: Layer Pruning
- Teacher: paraphrase-multilingual-MiniLM-L12-v2 (12 layers)
- Selected layers: [0, 1, 2, 3, 4, 5] (bottom half, syntactic-focused)
- Vocab pruning: 250K -> ~38K tokens (corpus-based, 18 target languages)

### Stage 2: Knowledge Distillation (v2)
- **Loss**: 0.3 * MSE + 2.0 * (1 - CosineSimilarity)
- **Training data**: MASSIVE dataset (90K multilingual sentences)
- **Optimizer**: AdamW (lr=5e-6, weight_decay=0.01)
- **Schedule**: Cosine annealing over 10 epochs
- **Batch size**: 64, GPU: NVIDIA RTX 5090

## Limitations

- Vocabulary pruning restricts the model to 18 target languages
- Designed for short dialogue utterances, not long documents
