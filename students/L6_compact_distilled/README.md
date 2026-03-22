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

# L6_compact_distilled

Ultra-compact multilingual sentence encoder (~71.2MB) for intent classification.
6 layers bottom + 20K vocab + distilled

## Performance

| Model | Size | MassiveIntent | MassiveScenario | Average |
|-------|------|--------------|-----------------|---------|
| Teacher (12L, full) | ~480MB | 55.52% | 61.01% | 58.27% |
| L6_bottom (38K vocab) | 98MB | 54.70% | 59.39% | 57.05% |
| **L6_compact_distilled** | **71.2MB** | **51.21%** | **58.33%** | **54.77%** |

## Model Details

| Property | Value |
|----------|-------|
| Teacher | paraphrase-multilingual-MiniLM-L12-v2 |
| Vocab | ~20,000 (frequency-based pruning, 97.4% coverage) |
| Size | 71.2MB |
| Distilled | Yes |

## Quick Start



## Distillation Details
- Loss: 0.3 * MSE + 2.0 * (1 - CosineSimilarity)  
- Epochs: 10, LR: 5e-6, Batch: 64
- Cosine-dominant loss preserves existing representations

