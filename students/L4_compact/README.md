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

# L4_compact

Ultra-compact multilingual sentence encoder (~57.7MB) for intent classification.
4 layers [0,4,7,11] + 20K vocab

## Performance

| Model | Size | MassiveIntent | MassiveScenario | Average |
|-------|------|--------------|-----------------|---------|
| Teacher (12L, full) | ~480MB | 55.52% | 61.01% | 58.27% |
| L6_bottom (38K vocab) | 98MB | 54.70% | 59.39% | 57.05% |
| **L4_compact** | **57.7MB** | **49.2%** | **52.69%** | **50.95%** |

## Model Details

| Property | Value |
|----------|-------|
| Teacher | paraphrase-multilingual-MiniLM-L12-v2 |
| Vocab | ~20,000 (frequency-based pruning, 97.4% coverage) |
| Size | 57.7MB |
| Distilled | No |

## Quick Start



