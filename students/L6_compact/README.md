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

# L6_compact

Ultra-compact multilingual sentence encoder (~71.2MB) for intent classification.
6 layers [0,1,2,3,4,5] + 20K vocab

## Performance

| Model | Size | MassiveIntent | MassiveScenario | Average |
|-------|------|--------------|-----------------|---------|
| Teacher (12L, full) | ~480MB | 55.52% | 61.01% | 58.27% |
| L6_bottom (38K vocab) | 98MB | 54.70% | 59.39% | 57.05% |
| **L6_compact** | **71.2MB** | **53.15%** | **57.42%** | **55.29%** |

## Model Details

| Property | Value |
|----------|-------|
| Teacher | paraphrase-multilingual-MiniLM-L12-v2 |
| Vocab | ~20,000 (frequency-based pruning, 97.4% coverage) |
| Size | 71.2MB |
| Distilled | No |

## Quick Start



