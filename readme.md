```markdown
# MOSAIC: Mapping Of Subjective Accounts into Interpreted Clusters

Topic modelling pipeline for consciousness-related textual data using BERTopic, BERT embeddings, and UMAP-HDBSCAN clustering.

## Overview

MOSAIC analyses subjective experiential reports through:
- Advanced NLP with BERT embeddings
- Dimensionality reduction via UMAP 
- Density-based clustering with HDBSCAN
- Topic coherence optimisation

## Structure

```
MOSAIC/
├── src/                    # Core functionality
│   ├── preprocessor.py     # Text cleaning, sentence splitting
│   ├── model.py           # BERTopic configuration
│   └── utils.py           # Metrics and helpers
├── configs/               # Experiment parameters
│   └── dreamachine.py    # Dataset-specific settings
└── scripts/              # Analysis tools
    └── grid_search.py    # Hyperparameter optimisation
```

### Source (`src/`)

- `preprocessor.py`
  - Text preprocessing and cleaning
  - Sentence splitting
  - Stopword removal
  - Custom tokenization

- `model.py`
  - BERTopic configuration
  - UMAP dimensionality reduction
  - HDBSCAN clustering
  - Topic extraction and validation

- `utils.py`
  - Coherence metrics
  - Grid search utilities
  - Helper functions

### Configs (`configs/`)

- `dreamachine.py`
  - Dataset-specific parameters
  - Model hyperparameters
  - Preprocessing settings

### Scripts (`scripts/`)

- `grid_search.py`
  - Hyperparameter optimisation
  - Model evaluation
  - Results logging

## Installation

```bash
git clone https://github.com/romybeaute/MOSAIC.git
cd MOSAIC
pip install -r requirements.txt
```

## Usage

Run grid search optimisation:
```python
python grid_search.py --dataset dreamachine --condition HS --sentences
```

Parameters:
- `--dataset`: Dataset name
- `--condition`: Experimental condition [HS, DL, HW]
- `--sentences`: Enable sentence-level analysis
- `--reduced_GS`: Use reduced parameter grid

## Citation

If using this code, please cite:
[Citation details to be added]
```