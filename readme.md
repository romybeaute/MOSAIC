# MOSAIC: Multimodal Ontological State Analysis in Consciousness

Python pipeline for topic modeling of consciousness-related textual data using BERTopic.

## Structure

MOSAIC/
├── src/
│   ├── preprocessor.py      # Text preprocessing and cleaning
│   ├── model.py            # BERTopic model configuration
│   └── utils.py            # Helper functions and metrics
├── scripts/
│   ├── grid_search.py      # Hyperparameter optimization
│   ├── visualization.py    # (Planned) Visualization tools
│   └── cluster_analysis.py # (Planned) Advanced analysis
└── requirements.txt

### Source (`src/`)

- `preprocessor.py`
 - Text cleaning and preprocessing
 - Sentence splitting
 - Stopword removal
 - Tokenization

- `model.py` 
 - BERTopic configuration
 - Model parameters
 - Topic extraction
 - Embedding generation

- `utils.py`
 - Coherence metrics
 - Evaluation utilities
 - Helper functions

### Scripts (`scripts/`)

- `grid_search.py`
 - Hyperparameter optimization
 - Model evaluation
 - Results logging

- `visualization.py`
 - Topic visualization
 - Embedding plots
 - Hierarchical clustering dendrograms

- `cluster_analysis.py` 
 - Topic hierarchy analysis
 - Cluster validation
 - Semantic interpretation

## Setup

```bash
pip install -r requirements.txt


## Usage
from src.model import setup_topic_model
from scripts.grid_search import run_grid_search

# Run grid search
results = run_grid_search(data, condition='HS')