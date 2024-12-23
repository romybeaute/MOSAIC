# MOSAIC: Mapping Of Subjective Accounts into Interpreted Clusters

Python pipeline for topic modeling of consciousness-related textual data using BERTopic.

## Structure

MOSAIC/
├── src/     # Core pipeline, data-agnostic
│   ├── preprocessor.py      # Text preprocessing and cleaning
│   ├── model.py            # BERTopic model configuration
│   └── utils.py            # Helper functions and metrics
├── configs/                # Data-specific configurations
│   ├── HS_config.py       # High Sensory parameters
│   ├── DL_config.py       # Deep Listening parameters
│   └── HW_config.py
├── scripts/               # Imports configs based on condition
│   ├── grid_search.py      # Hyperparameter optimization, 
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