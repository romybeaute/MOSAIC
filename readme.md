# MOSAIC: Mapping Of Subjective Accounts into Interpreted Clusters

A comprehensive topic modeling pipeline for consciousness-related textual data across multiple datasets, using BERTopic, BERT embeddings, and UMAP-HDBSCAN clustering with multilingual support.

## Overview

MOSAIC is a research framework that analyzes subjective experiential reports from various consciousness studies through:
- Advanced NLP with BERT embeddings and multilingual models
- Dimensionality reduction via UMAP
- Density-based clustering with HDBSCAN
- Hyperparameter optimization with Optuna
- Topic coherence optimization
- Large Language Model integration with Llama CPP for deeper insights
- Support for multiple datasets and languages

## Supported Datasets

This repository contains analysis pipelines for several consciousness research datasets:

- **Dreamachine**: Stroboscopic light-induced altered states of consciousness
- **Inner Speech**: Japanese phenomenological reports on inner speech experiences
- **Depression/MPE**: Mental health and psychological experience reports  
- **NDE**: Near-death experience accounts
- **Ganzfeld**: Sensory deprivation experimental reports

## Project Structure

```
MOSAIC/
├── src/                           # Core functionality
│   ├── preprocessor.py            # Text cleaning, sentence splitting
│   ├── model.py                   # BERTopic configuration
│   ├── utils.py                   # Metrics and helper functions
│   ├── optuna_search.py           # Hyperparameter search with Optuna
│   └── optuna_search_allmetrics.py # Multi-objective optimization
├── configs/                       # Dataset-specific configurations
│   └── dreamachine2.py           # Dreamachine dataset settings
├── preproc/                       # Data preprocessing utilities
│   ├── prepare_data.ipynb        # Data preparation notebook
│   └── preprocess_data_*.ipynb   # Dataset-specific preprocessing
├── scripts/                       # Analysis notebooks and tools
├── EVAL/                          # Model evaluation and analysis
│   ├── dreamachine/              # Dreamachine-specific evaluations
│   │   ├── demographics.ipynb    # Demographic analysis
│   │   └── stability_tests/      # Model stability testing
│   ├── conditions_similarity.ipynb # Cross-condition comparisons
│   └── optuna_search/            # Hyperparameter optimization results
├── MULTILINGUAL/                  # Multilingual analysis pipeline
│   ├── DREAMACHINE/              # Multilingual Dreamachine analysis
│   ├── INNERSPEECH/              # Japanese inner speech analysis
│   │   ├── app.py               # Streamlit dashboard
│   │   ├── app_hosted.py        # Hosted version of dashboard
│   │   └── local_translator.py  # Local translation utilities
│   ├── translate/                # Translation utilities
│   └── prepare_data.ipynb        # Multilingual data preparation
├── DATA/                          # Local data storage
├── pyproject.toml                # Project configuration and dependencies
├── requirements.txt              # Python dependencies
└── .mosaicvenv/                  # Virtual environment
```

## Key Features

### Core Analysis Pipeline
- **Preprocessing**: Text cleaning, sentence splitting, duplicate removal
- **Embedding**: Support for multiple transformer models (Qwen, E5, BGE, etc.)
- **Clustering**: UMAP dimensionality reduction + HDBSCAN clustering  
- **Topic Modeling**: BERTopic with custom representation models
- **Evaluation**: Coherence metrics, stability testing, bootstrap analysis

### Multilingual Support
- Translation pipelines for non-English datasets
- Support for Japanese text processing
- API-based and local translation options using Llama models

### Interactive Dashboards
- Streamlit applications for real-time analysis ([`MULTILINGUAL/INNERSPEECH/app.py`](MULTILINGUAL/INNERSPEECH/app.py))
- Parameter tuning interfaces
- Visualization tools with datamapplot integration

### Hyperparameter Optimization
- Optuna-based search for optimal model parameters
- Multi-objective optimization across multiple metrics
- Dataset-specific parameter spaces

## Installation

1. Clone the repository:
```bash
git clone https://github.com/romybeaute/MOSAIC.git
cd MOSAIC
```

2. Create and activate virtual environment:
```bash
python3 -m venv .mosaicvenv
source .mosaicvenv/bin/activate  # On Windows: .mosaicvenv\Scripts\activate
```

3. Install the package in development mode:
```bash
pip install -e .
```

This will install all dependencies specified in [`pyproject.toml`](pyproject.toml) and make the MOSAIC package available for import.

## Usage

### Basic Analysis
The primary way to use MOSAIC is through Jupyter notebooks in the [`scripts/`](scripts/) directory and dataset-specific folders:

```bash
# Navigate to analysis notebooks
cd EVAL/dreamachine/
jupyter lab demographics.ipynb
```

### Hyperparameter Optimization
Run hyperparameter search using Optuna:

```bash
python src/optuna_search.py --dataset dreamachine --condition DL --sentences --n_trials 200
```

Parameters:
- `--dataset`: Dataset name (dreamachine, innerspeech, etc.)
- `--condition`: Experimental condition (HS, DL, HW)
- `--sentences`: Enable sentence-level analysis
- `--n_trials`: Number of optimization trials

### Interactive Dashboard
Launch the Streamlit dashboard for interactive analysis:

```bash
streamlit run MULTILINGUAL/INNERSPEECH/app.py
```

### Translation Pipeline
For multilingual datasets, use the translation utilities:

```bash
python MULTILINGUAL/translate/local_translator.py \
    --dataset nde \
    --input-csv NDE_reflection_reports.csv \
    --text-column reflection_answer \
    --model llama \
    --task translate \
    --num-samples 100
```

## Configuration

Dataset-specific configurations are stored in the [`configs/`](configs/) directory. Example configuration for Dreamachine dataset in [`configs/dreamachine2.py`](configs/dreamachine2.py):

```python
class DreamachineConfig:
    def __init__(self):
        self.transformer_model = "Qwen/Qwen3-Embedding-0.6B"
        self.ngram_range = (1, 3)
        self.top_n_words = 15
        # ... other parameters
```

## Development

The project uses a modular structure:
- Core functionality in [`src/`](src/)
- Dataset-specific code in respective folders
- Shared utilities in [`DATA/helpers.py`](DATA/helpers.py)
- Path management through `mosaic.path_utils`

## Citation

If using this code, please cite:
- Beauté, R. et al. (2024). Analysing the phenomenology of stroboscopically induced phenomena using natural language topic modelling
- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure

## License

See [`LICENSE`](LICENSE) for details.