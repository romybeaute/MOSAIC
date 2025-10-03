
# MOSAIC: Mapping Of Subjective Accounts into Interpreted Clusters

Topic modelling pipeline for consciousness-related textual data using BERTopic, BERT embeddings, and UMAP-HDBSCAN clustering.

## Overview

MOSAIC analyses subjective experiential reports through:
- Advanced NLP with BERT embeddings
- Dimensionality reduction via UMAP
- Density-based clustering with HDBSCAN
- Hyperparameter optimisation with Optuna
- Topic coherence optimisation
- Large Language Model integration with Llama CPP for deeper insights.

## Structure

```
MOSAIC/
├── src/                    # Core functionality
│   ├── preprocessor.py     # Text cleaning, sentence splitting
│   ├── model.py            # BERTopic configuration
│   ├── utils.py            # Metrics and helpers
│   ├── optuna_search.py    # Hyperparameter search with Optuna
│   ├── llama_CPP_custom.py # Integration with Llama CPP
│   ├── prepare_data.ipynb  # Notebook for data preparation
│   └── preprocess_data_API.ipynb # Notebook for preprocessing data from an API
├── configs/                # Experiment parameters
│   └── dreamachine2.py     # Dataset-specific settings
├── scripts/                # Analysis tools
│   ├── dreamachine.ipynb   # Jupyter notebook for analysis
│   └── ...                 # Other analysis notebooks
└── EVAL/                   # Evaluation scripts and results
    ├── conditions_similarity.ipynb
    └── stability_tests/
```

### Source (`src/`)

- preprocessor.py: Text preprocessing and cleaning, and sentence splitting.
- model.py: BERTopic configuration, UMAP dimensionality reduction, and HDBSCAN clustering.
- utils.py: Coherence metrics and helper functions.
- optuna_search.py: Hyperparameter optimisation using Optuna.
- llama_CPP_custom.py: Custom functions for interacting with Llama CPP.
- prepare_data.ipynb: Notebook for preparing the data.
- preprocess_data_API.ipynb: Notebook for preprocessing data from an API.

### Configs (`configs/`)

- `dreamachine2.py`
  - Dataset-specific parameters
  - Model hyperparameters
  - Preprocessing settings

### Scripts (`scripts/`)

- This directory contains Jupyter notebooks for running experiments and analyzing results.


### Scripts (`EVAL/`)
- This directory contains scripts and notebooks for evaluating the model's performance, including stability tests and similarity analyses.



## Installation

```bash
git clone https://github.com/romybeaute/MOSAIC.git
cd MOSAIC
# Create and activate virtual environment
python3 -m venv .mosaicvenv
source .mosaicvenv/bin/activate
pip install -e .



# Install dependencies
pip install pandas sentence-transformers scikit-learn tqdm nltk bertopic umap-learn hdbscan gensim
or
pip install -r requirements.txt
```

## Usage

The primary way to use MOSAIC is through the Jupyter notebooks in the scripts/ directory. These notebooks provide a step-by-step guide for running the topic modeling pipeline, from data preprocessing to model evaluation.

To run the hyperparameter optimisation, you can use the optuna_search.py script:

```
python src/optuna_search.py --dataset your_dataset --condition COND --sentences
```

Parameters:
- `--dataset`: Dataset name
- `--condition`: Experimental condition [HS, DL, HW]
- `--sentences`: Enable sentence-level analysis

## Citation

If using this code, please cite:
- Analysing the phenomenology of stroboscopically induced phenomena using natural language topic modelling (Beauté et al.,2024)
- BERTopic: Neural topic modeling with a class-based TF-IDF procedure (M. Grootendorst, 2022)
```