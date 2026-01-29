#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : optuna_search.py
# description     : Run Optuna multi-objective optimization for BERTopic hyperparameters
#                   - Works with any preprocessed dataset
#                   - OPTIONAL: Use dataset config for transformer, stop_words, ngram_range
#                   - OPTIONAL: Custom search space from config
# author          : Romy Beauté (r.beaut@sussex.ac.uk)
# date            : 2026-01-29
# version         : 4 (With Config Support)
# usage           : python optuna_search.py --dataset dreamachine_DL --use-config --sentences --n_trials 100
# python_version  : 3.12.3
# ==============================================================================

import argparse
import pandas as pd
import os
import sys
import time
import csv
import importlib
from pathlib import Path
import optuna
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from optuna.samplers import NSGAIISampler

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from .model import run_bertopic
from .preprocessing.preprocessing import split_sentences, basic_preprocess

os.environ["TOKENIZERS_PARALLELISM"] = "True"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


class OptunaSearchBERTopic:
    def __init__(self, dataset="dreamachine_DL", use_config=False, condition=None, use_sentences=True):
        """
        Initialize Optuna search for any preprocessed dataset.
        
        Args:
            dataset: Dataset name (e.g., 'dreamachine_DL', 'MPE', 'innerspeech')
            use_config: Whether to load config file for this dataset (mosaic/configs/{dataset}.py)
            condition: Condition/variant (e.g., 'DL', 'HS', or None)
            use_sentences: Whether to split into sentences
        
        Config files (mosaic/configs/{dataset}.py) can contain:
        - transformer_model: Which embedding model to use
        - ngram_range: N-gram range for vectorizer
        - extended_stop_words: Custom stop words
        - max_df / min_df: Vectorizer thresholds
        - search_space: Custom search space for Optuna (optional)
        
        If use_config=True and config exists, these values override defaults.
        If use_config=False or config doesn't exist, hardcoded defaults are used.
        """
        self.dataset = dataset
        self.use_config = use_config
        self.condition = condition
        self.use_sentences = use_sentences
        self.top_n_words = 15
        self.random_seed = 42

        # Load config if requested
        self.config = None
        if self.use_config:
            self.config = self._load_config()
        
        # Setup vectorizer and embedding settings from config or defaults
        self._setup_vectorizer_settings()
        self._setup_embedding_model()
        
        # Load search space from config if available
        self.search_space = self._load_search_space_from_config() if self.use_config else None
        
        self.data = None
        self.embeddings = None
        self.vectorizer_model = None
        
        self.setup_paths()
        self.setup_models()

    def _load_config(self):
        """
        Load dataset-specific config file.
        
        Config file location: mosaic/configs/{dataset}.py
        Must have a 'config' object with attributes like:
        - transformer_model
        - ngram_range
        - extended_stop_words
        - max_df, min_df
        - search_space (optional)
        
        Returns:
            Config object, or None if not found
        """
        try:
            config_module = importlib.import_module(f'mosaic.configs.{self.dataset}')
            print(f"✓ Loaded config from mosaic/configs/{self.dataset}.py")
            return config_module.config
        except (ImportError, AttributeError) as e:
            print(f"⚠ Config file not found: mosaic/configs/{self.dataset}.py")
            print(f"  Using default parameters instead")
            return None

    def _setup_vectorizer_settings(self):
        """Setup vectorizer parameters from config or defaults."""
        if self.config and hasattr(self.config, 'transformer_model'):
            self.transformer_model_name = self.config.transformer_model
            print(f"  Using transformer from config: {self.transformer_model_name}")
        else:
            self.transformer_model_name = "sentence-transformers/all-MiniLM-L6-v2"
            print(f"  Using default transformer: {self.transformer_model_name}")
        
        if self.config and hasattr(self.config, 'ngram_range'):
            self.ngram_range = self.config.ngram_range
            print(f"  Using ngram_range from config: {self.ngram_range}")
        else:
            self.ngram_range = (1, 2)
            print(f"  Using default ngram_range: {self.ngram_range}")
        
        if self.config and hasattr(self.config, 'extended_stop_words'):
            self.stop_words = self.config.extended_stop_words
            print(f"  Using {len(self.stop_words)} stop words from config")
        else:
            self.stop_words = 'english'  # sklearn default
            print(f"  Using sklearn's default English stop words")
        
        if self.config and hasattr(self.config, 'max_df'):
            self.max_df = self.config.max_df
        else:
            self.max_df = 0.95
        
        if self.config and hasattr(self.config, 'min_df'):
            self.min_df = self.config.min_df
        else:
            self.min_df = 2

    def _setup_embedding_model(self):
        """Setup embedding model from config or defaults."""
        # This will be initialized in setup_models()
        pass

    def _load_search_space_from_config(self):
        """
        Load custom search space from config if available.
        
        Returns:
            Dict with search space ranges, or None to use defaults
        """
        if self.config and hasattr(self.config, 'search_space'):
            print(f"✓ Using custom search space from config")
            return self.config.search_space
        return None

    def setup_paths(self):
        """Setup data and results paths based on dataset and condition."""
        preprocessed_dir = Path("DATA") / "preprocessed"
        
        self.data_path = self._find_preprocessed_file(preprocessed_dir)
        
        if not self.data_path:
            raise FileNotFoundError(
                f"No preprocessed file found for dataset '{self.dataset}' in {preprocessed_dir}"
            )
        
        print(f"Using preprocessed data: {self.data_path}")
        
        # Setup results path
        config_suffix = "_with_config" if self.use_config and self.config else ""
        transformer_model_name = "custom" if self.use_config and self.config else "minilm"
        
        self.results_path = os.path.join(
            f"results/optuna", 
            f"OPTUNA_results_{self.dataset}"
            + ('_sentences' if self.use_sentences else '')
            + config_suffix
            + f'_{transformer_model_name}.csv'
        )
        
        self.study_db_path = os.path.join(
            f"results/optuna",
            f"optuna_study_{self.dataset}_{transformer_model_name}_multiobj.db"
        )
        
        print(f"Results will be saved to: {self.results_path}")

    def _find_preprocessed_file(self, preprocessed_dir: Path) -> Path:
        """Find a preprocessed CSV file matching the dataset name."""
        if not preprocessed_dir.exists():
            return None
        
        patterns = [
            f"{self.dataset}_preprocessed.csv",
            f"{self.dataset}_cleaned_llama.csv",
            f"{self.dataset}_cleaned_API.csv",
        ]
        
        for pattern in patterns:
            file_path = preprocessed_dir / pattern
            if file_path.exists():
                return file_path
        
        matches = list(preprocessed_dir.glob(f"{self.dataset}*.csv"))
        if matches:
            return matches[0]
        
        return None

    def setup_models(self):
        """Setup embedding and vectorizer models."""
        self.embedding_model = SentenceTransformer(self.transformer_model_name)
        self.vectorizer_model = CountVectorizer(
            ngram_range=self.ngram_range,
            stop_words=self.stop_words if isinstance(self.stop_words, str) else list(self.stop_words),
            max_df=self.max_df,
            min_df=self.min_df,
            lowercase=True
        )

    def load_data(self):
        """Load data from preprocessed CSV."""
        df = pd.read_csv(self.data_path)
        
        if 'cleaned_reflection' in df.columns:
            texts = df['cleaned_reflection'].dropna().reset_index(drop=True)
        elif 'reflection_answer' in df.columns:
            texts = df['reflection_answer'].dropna().reset_index(drop=True)
        elif 'sentences' in df.columns:
            texts = df['sentences'].dropna().reset_index(drop=True)
        else:
            text_cols = [col for col in df.columns if col not in ['id', 'index']]
            if text_cols:
                texts = df[text_cols[0]].dropna().reset_index(drop=True)
            else:
                raise ValueError(f"No suitable text column found in {self.data_path}")
        
        print(f"Loaded {len(texts)} texts from {self.data_path}")
        
        if self.use_sentences:
            print("Splitting into sentences...")
            texts, _ = split_sentences(texts.tolist())
            min_words = 2
            texts = [s for s in texts if len(s.split()) >= min_words]
            
            seen = set()
            texts = [s for s in texts if not (s in seen or seen.add(s))]
            
            print(f"After sentence splitting: {len(texts)} sentences")
        
        return texts

    def initialize_results_file(self):
        """Init results CSV with headers for multi-objective optimization."""
        os.makedirs(os.path.dirname(self.results_path), exist_ok=True)
        if not os.path.exists(self.results_path):
            pd.DataFrame(columns=[
                'trial_number', 'objective_embed_coherence', 'objective_cv',
                'n_components', 'n_neighbors', 'min_dist', 'min_cluster_size', 'min_samples',
                'embedding_coherence_attr', 'coherence_score_cv_attr',
                'n_topics'
            ]).to_csv(self.results_path, index=False)

    def _get_default_search_space(self, trial):
        """Default search space (used if no config)."""
        return {
            'n_components': trial.suggest_int('n_components', 5, 25),
            'n_neighbors': trial.suggest_int('n_neighbors', 10, 35),
            'min_dist': trial.suggest_float('min_dist', 0.0, 0.05, step=0.005),
            'min_cluster_size': trial.suggest_int('min_cluster_size', 10, 50),
            'min_samples': trial.suggest_int('min_samples', 5, 25),
        }

    def _define_search_space(self, trial):
        """
        Define hyperparameter search space.
        Uses custom search space from config if available, otherwise uses defaults.
        """
        if self.search_space:
            # Use custom search space from config
            return {
                'n_components': trial.suggest_int('n_components', 
                                                  self.search_space['n_components'][0],
                                                  self.search_space['n_components'][1]),
                'n_neighbors': trial.suggest_int('n_neighbors',
                                                 self.search_space['n_neighbors'][0],
                                                 self.search_space['n_neighbors'][1]),
                'min_dist': trial.suggest_float('min_dist',
                                                self.search_space['min_dist'][0],
                                                self.search_space['min_dist'][1],
                                                step=0.005),
                'min_cluster_size': trial.suggest_int('min_cluster_size',
                                                      self.search_space['min_cluster_size'][0],
                                                      self.search_space['min_cluster_size'][1]),
                'min_samples': trial.suggest_int('min_samples',
                                                 self.search_space['min_samples'][0],
                                                 self.search_space['min_samples'][1]),
            }
        else:
            return self._get_default_search_space(trial)

    def objective(self, trial):
        """The objective function for Optuna to optimize."""
        try:
            params = self._define_search_space(trial)

            model, topics, coherence_score, embedding_coherence = run_bertopic(
                data=self.data, embeddings=self.embeddings, vectorizer_model=self.vectorizer_model,
                embedding_model=self.embedding_model, n_neighbors=params['n_neighbors'],
                n_components=params['n_components'], min_dist=params['min_dist'],
                min_cluster_size=params['min_cluster_size'], min_samples=params['min_samples'],
                top_n_words=self.top_n_words, random_seed=self.random_seed
            )

            embedding_coherence = float(embedding_coherence)
            coherence_score = float(coherence_score)

            trial.set_user_attr('embedding_coherence', embedding_coherence)
            trial.set_user_attr('coherence_score', coherence_score)
            trial.set_user_attr('n_topics', len(set(topics)))

            return embedding_coherence, coherence_score
            
        except Exception as e:
            print(f"Error in trial {trial.number} with parameters {trial.params}: {str(e)}")
            raise optuna.exceptions.TrialPruned()

    def save_callback(self, study, trial):
        """Callback function to save results of each trial to a CSV file."""
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        
        result_row = [
            trial.number,
            trial.values[0],
            trial.values[1],
            trial.params['n_components'], trial.params['n_neighbors'],
            trial.params['min_dist'], trial.params['min_cluster_size'],
            trial.params['min_samples'],
            trial.user_attrs['embedding_coherence'],
            trial.user_attrs['coherence_score'],
            trial.user_attrs['n_topics']
        ]
        
        with open(self.results_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(result_row)

    def run_optimization(self, n_trials=100):
        """Runs the multi-objective Optuna optimization."""
        print(f"\n{'='*80}")
        print(f"OPTUNA OPTIMIZATION FOR {self.dataset.upper()}")
        if self.use_config and self.config:
            print(f"(Using config from mosaic/configs/{self.dataset}.py)")
        print(f"{'='*80}\n")
        
        self.data = self.load_data()
        print("\nGenerating sentence embeddings...")
        self.embeddings = self.embedding_model.encode(self.data, show_progress_bar=True)
        print("Embeddings generated.")

        self.initialize_results_file()
        
        study_name = f"bertopic-{self.dataset}-multiobj-optimization"
        storage_name = f"sqlite:///{self.study_db_path}"

        try:
            study = optuna.load_study(study_name=study_name, storage=storage_name)
            print(f"\nLoaded existing study with {len(study.trials)} trials.")
        except KeyError:
            print(f"\nCreating new study for {self.dataset}.")
            sampler = NSGAIISampler(seed=self.random_seed)
            directions = ['maximize', 'maximize']

            study = optuna.create_study(
                study_name=study_name,
                storage=storage_name,
                sampler=sampler,
                directions=directions)
                
        start_time = time.time()
        study.optimize(
            self.objective, 
            n_trials=n_trials, 
            callbacks=[self.save_callback]
        )
        
        elapsed = time.time() - start_time
        print(f"\nOptimization completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

        print("\n" + "="*80)
        print("PARETO FRONT (BEST TRADE-OFF TRIALS)")
        print("="*80)
        print(f"Found {len(study.best_trials)} optimal trials.\n")
        
        for i, trial in enumerate(study.best_trials):
            print(f"Solution {i+1} (Trial {trial.number}):")
            print(f"  Embedding Coherence: {trial.values[0]:.4f} ↑")
            print(f"  C_v Coherence:       {trial.values[1]:.4f} ↑")
            print(f"  # Topics:            {trial.user_attrs['n_topics']}")
            print("  Parameters:")
            for key, value in sorted(trial.params.items()):
                print(f"    • {key}: {value}")
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run BERTopic multi-objective optimization with Optuna",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Without config (uses defaults)
  python optuna_search.py --dataset dreamachine_DL --sentences --n_trials 100
  
  # With config (reads transformer, stop_words, ngram_range from mosaic/configs/dreamachine_DL.py)
  python optuna_search.py --dataset dreamachine_DL --use-config --sentences --n_trials 100
  
  # Test with small number of trials
  python optuna_search.py --dataset dreamachine_DL --use-config --sentences --n_trials 5
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='dreamachine_DL',
        help='Dataset name (e.g., dreamachine_DL, MPE, innerspeech)'
    )
    parser.add_argument(
        '--use-config',
        action='store_true',
        help='Load config from mosaic/configs/{dataset}.py if available'
    )
    parser.add_argument(
        '--condition',
        type=str,
        default=None,
        help='Condition/variant (optional, for results naming)'
    )
    parser.add_argument(
        '--sentences',
        action='store_true',
        help='Split text into sentences before optimization'
    )
    parser.add_argument(
        '--n_trials',
        type=int,
        default=100,
        help='Number of optimization trials (default: 100)'
    )
    
    args = parser.parse_args()

    try:
        optuna_search = OptunaSearchBERTopic(
            dataset=args.dataset,
            use_config=args.use_config,
            condition=args.condition,
            use_sentences=args.sentences
        )
        
        optuna_search.run_optimization(n_trials=args.n_trials)
    
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
