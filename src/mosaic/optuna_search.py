#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : optuna_search.py
# description     : Run Optuna multi-objective optimization for BERTopic
#                   - Prioritizes Config file > Condition Defaults > Generic Defaults
#                   - Auto-discovers best data file (API > Llama > Raw)
# author          : Romy Beauté (corrected by Coding Partner)
# date            : 2026-01-30
# version         : 5 (Merged & Fixed)
# usage           : python optuna_search.py --dataset dreamachine --condition DL --use-config --sentences --n_trials 100
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

# Handle local vs installed package imports
try:
    from mosaic.model import run_bertopic
    from mosaic.preprocessing.preprocessing import split_sentences
except ImportError:
    from src.mosaic.model import run_bertopic
    from src.mosaic.preprocessing.preprocessing import split_sentences

os.environ["TOKENIZERS_PARALLELISM"] = "True"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


class OptunaSearchBERTopic:
    def __init__(self, dataset="dreamachine", use_config=True, condition=None, use_sentences=True):
        self.dataset = dataset
        self.use_config = use_config
        self.condition = condition
        self.use_sentences = use_sentences
        self.random_seed = 42
        
        # 1. Load Configuration
        self.config = self._load_config() if self.use_config else None
        
        # 2. Setup Settings (Prioritizing Config)
        self._setup_embedding_settings()
        self._setup_vectorizer_settings()
        
        # 3. Setup Paths (Smart Discovery)
        self.data_path = None
        self.results_path = None
        self.study_db_path = None
        self.setup_paths()
        
        # 4. Initialize Models
        self.setup_models()

    def _load_config(self):
        """Dynamically load the dataset configuration."""
        try:
            # Assumes mosaic/configs/dataset.py structure
            module_name = f"mosaic.configs.{self.dataset}"
            config_module = importlib.import_module(module_name)
            print(f"Loaded config from {module_name}")
            return config_module.config
        except ImportError:
            print(f"Config for '{self.dataset}' not found. Using defaults.")
            return None
        except AttributeError:
            print(f"Module loaded but 'config' object missing. Using defaults.")
            return None

    def _setup_embedding_settings(self):
        """Determine which transformer model to use."""
        if self.config and hasattr(self.config, 'transformer_model'):
            self.transformer_model_name = self.config.transformer_model
            print(f"  - Model (Config): {self.transformer_model_name}")
        else:
            self.transformer_model_name = "Qwen/Qwen3-Embedding-0.6B"
            print(f"  - Model (Default): {self.transformer_model_name}")

    def _setup_vectorizer_settings(self):
        """Setup vectorizer parameters."""
        # N-grams
        if self.config and hasattr(self.config, 'ngram_range'):
            self.ngram_range = self.config.ngram_range
        else:
            self.ngram_range = (1, 2)

        # Stopwords
        if self.config and hasattr(self.config, 'extended_stop_words'):
            self.stop_words = list(self.config.extended_stop_words)
        else:
            self.stop_words = 'english'

        # DF Thresholds
        self.max_df = getattr(self.config, 'max_df', 0.95)
        self.min_df = getattr(self.config, 'min_df', 2)
        self.top_n_words = getattr(self.config, 'top_n_words', 15)

    def _find_preprocessed_file(self, preprocessed_dir: Path) -> Path:
        """
        Smart discovery of the best data file.
        PRIORITY: API Cleaned > Llama Cleaned > Generic Preprocessed
        """
        if not preprocessed_dir.exists():
            return None
        
        # Priority list (Highest quality first)
        patterns = [
            f"{self.dataset}_cleaned_API.csv",                              # Generic API
            f"{self.dataset}_{self.condition}_cleaned_llama.csv",
            f"{self.dataset}_cleaned_llama.csv",
            f"{self.dataset}_preprocessed.csv"
        ]
        
        # Check specific patterns
        for pattern in patterns:
            file_path = preprocessed_dir / pattern
            if file_path.exists():
                # Double check if other files exist to warn user
                all_matches = list(preprocessed_dir.glob(f"{self.dataset}*.csv"))
                if len(all_matches) > 1:
                    print(f"  ! Note: Multiple files found for '{self.dataset}'.")
                    print(f"  ! Selected highest priority: {pattern}")
                return file_path
        
        # Fallback: Wildcard search
        matches = list(preprocessed_dir.glob(f"{self.dataset}*.csv"))
        if matches:
            print(f"  ! Warning: No standard filename match. Using found file: {matches[0].name}")
            return matches[0]
            
        return None

    def setup_paths(self):
        # 1. Data Path
        # robustly find the project root by looking 3 levels up
        current_dir = Path(__file__).resolve().parent
        possible_roots = [
            current_dir / "DATA",                   # Same dir
            current_dir.parent / "DATA",            # src/DATA
            current_dir.parent.parent / "DATA",     # MOSAIC/DATA (Likely correct)
            current_dir.parent.parent.parent / "DATA" 
        ]
        
        data_file = None
        
        # Scan all possible DATA locations
        for root in possible_roots:
            if not root.exists():
                continue
                
            # Check root/{dataset} folder
            if (root / self.dataset).exists():
                found = self._find_preprocessed_file(root / self.dataset)
                if found:
                    data_file = found
                    print(f"  (Found data in: {root / self.dataset})")
                    break
                    
            # Check root/preprocessed folder
            if (root / "preprocessed").exists():
                found = self._find_preprocessed_file(root / "preprocessed")
                if found:
                    data_file = found
                    print(f"  (Found data in: {root / 'preprocessed'})")
                    break
            
            # Check root directly
            found = self._find_preprocessed_file(root)
            if found:
                data_file = found
                break

        if not data_file:
            # Print where we looked to help debug
            checked = [str(p) for p in possible_roots]
            raise FileNotFoundError(f"Could not find data for '{self.dataset}'. \nChecked in: {checked}")
            
        self.data_path = str(data_file)
        print(f"✓ Using data file: {self.data_path}")

        # 2. Results Path (Save relative to the found DATA root's parent, i.e., project root)
        # If we found data at MOSAIC/DATA, we want results at MOSAIC/results/optuna
        project_root = Path(data_file).parent.parent 
        if "preprocessed" in str(Path(data_file).parent): # Handle preprocessed subfolder
             project_root = project_root.parent

        sanitized_model = self.transformer_model_name.replace('/', '_')
        
        # Fallback if project root calculation is weird, use current script relative path
        if not project_root.exists():
            project_root = Path("results/optuna")
        else:
            project_root = project_root / "results/optuna"
            
        results_dir = project_root / self.dataset
        results_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_path = str(results_dir / f"OPTUNA_{self.condition}_{sanitized_model}_results.csv")
        self.study_db_path = str(results_dir / f"OPTUNA_{self.condition}_{sanitized_model}.db")
        
        print(f"✓ Results will be saved to: {self.results_path}")

    def setup_models(self):
        self.embedding_model = SentenceTransformer(self.transformer_model_name)
        self.vectorizer_model = CountVectorizer(
            ngram_range=self.ngram_range,
            stop_words=self.stop_words,
            max_df=self.max_df,
            min_df=self.min_df,
            lowercase=True
        )

    def load_data(self):
        df = pd.read_csv(self.data_path)
        # Try to find the text column intelligently
        cols = df.columns
        text_col = next((c for c in ['cleaned_reflection', 'reflection_answer', 'text'] if c in cols), None)
        
        if not text_col:
            raise ValueError(f"Could not find text column in {cols}")
            
        texts = df[text_col].dropna().reset_index(drop=True)
        
        if self.use_sentences:
            texts, _ = split_sentences(texts.tolist())
            min_words = 2
            texts = [s for s in texts if len(s.split()) >= min_words]
            # Deduplicate
            seen = set()
            texts = [s for s in texts if not (s in seen or seen.add(s))]
            
        return texts

    def initialize_results_file(self):
        if not os.path.exists(self.results_path):
            pd.DataFrame(columns=[
                'trial_number', 'objective_embed_coherence', 'objective_cv',
                'n_components', 'n_neighbors', 'min_dist', 'min_cluster_size', 'min_samples',
                'embedding_coherence_attr', 'coherence_score_cv_attr', 'n_topics'
            ]).to_csv(self.results_path, index=False)

    def _define_search_space(self, trial):
        """
        Priority:
        1. Config 'search_space' (if exists)
        2. Config 'get_default_params(condition)' (if exists - fixed point, not range, treated as narrow range)
        3. Hardcoded defaults based on Condition (DL vs HS)
        """
        
        # 1. Try Config Search Space
        if self.config and hasattr(self.config, 'search_space') and self.config.search_space:
            s = self.config.search_space
            # Helper to check if it's a range tuple or single value
            def get_range(key, default_low, default_high):
                val = s.get(key)
                if isinstance(val, (tuple, list)): return val[0], val[1]
                return default_low, default_high

            return {
                'n_components': trial.suggest_int('n_components', *get_range('n_components', 5, 20)),
                'n_neighbors': trial.suggest_int('n_neighbors', *get_range('n_neighbors', 10, 35)),
                'min_dist': trial.suggest_float('min_dist', *get_range('min_dist', 0.0, 0.1), step=0.005),
                'min_cluster_size': trial.suggest_int('min_cluster_size', *get_range('min_cluster_size', 10, 50)),
                'min_samples': trial.suggest_int('min_samples', *get_range('min_samples', 5, 20)),
            }

        # 2. Hardcoded Defaults based on Condition (Fallback)
        # This matches your Dreamachine requirements
        if self.condition == 'DL':
            return {
                'n_components': trial.suggest_int('n_components', 5, 15),
                'n_neighbors': trial.suggest_int('n_neighbors', 5, 15),
                'min_dist': trial.suggest_float('min_dist', 0.0, 0.05, step=0.005),
                'min_cluster_size': trial.suggest_int('min_cluster_size', 7, 10),
                'min_samples': trial.suggest_int('min_samples', 5, 10),
            }
        elif self.condition == 'HS':
            return {
                'n_components': trial.suggest_int('n_components', 5, 20),
                'n_neighbors': trial.suggest_int('n_neighbors', 5, 25),
                'min_dist': trial.suggest_float('min_dist', 0.0, 0.05, step=0.005),
                'min_cluster_size': trial.suggest_int('min_cluster_size', 8, 20),
                'min_samples': trial.suggest_int('min_samples', 5, 15),
            }
        else:
            # Generic
            return {
                'n_components': trial.suggest_int('n_components', 5, 30),
                'n_neighbors': trial.suggest_int('n_neighbors', 10, 40),
                'min_dist': trial.suggest_float('min_dist', 0.0, 0.1, step=0.005),
                'min_cluster_size': trial.suggest_int('min_cluster_size', 10, 50),
                'min_samples': trial.suggest_int('min_samples', 5, 25),
            }

    def objective(self, trial):
        try:
            params = self._define_search_space(trial)

            model, topics, coherence_score, embedding_coherence = run_bertopic(
                data=self.data, 
                embeddings=self.embeddings, 
                vectorizer_model=self.vectorizer_model,
                embedding_model=self.embedding_model, 
                n_neighbors=params['n_neighbors'],
                n_components=params['n_components'], 
                min_dist=params['min_dist'],
                min_cluster_size=params['min_cluster_size'], 
                min_samples=params['min_samples'],
                top_n_words=self.top_n_words, 
                random_seed=self.random_seed
            )

            embedding_coherence = float(embedding_coherence)
            coherence_score = float(coherence_score)

            # Store metrics
            trial.set_user_attr('embedding_coherence', embedding_coherence)
            trial.set_user_attr('coherence_score', coherence_score)
            trial.set_user_attr('n_topics', len(set(topics)))

            return embedding_coherence, coherence_score
            
        except Exception as e:
            # Prune trials that fail (e.g., cannot find topics)
            # print(f"Trial {trial.number} pruned: {e}")
            raise optuna.exceptions.TrialPruned()

    def save_callback(self, study, trial):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        
        result_row = [
            trial.number,
            trial.values[0],
            trial.values[1],
            trial.params['n_components'], trial.params['n_neighbors'],
            trial.params['min_dist'], trial.params['min_cluster_size'],
            trial.params['min_samples'],
            trial.user_attrs.get('embedding_coherence', 0),
            trial.user_attrs.get('coherence_score', 0),
            trial.user_attrs.get('n_topics', 0)
        ]
        
        with open(self.results_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(result_row)

    def run_optimization(self, n_trials=100):
        self.data = self.load_data()
        print("Generating embeddings...")
        self.embeddings = self.embedding_model.encode(self.data, show_progress_bar=True)
        
        self.initialize_results_file()
        
        study_name = f"{self.dataset}-{self.condition}-multiobj"
        storage_name = f"sqlite:///{self.study_db_path}"

        try:
            study = optuna.load_study(study_name=study_name, storage=storage_name)
            print(f"Resuming study '{study_name}' ({len(study.trials)} trials).")
        except KeyError:
            print(f"Starting new study '{study_name}'.")
            sampler = NSGAIISampler(seed=self.random_seed)
            study = optuna.create_study(
                study_name=study_name,
                storage=storage_name,
                sampler=sampler,
                directions=['maximize', 'maximize']
            )
                
        study.optimize(self.objective, n_trials=n_trials, callbacks=[self.save_callback])
        
        print("\n--- Pareto Front ---")
        for i, trial in enumerate(study.best_trials):
            print(f"\nSolution {i+1}: Embed_Coh={trial.values[0]:.3f}, CV={trial.values[1]:.3f}")
            print(f"  Params: {trial.params}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dreamachine')
    parser.add_argument('--condition', type=str, default='DL', help='Sub-condition (DL, HS)')
    parser.add_argument('--use-config', action='store_true', default=True, help='Use mosaic.configs')
    parser.add_argument('--sentences', action='store_true', help='Split sentences')
    parser.add_argument('--n_trials', type=int, default=100)
    args = parser.parse_args()

    # Pass args to class
    search = OptunaSearchBERTopic(
        dataset=args.dataset,
        use_config=args.use_config,
        condition=args.condition,
        use_sentences=args.sentences
    )
    
    search.run_optimization(n_trials=args.n_trials)