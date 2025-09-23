#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : optuna_search.py
# description     : Run Optuna optimization for BERTopic hyperparameters
# author          : Romy Beauté (Adapted for Optuna by Coding Partner)
# date            : 2025-09-23
# version         : 1
# usage           : python optuna_search.py --condition DL --sentences --n_trials 100
# python_version  : 3.12.3
# ==============================================================================

import argparse
import pandas as pd
import os
import sys
import time
import csv
import optuna
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

# --- Ensure project root is in the path to import custom modules ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from src.model import run_bertopic
from src.preprocessor import split_sentences

# --- Environment variable setup ---
os.environ["TOKENIZERS_PARALLELISM"] = "True"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

class OptunaSearchBERTopic:
    def __init__(self, dataset="dreamachine", condition="DL", use_sentences=True):
        self.dataset = dataset
        self.condition = condition
        self.use_sentences = use_sentences
        self.top_n_words = 15

        # --- Load appropriate dataset config ---
        if dataset == "dreamachine":
            from configs.dreamachine2 import config # Import config instance
            self.dataset_config = config
        else:
            raise ValueError(f"Unrecognised dataset: {dataset}")
        
        # --- Pre-calculate these to be used in the objective function ---
        self.data = None
        self.embeddings = None
        self.vectorizer_model = None
        
        self.setup_paths()
        self.setup_models()

    def setup_paths(self):
        """Setup data and results paths based on dataset and condition."""
        self.data_path = os.path.join("DATA", f"{self.dataset}/{self.condition}_reflections_APIcleaned.csv")
        self.results_path = os.path.join(
            f"EVAL/{self.dataset}", 
            f"OPTUNA_results_{self.condition}"
            + ('_sentences' if self.use_sentences else '')
            + '.csv'
        )

    def setup_models(self):
        """Initialize models using dataset specific configs."""
        self.embedding_model = SentenceTransformer(self.dataset_config.transformer_model)
        self.vectorizer_model = CountVectorizer(
            ngram_range=self.dataset_config.ngram_range,
            stop_words=list(self.dataset_config.extended_stop_words),
            max_df=self.dataset_config.max_df,
            min_df=self.dataset_config.min_df,
            lowercase=True
        )

    def load_data(self):
        """Load and preprocess data."""
        df_reports = pd.read_csv(self.data_path, sep=',')['cleaned_reflection'].dropna().reset_index(drop=True)
        
        if self.use_sentences:
            df_reports, _ = split_sentences(df_reports)
            print(f"Split into {len(df_reports)} sentences")

            min_words = 2
            df_reports = [s for s in df_reports if len(s.split()) >= min_words]
            print(f"After removing short sentences (<{min_words} words), {len(df_reports)} remain.")

            seen = set()
            df_reports = [s for s in df_reports if not (s in seen or seen.add(s))]
            print(f"After removing duplicates, {len(df_reports)} remain.")
        
        return df_reports

    def initialize_results_file(self):
        """Initialize results CSV with headers."""
        os.makedirs(os.path.dirname(self.results_path), exist_ok=True)
        if not os.path.exists(self.results_path):
            # The 'value' column is now explicitly named 'embedding_coherence' for clarity
            pd.DataFrame(columns=[
                'trial_number', 'embedding_coherence', 'n_components', 'n_neighbors', 'min_dist',
                'min_cluster_size', 'min_samples',
                'coherence_score', 'coherence_score_umass', 'n_topics'
            ]).to_csv(self.results_path, index=False)

    def _define_search_space(self, trial):
        """Define the hyperparameter search space for Optuna based on the condition."""
        if self.condition == 'DL':
            params = {
                'n_components': trial.suggest_int('n_components', 2, 10),
                'n_neighbors': trial.suggest_int('n_neighbors', 5, 15),
                'min_dist': trial.suggest_float('min_dist', 0.0, 0.2, step=0.05),
                'min_cluster_size': trial.suggest_int('min_cluster_size', 3, 15),
                'min_samples': trial.suggest_int('min_samples', 2, 10),
            }
        elif self.condition == 'HS':
            params = {
                'n_components': trial.suggest_int('n_components', 5, 20),
                'n_neighbors': trial.suggest_int('n_neighbors', 10, 30),
                'min_dist': trial.suggest_float('min_dist', 0.0, 0.2, step=0.05),
                'min_cluster_size': trial.suggest_int('min_cluster_size', 5, 20),
                'min_samples': trial.suggest_int('min_samples', 5, 15),
            }
        else: # Generic default
            params = {
                'n_components': trial.suggest_int('n_components', 5, 25),
                'n_neighbors': trial.suggest_int('n_neighbors', 10, 35),
                'min_dist': trial.suggest_float('min_dist', 0.0, 0.2, step=0.05),
                'min_cluster_size': trial.suggest_int('min_cluster_size', 10, 50),
                'min_samples': trial.suggest_int('min_samples', 5, 25),
            }
        return params

    def objective(self, trial):
        """The objective function for Optuna to optimize."""
        try:
            # 1. Suggest hyperparameters
            params = self._define_search_space(trial)

            # 2. Run the BERTopic model
            model, topics, coherence_score, coherence_score_umass, embedding_coherence = run_bertopic(
                data=self.data,
                embeddings=self.embeddings,
                vectorizer_model=self.vectorizer_model,
                embedding_model=self.embedding_model,
                n_neighbors=params['n_neighbors'],
                n_components=params['n_components'],
                min_dist=params['min_dist'],
                min_cluster_size=params['min_cluster_size'],
                min_samples=params['min_samples'],
                top_n_words=self.top_n_words
            )

            # 3. Store additional metrics in the trial for later analysis
            trial.set_user_attr('coherence_score', coherence_score)
            trial.set_user_attr('coherence_score_umass', coherence_score_umass)
            trial.set_user_attr('n_topics', len(set(topics)))
            
            # 4. Return the metric to be maximized
            # This is the value Optuna will work to maximize.
            return embedding_coherence

        except Exception as e:
            print(f"Error in trial {trial.number} with parameters {trial.params}: {str(e)}")
            # Tell Optuna this trial failed so it can be ignored
            raise optuna.exceptions.TrialPruned()

    def save_callback(self, study, trial):
        """Callback function to save results of each trial to a CSV file."""
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return

        # Prepare data for CSV
        result_row = [
            trial.number,
            trial.value,  # This is the embedding_coherence score returned from the objective function
            trial.params['n_components'],
            trial.params['n_neighbors'],
            trial.params['min_dist'],
            trial.params['min_cluster_size'],
            trial.params['min_samples'],
            trial.user_attrs['coherence_score'],
            trial.user_attrs['coherence_score_umass'],
            trial.user_attrs['n_topics']
        ]
        
        # Append to the CSV file
        with open(self.results_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(result_row)

    def run_optimization(self, n_trials=100):
        """Execute the Optuna optimization."""
        self.data = self.load_data()
        print("Generating sentence embeddings...")
        self.embeddings = self.embedding_model.encode(self.data, show_progress_bar=True)
        print("Embeddings generated.")

        self.initialize_results_file()

        # Create a study object and specify the direction as 'maximize'
        study = optuna.create_study(direction='maximize')
        
        start_time = time.time()
        study.optimize(
            self.objective, 
            n_trials=n_trials, 
            callbacks=[self.save_callback]
        )
        
        print(f"\nOptimization completed in {time.time() - start_time:.2f} seconds")

        # Print the best results
        print("\n--- Best Trial Found ---")
        best_trial = study.best_trial
        print(f"  🏆 Score (Embedding Coherence): {best_trial.value:.4f}")
        print("  Best Parameters: ")
        for key, value in best_trial.params.items():
            print(f"    - {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BERT topic modeling with Optuna optimization")
    parser.add_argument('--dataset', type=str, default='dreamachine', help='Dataset to use')
    parser.add_argument('--condition', type=str, default='DL', choices=['HS', 'DL', 'HW'])
    parser.add_argument('--sentences', action='store_true', help='Split text into sentences')
    parser.add_argument('--n_trials', type=int, default=100, help='Number of optimization trials')
    args = parser.parse_args()

    optuna_search = OptunaSearchBERTopic(
        dataset=args.dataset,
        condition=args.condition,
        use_sentences=args.sentences
    )
    
    optuna_search.run_optimization(n_trials=args.n_trials)

    # python src/optuna_search.py --dataset dreamachine --condition DL --sentences --n_trials 2