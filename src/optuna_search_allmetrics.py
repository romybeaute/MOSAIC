#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : optuna_search.py
# description     : Run Optuna multi-objective optimization for BERTopic hyperparameters
# author          : Romy Beauté (Adapted for Optuna by Coding Partner)
# date            : 2025-09-23
# version         : 2 (Multi-Objective)
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
# CHANGED: Import the multi-objective sampler
from optuna.samplers import TPESampler, NSGAIISampler

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

# Make sure you have the updated utils.py with this function
from src.model import run_bertopic
from preproc.preprocessor import split_sentences

os.environ["TOKENIZERS_PARALLELISM"] = "True"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

class OptunaSearchBERTopic:
    def __init__(self, dataset="dreamachine", condition="DL", use_sentences=True):
        self.dataset = dataset
        self.condition = condition
        self.use_sentences = use_sentences
        self.top_n_words = 15
        self.random_seed = 42

        if dataset == "dreamachine":
            from configs.dreamachine2 import config 
            self.dataset_config = config
        else:
            raise ValueError(f"Unrecognised dataset: {dataset}")
        
        self.data = None
        self.embeddings = None
        self.vectorizer_model = None
        
        self.setup_paths()
        self.setup_models()

    def setup_paths(self):
        """Setup data and results paths based on dataset and condition."""
        self.data_path = os.path.join("DATA", f"{self.dataset}/{self.condition}_reflections_APIcleaned.csv")
        sanitized_model_name = self.dataset_config.transformer_model.replace('/', '_')

        # CHANGED: Updated filename for multi-objective results
        self.results_path = os.path.join(
            f"EVAL/{self.dataset}", 
            f"OPTUNA_results_{self.condition}"
            + ('_sentences' if self.use_sentences else '')
            + f'_multiobj_{sanitized_model_name}.csv'
        )
        
        self.study_db_path = os.path.join(
            f"EVAL/{self.dataset}",
            f"optuna_study_{self.condition}_{sanitized_model_name}_allmeasures.db"
        )
        print(f"Path to Optuna DB: {self.study_db_path}")

    def setup_models(self):
        self.embedding_model = SentenceTransformer(self.dataset_config.transformer_model)
        self.vectorizer_model = CountVectorizer(
            ngram_range=self.dataset_config.ngram_range,
            stop_words=list(self.dataset_config.extended_stop_words),
            max_df=self.dataset_config.max_df,
            min_df=self.dataset_config.min_df,
            lowercase=True
        )

    def load_data(self):
        df_reports = pd.read_csv(self.data_path, sep=',')['cleaned_reflection'].dropna().reset_index(drop=True)
        if self.use_sentences:
            df_reports, _ = split_sentences(df_reports)
            min_words = 2
            df_reports = [s for s in df_reports if len(s.split()) >= min_words]
            seen = set()
            df_reports = [s for s in df_reports if not (s in seen or seen.add(s))]
        return df_reports

    def initialize_results_file(self):
        """Init results CSV with headers for multi-objective optimization."""
        os.makedirs(os.path.dirname(self.results_path), exist_ok=True)
        if not os.path.exists(self.results_path):
            # CHANGED: Update headers for multi-objective results and new metrics
            pd.DataFrame(columns=[
                'trial_number', 'objective_embed_coherence', 'objective_cv',
                'n_components', 'n_neighbors', 'min_dist', 'min_cluster_size', 'min_samples',
                'embedding_coherence_attr', 'coherence_score_cv_attr',
                # 'inter_topic_similarity_attr', 
                'n_topics'
            ]).to_csv(self.results_path, index=False)
            
    # def _define_search_space(self, trial):
    #     """Define the hyperparameter search space for Optuna based on the condition."""
    #     # Search space definitions remain the same
    #     if self.condition == 'DL':
    #         params = {
    #             'n_components': trial.suggest_int('n_components', 5, 15),
    #             'n_neighbors': trial.suggest_int('n_neighbors', 5, 15),
    #             'min_dist': trial.suggest_float('min_dist', 0.0, 0.05,step=0.005),
    #             'min_cluster_size': trial.suggest_int('min_cluster_size', 5, 10),
    #             'min_samples': trial.suggest_int('min_samples', 3, 10),
    #         }
    #     elif self.condition == 'HS':
    #         params = {
    #             'n_components': trial.suggest_int('n_components', 10, 20),
    #             'n_neighbors': trial.suggest_int('n_neighbors', 10, 20),
    #             'min_dist': trial.suggest_float('min_dist', 0.0, 0.05,step=0.005),
    #             'min_cluster_size': trial.suggest_int('min_cluster_size', 10, 15),
    #             'min_samples': trial.suggest_int('min_samples', 5, 10),
    #         }
    #     else: # Generic default
    #         params = {
    #             'n_components': trial.suggest_int('n_components', 5, 25),
    #             'n_neighbors': trial.suggest_int('n_neighbors', 10, 35),
    #             'min_dist': trial.suggest_float('min_dist', 0.0, 0.05,step=0.005),
    #             'min_cluster_size': trial.suggest_int('min_cluster_size', 10, 50),
    #             'min_samples': trial.suggest_int('min_samples', 5, 25),
    #         }
    #     return params


    def _define_search_space(self, trial):
        """Define the hyperparameter search space for Optuna based on the condition."""
        # Search space definitions remain the same
        if self.condition == 'DL':
            params = {
                'n_components': trial.suggest_int('n_components', 5, 15),
                'n_neighbors': trial.suggest_int('n_neighbors', 5, 15),
                'min_dist': trial.suggest_float('min_dist', 0.0, 0.05,step=0.005),
                'min_cluster_size': trial.suggest_int('min_cluster_size', 7, 10),
                'min_samples': trial.suggest_int('min_samples', 5, 10),
            }
        elif self.condition == 'HS':
            params = {
                'n_components': trial.suggest_int('n_components', 5, 20),
                'n_neighbors': trial.suggest_int('n_neighbors', 5, 25),
                'min_dist': trial.suggest_float('min_dist', 0.0, 0.05,step=0.005),
                'min_cluster_size': trial.suggest_int('min_cluster_size', 8, 20),
                'min_samples': trial.suggest_int('min_samples', 5, 15),
            }
        else: # Generic default
            params = {
                'n_components': trial.suggest_int('n_components', 5, 25),
                'n_neighbors': trial.suggest_int('n_neighbors', 10, 35),
                'min_dist': trial.suggest_float('min_dist', 0.0, 0.05,step=0.005),
                'min_cluster_size': trial.suggest_int('min_cluster_size', 10, 50),
                'min_samples': trial.suggest_int('min_samples', 5, 25),
            }
        return params
    
    def _define_search_space(self, trial):
        """Define the hyperparameter search space for Optuna based on the condition."""
        # Search space definitions remain the same
        if self.condition == 'DL':
            params = {
                'n_components': trial.suggest_int('n_components', 5, 15),
                'n_neighbors': trial.suggest_int('n_neighbors', 5, 15),
                'min_dist': trial.suggest_float('min_dist', 0.0, 0.05,step=0.005),
                'min_cluster_size': trial.suggest_int('min_cluster_size', 7, 10),
                'min_samples': trial.suggest_int('min_samples', 5, 10),
            }
        elif self.condition == 'HS':
            params = {
                'n_components': trial.suggest_int('n_components', 13, 20),
                'n_neighbors': trial.suggest_int('n_neighbors', 15, 26),
                'min_dist': trial.suggest_float('min_dist', 0.015, 0.02,step=0.005),
                'min_cluster_size': trial.suggest_int('min_cluster_size', 10, 10),
                'min_samples': trial.suggest_int('min_samples', 8, 8),
            }
        else: # Generic default
            params = {
                'n_components': trial.suggest_int('n_components', 5, 25),
                'n_neighbors': trial.suggest_int('n_neighbors', 10, 35),
                'min_dist': trial.suggest_float('min_dist', 0.0, 0.05,step=0.005),
                'min_cluster_size': trial.suggest_int('min_cluster_size', 10, 50),
                'min_samples': trial.suggest_int('min_samples', 5, 25),
            }
        return params


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

            # --- CHANGED: Convert all metric values to standard Python floats ---
            embedding_coherence = float(embedding_coherence)
            coherence_score = float(coherence_score)

            # Store all metrics as user attributes for comprehensive logging
            trial.set_user_attr('embedding_coherence', embedding_coherence)
            trial.set_user_attr('coherence_score', coherence_score)
            # trial.set_user_attr('inter_topic_similarity', inter_topic_sim)
            trial.set_user_attr('n_topics', len(set(topics)))

            # Return a tuple of metrics for multi-objective optimization
            return embedding_coherence, coherence_score
            
        except Exception as e:
            print(f"Error in trial {trial.number} with parameters {trial.params}: {str(e)}")
            raise optuna.exceptions.TrialPruned()

    def save_callback(self, study, trial):
        """Callback function to save results of each trial to a CSV file."""
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        
        # CHANGED: trial.value is now trial.values (a list) for multi-objective
        result_row = [
            trial.number,
            trial.values[0],  # Objective 1: embedding_coherence
            trial.values[1],  # Objective 2: coherence_score (c_v)
            trial.params['n_components'], trial.params['n_neighbors'],
            trial.params['min_dist'], trial.params['min_cluster_size'],
            trial.params['min_samples'],
            trial.user_attrs['embedding_coherence'],
            trial.user_attrs['coherence_score'],
            # trial.user_attrs['inter_topic_similarity'],
            trial.user_attrs['n_topics']
        ]
        
        with open(self.results_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(result_row)

    def run_optimization(self, n_trials=100):
        """Runs the multi-objective Optuna optimization."""
        self.data = self.load_data()
        print("Generating sentence embeddings...")
        self.embeddings = self.embedding_model.encode(self.data, show_progress_bar=True)
        print("Embeddings generated.")

        self.initialize_results_file()
        
        study_name = f"bertopic-{self.condition}-multiobj-optimization"
        storage_name = f"sqlite:///{self.study_db_path}"

        try:
            study = optuna.load_study(study_name=study_name, storage=storage_name)
            print(f"Loaded existing study '{study_name}' with {len(study.trials)} trials.")
        except KeyError:
            print(f"Creating new study '{study_name}'.")
            # CHANGED: Use NSGAIISampler and specify optimization directions
            sampler = NSGAIISampler(seed=self.random_seed)
            directions = ['maximize', 'maximize']# 'minimize']

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
        
        print(f"\nOptimization completed in {time.time() - start_time:.2f} seconds")

        # CHANGED: Print the Pareto front (the set of best trials)
        print("\n--- Pareto Front (Best Trade-off Trials) ---")
        print(f"Found {len(study.best_trials)} optimal trials.")
        for i, trial in enumerate(study.best_trials):
            print(f"\n--- Trial {trial.number} (Solution {i+1}) ---")
            print(f"  🎯 Objectives:")
            print(f"    - Embedding Coherence: {trial.values[0]:.4f} (Higher is better)")
            print(f"    - C_v Coherence:       {trial.values[1]:.4f} (Higher is better)")
            # print(f"    - Inter-Topic Sim:     {trial.values[2]:.4f} (Lower is better)")
            print("  🔧 Parameters:")
            for key, value in trial.params.items():
                print(f"    - {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BERTopic multi-objective optimization with Optuna")
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


#python src/optuna_search_allmetrics.py --dataset dreamachine --condition HS --sentences --n_trials 200
#python src/optuna_search_allmetrics.py --dataset dreamachine --condition DL --sentences --n_trials 2