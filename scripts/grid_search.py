#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : grid_search.py
# description     : Run grid search for BERTopic hyperparameters
# author          : Romy Beauté, Guillaume Dumas
# date            : 2024-09-18
# version         : 2
# usage           : python grid_search.py --condition DL --reduced_GS --sentences
#                   python grid_search.py --condition HS --sentences
# python_version  : 3.10
# ==============================================================================

import argparse
import pandas as pd
from tqdm import tqdm
import os
import sys
import time
import csv
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from src.model import run_bertopic
from src.preprocessor import split_sentences
from src.utils import get_params_grid


os.environ["TOKENIZERS_PARALLELISM"] = "True"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"




class GridSearchBERTopic:
    def __init__(self, dataset="dreamachine",condition="DL", reduced_GS=False, use_sentences=True):
        self.dataset = dataset
        self.condition = condition
        self.reduced_GS = reduced_GS
        self.use_sentences = use_sentences
        self.random_seed = 42
        self.top_n_words = 15

        #load apprioriate dataset config
        if dataset == "dreamachine":
            from configs.dreamachine2 import config #import config instance defined in dreamachine.py
            self.dataset_config = config
        else:
            raise ValueError(f"Unrecognised dataset: {dataset}")
        
        self.setup_paths()
        self.setup_models()

    
    def setup_paths(self):
        """setup data and results paths based on dataset and condition."""

        self.data_path = os.path.join("DATA", f"{self.dataset}/{self.condition}_reflections_APIcleaned.csv")
        
        self.results_path = os.path.join(
            f"EVAL/{self.dataset}", 
            f"GS_results_{self.condition}"
            + ('_reduced' if self.reduced_GS else '')
            + ('_sentences' if self.use_sentences else '')
            + '.csv'
        )

    def setup_models(self):
        """init models and parameters using spec dataset configs."""
        self.embedding_model = SentenceTransformer(self.dataset_config.transformer_model)
        self.vectorizer_models = {
            'default': CountVectorizer(
                ngram_range=self.dataset_config.ngram_range,
                stop_words=list(self.dataset_config.extended_stop_words),
                max_df=self.dataset_config.max_df,
                min_df=self.dataset_config.min_df,
                lowercase=True #lowercase all words to make sure stopwords not case sensitive
            )
        }


    def load_data(self):
        """Load and preprocess data."""
        # df_reports = pd.read_csv(self.data_path, sep='\t')['reflection_answer']
        df_reports = pd.read_csv(self.data_path, sep=',')['cleaned_reflection'].dropna().reset_index(drop=True)
        print(df_reports)
        
        if self.use_sentences:
            df_reports,_ = split_sentences(df_reports)
            print(f"Split into {len(df_reports)} sentences")

            # Remove sentences with less than 2 words
            min_words = 2
            df_reports = [s for s in df_reports if len(s.split()) >= min_words]
            print(f"After removing short sentences (<{min_words} words), {len(df_reports)} remain.")

            # Remove duplicate sentences
            seen = set()
            df_reports = [s for s in df_reports if not (s in seen or seen.add(s))]
            print(f"After removing duplicates, {len(df_reports)} remain.")

        
        # return [clean_text(doc) for doc in df_reports]
        return df_reports
    

    def initialize_results_file(self):
        """Initialize results CSV with headers."""
        os.makedirs(os.path.dirname(self.results_path), exist_ok=True)
        if not os.path.exists(self.results_path):
            pd.DataFrame(columns=[
                'n_components', 'n_neighbors', 'min_dist',
                'min_cluster_size', 'min_samples', 'top_n_words',
                'coherence_score', 'coherence_score_umass', 'embedding_coherence','n_topics'
            ]).to_csv(self.results_path, index=False)


    def run_grid_search(self):
        """Execute grid search for all vectorizer models."""
        data = self.load_data()

        #generate embeddings once before the search 
        print("Generating sentence embeddings...")
        embeddings = self.embedding_model.encode(data, show_progress_bar=True)
        print("Embeddings generated.")

        self.initialize_results_file()
        
        results = {}
        for ngram_key, vectorizer_model in self.vectorizer_models.items():
            print(f"\nRunning grid search for {ngram_key}...")
            results[ngram_key] = self._run_single_search(data, embeddings,vectorizer_model)
        
        return results
    


    def _run_single_search(self, data, embeddings, vectorizer_model):
        """Execute grid search for a single vectorizer model."""
        umap_combinations, hdbscan_combinations = get_params_grid(
            self.dataset_config, 
            self.condition,
            self.reduced_GS
        )
        top_n_words_options = [self.dataset_config.top_n_words]
        start_time = time.time()
        
        for umap_config in tqdm(umap_combinations):
            for hdbscan_config in hdbscan_combinations:
                for top_n_words in top_n_words_options:
                    try:
                        n_components, n_neighbors, min_dist = umap_config
                        min_cluster_size, min_samples = hdbscan_config
                        
                        model, topics, coherence_score, coherence_score_umass, embedding_coherence = run_bertopic(
                            data=data,
                            embeddings=embeddings,
                            vectorizer_model=vectorizer_model,
                            embedding_model=self.embedding_model,
                            n_neighbors=n_neighbors,
                            n_components=n_components,
                            min_dist=min_dist,
                            min_cluster_size=min_cluster_size,
                            min_samples=min_samples,
                            top_n_words=top_n_words
                        )
                        
                        self._save_result(
                            n_components, n_neighbors, min_dist,
                            min_cluster_size, min_samples, top_n_words,
                            coherence_score, coherence_score_umass,embedding_coherence,
                            len(set(topics))
                        )
                        
                    except Exception as e:
                        print(f"Error with parameters {umap_config}, {hdbscan_config}, "
                              f"top_n_words={top_n_words}: {str(e)}")
                        continue

        print(f"\nGrid search completed in {time.time() - start_time:.2f} seconds")
        return pd.read_csv(self.results_path).sort_values('embedding_coherence', ascending=False)
    

    def save_best_model(self, model, score):
        model_path = os.path.join(
            "models",
            f"best_model_{self.condition}_{score:.3f}.pkl"
        )
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
    

    # def _save_result(self, n_components, n_neighbors, min_dist, min_cluster_size, 
    #                 min_samples, top_n_words, coherence_score, coherence_score_umass, embedding_coherence, n_topics):
    #     """Save a single result to CSV."""
    #     result_df = pd.DataFrame([{
    #         'n_components': n_components,
    #         'n_neighbors': n_neighbors,
    #         'min_dist': min_dist,
    #         'min_cluster_size': min_cluster_size,
    #         'min_samples': min_samples,
    #         'top_n_words': top_n_words,
    #         'coherence_score': coherence_score,
    #         'coherence_score_umass': coherence_score_umass,
    #         'embedding_coherence':embedding_coherence,
    #         'n_topics': n_topics
    #     }])
    #     result_df.to_csv(self.results_path, mode='a', header=False, index=False)

    def _save_result(self, n_components, n_neighbors, min_dist, min_cluster_size,
                    min_samples, top_n_words, coherence_score, coherence_score_umass, embedding_coherence, n_topics):
        """Save a single result to CSV using the csv module for safety."""
        
        # Define the row as a list in the correct order
        result_row = [
            n_components, n_neighbors, min_dist,
            min_cluster_size, min_samples, top_n_words,
            coherence_score, coherence_score_umass, embedding_coherence, n_topics
        ]
        
        # Use the 'csv' module to safely append the row
        with open(self.results_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(result_row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BERT topic modeling with grid search")
    parser.add_argument('--dataset', type=str, default='dreamachine', help='Dataset to use')
    parser.add_argument('--condition', type=str, default='DL', choices=['HS', 'DL', 'HW'])
    parser.add_argument('--reduced_GS', action='store_true')
    parser.add_argument('--sentences', action='store_true')
    args = parser.parse_args()

    grid_search = GridSearchBERTopic(
        dataset=args.dataset,
        condition=args.condition,
        reduced_GS=args.reduced_GS,
        use_sentences=args.sentences
    )
    
    results = grid_search.run_grid_search()
    for ngram_key, result_df in results.items():
        print(result_df.head())


# python scripts/grid_search.py --dataset dreamachine --condition DL --sentences --reduced_GS