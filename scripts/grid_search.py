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
import time
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer


from src.model import run_bertopic
from src.preprocessor import split_sentences, clean_text
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

        #load apprioriate dataset config
        if dataset == "dreamachine":
            from configs.dreamachine import config #import config instance defined in dreamachine.py
            self.dataset_config = config
        else:
            raise ValueError(f"Unrecognised dataset: {dataset}")
        
        self.setup_paths()
        self.setup_models()

    
    def setup_paths(self):
        """setup data and results paths based on dataset and condition."""

        self.data_path = os.path.join("DATA", f"{self.dataset}/{self.condition}.csv")
        
        self.results_path = os.path.join(
            "RESULTS", 
            f"GS_results_{self.condition}"
            + ('_reduced' if self.reduced_GS else '')
            + ('_sentences' if self.use_sentences else '')
            + '.csv'
        )

    def setup_models(self):
        """init models and parameters using spec dataset configs."""
        self.embedding_model = SentenceTransformer(self.dataset_config.transformer_model)
        self.vectorizer_models = {
            '1_2_gram': CountVectorizer(
                ngram_range=self.dataset_config.ngram_range,
                stop_words=list(self.dataset_config.extended_stop_words),
                max_df=self.dataset_config.max_df,
                min_df=self.dataset_config.min_df
            )
        }


    def load_data(self):
        """Load and preprocess data."""
        df_reports = pd.read_csv(self.data_path, sep='\t')['reflection_answer']
        
        if self.use_sentences:
            df_reports = split_sentences(df_reports)
            print(f"Split into {len(df_reports)} sentences")
        
        return [clean_text(doc) for doc in df_reports]
    

    def initialize_results_file(self):
        """Initialize results CSV with headers."""
        os.makedirs(os.path.dirname(self.results_path), exist_ok=True)
        if not os.path.exists(self.results_path):
            pd.DataFrame(columns=[
                'n_components', 'n_neighbors', 'min_dist',
                'min_cluster_size', 'min_samples', 'top_n_words',
                'coherence_score', 'coherence_score_umass', 'n_topics'
            ]).to_csv(self.results_path, index=False)


    def run_grid_search(self):
        """Execute grid search for all vectorizer models."""
        data = self.load_data()
        self.initialize_results_file()
        
        results = {}
        for ngram_key, vectorizer_model in self.vectorizer_models.items():
            print(f"\nRunning grid search for {ngram_key}...")
            results[ngram_key] = self._run_single_search(data, vectorizer_model)
        
        return results
    


    def _run_single_search(self, data, vectorizer_model):
        """Execute grid search for a single vectorizer model."""
        umap_combinations, hdbscan_combinations = get_params_grid(
            self.dataset_config, 
            self.condition,
            self.reduced_GS
        )
        top_n_words_options = [self.top_n_words]
        start_time = time.time()
        
        for umap_config in tqdm(umap_combinations):
            for hdbscan_config in hdbscan_combinations:
                for top_n_words in top_n_words_options:
                    try:
                        n_components, n_neighbors, min_dist = umap_config
                        min_cluster_size, min_samples = hdbscan_config
                        
                        model, topics, coherence_score, coherence_score_umass = run_bertopic(
                            data=data,
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
                            coherence_score, coherence_score_umass,
                            len(set(topics))
                        )
                        
                    except Exception as e:
                        print(f"Error with parameters {umap_config}, {hdbscan_config}, "
                              f"top_n_words={top_n_words}: {str(e)}")
                        continue

        print(f"\nGrid search completed in {time.time() - start_time:.2f} seconds")
        return pd.read_csv(self.results_path).sort_values('coherence_score', ascending=False)
    

    def save_best_model(self, model, score):
        model_path = os.path.join(
            "models",
            f"best_model_{self.condition}_{score:.3f}.pkl"
        )
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
    

    def _save_result(self, n_components, n_neighbors, min_dist, min_cluster_size, 
                    min_samples, top_n_words, coherence_score, coherence_score_umass, n_topics):
        """Save a single result to CSV."""
        result_df = pd.DataFrame([{
            'n_components': n_components,
            'n_neighbors': n_neighbors,
            'min_dist': min_dist,
            'min_cluster_size': min_cluster_size,
            'min_samples': min_samples,
            'top_n_words': top_n_words,
            'coherence_score': coherence_score,
            'coherence_score_umass': coherence_score_umass,
            'n_topics': n_topics
        }])
        result_df.to_csv(self.results_path, mode='a', header=False, index=False)


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
        print(f"\nTop results for {ngram_key}:")
        print(result_df.head())


# python grid_search.py --dataset dreamachine --condition HS --sentences --reduced_GS