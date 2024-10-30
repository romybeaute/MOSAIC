#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : grid_search_colyra.py
# description     : Run grid search for BERTopic hyperparameters
# author          : Romy Beauté, Guillaume Dumas
# date            : 2024-09-18
# version         : 2
# usage           : python grid_search_colyra.py --condition DL --reduced_GS --sentences
#                   python grid_search_colyra.py --condition HS --sentences
# original file   : https://colab.research.google.com/drive/1Kz_TAUAQgP9ZHo_QILVaK0n59cpVVBi1
# python_version  : 3.10
# ==============================================================================

import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import re
import itertools
import time
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer

from gensim.corpora.dictionary import Dictionary
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel

# BERTopic Libraries
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, TextGeneration
from torch import bfloat16
import transformers
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap.umap_ import UMAP
from hdbscan import HDBSCAN

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

from META_helpers_colyra import split_sentences


os.environ["TOKENIZERS_PARALLELISM"] = "True"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
random_seed = 42

# reduced_custom_stopwords = {'like','felt','experience'}
reduced_custom_stopwords = {}

sentence_transformer_model = "all-mpnet-base-v2" #"paraphrase-MiniLM-L6-v2"# #"BAAI/bge-small-en" "all-MiniLM-L6-v2'"
embedding_model = SentenceTransformer(sentence_transformer_model)


def hyperparams(len_dataset): #extended version (modified 11/09/24)
    '''
    Defined in helpers/BERT_helpers.py
    '''
    if args.condition == 'HS':
        return {'umap_params': {
                'n_components': list(range(2, 21)), #2 to 20 step 1 (default to 2)
                'n_neighbors': [5,10,15,20,25,30],
                'min_dist': [0.0,0.01,0.05], #default to 1.0
            },
            'hdbscan_params': {
                'min_cluster_size': [5,10,15], #default to 10
                'min_samples': [None,5],
            }}
    elif args.condition == 'DL':
        return {'umap_params': {
                'n_components': list(range(2, 21)), #default to 2
                'n_neighbors': [5,10,15,20,25,30],
                'min_dist': [0.0,0.01,0.05], #default to 1.0
            },
            'hdbscan_params': {
                'min_cluster_size': [5,10,15], #default to 10
                'min_samples': [None,5],
            }}
    else:
        return {'umap_params': {
                'n_components': [2,4,6,8,10,12,14,16,18,20], #default to 2, A higher number of components (8-12) can help capture more nuanced relationships in a large dataset, potentially leading to more coherent topics.
                'n_neighbors': [10,15,20,25,30,35,40], #For a dataset of this size, values between 15-25 should provide a good balance between local and global structure preservation.
                'min_dist': [0.0,0.01,0.05], #default to 1.0,  Lower values (0.0 or 0.01) tend to create more compact clusters, which can be beneficial for topic coherence.
            },
            'hdbscan_params': {
                'min_cluster_size': [10,20,30,40,50,60], #default to 10, A higher value (50-100) can help reduce the number of small, noisy clusters, potentially leading to more coherent topics.
                'min_samples': [None,10], #Using None (the default) or a small value like 5-10 can help balance between noise reduction and topic discovery.
            }}




def hyperparams_reduced(len_dataset): #extended version (modified 11/09/24)
    '''
    Defined in helpers/BERT_helpers.py
    '''
    if args.condition == 'HS':
        return {'umap_params': {
                # 'n_components': [2,4,6,8,10,12,14,16,18,20], #default to 2
                'n_neighbors': [5,10,15,20], #Heuristics: Small values (<5) focus too much on local structure and Large values (>50) may blur local distinctions
                'min_dist': [0.0,0.01,0.05], #default to 1.0
            },
            'hdbscan_params': {
                'min_cluster_size': [5,10], #default to 10
                'min_samples': [5],
            }}
    elif args.condition == 'DL':
        return {'umap_params': {
                # 'n_components': [2,4,6,8,10,12,14,16,18,20], #default to 2
                'n_components': list(range(2, 21)),
                'n_neighbors': [5,10,15,20],
                'min_dist': [0.0,0.01,0.05], #default to 1.0
            },
            'hdbscan_params': {
                'min_cluster_size': [5,10], #default to 10
                'min_samples': [5],
            }}
    else:
        return {'umap_params': {
                'n_components': [8,10,12,14,16], #default to 2, A higher number of components (8-12) can help capture more nuanced relationships in a large dataset, potentially leading to more coherent topics.
                'n_neighbors': [15,20,25], #For a dataset of this size, values between 15-25 should provide a good balance between local and global structure preservation.
                'min_dist': [0.0,0.01], #default to 1.0,  Lower values (0.0 or 0.01) tend to create more compact clusters, which can be beneficial for topic coherence.
            },
            'hdbscan_params': {
                'min_cluster_size': [50,75,100], #default to 10, A higher value (50-100) can help reduce the number of small, noisy clusters, potentially leading to more coherent topics.
                'min_samples': [None,10], #Using None (the default) or a small value like 5-10 can help balance between noise reduction and topic discovery.
            }}







#############################################################################
################ COHERENCE METRICS ##########################################
#############################################################################


# calculate coherence using BERTopic's model
def calculate_coherence(topic_model, data):

    topics, _ = topic_model.fit_transform(data)
    # Preprocess Documents
    documents = pd.DataFrame({"Document": data,
                          "ID": range(len(data)),
                          "Topic": topics})
    documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})

    #Extracting the vectorizer and embedding model from BERTopic model
    vectorizer = topic_model.vectorizer_model #CountVectorizer of BERTopic model
    tokenizer = vectorizer.build_tokenizer()

    # Extract features for Topic Coherence evaluation
    tokens = [tokenizer(doc) for doc in data]

    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]

    topic_words = [[word for word, _ in topic_model.get_topic(topic_id)] for topic_id in range(len(set(topics))-1)]

    print("Topics:", topic_words)
    coherence_model = CoherenceModel(topics=topic_words,
                                     texts=tokens,
                                     corpus=corpus,
                                     dictionary=dictionary,
                                     coherence='c_v')
    coherence_score = coherence_model.get_coherence()

    coherence_model_umass = CoherenceModel(topics=topic_words, 
                                 texts=tokens, 
                                 corpus=corpus,
                                 dictionary=dictionary, 
                                 coherence='u_mass')
    coherence_score_umass = coherence_model_umass.get_coherence()

    return coherence_score,coherence_score_umass


def get_params_grid(len_dataset, reduced=False): #reduced set to true to test with reduced hyperparams combnations
    if reduced:
        hyperparams_dict = hyperparams_reduced(len_dataset)
    else:
        hyperparams_dict = hyperparams(len_dataset)

    # Generate all combinations of hyperparameters
    umap_combinations = list(itertools.product(
    hyperparams_dict['umap_params']['n_components'],
    hyperparams_dict['umap_params']['n_neighbors'],
    hyperparams_dict['umap_params']['min_dist']))

    hdbscan_combinations = list(itertools.product(
    hyperparams_dict['hdbscan_params']['min_cluster_size'],
    hyperparams_dict['hdbscan_params']['min_samples']))

    print("Total number of combinations:", len(umap_combinations)*len(hdbscan_combinations)*2) # *2 to acocunt for len(top_n_words_options)
    return umap_combinations, hdbscan_combinations



def run_grid_search(data, vectorizer_model, embedding_model, condition, reduced_GS=False, sentences=False, store_results=True):

    umap_combinations, hdbscan_combinations =get_params_grid(len(data),reduced=reduced_GS)
    top_n_words_options = [5,10]  # New parameter options

    start_time = time.time()

    # Nested loop to iterate over each combination of UMAP and HDBSCAN parameters
    results = []

    for umap_config in tqdm(umap_combinations):
        for hdbscan_config in hdbscan_combinations:
            for top_n_words in top_n_words_options:
                try:
                    # Unpack the parameter sets
                    n_components, n_neighbors, min_dist = umap_config
                    min_cluster_size, min_samples = hdbscan_config

                    
                    # Execute the run_bertopic function using unpacked parameters
                    model, topics, coherence_score,coherence_score_umass = run_bertopic(
                        data=data,
                        vectorizer_model=vectorizer_model,
                        embedding_model=embedding_model,
                        n_neighbors=n_neighbors,
                        n_components=n_components,
                        min_dist=min_dist,
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                        top_n_words=top_n_words
                    )
                    # Store results
                    results.append({
                        'n_components': n_components,
                        'n_neighbors': n_neighbors,
                        'min_dist': min_dist,
                        'min_cluster_size': min_cluster_size,
                        'min_samples': min_samples,
                        'top_n_words': top_n_words,
                        'coherence_score': coherence_score,
                        'cohenrece_score_umass': coherence_score_umass,
                        'n_topics':len(np.unique(topics))
                    })
                except Exception as e:
                    print(f"Error with parameters {umap_config}, {hdbscan_config}, top_n_words={top_n_words}: {e}")
                    continue

    results_df = pd.DataFrame(results).sort_values(by='coherence_score', ascending=False)
    print(f"Grid search completed in {time.time() - start_time:.2f} seconds")

    if store_results:
        os.makedirs('RESULTS', exist_ok=True) # Create directory if it doesn't exist
        name_file = f'RESULTS/grid_search_results_{condition}_seed{random_seed}.csv'
        if reduced_GS:
            name_file = name_file.replace('.csv','_reduced.csv')
        if sentences:
            name_file = name_file.replace('.csv','_sentences.csv')

        if os.path.isfile(name_file):
            existing_df = pd.read_csv(name_file)
            combined_df = pd.concat([existing_df, results_df], ignore_index=True)
            results_df.to_csv(name_file, mode='a', header=False, index=False)
            param_columns = ['n_components', 'n_neighbors', 'min_dist', 
                           'min_cluster_size', 'min_samples', 'top_n_words']
            combined_df = combined_df.drop_duplicates(subset=param_columns, keep='last')
            # re sort the file by coherence score
            combined_df = combined_df.sort_values(by='coherence_score', ascending=False)
            combined_df.to_csv(name_file, index=False)
            results_df = combined_df

        else:
            results_df.to_csv(name_file, index=False) #file doesnt exist, create it to save current results

    return results_df


def run_bertopic(data,
         vectorizer_model,
         embedding_model,
         n_neighbors, 
         n_components, 
         min_dist, 
         min_cluster_size, 
         min_samples,
         top_n_words, #default to 10
         nr_topics = "auto"):
    
    '''
    Defined in BERT_DM/BERTopic_hypertuned_multiprocessing.py

    Return : 
    - model : implemented BERTopic model with fine-tuned hyperparameters
    - topics : contains a one-to-one mapping of inputs to their modeled topic (or cluster)
    - probs : contains the probability of each document belonging to their assigned topic

    '''
    print(f"Received parameters: n_neighbors={n_neighbors}, n_components={n_components}, min_dist={min_dist}, min_cluster_size={min_cluster_size}, min_samples={min_samples},top_n_words={top_n_words}")


    # ********** Instanciate BERTOPIC **********

    umap_model = UMAP(n_neighbors=n_neighbors,
                      n_components=n_components,
                      min_dist=min_dist,
                      metric = 'cosine',
                      random_state=random_seed) # rdm seed for reportability
    
    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, 
                            min_samples=min_samples,
                            cluster_selection_epsilon=0.03, #reduce number of outliers identified
                            gen_min_span_tree=True,
                            prediction_data=True) 
    
    model = BERTopic(
        umap_model=umap_model,
        low_memory=True,
        hdbscan_model=hdbscan_model,
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        top_n_words=top_n_words,
        nr_topics= nr_topics,#default to None
        language='english',
        calculate_probabilities=True,
        verbose=True)

    # ********** Fit BERTOPIC **********
    topics,_ = model.fit_transform(data) 
    # coherence_score = calculate_coherence(model, data)
    coherence_score,coherence_score_umass = calculate_coherence(model, data)
    print("Coherence Score (CV):", coherence_score)
    print("Coherence Score (UMass):", coherence_score_umass)
    print(f"n = {len(np.unique(topics))} topics extracted")

    return model, topics, coherence_score,coherence_score_umass




def main(args):
    # Setup the paths based on the condition
    if args.condition == 'HW':
        reports_path = os.path.join("DATA", f"{args.condition}_reflections.csv")
    else:
        reports_path = os.path.join("DATA", f"{args.condition}_reflections_cleaned.csv")
    df_reports = pd.read_csv(reports_path,sep='\t')['reflection_answer']

    if args.sentences:
        df_reports = split_sentences(df_reports)
        print("Splitting sentences...")
        print(f"N = {len(df_reports)} sentences")
        

    # Configuration based on command line arguments
    extended_stop_words = set(stopwords.words('english')).union(reduced_custom_stopwords)
    embedding_model = SentenceTransformer(args.sentence_transformer_model)
    # vectorizer_model = CountVectorizer(ngram_range=(1,3), stop_words=list(extended_stop_words))

    vectorizer_models = {
        '1_2_gram': CountVectorizer(ngram_range=(1, 2), stop_words=list(extended_stop_words),max_df=0.9,min_df=2),
        # '1_3_gram': CountVectorizer(ngram_range=(1, 3), stop_words=list(extended_stop_words))
        }
    # Run grid search
    results = {}
    for ngram_key, vectorizer_model in vectorizer_models.items():
        print(f"Running grid search for {ngram_key}...")
        results[ngram_key] = run_grid_search(df_reports, vectorizer_model, embedding_model, args.condition, reduced_GS=args.reduced_GS, sentences=args.sentences, store_results=True)
    
    for ngram_key, result_df in results.items():
        print(f"Top results for {ngram_key} n-gram:")
        print(result_df.head(5))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BERT topic modeling with grid search.")
    parser.add_argument('--condition', type=str, default='DL', choices=['HS', 'DL', 'HW'], help='Condition to process')
    parser.add_argument('--reduced_GS', action='store_true', help='Flag to use reduced grid search parameters')
    parser.add_argument('--stopwords', type=set, default=set(), help='Additional stopwords to use')
    parser.add_argument('--sentence_transformer_model', type=str, default="all-mpnet-base-v2", help='Sentence transformer model to use')
    parser.add_argument('--sentences', action='store_true', help='Flag to split sentences')
    args = parser.parse_args()

    main(args)


# python grid_search_colyra.py --condition HS --sentences --reduced_GS