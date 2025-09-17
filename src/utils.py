#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : utils.py
# description     : Define helpers functions
#                   Condition can be either HS,DL or HW (if for Dreamachine dataset)
# author          : Romy, Beauté (r.beaut@sussex.ac.uk)
# date            : 2024-07-25
# ==============================================================================





#############################################################################
################ COHERENCE METRICS ##########################################
#############################################################################

from gensim.corpora.dictionary import Dictionary
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel



def calculate_coherence(topic_model, data):
    # topics, _ = topic_model.fit_transform(data)
    # topics = topic_model.topics_
    unique_topics = sorted(list(set(topic_model.topics_)))
    if -1 in unique_topics:  # Remove outlier topic if present
        unique_topics.remove(-1)
    
    # get topic words and filter out empty topics
    topic_words = []
    for topic_id in unique_topics:
        words = [word for word, _ in topic_model.get_topic(topic_id)]
        # only include topics that have at least one non-empty word
        if any(word.strip() for word in words):
            topic_words.append(words)
    
    if not topic_words:
        print("No valid topics found for coherence calculation")
        return float('nan'), float('nan')
    
    # extract features for Topic Coherence evaluation
    vectorizer = topic_model.vectorizer_model
    tokenizer = vectorizer.build_tokenizer()
    tokens = [tokenizer(doc) for doc in data]
    
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]

    try:
        coherence_model = CoherenceModel(
            topics=topic_words,
            texts=tokens,
            corpus=corpus,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()

        coherence_model_umass = CoherenceModel(
            topics=topic_words,
            texts=tokens,
            corpus=corpus,
            dictionary=dictionary,
            coherence='u_mass'
        )
        coherence_score_umass = coherence_model_umass.get_coherence()
        
        print(f"Number of valid topics used for coherence calculation (excludes outliers -1): {len(topic_words)}")
        return coherence_score, coherence_score_umass
        
    except Exception as e:
        print(f"Error calculating coherence: {str(e)}")
        return float('nan'), float('nan')





#############################################################################
################ HYPERPARAMETERS ############################################
#############################################################################

import itertools


def get_params_grid(dataset_config, condition,reduced=False): #reduced set to true to test with reduced hyperparams combnations
    """Generate parameter grid from dataset config."""
    params = dataset_config.get_params(condition, reduced)
    
   #  Generate all combinations of hyperparameters
    umap_combinations = list(itertools.product(
        params['umap_params']['n_components'],
        params['umap_params']['n_neighbors'],
        params['umap_params']['min_dist']
    ))

    hdbscan_combinations = list(itertools.product(
        params['hdbscan_params']['min_cluster_size'],
        params['hdbscan_params']['min_samples']
    ))

    print("Total number of combinations:", len(umap_combinations)*len(hdbscan_combinations)) 
    return umap_combinations, hdbscan_combinations

