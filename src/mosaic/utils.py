#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : utils.py
# description     : Define helpers functions
# author          : Romy, Beauté (r.beaut@sussex.ac.uk)
# date            : 2024-07-25
# ==============================================================================



import pandas as pd
import json
import pickle
import matplotlib.pyplot as plt


def load_data_file(raw_file):
    """
    Load data from a raw file into a pandas DataFrame.
    Supports CSV, TSV, JSON, Excel, Pickle, Parquet, and TXT formats.
    """
    
    suffix = raw_file.suffix.lower()

    if suffix in [".csv"]:
        df = pd.read_csv(raw_file)

    elif suffix in [".tsv", ".tab"]:
        df = pd.read_csv(raw_file, sep="\t")

    elif suffix in [".json"]:
        with raw_file.open() as f:
            data = json.load(f)
        # Convert JSON list/dict to DataFrame if possible
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.json_normalize(data)

    elif suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(raw_file)

    elif suffix in [".pkl", ".pickle"]:
        with raw_file.open("rb") as f:
            data = pickle.load(f)
        # Try to convert to DataFrame if it’s a list/dict
        if isinstance(data, dict) or isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            raise TypeError("Pickle loaded but cannot be converted to DataFrame.")

    elif suffix in [".parquet"]:
        df = pd.read_parquet(raw_file)

    elif suffix in [".txt"]:
        # Fallback: load as plain text
        df = pd.DataFrame({"text": raw_file.read_text().splitlines()})

    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    print("\nLoaded data shape:", df.shape)
    return df



#############################################################################
################ COHERENCE METRICS ##########################################
#############################################################################

from gensim.corpora.dictionary import Dictionary
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import itertools




def calculate_embedding_coherence(model, docs, embeddings):
    """
    Calculate the average intra-topic cosine similarity for a BERTopic model.

    Args:
        model (BERTopic): A fitted BERTopic model.
        docs (list of str): The original documents used to fit the model.
        embeddings (np.ndarray): The document embeddings for the training data.

    Returns:
        float: The mean coherence score across all topics.
    """
    # Group documents and their embeddings by topic
    # The original documents ('docs') must be used here to match the length of model.topics_
    documents_df = pd.DataFrame({"Doc": docs, "Topic": model.topics_})
    
    # Ensure embeddings are a numpy array
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings)
        
    documents_df['Embedding'] = list(embeddings)

    # Calculate coherence for each topic
    topic_coherence_scores = []
    for topic_id in documents_df['Topic'].unique():
        if topic_id == -1:  # Skip outlier topic
            continue
        
        topic_docs_df = documents_df[documents_df['Topic'] == topic_id]
        
        if len(topic_docs_df) < 2: # Cannot calculate coherence for a single document
            continue

        topic_embeddings = np.vstack(topic_docs_df['Embedding'].values)
        
        # Calculate pairwise similarity and get the mean of the upper triangle
        similarity_matrix = cosine_similarity(topic_embeddings)
        upper_triangle_mean = np.mean(similarity_matrix[np.triu_indices(len(topic_docs_df), k=1)])
        topic_coherence_scores.append(upper_triangle_mean)

    # Return the mean coherence across all topics, or 0.0 if no valid topics found
    return np.mean(topic_coherence_scores) if topic_coherence_scores else 0.0





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
        return float('nan')
    
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

        
        print(f"Number of valid topics used for coherence calculation (excludes outliers -1): {len(topic_words)}")
        return coherence_score
        
    except Exception as e:
        print(f"Error calculating coherence: {str(e)}")
        return float('nan')





#############################################################################
################ HYPERPARAMETERS ############################################
#############################################################################




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




#Get PARETO FRONT for hyperparameter tuning results (from optuna results)
def get_pareto_front(df, col1, col2):
    """
    Identifies the rows that are on the Pareto front (non-dominated)
    we want to MAXIMISE both columns.
    """
    population = df[[col1, col2]].values
    is_pareto = np.ones(population.shape[0], dtype=bool)
    
    for i, c in enumerate(population):
        if is_pareto[i]:
            # Check if current point 'c' is dominated by any other point 'population[j]'
            # Dominated means: other point is >= in all objectives AND > in at least one
            lower_or_equal = np.all(population >= c, axis=1)
            strictly_greater = np.any(population > c, axis=1)
            
            if np.any(lower_or_equal & strictly_greater):
                is_pareto[i] = False

        pareto_solution = df[is_pareto].sort_values(by=col1)
    print(f"Found {len(pareto_solution)} solutions on the Pareto Front:\n")
    print(pareto_solution[['trial_number', 'embedding_coherence', 'objective_cv', 'n_topics']].to_string(index=False))

    plt.figure(figsize=(10, 6))
    plt.scatter(df['embedding_coherence'], df['objective_cv'], c='gray', alpha=0.4, label='All Trials')
    plt.scatter(pareto_solution['embedding_coherence'], pareto_solution['objective_cv'], c='red', s=50, label='Pareto Front')
    plt.plot(pareto_solution['embedding_coherence'], pareto_solution['objective_cv'], c='red', linestyle='--', alpha=0.5)

    plt.xlabel('Embedding Coherence')
    plt.ylabel('CV Score')
    plt.title('Pareto Front: Coherence vs CV')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

    
    return pareto_solution