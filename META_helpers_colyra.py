#define helpers functions

import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import os
import re
import itertools
import time

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer
import nltk
nltk.download('wordnet')

from gensim.corpora.dictionary import Dictionary
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from sklearn.feature_extraction.text import CountVectorizer


# BERTopic Libraries
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN


from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV


reduced_custom_stopwords = {'like','felt'}

stop_words = set(stopwords.words('english'))
extended_stop_words = stop_words.union(reduced_custom_stopwords) #load custom stopwords from BERT_helpers.py

sentence_transformer_model = "all-mpnet-base-v2" #"BAAI/bge-small-en" "all-MiniLM-L6-v2'"

random_state = 22

def hyperparams(len_dataset):
    '''
    Defined in helpers/BERT_helpers.py
    '''
    return {'umap_params': {
            'n_components': [2,3,4,5,7,10], #default to 2 
            'n_neighbors': [3,4,5,7,10,12],
            'min_dist': [0.0,0.01,0.025], #default to 1.0
        },
        'hdbscan_params': {
            #list of 3 values : 1% len_dataset,10 (default value), 5% len_dataset
            'min_cluster_size': [int(len_dataset*0.025),10,int(len_dataset*0.05)],
            'min_samples': [int(len_dataset*0.025),10,int(len_dataset*0.05)] #default to None
        }
    }



def hyperparams_reduced(len_dataset):
    '''
    Defined in helpers/BERT_helpers.py
    '''
    return {'umap_params': {
            'n_components': [2,3,5,7], #default to 2 
            'n_neighbors': [3,5,7,10,12,15],
            'min_dist': [0.01], #default to 1.0
        },
        'hdbscan_params': {
            # list of 3 values : 1% len_dataset,10 (default value), 5% len_dataset
            'min_cluster_size': [int(len_dataset*0.05)],
            'min_samples': [None]  #default to None
        }
    }

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
    # analyzer = vectorizer.build_analyzer() #allows for n-gram tokenization
    
    # Extract features for Topic Coherence evaluation
    words = vectorizer.get_feature_names_out()
    tokens = [tokenizer(doc) for doc in data]
    # tokens = [analyzer(doc) for doc in data]

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
    return coherence_score


def get_params_grid(len_dataset,reduced=False): #reduced set to true to test with reduced hyperparams combnations
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

    print("Total number of combinations:", len(umap_combinations)*len(hdbscan_combinations))
    return umap_combinations, hdbscan_combinations



def run_grid_search(data, vectorizer_model, embedding_model,HighSensory,HandWritten=False,reduced_GS=False,store_results=True):

    umap_combinations, hdbscan_combinations =get_params_grid(len(data),reduced=reduced_GS)

    start_time = time.time()
    
    # Nested loop to iterate over each combination of UMAP and HDBSCAN parameters
    results = []

    for umap_config in tqdm(umap_combinations):
        for hdbscan_config in hdbscan_combinations:
            try:
                # Unpack the parameter sets
                n_components, n_neighbors, min_dist = umap_config
                min_cluster_size, min_samples = hdbscan_config
                
                # Execute the main function using unpacked parameters
                model, topics, coherence_score = main(
                    data=data,
                    vectorizer_model=vectorizer_model,
                    embedding_model=embedding_model,
                    n_neighbors=n_neighbors,
                    n_components=n_components,
                    min_dist=min_dist,
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples
                )
                # Store results
                results.append({
                    'n_components': n_components,
                    'n_neighbors': n_neighbors,
                    'min_dist': min_dist,
                    'min_cluster_size': min_cluster_size,
                    'min_samples': min_samples,
                    'coherence_score': coherence_score,
                    'n_topics':len(np.unique(topics)),
                    'model': model
                })
            except Exception as e:
                print(f"Error with parameters {umap_config}, {hdbscan_config}: {e}")
                continue
    results_df = pd.DataFrame(results).sort_values(by='coherence_score', ascending=False)
    print(f"Grid search completed in {time.time() - start_time:.2f} seconds")

    if store_results:
        name_file = f'RESULTS/grid_search_results_{"HighSensory" if HighSensory else "DeepListening" if not HandWritten else "HandWritten"}_seed{random_state}.csv'
        if os.path.isfile(name_file):
            results_df.to_csv(name_file, mode='a', header=False, index=False)
            # re sort the file by coherence score
            results_df = pd.read_csv(name_file).sort_values(by='coherence_score', ascending=False)

        else:
            results_df.to_csv(name_file, index=False)

    return results_df

def main(data,
         vectorizer_model,
         embedding_model,
         n_neighbors, 
         n_components, 
         min_dist, 
         min_cluster_size, 
         min_samples=None,
         top_n_words = 5,
         nr_topics = None):
    
    '''
    Defined in BERT_DM/BERTopic_hypertuned_multiprocessing.py

    Return : 
    - model : implemented BERTopic model with fine-tuned hyperparameters
    - topics : contains a one-to-one mapping of inputs to their modeled topic (or cluster)
    - probs : contains the probability of each document belonging to their assigned topic

    '''
    print(f"Received parameters: n_neighbors={n_neighbors}, n_components={n_components}, min_dist={min_dist}, min_cluster_size={min_cluster_size}, min_samples={min_samples}")


    # ********** Instanciate BERTOPIC **********

    umap_model = UMAP(n_neighbors=n_neighbors,
                      n_components=n_components,
                      min_dist=min_dist,
                      metric = 'cosine',
                      random_state=random_state) # rdm seed for reportability
    
    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, 
                            min_samples=min_samples,
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
    coherence_score = calculate_coherence(model, data)
    print("Coherence Score:", coherence_score)
    print(f"n = {len(np.unique(topics))} topics extracted")

    return model, topics, coherence_score
