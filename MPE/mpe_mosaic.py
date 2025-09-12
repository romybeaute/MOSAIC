import pandas as pd
import numpy as np
import re
import streamlit as st
import nltk
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

# BERTopic and related imports
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, LlamaCPP
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

# Visualization
import datamapplot
import matplotlib.pyplot as plt

# Ensure the punkt tokenizer is downloaded
@st.cache_resource
def download_nltk_punkt():
    """Downloads the NLTK punkt tokenizer if not already present."""
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        nltk.download('punkt')

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_and_prepare_data(csv_path):
    """
    Loads data from a CSV, preprocesses it, and tokenizes it into sentences.
    This function is cached to avoid reloading and reprocessing the data on every run.
    """
    download_nltk_punkt() # Ensure tokenizer is available

    df = pd.read_csv(csv_path)
    df.dropna(subset=['phen_report_english'], inplace=True)
    df = df[df.phen_report_english.str.strip() != '']

    reports = df['phen_report_english'].tolist()
    reports_sentences = [nltk.sent_tokenize(report) for report in reports]
    sentences = [sentence for report in reports_sentences for sentence in report]
    return sentences

# --- Model Loading (with Caching) ---
@st.cache_resource
def load_llm_model():
    """
    Downloads and loads the Llama GGUF model.
    Using st.cache_resource to ensure the model is loaded only once.
    """
    model_name_or_path = "NousResearch/Meta-Llama-3-8B-Instruct-GGUF"
    model_basename = "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
    model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename, cache_dir="model")
    
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1, # Offload all possible layers to GPU
        n_ctx=4096,
        stop=["Q:", "\n"],
        verbose=False
    )
    return llm

@st.cache_resource
def load_embedding_model(model_name="BAAI/bge-small-en-v1.5"):
    """
    Loads a SentenceTransformer embedding model.
    Cached to prevent reloading on each interaction.
    """
    return SentenceTransformer(model_name)

# --- Core Topic Modeling ---
@st.cache_data
def get_embeddings(docs, model_name="BAAI/bge-small-en-v1.5"):
    """
    Generates and caches document embeddings.
    """
    embedding_model = load_embedding_model(model_name)
    embeddings = embedding_model.encode(docs, show_progress_bar=False)
    return embeddings

@st.cache_data
def perform_topic_modeling(_docs, _embeddings, umap_params, hdbscan_params, use_vectorizer):
    """
    Performs the main BERTopic modeling.
    The function is cached, so it only reruns if the parameters change.
    The leading underscores in args tell Streamlit to hash the data instead of the object id.
    """
    llm = load_llm_model()

    # Define Representation Model
    prompt = """Q:
I have a topic that contains the following documents:
[DOCUMENTS]
The topic is described by the following keywords: '[KEYWORDS]'.
Based on the above information, can you give a short label of the topic of at most 5 words?
A:
"""
    representation_model = {
        "KeyBERT": KeyBERTInspired(),
        "LLM": LlamaCPP(llm, prompt=prompt),
    }
    
    # Define Sub-models based on parameters
    umap_model = UMAP(
        n_neighbors=umap_params['n_neighbors'],
        n_components=umap_params['n_components'],
        min_dist=umap_params['min_dist'],
        metric='cosine',
        random_state=42
    )
    
    hdbscan_model = HDBSCAN(
        min_cluster_size=hdbscan_params['min_cluster_size'],
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )

    vectorizer_model = CountVectorizer(ngram_range=(1, 3), stop_words='english') if use_vectorizer else None

    # Create and train BERTopic model
    topic_model = BERTopic(
        embedding_model=load_embedding_model(), # Re-use the cached model
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        top_n_words=10,
        verbose=True
    )

    topics, probs = topic_model.fit_transform(_docs, _embeddings)

    # Generate labels
    llm_labels_raw = [label[0][0].split("\n")[0].replace('"', '') for label in topic_model.get_topics(full=True)["LLM"].values()]
    llm_labels_cleaned = [re.sub(r'\W+', ' ', label).strip() for label in llm_labels_raw]
    llm_labels = [label if label else "Unlabelled" for label in llm_labels_cleaned]
    
    all_labels = [llm_labels[topic + topic_model._outliers] if topic != -1 else "Unlabelled" for topic in topics]
    
    # Pre-reduce embeddings for visualization
    reduced_embeddings = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine', random_state=42).fit_transform(_embeddings)

    return topic_model, reduced_embeddings, all_labels

def create_plot(reduced_embeddings, all_labels):
    """Generates the datamapplot visualization."""
    fig, ax = datamapplot.create_plot(
        reduced_embeddings,
        all_labels,
        label_font_size=11,
        title="MPE dimensions of experience",
        sub_title="Topics labeled with `llama-3-8b-instruct`",
        label_wrap_width=10
    )
    return fig