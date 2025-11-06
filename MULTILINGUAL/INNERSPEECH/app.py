"""
File: app.py
Description: Streamlit app for advanced topic modeling on Innerspeech dataset with BERTopic and Llama integration.
Last Modified: 06/11/2025
@author: r.beaut@sussex.ac.uk
"""
from pathlib import Path
import sys
from mosaic.path_utils import CFG, raw_path, proc_path, eval_path, project_root


import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import nltk
import json

# BERTopic and related imports
from bertopic import BERTopic
from bertopic.representation import LlamaCPP
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

# Visualization
import datamapplot
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt

ROOT = project_root()
sys.path.append(str(ROOT / "MULTILINGUAL"))  # only if needed

# --- 1. CONFIGURATION & DEFAULTS ---
st.set_page_config(page_title="Advanced Topic Modeling", layout="wide")


################################################ DATASET CONFIGURATION ################################################
# --- File and Model paths (NOW USING MOSAIC PATH UTILS) ---
# The Innerspeech raw folder on Box is INNERSPEECH
DATASET = "INNERSPEECH"

RAW_DIR  = raw_path(DATASET)        # e.g., ~/Box-Box/TMDATA/INNERSPEECH
PROC_DIR = proc_path(DATASET,'preprocessed')       # e.g., ~/Projects/MOSAIC/DATA/innerspeech
EVAL_DIR = eval_path(DATASET)       # e.g., ~/Projects/MOSAIC/DATA/EVAL/innerspeech
CACHE_DIR = str(PROC_DIR / "cache")

# Ensure processed directories exist
(PROC_DIR).mkdir(parents=True, exist_ok=True)
(Path(CACHE_DIR)).mkdir(parents=True, exist_ok=True)
(Path(EVAL_DIR)).mkdir(parents=True, exist_ok=True)

DATASETS = {
    "API Translation (Batched)": str(PROC_DIR / "innerspeech_translated_batched_API.csv"),
    "Local Translation (Llama)": str(PROC_DIR / "innerspeech_dataset_translated_llama.csv"),
}

HISTORY_FILE = str(PROC_DIR / "run_history.json")

# # --- File and Model paths ---
# DATASETS = {
#     "API Translation (Batched)": '/Users/rbeaute/Projects/MOSAIC/DATA/multilingual/innerspeech_translated_batched_API.csv',
#     "Local Translation (Llama)": '/Users/rbeaute/Projects/MOSAIC/DATA/multilingual/japanese/innerspeech/innerspeech_dataset_translated_llama.csv'
# }

# HISTORY_FILE = "run_history.json"
# CACHE_DIR = "/Users/rbeaute/Projects/MOSAIC/DATA/multilingual/english/innerspeech/cache"


################################################################################################################################################


EMBEDDING_MODELS = (
    "intfloat/multilingual-e5-large-instruct",
    "Qwen/Qwen3-Embedding-0.6B",
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-mpnet-base-v2"
)

DEFAULT_PARAMS = {
    'embedding_model': EMBEDDING_MODELS[0],
    'split_into_sentences': True,
    'use_vectorizer': True,
    'ngram_min': 1, 'ngram_max': 3,
    'min_df': 1,
    'stopwords': None,
    'umap_neighbors': 15, 'umap_components': 5, 'min_dist': 0.0,
    'hdbscan_min_cluster_size': 40, 'hdbscan_min_samples': 30,
    'bt_nr_topics': 'auto', 'bt_top_n_words': 15
}

# --- 2. CORE LOGIC & CACHED FUNCTIONS ---

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            try: return json.load(f)
            except json.JSONDecodeError: return []
    return []

def save_history(history_data):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history_data, f, indent=4)

@st.cache_resource
def load_embedding_model(model_name):
    st.info(f"Loading embedding model '{model_name}' into memory...")
    return SentenceTransformer(model_name)

@st.cache_resource
def load_llm_model():
    model_name_or_path = "NousResearch/Meta-Llama-3-8B-Instruct-GGUF"
    model_basename = "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
    model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)
    return Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=8192, stop=["Q:", "\n"], verbose=False)

@st.cache_data
def load_precomputed_data(docs_file, embeddings_file):
    docs = np.load(docs_file, allow_pickle=True).tolist()
    embeddings = np.load(embeddings_file, allow_pickle=True)
    return docs, embeddings

def get_config_hash(config):
    return json.dumps(config, sort_keys=True)

@st.cache_data
def perform_topic_modeling(_docs, _embeddings, config_hash):
    # This function remains the same
    config = json.loads(config_hash)
    if 'ngram_range' in config['vectorizer_params']:
        config['vectorizer_params']['ngram_range'] = tuple(config['vectorizer_params']['ngram_range'])
    llm = load_llm_model()
    prompt = """Q:
You are an expert in micro-phenomenology. The following documents are reflections from participants about their experience. I have a topic that contains the following documents:
[DOCUMENTS]
The topic is described by the following keywords: '[KEYWORDS]'.
Based on the above information, give an informative, short label for this topic, between 5 and 10 words.
Instructions for your response:
- Do NOT start the label with 'experiences of'.
- Your response MUST be only the label itself.
- Do NOT include any introductory phrases, explanations, or quotation marks in your output.
A:"""
    representation_model = {"LLM": LlamaCPP(llm, prompt=prompt, nr_docs=25, doc_length=300, tokenizer="whitespace")}
    umap_model = UMAP(random_state=42, metric='cosine', **config['umap_params'])
    hdbscan_model = HDBSCAN(metric='euclidean', prediction_data=True, **config['hdbscan_params'])
    vectorizer_model = CountVectorizer(**config['vectorizer_params']) if config['use_vectorizer'] else None
    nr_topics_val = None if config['bt_params']['nr_topics'] == 'auto' else int(config['bt_params']['nr_topics'])
    topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, vectorizer_model=vectorizer_model, representation_model=representation_model, top_n_words=config['bt_params']['top_n_words'], nr_topics=nr_topics_val, verbose=False)
    topics, _ = topic_model.fit_transform(_docs, _embeddings)
    info = topic_model.get_topic_info()
    outlier_perc = (info.Count[info.Topic == -1].iloc[0] / info.Count.sum()) * 100 if -1 in info.Topic.values else 0
    llm_labels_raw = [label[0][0] for label in topic_model.get_topics(full=True)["LLM"].values()]
    llm_labels_cleaned = [label.split(':')[-1].strip().strip('"').strip('.').strip() if ':' in label else label.strip().strip('"').strip('.').strip() for label in llm_labels_raw]
    final_topic_labels = [label if label else "Unlabelled" for label in llm_labels_cleaned]
    all_labels = [final_topic_labels[topic + topic_model._outliers] if topic != -1 else "Unlabelled" for topic in topics]
    reduced_embeddings = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine', random_state=42).fit_transform(_embeddings)
    return topic_model, reduced_embeddings, all_labels, len(info) - 1, outlier_perc

def generate_and_save_embeddings(csv_path, docs_file, embeddings_file, selected_embedding_model, split_sentences, device):
    granularity_text = "sentences" if split_sentences else "reports"
    if not os.path.exists(docs_file):
        st.info(f"Preparing docs for {os.path.basename(csv_path)} at {granularity_text} level...")
        with st.spinner('Reading data...'):
            df = pd.read_csv(csv_path)
            df.dropna(subset=['reflection_answer_english'], inplace=True)
            df = df[df.reflection_answer_english.str.strip() != '']
            reports = df['reflection_answer_english'].tolist()
        if split_sentences:
            with st.spinner('Splitting into sentences...'):
                try: nltk.data.find('tokenizers/punkt')
                except nltk.downloader.DownloadError: st.info("Downloading 'punkt' tokenizer..."); nltk.download('punkt')
                sentences = [s for r in reports for s in nltk.sent_tokenize(r)]
            with st.spinner('Filtering short sentences...'):
                docs = [s for s in sentences if len(s.split()) > 2]
        else: docs = reports
        with st.spinner(f"Saving {len(docs)} documents..."):
            np.save(docs_file, np.array(docs, dtype=object))
        st.success("Document preparation complete.")
    else: docs = np.load(docs_file, allow_pickle=True).tolist()
    
    st.info(f"Generating embeddings with '{selected_embedding_model}' on {device} for {len(docs)} documents...")
    with st.spinner("This can take a while..."):
        embedding_model_obj = load_embedding_model(selected_embedding_model)
        
        encode_device = None
        batch_size = 32 # Default for GPU
        if device == 'CPU':
            encode_device = 'cpu'
            batch_size = 64 # Use a larger batch size for CPU
        
        embeddings = embedding_model_obj.encode(docs, show_progress_bar=True, batch_size=batch_size, device=encode_device)
        
    with st.spinner(f"Saving embeddings..."):
        np.save(embeddings_file, embeddings)
    st.success("Embedding generation complete!"); st.balloons(); st.rerun()

# --- 4. MAIN APP INTERFACE ---
st.title("Topic Modelling Dashboard for Innerspeech Data: Fine-Tuning with visualisation")

# --- SIDEBAR ---
st.sidebar.header("Data Source & Model")
selected_dataset_name = st.sidebar.selectbox("Choose a dataset", options=list(DATASETS.keys()))
selected_embedding_model = st.sidebar.selectbox("Choose an embedding model", options=EMBEDDING_MODELS)

st.sidebar.markdown("[See model performance on the MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)")
# (MODEL_SPECIFICATIONS dictionary remains the same)
MODEL_SPECIFICATIONS = {"intfloat/multilingual-e5-large-instruct": {"description": "A powerful multilingual model, great for diverse languages. Requires more computational resources.","size": "2.24 GB","speed": "Medium",},"Qwen/Qwen3-Embedding-0.6B": {"description": "A very strong multilingual embedding model from Alibaba Cloud. Recommended for best results.","size": "0.6B parameters","speed": "Fast",},"BAAI/bge-small-en-v1.5": {"description": "A fast and efficient English-only model. Good for quick experiments on English text.","size": "67 MB","speed": "Very Fast",},"sentence-transformers/all-mpnet-base-v2": {"description": "A classic, well-balanced model for English. A solid baseline choice.","size": "438 MB","speed": "Fast",}}
spec = MODEL_SPECIFICATIONS.get(selected_embedding_model)
if spec: st.sidebar.info(f"**Model Details:**\n- **Description**: {spec['description']}\n- **Size**: {spec['size']}\n- **Speed**: {spec['speed']}")

st.sidebar.header("Data Preparation")

# --- NEW DEVICE SELECTOR ---
selected_device = st.sidebar.radio(
    "Select processing device",
    ['GPU (MPS)', 'CPU'],
    index=0,
    help="Choose 'CPU' if you encounter 'MPS backend out of memory' errors. CPU is slower but has access to more memory."
)

selected_granularity = st.sidebar.checkbox("Split reports into sentences", value=DEFAULT_PARAMS['split_into_sentences'])
granularity_label = "Sentences" if selected_granularity else "Reports"

st.sidebar.header("Performance Tuning")
subsample_perc = st.sidebar.slider("Subsample for tuning (%)", 10, 100, 100, 5, help="For faster tuning, use a fraction of data.")

def get_precomputed_filenames(csv_path, model_name, split_sentences):
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    model_safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', model_name)
    granularity_suffix = "sentences" if split_sentences else "reports"
    docs_filename = f"precomputed_{base_name}_{granularity_suffix}_docs.npy"
    embeddings_filename = f"precomputed_{base_name}_{model_safe_name}_{granularity_suffix}_embeddings.npy"
    # docs_file_path = os.path.join(CACHE_DIR, docs_filename)
    docs_file_path = str(Path(CACHE_DIR) / docs_filename)
    embeddings_file_path = str(Path(CACHE_DIR) / embeddings_filename)
    return docs_file_path, embeddings_file_path


CSV_PATH = DATASETS[selected_dataset_name]
DOCS_FILE, EMBEDDINGS_FILE = get_precomputed_filenames(CSV_PATH, selected_embedding_model, selected_granularity)

if not os.path.exists(EMBEDDINGS_FILE):
    st.warning(f"Pre-computed data for **'{granularity_label}'** with model **'{selected_embedding_model}'** not found.")
    if st.button("Prepare Data for this Configuration"):
        # Pass the selected device to the function
        generate_and_save_embeddings(CSV_PATH, DOCS_FILE, EMBEDDINGS_FILE, selected_embedding_model, selected_granularity, selected_device)
else:
    # (The rest of the script remains the same)
    docs, embeddings = load_precomputed_data(DOCS_FILE, EMBEDDINGS_FILE)
    if subsample_perc < 100:
        st.warning(f"⚠️ Analysis is running on a {subsample_perc}% subsample of the data.")
        sample_size = int(len(docs) * (subsample_perc / 100))
        random_indices = np.random.choice(len(docs), size=sample_size, replace=False)
        docs = [docs[i] for i in random_indices]
        embeddings = np.array(embeddings)[random_indices]
    doc_count_label = f"{len(docs)} ({subsample_perc}%)" if subsample_perc < 100 else f"{len(docs)}"
    st.metric("Documents to Analyze", doc_count_label, granularity_label)
    st.sidebar.header("Model Parameters")
    use_vectorizer = st.sidebar.checkbox("Use CountVectorizer", value=DEFAULT_PARAMS['use_vectorizer'])
    with st.sidebar.expander("CountVectorizer Parameters", expanded=True):
        ngram_min = st.slider("Min N-gram", 1, 5, DEFAULT_PARAMS['ngram_min'], help="The minimum size of word sequences to consider as a keyword. A value of 1 means single words (unigrams).")
        ngram_max = st.slider("Max N-gram", 1, 5, DEFAULT_PARAMS['ngram_max'], help="The maximum size of word sequences. A value of 3 means sequences of up to three words (trigrams) can be keywords.")
        min_df = st.slider("Min Doc Frequency", 1, 50, DEFAULT_PARAMS['min_df'], step=5, help="The minimum number of documents a word must appear in to be considered a keyword. Helps remove very rare words.")
        stopwords = st.select_slider("Stopwords", ['english', None], value=DEFAULT_PARAMS['stopwords'], help="Removes common words (like 'the', 'is', 'a'). Select 'english' for English text. Select 'None' to not remove any.")
    with st.sidebar.expander("UMAP Parameters", expanded=True):
        umap_neighbors = st.slider("n_neighbors", 2, 50, DEFAULT_PARAMS['umap_neighbors'], step=5, help="Controls how UMAP balances local versus global structure. Lower values focus on local details, higher values on the bigger picture.")
        umap_components = st.slider("n_components", 2, 50, DEFAULT_PARAMS['umap_components'], step=5, help="The number of dimensions to reduce the data to before clustering. The default is often sufficient.")
        umap_min_dist = st.slider("min_dist", 0.0, 0.99, DEFAULT_PARAMS['min_dist'], step=0.01, help="Controls how tightly UMAP packs points together. Lower values create more dense clusters, higher values create more dispersed ones.")
    with st.sidebar.expander("HDBSCAN Parameters", expanded=True):
        hdbscan_min_cluster_size = st.slider("min_cluster_size", 10, 200, DEFAULT_PARAMS['hdbscan_min_cluster_size'], step=5, help="The smallest number of documents required to form a distinct topic. Higher values lead to fewer, broader topics.")
        hdbscan_min_samples = st.slider("min_samples", 2, 100, DEFAULT_PARAMS['hdbscan_min_samples'], step=5, help="How conservative the clustering is. Higher values make the model more likely to declare documents as outliers rather than assigning them to a topic.")
    with st.sidebar.expander("BERTopic Parameters", expanded=True):
        bt_nr_topics = st.text_input("Number of Topics (nr_topics)", value=DEFAULT_PARAMS['bt_nr_topics'], help="Set to a specific number to merge topics until you have that many. Set to 'auto' to let HDBSCAN decide.")
        bt_top_n_words = st.slider("Top N Words", 5, 25, DEFAULT_PARAMS['bt_top_n_words'], help="The number of keywords to generate for each topic's basic representation.")
    current_config = {"embedding_model": selected_embedding_model, "granularity": granularity_label, "subsample_percent": subsample_perc, "use_vectorizer": use_vectorizer, "vectorizer_params": {'ngram_range': (ngram_min, ngram_max), 'min_df': min_df, 'stop_words': stopwords}, "umap_params": {'n_neighbors': umap_neighbors, 'n_components': umap_components, 'min_dist': umap_min_dist}, "hdbscan_params": {'min_cluster_size': hdbscan_min_cluster_size, 'min_samples': hdbscan_min_samples}, "bt_params": {'nr_topics': bt_nr_topics, 'top_n_words': bt_top_n_words}}
    run_button = st.sidebar.button("Run Analysis", type="primary")
    main_tab, history_tab = st.tabs(["Main Results", "Run History"])
    if 'history' not in st.session_state: st.session_state.history = load_history()
    if run_button:
        with st.spinner('Performing topic modeling...'):
            topic_model, reduced_embeddings, all_labels, num_topics, outlier_perc = perform_topic_modeling(docs, embeddings, get_config_hash(current_config))
            st.session_state.latest_results = (topic_model, reduced_embeddings, all_labels)
            history_entry = {"timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), "config": current_config, "num_topics": num_topics, "outlier_perc": f"{outlier_perc:.2f}%", "llm_labels": [l for l in topic_model.get_topic_info().Name.values if "Unlabelled" not in l and "outlier" not in l]}
            st.session_state.history.insert(0, history_entry)
            save_history(st.session_state.history)
            st.rerun()
    with main_tab:
        if 'latest_results' in st.session_state:
            st.header("Topic Visualization")
            fig, _ = datamapplot.create_plot(st.session_state.latest_results[1], st.session_state.latest_results[2])
            st.pyplot(fig)
            st.subheader("Topic Information")
            st.dataframe(st.session_state.latest_results[0].get_topic_info())
        else: st.info("Click 'Run Analysis' in the sidebar to generate results.")
    with history_tab:
        st.header("Run History")
        if not st.session_state.history: st.info(f"No runs yet. History will be saved to `{HISTORY_FILE}`.")
        else:
            for i, entry in enumerate(st.session_state.history):
                config_data = entry.get('config', {})
                title_model = config_data.get('embedding_model', entry.get('embedding_model', 'Unknown'))
                title_granularity = config_data.get('granularity', entry.get('granularity', 'Unknown'))
                title = f"Run {len(st.session_state.history)-i} ({entry.get('timestamp', 'N/A')}) - {title_model} ({title_granularity})"
                with st.expander(title):
                    st.write(f"**Topics:** `{entry.get('num_topics')}` | **Outliers:** `{entry.get('outlier_perc')}`")
                    st.write("**LLM-Generated Labels:**")
                    st.write(entry.get('llm_labels', []))
                    with st.expander("Show full configuration for this run"): st.json(config_data)