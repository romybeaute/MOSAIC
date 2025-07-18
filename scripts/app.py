
import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import nltk
import json


from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, LlamaCPP
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN


import datamapplot
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt


st.set_page_config(page_title="Topic Modelling Visualisation for hyperparameter tuning and interpretability", layout="wide")


DATASETS = {
    "High Sensory (HS)": '/Users/rbeaute/Projects/MOSAIC/DATA/multilingual/english/dreamachine/HS_reflections_cleaned.csv',
    "Deep Listening (DL)": '/Users/rbeaute/Projects/MOSAIC/DATA/multilingual/english/dreamachine/DL_reflections_cleaned.csv'
}
HISTORY_FILE = "run_history.json"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# --- Default Parameters for Reset Functionality ---
DEFAULT_PARAMS = {
    'use_vectorizer': True,
    'ngram_min': 1, 'ngram_max': 3,
    'min_df': 1, 'max_df': 1.0,
    'stopwords': None,
    'umap_neighbors': 15, 'umap_components': 5, 'umap_min_dist': 0.0,
    'hdbscan_min_cluster_size': 40, 'hdbscan_min_samples': 40,
    'bt_nr_topics': 'auto', 'bt_top_n_words': 10
}


def load_history():
    """Loads the run history from a JpriSON file."""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return [] # Return empty list if file is corrupted or empty
    return []

def save_history(history_data):
    """Saves the run history to a JSON file."""
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history_data, f, indent=4)

@st.cache_resource
def load_llm_model():
    model_name_or_path = "NousResearch/Meta-Llama-3-8B-Instruct-GGUF"
    model_basename = "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
    model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename, cache_dir="model")
    return Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=4096, stop=["Q:", "\n"], verbose=False)

# caching per-dataset by passing file paths
@st.cache_data
def load_precomputed_data(docs_file, embeddings_file):
    """Loads docs and embeddings from specified file paths."""
    docs = np.load(docs_file, allow_pickle=True).tolist()
    embeddings = np.load(embeddings_file, allow_pickle=True)
    return docs, embeddings

def get_config_hash(config):
    """Creates a hashable representation of the config dict for caching."""
    return json.dumps(config, sort_keys=True)

@st.cache_data
def perform_topic_modeling(_docs, _embeddings, config_hash):
    """Performs the main BERTopic modeling. Uses a config hash to manage caching."""
    config = json.loads(config_hash)

    if 'ngram_range' in config['vectorizer_params']:
        config['vectorizer_params']['ngram_range'] = tuple(config['vectorizer_params']['ngram_range'])

    llm = load_llm_model()

    prompt = "Q:\nYou are an expert in micro-phenomenology. The following documents are reflections from participants about their minimal phenomenal experiences. I have a topic that contains the following documents:\n[DOCUMENTS]\nThe topic is described by the following keywords: '[KEYWORDS]'.\nBased on the above information, can you give an informative label of the topic of at most 10 words?\nA:"
    representation_model = {"LLM": LlamaCPP(llm, prompt=prompt)}

    umap_model = UMAP(random_state=42, metric='cosine', **config['umap_params'])
    hdbscan_model = HDBSCAN(metric='euclidean', prediction_data=True, **config['hdbscan_params'])
    vectorizer_model = CountVectorizer(**config['vectorizer_params']) if config['use_vectorizer'] else None

    nr_topics_val = None if config['bt_params']['nr_topics'] == 'auto' else int(config['bt_params']['nr_topics'])

    topic_model = BERTopic(
        umap_model=umap_model, hdbscan_model=hdbscan_model, vectorizer_model=vectorizer_model,
        representation_model=representation_model, top_n_words=config['bt_params']['top_n_words'],
        nr_topics=nr_topics_val, verbose=False
    )
    topics, _ = topic_model.fit_transform(_docs, _embeddings)

    info = topic_model.get_topic_info()
    outlier_perc = (info.Count[info.Topic == -1].iloc[0] / info.Count.sum()) * 100 if -1 in info.Topic.values else 0
    llm_labels_raw = [label[0][0].split("\n")[0].replace('"', '') for label in topic_model.get_topics(full=True)["LLM"].values()]
    llm_labels_cleaned = [re.sub(r'\W+', ' ', label).strip() for label in llm_labels_raw]

    final_topic_labels = [label if label else "Unlabelled" for label in llm_labels_cleaned]
    all_labels = [final_topic_labels[topic + topic_model._outliers] if topic != -1 else "Unlabelled" for topic in topics]

    reduced_embeddings = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine', random_state=42).fit_transform(_embeddings)

    return topic_model, reduced_embeddings, all_labels, len(info) -1, outlier_perc

@st.cache_data
def generate_hdbscan_tree(_embeddings, umap_params, hdbscan_params):
    """Generates the HDBSCAN condensed tree plot for pre-analysis."""
    umap_model = UMAP(random_state=42, **umap_params)
    u = umap_model.fit_transform(_embeddings)

    clusterer = HDBSCAN(prediction_data=True, **hdbscan_params)
    clusterer.fit(u)

    clusterer.condensed_tree_.plot(select_clusters=True)
    plt.title("HDBSCAN Condensed Tree")
    fig = plt.gcf()
    return fig

# ONE-TIME DATA PREPARATION (if pre-computed files do not exist)

def generate_and_save_embeddings(csv_path, docs_file, embeddings_file):
    """
    Performs one-time data preparation and saves to dataset-specific files.
    """
    st.info(f"Performing one-time data preparation for {os.path.basename(csv_path)}...")

    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        st.info("Downloading 'punkt' tokenizer...")
        nltk.download('punkt')
        st.info("Download complete.")

    with st.spinner('Reading data and splitting into sentences...'):
        df = pd.read_csv(csv_path, quoting=3, on_bad_lines='skip')

        df['reflection_answer'] = df['reflection_answer'].str.replace('"', '', regex=False) # Remove quotes 
        # df = df[df['reflection_answer'].str.split().str.len() > 1] # delete lines that have only one word
        df.reset_index(drop=True, inplace=True)
        reports = df['reflection_answer'].tolist()
        print(f"Loaded {len(reports)} (translated) documents for BERTopic modeling.")

        reports_sentences = [nltk.sent_tokenize(report) for report in reports]
        sentences = [sentence for report in reports_sentences for sentence in report]

    with st.spinner('Filtering short sentences...'):
        min_word_count = 2
        filtered_sentences = [s for s in sentences if len(s.split()) > min_word_count]
    docs = filtered_sentences

    with st.spinner(f"Generating embeddings for {len(docs)} sentences with '{EMBEDDING_MODEL}'..."):
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        embeddings = embedding_model.encode(docs, show_progress_bar=True)

    with st.spinner(f"Saving pre-computed data to {docs_file} and {embeddings_file}..."):
        np.save(docs_file, np.array(docs, dtype=object))
        np.save(embeddings_file, embeddings)

    st.success("Data preparation and saving complete! The app will now reload.")
    st.balloons()

# MAIN APP INTERFACE 
st.title("Dreamachine Topic Modelling Dashboard for hyperparameter tuning and interpretability")


st.sidebar.header("Data Source")
selected_dataset_name = st.sidebar.selectbox(
    "Choose a dataset",
    options=list(DATASETS.keys()),
    key="dataset_selector"
)
CSV_PATH = DATASETS[selected_dataset_name]

def get_precomputed_filenames(csv_path):
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    docs_file = f"precomputed_{base_name}_docs.npy"
    embeddings_file = f"precomputed_{base_name}_embeddings.npy"
    return docs_file, embeddings_file

DOCS_FILE, EMBEDDINGS_FILE = get_precomputed_filenames(CSV_PATH)

st.sidebar.info(f"Using dataset: `{os.path.basename(CSV_PATH)}`")

if not os.path.exists(EMBEDDINGS_FILE):
    st.warning(f"Pre-computed data for '{selected_dataset_name}' not found.")
    if st.button(f"Prepare Data for '{selected_dataset_name}' (One-Time Setup)"):
        generate_and_save_embeddings(CSV_PATH, DOCS_FILE, EMBEDDINGS_FILE)
        st.rerun()
else:

    docs, embeddings = load_precomputed_data(DOCS_FILE, EMBEDDINGS_FILE)

    # --- Sidebar UI for model parameters ---
    st.sidebar.header("Model Parameters")
    if 'params' not in st.session_state:
        st.session_state.params = DEFAULT_PARAMS.copy()

    def update_param(key):
        st.session_state.params[key] = st.session_state[f"widget_{key}"]

    st.sidebar.checkbox("Use CountVectorizer", key="widget_use_vectorizer", value=st.session_state.params['use_vectorizer'], on_change=update_param, args=('use_vectorizer',))
    with st.sidebar.expander("CountVectorizer Parameters"):
        st.slider("Min N-gram", 1, 5, key="widget_ngram_min", value=st.session_state.params['ngram_min'], on_change=update_param, args=('ngram_min',))
        st.slider("Max N-gram", 1, 5, key="widget_ngram_max", value=st.session_state.params['ngram_max'], on_change=update_param, args=('ngram_max',))
        st.slider("Min Doc Frequency (min_df)", 1, 50, key="widget_min_df", value=st.session_state.params['min_df'], on_change=update_param, args=('min_df',))
        st.select_slider("Stopwords", ['english', None], key="widget_stopwords", value=st.session_state.params['stopwords'], on_change=update_param, args=('stopwords',))
    with st.sidebar.expander("UMAP Parameters"):
        st.slider("n_neighbors", 2, 50, key="widget_umap_neighbors", value=st.session_state.params['umap_neighbors'], on_change=update_param, args=('umap_neighbors',))
        st.slider("n_components", 2, 50, key="widget_umap_components", value=st.session_state.params['umap_components'], on_change=update_param, args=('umap_components',))
        st.slider("min_dist", 0.0, 0.99, key="widget_umap_min_dist", value=st.session_state.params['umap_min_dist'], on_change=update_param, args=('umap_min_dist',), step=0.01)
    with st.sidebar.expander("HDBSCAN Parameters"):
        st.slider("min_cluster_size", 10, 200, key="widget_hdbscan_min_cluster_size", value=st.session_state.params['hdbscan_min_cluster_size'], on_change=update_param, args=('hdbscan_min_cluster_size',))
        st.slider("min_samples", 1, 100, key="widget_hdbscan_min_samples", value=st.session_state.params['hdbscan_min_samples'], on_change=update_param, args=('hdbscan_min_samples',))
    with st.sidebar.expander("BERTopic Parameters"):
        st.text_input("Number of Topics (nr_topics)", key="widget_bt_nr_topics", value=st.session_state.params['bt_nr_topics'], on_change=update_param, args=('bt_nr_topics',))
        st.caption("Enter 'auto' or a number.")
        st.slider("Top N Words", 5, 25, key="widget_bt_top_n_words", value=st.session_state.params['bt_top_n_words'], on_change=update_param, args=('bt_top_n_words',))

    col1, col2 = st.sidebar.columns(2)
    run_button = col1.button("Run Analysis", type="primary")
    if col2.button("Reset Params"):
        st.session_state.params = DEFAULT_PARAMS.copy()
        st.rerun()

    main_tab, hdbscan_tab, history_tab = st.tabs(["Main Results", "HDBSCAN Analysis", "Run History"])

    # load history from file at the start 
    if 'history' not in st.session_state:
        st.session_state.history = load_history()

    current_config = {
        "use_vectorizer": st.session_state.params['use_vectorizer'],
        "vectorizer_params": {'ngram_range': tuple((st.session_state.params['ngram_min'], st.session_state.params['ngram_max'])), 'min_df': st.session_state.params['min_df'], 'stop_words': st.session_state.params['stopwords']},
        "umap_params": {'n_neighbors': st.session_state.params['umap_neighbors'], 'n_components': st.session_state.params['umap_components'], 'min_dist': st.session_state.params['umap_min_dist']},
        "hdbscan_params": {'min_cluster_size': st.session_state.params['hdbscan_min_cluster_size'], 'min_samples': st.session_state.params['hdbscan_min_samples']},
        "bt_params": {'nr_topics': st.session_state.params['bt_nr_topics'], 'top_n_words': st.session_state.params['bt_top_n_words']}
    }

    if run_button:
        with st.spinner('Performing topic modeling... This may take a moment.'):
            topic_model, reduced_embeddings, all_labels, num_topics, outlier_perc = perform_topic_modeling(docs, embeddings, get_config_hash(current_config))
            st.session_state.latest_results = (topic_model, reduced_embeddings, all_labels)

            history_entry = {
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dataset_name": selected_dataset_name,
                "config": current_config,
                "num_topics": num_topics,
                "outlier_perc": f"{outlier_perc:.2f}%",
                "llm_labels": [label for label in topic_model.get_topic_info().Name.values if "Unlabelled" not in label and "outlier" not in label]
            }
            st.session_state.history.insert(0, history_entry)
            save_history(st.session_state.history)

    with main_tab:
        st.header("Topic Visualization")
        if 'latest_results' in st.session_state:
            _, reduced_embeddings, all_labels = st.session_state.latest_results
            with st.spinner('Creating plot...'):
                fig, _ = datamapplot.create_plot(
                    reduced_embeddings, all_labels, title="Dimensions of Experience ('experiential topics')"
                )
                st.pyplot(fig)
            st.subheader("Topic Information")
            st.dataframe(st.session_state.latest_results[0].get_topic_info())
        else:
            st.info("Click 'Run Analysis' in the sidebar to generate results.")

    with hdbscan_tab:
        st.header("HDBSCAN Condensed Tree Analysis")
        st.info("This plot helps you choose an appropriate `min_cluster_size` by visualizing the cluster hierarchy.")
        if st.button("Generate HDBSCAN Tree"):
            with st.spinner("Generating tree..."):
                tree_fig = generate_hdbscan_tree(embeddings, current_config['umap_params'], current_config['hdbscan_params'])
                st.pyplot(tree_fig)

    with history_tab:
        st.header("Run History")
        if not st.session_state.history:
            st.info(f"No runs have been performed yet. History will be saved to `{HISTORY_FILE}`.")
        else:
            st.info(f"History is loaded from and saved to `{HISTORY_FILE}`.")
            for i, entry in enumerate(st.session_state.history):
                title = f"Run {len(st.session_state.history) - i} ({entry.get('timestamp', 'N/A')}) on '{entry.get('dataset_name', 'Unknown')}'"
                with st.expander(title):
                    st.write(f"**Topics Found:** {entry['num_topics']} | **Outlier Percentage:** {entry['outlier_perc']}")
                    st.write("**Configuration**")
                    st.json(entry['config'], expanded=False)
                    st.write("**LLM-Generated Labels:**")
                    st.write(entry['llm_labels'])