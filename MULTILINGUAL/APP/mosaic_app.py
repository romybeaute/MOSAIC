'''
Optimised Topic Modelling Parameter Explorer for Multilingual 
With cached computations for faster exploration and dataset reduction
'''

# set env variables to manage threading issues
import os
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
os.environ["MKL_THREADING_LAYER"] = "sequential"
import threading
threading.current_thread().name = "MainThread"

import pickle
import sys
import glob
import pandas as pd
import importlib
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
from nltk.tokenize import PunktSentenceTokenizer

import streamlit as st

try:
    nltk.download('punkt')
    nltk.download('punkt_tab')
except Exception as e:
    st.warning(f"Failed to download NLTK resources: {str(e)}")



project = "multilingual"
MOSAIC_dir = "/Users/rbeaute/Projects/MOSAIC"  
DATA_dir = os.path.join(MOSAIC_dir, f'DATA/{project}')
cache_dir = os.path.join(DATA_dir, "cache")

#import utility functions for multiling 
multiling_path = os.path.join(MOSAIC_dir, "MULTILINGUAL")
if multiling_path not in sys.path:
    sys.path.insert(0, multiling_path)

# import language processors from multiling_helpers
try:
    from multiling_helpers import EnglishProcessor, FrenchProcessor, JapaneseProcessor, PortugueseProcessor
except ImportError as e:
    st.error(f"Error importing language processors: {e}")


# Custom caching function to avoid Streamlit caching issues
def cached_function(func):
    """Simple caching decorator that uses dictionary to store results"""
    cache = {}
    
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    
    return wrapper


# Function to get available datasets for a specific language
def get_available_datasets(language):
    """Get list of available datasets for a given language"""
    language_dir = os.path.join(DATA_dir, language.lower())
    if os.path.exists(language_dir):
        # Get all subdirectories in the language directory
        datasets = [os.path.basename(d) for d in glob.glob(os.path.join(language_dir, "*")) 
                   if os.path.isdir(d)]
        return datasets
    else:
        return []


# Function to load raw data file
@cached_function
def load_raw_data(language, dataset):
    """Load raw test_reports.pkl file from the dataset directory"""
    dataset_path = os.path.join(DATA_dir, language.lower(), dataset)
    raw_data_path = os.path.join(dataset_path, f"{dataset}_reports.pkl")
    
    try:
        with open(raw_data_path, "rb") as f:
            raw_data = pickle.load(f)
        return raw_data
    except FileNotFoundError as e:
        st.error(f"Could not find raw data file ({dataset}_reports.pkl) for {dataset} in {language}: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error loading raw data: {str(e)}")
        return None



# supp caching functions for processed data
@cached_function
def load_processed_data(language, dataset):
    """Load preprocessed data and embeddings from cache"""
    dataset_cache_path = os.path.join(cache_dir, language.lower(), dataset)
    
    try:
        with open(os.path.join(dataset_cache_path, "processed_data.pkl"), "rb") as f:
            processed_data = pickle.load(f)
        with open(os.path.join(dataset_cache_path, "embeddings.pkl"), "rb") as f:
            embeddings = pickle.load(f)
        return processed_data, embeddings
    except FileNotFoundError as e:
        st.error(f"Could not find cache files for {dataset} in {language}: {str(e)}")
        return None, None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None


def display_data_sample(data, title="Sample Data"):
    """Display a sample of data in an appropriate format"""
    st.subheader(title)
    
    if data is None:
        st.warning("No data available to display.")
        return
        
    # Determine the data type and display accordingly
    if hasattr(data, 'shape'):  
        num_items = data.shape[0]
        st.write(f"Total items: **{num_items}**")
        
        st.dataframe(data.head(10)) #check first 10 items
        
    elif isinstance(data, list):  # For list
        num_items = len(data)
        st.write(f"Total items: **{num_items}**")
        
        # Display first 10 items
        if data and num_items > 0:
            if isinstance(data[0], dict):
                # if list of dictionaries
                df_sample = pd.DataFrame(data[:10])
                st.dataframe(df_sample)
            elif isinstance(data[0], str):
                # if a list of strings
                for i, item in enumerate(data[:10]):
                    with st.expander(f"Item #{i+1}"):
                        st.write(item)
            else:
                # for other data types
                st.write("First 10 reports:")
                for i, item in enumerate(data[:10]):
                    st.write(f"Item {i+1}: {str(item)[:500]}...")
    else:
        st.write(f"Data type: {type(data)}")
        st.write("Cannot display sample data for this data type.")


def main():
    # Set page config for title and icon
    st.set_page_config(
        page_title="Multilingual Topic Modelling Explorer",
        page_icon="🌐",
        layout="wide"
    )

    # Add GitHub link to upper right corner
    st.markdown(
        """
        <style>
        .github-corner {
            position: absolute;
            top: 0;
            right: 0;
        }
        </style>
        <a href="https://github.com/romybeaute/MOSAIC" class="github-corner" aria-label="View source on GitHub">
            <svg width="80" height="80" viewBox="0 0 250 250" style="fill:#FF8C00; color:#fff; position: absolute; top: 0; border: 0; right: 0;" aria-hidden="true">
                <path d="M0,0 L115,115 L130,115 L142,142 L250,250 L250,0 Z"></path>
                <path d="M128.3,109.0 C113.8,99.7 119.0,89.6 119.0,89.6 C122.0,82.7 120.5,78.6 120.5,78.6 C119.2,72.0 123.4,76.3 123.4,76.3 C127.3,80.9 125.5,87.3 125.5,87.3 C122.9,97.6 130.6,101.9 134.4,103.2" fill="currentColor" style="transform-origin: 130px 106px;" class="octo-arm"></path>
                <path d="M115.0,115.0 C114.9,115.1 118.7,116.5 119.8,115.4 L133.7,101.6 C136.9,99.2 139.9,98.4 142.2,98.6 C133.8,88.0 127.5,74.4 143.8,58.0 C148.5,53.4 154.0,51.2 159.7,51.0 C160.3,49.4 163.2,43.6 171.4,40.1 C171.4,40.1 176.1,42.5 178.8,56.2 C183.1,58.6 187.2,61.8 190.9,65.4 C194.5,69.0 197.7,73.2 200.1,77.6 C213.8,80.2 216.3,84.9 216.3,84.9 C212.7,93.1 206.9,96.0 205.4,96.6 C205.1,102.4 203.0,107.8 198.3,112.5 C181.9,128.9 168.3,122.5 157.7,114.1 C157.9,116.9 156.7,120.9 152.7,124.9 L141.0,136.5 C139.8,137.7 141.6,141.9 141.8,141.8 Z" fill="currentColor" class="octo-body"></path>
            </svg>
        </a>
        """,
        unsafe_allow_html=True
    )

    st.title("Multilingual Topic Modelling Explorer")
    st.markdown("Explore topic models across different languages and datasets.")

    # init session state 
    if "language" not in st.session_state:
        st.session_state.language = "English"
    
    if "dataset" not in st.session_state:
        st.session_state.dataset = None


    ####################################################################
    # LANGUAGE SELECTION
    ####################################################################

    st.sidebar.markdown("---")
    st.sidebar.title("Language")

    available_languages = ["English", "Japanese", "French", "Portuguese (Br)"]
    language = st.sidebar.selectbox(
        "Select Language",
        available_languages,
        index=available_languages.index(st.session_state.language)
    )

    if language != st.session_state.language:
        st.session_state.language = language
        st.session_state.dataset = None  # Reset dataset when language changes
    
    # Get available datasets for the selected language
    available_datasets = get_available_datasets(language)


    ####################################################################
    # DATASET SELECTION
    ####################################################################
    st.sidebar.markdown("---")
    st.sidebar.title("Dataset")
    
    if available_datasets:
        dataset_index = 0
        if st.session_state.dataset in available_datasets:
            dataset_index = available_datasets.index(st.session_state.dataset)
            
        dataset = st.sidebar.selectbox(
            "Select Dataset",
            available_datasets,
            index=dataset_index
        )
        st.session_state.dataset = dataset
        
        # Display selected dataset info
        st.write(f"Selected Language: **{language}**")
        st.write(f"Selected Dataset: **{dataset}**")
        
        ####################################################################
        # DATA DISPLAY TABS
        ####################################################################
        tab1, tab2 = st.tabs(["Raw Data", "Processed Data"])
        
        with tab1:
            # Load and display raw data
            st.info(f"Loading raw data for {dataset} in {language}...")
            raw_data = load_raw_data(language, dataset)
            
            if raw_data is not None:
                st.success("Raw data loaded successfully!")
                
                # Add button to split sentences
                col1, col2 = st.columns([1, 3])
                with col1:
                    split_sentences_button = st.button("Split Sentences", key="split_sentences_btn")
                
                # Create tabs for original and split data
                orig_tab, split_tab = st.tabs(["Original Data", "Split Sentences"])
                
                with orig_tab:
                    display_data_sample(raw_data, "Raw Data - First 10 Reports")
                
                with split_tab:
                    if split_sentences_button:
                        st.info("Splitting sentences from raw data...")
                        
                        try:
                            language_map = {
                                "English": EnglishProcessor(),
                                "Japanese": JapaneseProcessor(),
                                "French": FrenchProcessor(),
                                "Portuguese (Br)": PortugueseProcessor()
                            }
            
                            processor = language_map.get(language)
                            if not processor:
                                st.error(f"No processor available for {language}")
                                return
                
                            # Directly use the processor's split_sentences method
                            sentences, doc_map = processor.split_sentences(raw_data)

                            
                            st.success(f"Split {len(raw_data)} documents into {len(sentences)} sentences")
                            
                            # Display some basic statistics
                            doc_counts = {}
                            for doc_idx in doc_map:
                                doc_counts[doc_idx] = doc_counts.get(doc_idx, 0) + 1
                                
                            st.write("Sentences per document:")
                            doc_sentences_df = pd.DataFrame({
                                'Document': list(doc_counts.keys()),
                                'Sentence Count': list(doc_counts.values())
                            })
                            st.dataframe(doc_sentences_df)
                            
                            # Show the first 10 sentences with their document mapping
                            st.subheader("First 10 Sentences")
                            for i in range(min(10, len(sentences))):
                                st.write(f"**Document {doc_map[i]+1}, Sentence {i+1}:** {sentences[i]}")
                            
                            # Show comparison for better understanding
                            st.subheader("Original vs Split Comparison")
                            
                            # Show first 3 documents with their sentences
                            for i in range(min(3, len(raw_data))):
                                with st.expander(f"Example #{i+1} - Original vs Split"):
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.subheader("Original")
                                        # Display original document
                                        st.text_area("Text", raw_data[i], height=300, disabled=True)
                                    
                                    with col2:
                                        st.subheader("Split Sentences")
                                        # Get all sentences for this document
                                        doc_sentences = [sentences[j] for j, doc_idx in enumerate(doc_map) if doc_idx == i]
                                        
                                        st.write(f"Document contains {len(doc_sentences)} sentences:")
                                        for j, sentence in enumerate(doc_sentences):
                                            st.write(f"{j+1}. {sentence}")
                                            
                        except Exception as e:
                            st.error(f"Error while splitting sentences: {str(e)}")
            else:
                st.warning(f"No raw data found for {dataset}. Check the path: {os.path.join(DATA_dir, language.lower(), dataset, f'{dataset}_reports.pkl')}")
        
        with tab2:
            # Load and display processed data
            st.info(f"Loading processed data for {dataset} in {language}...")
            processed_data, embeddings = load_processed_data(language, dataset)
            
            if processed_data is not None:
                st.success("Processed data loaded successfully!")
                display_data_sample(processed_data, "Processed Data - First 10 Reports")
                
                if embeddings is not None:
                    st.subheader("Embeddings Information")
                    if isinstance(embeddings, pd.DataFrame):
                        st.write(f"Embeddings shape: {embeddings.shape}")
                    elif hasattr(embeddings, 'shape'):  # For numpy arrays
                        st.write(f"Embeddings shape: {embeddings.shape}")
                    else:
                        st.write(f"Embeddings type: {type(embeddings)}")
            else:
                st.warning(f"No processed data found for {dataset}.")
    else:
        st.sidebar.warning(f"No datasets available for {language}. Please select another language.")


if __name__ == "__main__":
    main()