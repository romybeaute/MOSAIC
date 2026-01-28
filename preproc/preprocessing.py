#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : preprocessing.py
# description     : Unified preprocessing module with multiple cleaning methods
#                   - Sentence splitting and basic cleaning
#                   - Local LLM cleaning (using llama-cpp-python)
#                   - API-based cleaning (using Google Generative AI)
# author          : Romy, Beauté (r.beaut@sussex.ac.uk)
# date            : 2024-07-25
# ==============================================================================

import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path

# NLTK imports
from nltk.tokenize import PunktSentenceTokenizer
import nltk

# Optional imports - handled gracefully if not installed
try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False

try:
    from huggingface_hub import hf_hub_download
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False


# =============================================================================
# SECTION 1: BASIC TEXT PREPROCESSING
# =============================================================================

def split_sentences(reflections):
    """
    Split list of texts into sentences and track which sentence belongs to which document.
    
    Parameters:
    -----------
    reflections : list
        A list of strings (documents/reflections)
        
    Returns:
    --------
    tuple
        (sentences, doc_map) where:
        - sentences is a list of all sentences from all documents
        - doc_map is a list of indices indicating which document each sentence belongs to
    """
    tokenizer = PunktSentenceTokenizer()
    sentences = []
    doc_map = [] 
    
    for doc_idx, reflection in enumerate(reflections):
        doc_sentences = tokenizer.tokenize(reflection)
        sentences.extend(doc_sentences)
        doc_map.extend([doc_idx] * len(doc_sentences))
    
    return sentences, doc_map


def basic_preprocess(texts, split_into_sentences=True, min_words=2):
    """
    Basic text preprocessing: split sentences, remove short/duplicate texts.
    
    Parameters:
    -----------
    texts : list
        A list of strings to preprocess
    split_into_sentences : bool
        Whether to split texts into individual sentences
    min_words : int
        Minimum number of words required to keep a text
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with 'reflection_answer' and 'cleaned_reflection' columns
    """
    # Divide into sentences if needed
    if split_into_sentences:
        texts, doc_map = split_sentences(texts)
    else:
        doc_map = list(range(len(texts)))
    
    initial_count = len(texts)
    print(f"\nSuccessfully loaded {initial_count} texts.")

    # Remove texts with less than min_words
    filtered_texts = []
    filtered_doc_map = []
    for text, doc_idx in zip(texts, doc_map):
        if len(text.split()) >= min_words:
            filtered_texts.append(text)
            filtered_doc_map.append(doc_idx)
    
    texts = filtered_texts
    doc_map = filtered_doc_map
    print(f"After removing short texts (< {min_words} words), {len(texts)} remain.")

    # Remove duplicates
    seen = set()
    final_texts = []
    final_doc_map = []
    for text, doc_idx in zip(texts, doc_map):
        if not (text in seen or seen.add(text)):
            final_texts.append(text)
            final_doc_map.append(doc_idx)
    
    print(f"After removing duplicates, {len(final_texts)} remain.")
    
    # Return as DataFrame like the other methods
    return pd.DataFrame({
        'reflection_answer': final_texts,
        'cleaned_reflection': final_texts
    })


# =============================================================================
# SECTION 2: LOCAL LLM PREPROCESSING (using llama-cpp-python)
# =============================================================================

def preprocess_with_local_llama(csv_path, output_path, model_config, 
                                text_column='reflection_answer', num_samples=None):
    """
    Cleans a CSV column using a local Llama model, optimized for GPU support (Metal, CUDA, etc).
    
    Note: This requires llama-cpp-python which needs a C++ compiler.
    For Mac: Works with Metal acceleration automatically
    For GPU: Works with CUDA if configured properly

    Parameters:
    -----------
    csv_path : str
        Path to the input CSV file.
    output_path : str
        Path to save the new CSV file with cleaned text.
    model_config : dict
        Dictionary containing:
        - 'repo': HuggingFace repo ID (e.g., 'NousResearch/Meta-Llama-3-8B-Instruct-GGUF')
        - 'filename': Model filename (e.g., 'Meta-Llama-3-8B-Instruct-Q4_K_M.gguf')
        - 'name': Model display name (e.g., 'Llama-3-8B-Instruct')
        - 'prompt_template': Prompt template with {text_to_clean} placeholder
    text_column : str
        The name of the column containing text to be cleaned.
    num_samples : int, optional
        Number of rows to process (for testing). Defaults to all.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with original text and cleaned_reflection column.
    """
    if not HAS_LLAMA_CPP:
        raise ImportError("llama-cpp-python is required. Install with: pip install llama-cpp-python")
    if not HAS_HF_HUB:
        raise ImportError("huggingface-hub is required. Install with: pip install huggingface-hub")
    
    print(f"\n{'='*80}")
    print(f"LOCAL LLAMA PREPROCESSING")
    print(f"{'='*80}")
    print(f"Input file: {os.path.basename(csv_path)}")
    print(f"Model: {model_config.get('name', 'Unknown')}")
    print(f"GPU acceleration: Enabled (n_gpu_layers=-1)")

    # Load data
    try:
        df = pd.read_csv(csv_path).dropna(subset=[text_column])
        print(f"Total rows available: {len(df)}")
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found.")
        return None
    except KeyError:
        print(f"Error: Column '{text_column}' not found in {csv_path}.")
        return None

    df_to_process = df.head(num_samples).copy() if num_samples else df.copy()
    sample_type = f"SAMPLE of {num_samples}" if num_samples else "FULL DATASET"
    print(f"Processing: {sample_type} ({len(df_to_process)} rows)")

    # Download and load model
    print(f"\nDownloading model '{model_config['filename']}'...")
    print("(~4GB, may take a few minutes on first run, cached afterwards)")
    try:
        model_path = hf_hub_download(
            repo_id=model_config['repo'],
            filename=model_config['filename']
        )
        print(f"✓ Model downloaded to: {model_path}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        return None
    
    print("\nLoading model with GPU acceleration...")
    try:
        llama = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=4096, verbose=False)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # Clean texts
    cleaned_texts = []
    texts_to_clean = df_to_process[text_column].tolist()
    prompt_template = model_config['prompt_template']
    error_count = 0
    
    print(f"\nStarting text cleaning...")
    for idx, text in enumerate(tqdm(texts_to_clean, desc="Cleaning texts with Llama")):
        if not isinstance(text, str) or not text.strip():
            cleaned_texts.append("Not applicable (empty)")
            continue
        
        try:
            prompt = prompt_template.format(text_to_clean=text)
            response = llama(prompt=prompt, max_tokens=1024, stop=["<|eot_id|>"], echo=False)
            raw_output = response['choices'][0]['text'].strip()
            
            # Remove preamble if present
            preamble = "Here is the cleaned text:"
            if raw_output.lower().startswith(preamble.lower()):
                cleaned_text = raw_output[len(preamble):].strip()
            else:
                cleaned_text = raw_output
            
            cleaned_texts.append(cleaned_text)
            
        except Exception as e:
            error_count += 1
            print(f"\n[Error {error_count}] Cleaning text {idx}: {str(e)[:100]}")
            cleaned_texts.append(f"Error during cleaning: {str(e)[:50]}")

    # Save results
    df_to_process['cleaned_reflection'] = cleaned_texts
    df_to_process.to_csv(output_path, index=False)
    
    print(f"\n{'='*80}")
    print(f"LOCAL LLAMA PREPROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Output saved to: {output_path}")
    print(f"Rows processed: {len(df_to_process)}")
    print(f"Errors encountered: {error_count}")
    print("\nFirst 3 samples:")
    for i in range(min(3, len(df_to_process))):
        print(f"\n[{i+1}] Original: {df_to_process[text_column].iloc[i][:60]}...")
        print(f"    Cleaned:  {df_to_process['cleaned_reflection'].iloc[i][:60]}...")
    
    return df_to_process


# =============================================================================
# SECTION 3: API-BASED PREPROCESSING (using Google Generative AI)
# =============================================================================

def clean_batch_with_gemini(texts):
    """
    Uses the Gemini API to clean a BATCH of texts in a single API call.

    Parameters:
    -----------
    texts : list
        A list of raw text strings.

    Returns:
    --------
    list
        A list of cleaned text strings.
    """
    if not HAS_GEMINI:
        raise ImportError("google-generativeai is required. Install with: pip install google-generativeai")
    
    import json
    
    # Create numbered list of texts
    numbered_texts = "\n".join([f"{i+1}. {text}" for i, text in enumerate(texts)])
    
    # Detailed prompt for batch processing
    prompt = f"""Please act as a data cleaning expert. Your task is to clean each of the following numbered texts.
    Follow these rules precisely:
    1. For each text, correct spelling mistakes, fix grammar, and remove artifacts like '\\n'.
    2. Do NOT change the original meaning or remove punctuation.
    3. Return the result as a single, valid JSON array of strings.
    4. The JSON array must have exactly {len(texts)} elements, where each string is a cleaned version of the corresponding input text.
    5. Do not include the numbers or any other commentary in your output, only the JSON array.

    TEXTS TO CLEAN:
    ---
    {numbered_texts}
    ---
    """
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        # Clean response and parse JSON
        cleaned_response_text = response.text.strip().replace("```json", "").replace("```", "")
        cleaned_texts = json.loads(cleaned_response_text)
        
        if len(cleaned_texts) == len(texts):
            return cleaned_texts
        else:
            print(f"Warning: Expected {len(texts)} results, got {len(cleaned_texts)}.")
            return ["Error: Mismatch in batch response"] * len(texts)
            
    except Exception as e:
        print(f"Error during batch cleaning: {e}")
        return [f"Error: {e}"] * len(texts)


def preprocess_with_gemini_api(csv_path, output_path, text_column='reflection_answer', 
                               batch_size=20, num_samples=None):
    """
    Cleans a CSV column using Google Gemini API in batches.
    
    Note: This requires a valid GOOGLE_API_KEY environment variable.
    Get your API key from: https://aistudio.google.com/app/apikey

    Parameters:
    -----------
    csv_path : str
        Path to the input CSV file.
    output_path : str
        Path to save the new CSV file with cleaned text.
    text_column : str
        The name of the column containing text to be cleaned.
    batch_size : int
        Number of texts to clean per API call. Default 20.
        (Reduce if getting timeouts: try 10 or 5)
    num_samples : int, optional
        Number of rows to process (for testing). Defaults to all.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with original text and cleaned_reflection column.
    """
    if not HAS_GEMINI:
        raise ImportError("google-generativeai is required. Install with: pip install google-generativeai")
    if HAS_DOTENV:
        load_dotenv()
    
    # Load API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found!\n"
            "Set it in .env file or as environment variable:\n"
            "  export GOOGLE_API_KEY=your_key_here\n"
            "Get API key from: https://aistudio.google.com/app/apikey"
        )
    
    genai.configure(api_key=api_key)
    
    print(f"\n{'='*80}")
    print(f"GOOGLE GEMINI API PREPROCESSING")
    print(f"{'='*80}")
    print(f"Input file: {os.path.basename(csv_path)}")
    print(f"Batch size: {batch_size} texts per API call")

    # Load data
    try:
        df = pd.read_csv(csv_path).dropna(subset=[text_column])
        print(f"Total rows available: {len(df)}")
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found.")
        return None
    except KeyError:
        print(f"Error: Column '{text_column}' not found in {csv_path}.")
        return None

    df_to_process = df.head(num_samples).copy() if num_samples else df.copy()
    sample_type = f"SAMPLE of {num_samples}" if num_samples else "FULL DATASET"
    print(f"Processing: {sample_type} ({len(df_to_process)} rows)")

    # Calculate number of API calls needed
    num_batches = (len(df_to_process) + batch_size - 1) // batch_size
    print(f"Will make ~{num_batches} API calls")

    # Process in batches
    texts_to_clean = df_to_process[text_column].tolist()
    batches = np.array_split(texts_to_clean, num_batches)
    
    all_cleaned = []
    error_count = 0
    print(f"\nStarting text cleaning with Gemini API...")
    
    for batch_idx, batch in enumerate(tqdm(batches, desc="Cleaning batches with Gemini")):
        try:
            cleaned = clean_batch_with_gemini(batch.tolist())
            all_cleaned.extend(cleaned)
        except Exception as e:
            error_count += 1
            print(f"\n[Error in batch {batch_idx}]: {str(e)[:100]}")
            # Add error markers for this batch
            all_cleaned.extend([f"API Error: {str(e)[:50]}"] * len(batch))

    # Save results
    df_to_process['cleaned_reflection'] = all_cleaned
    df_to_process.to_csv(output_path, index=False)
    
    print(f"\n{'='*80}")
    print(f"GEMINI API PREPROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Output saved to: {output_path}")
    print(f"Rows processed: {len(df_to_process)}")
    print(f"Batches completed: {num_batches}")
    print(f"API errors: {error_count}")
    print("\nFirst 3 samples:")
    for i in range(min(3, len(df_to_process))):
        print(f"\n[{i+1}] Original: {df_to_process[text_column].iloc[i][:60]}...")
        print(f"    Cleaned:  {df_to_process['cleaned_reflection'].iloc[i][:60]}...")
    
    return df_to_process


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_data(csv_path, text_column='reflection_answer', remove_na=True):
    """
    Load a CSV file and optionally remove NA values.
    
    Parameters:
    -----------
    csv_path : str
        Path to CSV file
    text_column : str
        Column name to load
    remove_na : bool
        Whether to drop NA values
        
    Returns:
    --------
    pd.DataFrame
        Loaded data
    """
    try:
        if remove_na:
            df = pd.read_csv(csv_path).dropna(subset=[text_column])
        else:
            df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows from {os.path.basename(csv_path)}")
        return df
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found.")
        return None
    except KeyError:
        print(f"Error: Column '{text_column}' not found.")
        return None


def compare_cleaning_results(original, cleaned, num_samples=5):
    """
    Display side-by-side comparison of original and cleaned texts.
    
    Parameters:
    -----------
    original : pd.Series
        Series of original texts
    cleaned : pd.Series
        Series of cleaned texts
    num_samples : int
        Number of samples to display
    """
    samples = np.random.choice(len(original), num_samples, replace=False)
    
    print("\n" + "="*80)
    print("CLEANING RESULTS COMPARISON")
    print("="*80)
    
    for i in samples:
        print(f"\n[Sample {i}]")
        print(f"ORIGINAL:\n{original.iloc[i]}\n")
        print(f"CLEANED:\n{cleaned.iloc[i]}")
        print("-"*80)