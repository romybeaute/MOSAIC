#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : preprocessing.py
# description     : Unified preprocessing module (Standard + Llama + Gemini)
#                   - Fixes ImportError (restored load_data)
#                   - Fixes Gemini Quota (forces 1.5-flash)
#                   - Fixes Llama Chatter (programmatic cleanup)
# author          : Romy, Beauté (r.beaut@sussex.ac.uk)
# date            : 2026-01-28
# ==============================================================================

import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import time
import json
from pathlib import Path
from nltk.tokenize import PunktSentenceTokenizer
import re

# Optional imports
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
# SECTION 1: BASIC TEXT PREPROCESSING (Structure)
# =============================================================================

def split_sentences(reflections):
    tokenizer = PunktSentenceTokenizer()
    sentences = []
    doc_map = [] 
    
    for doc_idx, reflection in enumerate(reflections):
        doc_sentences = tokenizer.tokenize(reflection)
        sentences.extend(doc_sentences)
        doc_map.extend([doc_idx] * len(doc_sentences))
    
    return sentences, doc_map


def basic_preprocess(texts, split_into_sentences=True, min_words=2):
    if split_into_sentences:
        texts, doc_map = split_sentences(texts)

    
    initial_count = len(texts)
    print(f"\nSuccessfully loaded {initial_count} texts.")

    texts = [re.sub(r'^\s*\d+[\.\)]\s*', '', text) for text in texts] #clean numbering
    

    filtered_texts = []
    for text in texts:
        if len(text.split()) >= min_words:
            filtered_texts.append(text)

    # Calculate removed stats
    removed_count = initial_count - len(filtered_texts)
    print(f"Threshold (min_words): {min_words}")
    print(f"Removed short texts:   {removed_count} ({(removed_count/initial_count)*100:.1f}%)")
    
    # Deduplicate while preserving order
    seen = set()
    final_texts = [x for x in filtered_texts if not (x in seen or seen.add(x))]

    duplicates_count = len(filtered_texts) - len(final_texts)
    print(f"Removed duplicates:    {duplicates_count}")
    print(f"Final count:           {len(final_texts)}")

    
    return pd.DataFrame({
        'sentences': final_texts
    })


# =============================================================================
# SECTION 2: LOCAL LLM PREPROCESSING (Llama)
# =============================================================================

def clean_llama_output_programmatically(text):
    """
    Safety net: Removes common introductory phrases if the LLM disobeys.
    """
    prefixes = [
        "Here is the cleaned and translated text:",
        "Here is the cleaned text:",
        "Here is the translation:",
        "Sure, here is the text:",
        "Cleaned text:",
        "Translation:",
        "Output:",
    ]
    
    cleaned = text.strip()
    
    # Remove quotes if the model wrapped output in them
    if cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned[1:-1]
        
    for p in prefixes:
        if cleaned.lower().startswith(p.lower()):
            cleaned = cleaned[len(p):].strip()
            
    return cleaned

def preprocess_with_local_llama(csv_path, output_path,
                                text_column='reflection_answer', num_samples=None):
    if not HAS_LLAMA_CPP or not HAS_HF_HUB:
        raise ImportError("llama-cpp-python and huggingface-hub are required.")
    
    print(f"\n{'='*80}\nLOCAL LLAMA PREPROCESSING\n{'='*80}")
    print(f"Input file: {os.path.basename(csv_path)}")

    # Load data
    try:
        df = pd.read_csv(csv_path).dropna(subset=[text_column])
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

    df_to_process = df.head(num_samples).copy() if num_samples else df.copy()
    print(f"Processing: {len(df_to_process)} rows")

    # Load Model
    try:
        model_path = hf_hub_download(
            repo_id='NousResearch/Meta-Llama-3-8B-Instruct-GGUF',
            filename='Meta-Llama-3-8B-Instruct-Q4_K_M.gguf'
        )
        llama = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=4096, verbose=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    cleaned_texts = []
    texts_to_clean = df_to_process[text_column].tolist()
    error_count = 0
    
    # Prompt with translation instruction
    prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert data cleaner. Your task is to clean the user's text.
Follow these rules precisely:
1. Translate the text into standard English if it is not already.
2. Correct spelling mistakes and fix grammar.
3. Remove artifacts and formatting like '\\n'.
4. Do NOT change the original meaning or punctuation of the text.
5. Your response must contain ONLY the cleaned text, without any introductory phrases or commentary.<|eot_id|><|start_header_id|>user<|end_header_id|>
6. If already in English, do not translate and DO NOT change the content; only correct errors and clean.
Clean the following text:

"{text_to_clean}"<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    
    print(f"\nStarting text cleaning...")
    for idx, text in enumerate(tqdm(texts_to_clean, desc="Cleaning texts with Llama")):
        if not isinstance(text, str) or not text.strip():
            cleaned_texts.append("")
            continue
        
        try:
            prompt = prompt_template.format(text_to_clean=text)
            response = llama(prompt=prompt, max_tokens=1024, stop=["<|eot_id|>"], echo=False)
            raw_output = response['choices'][0]['text'].strip()
            
            # Apply Safety Net
            final_text = clean_llama_output_programmatically(raw_output)
            cleaned_texts.append(final_text)
            
        except Exception as e:
            error_count += 1
            cleaned_texts.append(f"Error: {str(e)[:50]}")

    df_to_process['cleaned_reflection'] = cleaned_texts
    df_to_process.to_csv(output_path, index=False)
    
    print(f"\nOutput saved to: {output_path}")
    return df_to_process


# =============================================================================
# SECTION 3: API-BASED PREPROCESSING (GEMINI)
# =============================================================================

def get_best_gemini_model():
    """
    Forces the use of Flash models which have better free tier limits.
    """
    if not HAS_GEMINI:
        raise ImportError("google-generativeai is required.")
    
    available_models = []
    try:
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                available_models.append(m.name)
    except Exception as e:
        print(f"Error listing models: {e}")
        return "gemini-2.5-flash-lite"
    
    priority = ['gemini-2.5-flash-lite','gemini-2.0-flash-lite','gemini-1.5-flash']
    
    for p in priority:
        for m in available_models:
            if p in m:
                print(f"✓ Selected: {m} (Best for Free Tier)")
                return m
                
    return available_models[0]


def clean_batch_with_gemini(texts, model_name=None):
    if not HAS_GEMINI:
        raise ImportError("google-generativeai is required.")
    
    if model_name is None:
        model_name = get_best_gemini_model()
    
    numbered_texts = "\n".join([f"{i+1}. {text}" for i, text in enumerate(texts)])
    
    prompt = f"""Task: Translate and Clean.
Rules:
1. Translate every text into standard British English.
2. If the text is already English, correct spelling and grammar errors only.
3. Remove artifacts like '\\n'.
4. Do NOT change the original meaning or punctuation.
5. Return strictly a JSON array of strings. No markdown.
6. If already in English, do not translate and DO NOT change the content; only correct errors and clean.
7. Do not return anything else than the translated and/or cleaned texts.

TEXTS:
{numbered_texts}
"""
    
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        text_resp = response.text.strip()
        
        # Clean potential markdown
        if text_resp.startswith("```"):
            text_resp = text_resp.split("\n", 1)[1]
            if text_resp.endswith("```"): text_resp = text_resp[:-3]
            
        cleaned_texts = json.loads(text_resp)
        if len(cleaned_texts) == len(texts):
            return cleaned_texts
        return ["Error: Mismatch"] * len(texts)
            
    except Exception as e:
        # Check for quota error
        if "429" in str(e):
            print("\nQuota Exceeded (429). Waiting 60 seconds...")
            return ["Error: Quota Exceeded"] * len(texts)
        return [f"Error: {e}"] * len(texts)


def preprocess_with_gemini_api(csv_path, output_path, text_column='reflection_answer', 
                               batch_size=10, num_samples=None): 
    if not HAS_GEMINI: raise ImportError("google-generativeai required")
    if HAS_DOTENV: load_dotenv()
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key: raise ValueError("GOOGLE_API_KEY not found.")
    
    genai.configure(api_key=api_key)
    
    print(f"\n{'='*80}\nGEMINI API PREPROCESSING\n{'='*80}")
    print(f"Input file: {os.path.basename(csv_path)}")

    try:
        df = pd.read_csv(csv_path).dropna(subset=[text_column])
    except Exception:
        return None

    df_to_process = df.head(num_samples).copy() if num_samples else df.copy()
    texts = df_to_process[text_column].tolist()
    
    # Batch processing
    all_cleaned = []
    num_batches = (len(texts) + batch_size - 1) // batch_size
    batches = np.array_split(texts, num_batches)
    
    for batch in tqdm(batches, desc="Gemini Batches"):
        cleaned = clean_batch_with_gemini(batch.tolist())
        all_cleaned.extend(cleaned)

    df_to_process['cleaned_reflection'] = all_cleaned
    df_to_process.to_csv(output_path, index=False)
    
    print(f"Saved to: {output_path}")
    return df_to_process

# =============================================================================
# SECTION 4: UTILITY FUNCTIONS (RESTORED)
# =============================================================================

def load_data(csv_path, text_column='reflection_answer', remove_na=True):
    """
    Load a CSV file and optionally remove NA values.
    """
    try:
        if remove_na:
            df = pd.read_csv(csv_path).dropna(subset=[text_column])
        else:
            df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows from {os.path.basename(csv_path)}")
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None

def compare_cleaning_results(original, cleaned, num_samples=5):
    """
    Display side-by-side comparison of original and cleaned texts.
    """
    count = min(num_samples, len(original))
    samples = np.random.choice(len(original), count, replace=False)
    
    print("\n" + "="*80)
    print("CLEANING RESULTS COMPARISON")
    print("="*80)
    
    for i in samples:
        print(f"\n[Sample {i}]")
        print(f"ORIGINAL:\n{original.iloc[i]}\n")
        print(f"CLEANED:\n{cleaned.iloc[i]}")
        print("-"*80)