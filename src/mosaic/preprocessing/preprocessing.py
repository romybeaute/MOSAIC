#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : preprocessing.py
# description     : Unified preprocessing module (Standard + Llama + Gemini)
#                   - Error handling & logging for all methods
#                   - Deterministic Llama (temperature=0, seed=42, no cropping)
#                   - Optional max_text_length filtering
#                   - Auto-config dataset naming
# author          : Romy, Beauté (r.beaut@sussex.ac.uk)
# date            : 2026-01-29
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
                                text_column='reflection_answer', num_samples=None,
                                max_text_length=None, log_errors=True):
    """
    Preprocess with Llama + deterministic error handling.
    
    Args:
        csv_path: Path to input CSV
        output_path: Path to output CSV
        text_column: Column name containing text
        num_samples: Limit to N samples (None = all)
        max_text_length: Maximum characters per report (None = no limit, process all)
        log_errors: Save error log to .log file (True recommended)
    """
    if not HAS_LLAMA_CPP or not HAS_HF_HUB:
        raise ImportError("llama-cpp-python and huggingface-hub are required.")
    
    print(f"\n{'='*80}\nLOCAL LLAMA PREPROCESSING\n{'='*80}")
    print(f"Input file: {os.path.basename(csv_path)}")
    if max_text_length:
        print(f"Max text length: {max_text_length} characters")
    else:
        print(f"Max text length: No limit (process all reports)")

    # Load data
    try:
        df = pd.read_csv(csv_path).dropna(subset=[text_column])
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

    df_to_process = df.head(num_samples).copy() if num_samples else df.copy()
    print(f"Processing: {len(df_to_process)} reports")

    # Load Model
    try:
        model_path = hf_hub_download(
            repo_id='NousResearch/Meta-Llama-3-8B-Instruct-GGUF',
            filename='Meta-Llama-3-8B-Instruct-Q4_K_M.gguf'
        )
        # DETERMINISTIC SETTINGS (temperature=0, seed=42 for reproducibility)
        llama = Llama(
            model_path=model_path, 
            n_gpu_layers=-1, 
            n_ctx=4096, 
            verbose=False,
            seed=42  # Fixed seed for reproducibility
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    cleaned_texts = []
    texts_to_clean = df_to_process[text_column].tolist()
    error_count = 0
    skipped_count = 0
    error_log = []
    
    # Prompt - DETERMINISTIC AND FAITHFUL TO ORIGINAL
    prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert data cleaner. Your task is to clean the user's text.
Follow these rules PRECISELY:
1. Translate the text into standard English if it is not already.
2. Correct spelling mistakes and fix grammar.
3. Remove artifacts and formatting like '\\n'.
4. Do NOT change the original meaning, punctuation, or structure of the text.
5. Do NOT crop, truncate, or remove any sentences or paragraphs.
6. Your response must contain ONLY the cleaned text, without any introductory phrases or commentary.<|eot_id|><|start_header_id|>user<|end_header_id|>
If already in English, do not translate and DO NOT change the content; only correct errors and clean.
Clean the following text:

"{text_to_clean}"<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    
    print(f"\nStarting text cleaning...")
    for idx, text in enumerate(tqdm(texts_to_clean, desc="Cleaning texts with Llama")):
        if not isinstance(text, str) or not text.strip():
            cleaned_texts.append("")
            continue
        
        # Check text length
        text_length = len(text)
        if max_text_length and text_length > max_text_length:
            error_msg = f"Report {idx}: Text too long ({text_length} > {max_text_length} chars)"
            error_log.append(error_msg)
            skipped_count += 1
            cleaned_texts.append(f"[SKIPPED - Text too long: {text_length} chars]")
            continue
        
        try:
            prompt = prompt_template.format(text_to_clean=text)
            # DETERMINISTIC INFERENCE (temperature=0 for greedy decoding)
            response = llama(
                prompt=prompt, 
                max_tokens=8192,  # Large enough for most reports, NO CROPPING
                temperature=0.0,  # DETERMINISTIC: always pick same token
                top_p=1.0,        # Use all tokens
                top_k=40,
                stop=["<|eot_id|>"], 
                echo=False,
                seed=42            # Fixed seed
            )
            raw_output = response['choices'][0]['text'].strip()
            
            # Apply Safety Net (minimal cleanup)
            final_text = clean_llama_output_programmatically(raw_output)
            cleaned_texts.append(final_text)
            
        except Exception as e:
            error_count += 1
            error_type = type(e).__name__
            error_msg = f"Report {idx}: {error_type}: {str(e)[:100]}"
            error_log.append(error_msg)
            cleaned_texts.append(f"[ERROR - {error_type}]")

    df_to_process['cleaned_reflection'] = cleaned_texts
    df_to_process.to_csv(output_path, index=False)
    
    # Save error log if there were errors
    if log_errors and error_log:
        log_path = str(output_path).replace('.csv', '_errors.log')
        with open(log_path, 'w') as f:
            f.write(f"Preprocessing Error Log\n")
            f.write(f"{'='*80}\n")
            f.write(f"Dataset: {os.path.basename(csv_path)}\n")
            f.write(f"Method: Llama (Deterministic, No Cropping)\n")
            if max_text_length:
                f.write(f"Max text length: {max_text_length} chars\n")
            else:
                f.write(f"Max text length: No limit\n")
            f.write(f"Total reports: {len(texts_to_clean)}\n")
            f.write(f"Successfully processed: {len(texts_to_clean) - error_count - skipped_count}\n")
            f.write(f"Errors: {error_count}\n")
            f.write(f"Skipped (too long): {skipped_count}\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"ERROR DETAILS:\n")
            for err in error_log:
                f.write(f"{err}\n")
        print(f"\n⚠ Error log saved to: {log_path}")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total reports: {len(texts_to_clean)}")
    print(f"Successfully processed: {len(texts_to_clean) - error_count - skipped_count}")
    print(f"Errors: {error_count}")
    print(f"Skipped (too long): {skipped_count}")
    print(f"Output saved to: {output_path}")
    print(f"{'='*80}\n")
    
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
                               batch_size=10, num_samples=None, max_text_length=None,
                               log_errors=True): 
    if not HAS_GEMINI: raise ImportError("google-generativeai required")
    if HAS_DOTENV: load_dotenv()
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key: raise ValueError("GOOGLE_API_KEY not found.")
    
    genai.configure(api_key=api_key)
    
    print(f"\n{'='*80}\nGEMINI API PREPROCESSING\n{'='*80}")
    print(f"Input file: {os.path.basename(csv_path)}")
    if max_text_length:
        print(f"Max text length: {max_text_length} characters")
    else:
        print(f"Max text length: No limit (process all reports)")

    try:
        df = pd.read_csv(csv_path).dropna(subset=[text_column])
    except Exception:
        return None

    df_to_process = df.head(num_samples).copy() if num_samples else df.copy()
    print(f"Processing: {len(df_to_process)} reports")
    
    texts = df_to_process[text_column].tolist()
    skipped_count = 0
    error_log = []
    
    # Filter texts by length and log skipped ones
    filtered_texts = []
    for idx, text in enumerate(texts):
        text_length = len(text)
        if max_text_length and text_length > max_text_length:
            error_msg = f"Report {idx}: Text too long ({text_length} > {max_text_length} chars)"
            error_log.append(error_msg)
            skipped_count += 1
            filtered_texts.append(f"[SKIPPED - Text too long: {text_length} chars]")
        else:
            filtered_texts.append(text)
    
    texts = filtered_texts
    
    # Batch processing
    all_cleaned = []
    num_batches = (len(texts) + batch_size - 1) // batch_size
    batches = np.array_split(texts, num_batches)
    
    for batch in tqdm(batches, desc="Gemini Batches"):
        cleaned = clean_batch_with_gemini(batch.tolist())
        all_cleaned.extend(cleaned)

    df_to_process['cleaned_reflection'] = all_cleaned
    df_to_process.to_csv(output_path, index=False)
    
    # Save error log
    if log_errors and error_log:
        log_path = str(output_path).replace('.csv', '_errors.log')
        with open(log_path, 'w') as f:
            f.write(f"Preprocessing Error Log\n")
            f.write(f"{'='*80}\n")
            f.write(f"Dataset: {os.path.basename(csv_path)}\n")
            f.write(f"Method: Gemini API\n")
            if max_text_length:
                f.write(f"Max text length: {max_text_length} chars\n")
            else:
                f.write(f"Max text length: No limit\n")
            f.write(f"Total reports: {len(texts)}\n")
            f.write(f"Skipped (too long): {skipped_count}\n")
            f.write(f"Successfully processed: {len(texts) - skipped_count}\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"DETAILS:\n")
            for err in error_log:
                f.write(f"{err}\n")
        print(f"\n⚠ Error log saved to: {log_path}")
    
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
        print(f"Loaded {len(df)} reports from {os.path.basename(csv_path)}")
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


# =============================================================================
# SECTION 5: AUTO-CONFIGURATION & CLI
# =============================================================================

def build_dataset_config(data_dir="DATA"):
    """
    Build dataset configuration dictionary dynamically from raw CSV files.
    Returns dict mapping dataset_name -> {input, output_api, output_local, output_basic}
    """
    data_path = Path(data_dir)
    raw_dir = data_path / "raw"
    
    if not raw_dir.exists():
        raise FileNotFoundError(f"DATA/raw directory not found at {raw_dir}")
    
    datasets = {}
    raw_files = sorted(raw_dir.glob("*_raw.csv"))
    
    for file_path in raw_files:
        filename = file_path.name
        dataset_name = filename.rsplit('_raw.csv', 1)[0]
        
        datasets[dataset_name] = {
            'input': filename,
            'output_api': f"{dataset_name}_cleaned_API.csv",
            'output_local': f"{dataset_name}_cleaned_llama.csv",
            'output_basic': f"{dataset_name}_preprocessed.csv"
        }
    
    return datasets


def get_data_paths(dataset_name, data_dir="DATA", method="basic", sample=None):
    """
    Get full paths for input/output files based on dataset name and method.
    
    Args:
        dataset_name: Name of dataset (e.g., 'dreamachine_DL')
        data_dir: Path to DATA directory
        method: 'basic', 'llama', or 'gemini'
        sample: Optional sample size (e.g., 5, 10). If provided, appends to filename
    
    Returns:
        tuple: (input_path, output_path)
    """
    data_path = Path(data_dir)
    raw_dir = data_path / "raw"
    preproc_dir = data_path / "preprocessed"
    
    preproc_dir.mkdir(exist_ok=True, parents=True)
    
    config = build_dataset_config(data_dir)
    
    if dataset_name not in config:
        available = ", ".join(config.keys())
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available}")
    
    input_path = raw_dir / config[dataset_name]['input']
    
    if method == "basic":
        output_name = config[dataset_name]['output_basic']
    elif method == "llama":
        output_name = config[dataset_name]['output_local']
    elif method == "gemini":
        output_name = config[dataset_name]['output_api']
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Add sample size to filename if specified
    if sample:
        output_name = output_name.replace('.csv', f'_sample{sample}.csv')
    
    output_path = preproc_dir / output_name
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    return str(input_path), str(output_path)


# =============================================================================
# SECTION 6: CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Preprocess text data with automatic path configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic preprocessing on full dataset
  python preprocessing.py --dataset dreamachine_DL --method basic
  
  # Test with 10 samples before full run
  python preprocessing.py --dataset dreamachine_DL --method basic --sample 10
  
  # Llama preprocessing (deterministic, no cropping)
  python preprocessing.py --dataset MPE --method llama --sample 5
  
  # Gemini API preprocessing
  python preprocessing.py --dataset innerspeech --method gemini
  
  # With optional length limit (skip reports > 10000 chars)
  python preprocessing.py --dataset dreamachine_DL --method llama --max-text-length 10000
  
  # Without error logging
  python preprocessing.py --dataset dreamachine_DL --method llama --no-error-log
  
  # List available datasets
  python preprocessing.py --list-datasets
        """
    )
    
    parser.add_argument(
        "--dataset",
        required=False,
        help="Dataset name (e.g., dreamachine_DL, MPE, innerspeech)"
    )
    parser.add_argument(
        "--method",
        choices=["basic", "llama", "gemini"],
        default="basic",
        help="Preprocessing method (default: basic)"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Test with reduced sample size (e.g., 10). None = full dataset"
    )
    parser.add_argument(
        "--data-dir",
        default="DATA",
        help="Path to DATA directory (default: DATA)"
    )
    parser.add_argument(
        "--text-column",
        default="reflection_answer",
        help="Name of text column in CSV (default: reflection_answer)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for Gemini API (default: 10)"
    )
    parser.add_argument(
        "--max-text-length",
        type=int,
        default=None,
        help="Maximum characters per report (None = no limit, process all)"
    )
    parser.add_argument(
        "--no-error-log",
        action="store_true",
        help="Don't save error log file"
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available datasets and exit"
    )
    
    args = parser.parse_args()
    
    # Handle list-datasets
    if args.list_datasets:
        try:
            config = build_dataset_config(args.data_dir)
            print("\n" + "="*80)
            print("AVAILABLE DATASETS")
            print("="*80)
            for name in sorted(config.keys()):
                print(f"  • {name}")
            print("="*80 + "\n")
        except Exception as e:
            print(f"Error: {e}")
        exit(0)
    
    # Require dataset for processing
    if not args.dataset:
        parser.print_help()
        print("\nError: --dataset is required (unless using --list-datasets)")
        exit(1)
    
    # Get paths
    try:
        input_path, output_path = get_data_paths(
            args.dataset, 
            data_dir=args.data_dir, 
            method=args.method,
            sample=args.sample
        )
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
    
    # Print configuration
    print("\n" + "="*80)
    print("PREPROCESSING CONFIGURATION")
    print("="*80)
    print(f"Dataset:        {args.dataset}")
    print(f"Method:         {args.method}")
    print(f"Input:          {input_path}")
    print(f"Output:         {output_path}")
    if args.sample:
        print(f"Sample size:    {args.sample} (TEST MODE)")
    else:
        print(f"Sample size:    Full dataset")
    if args.max_text_length:
        print(f"Max text length: {args.max_text_length} characters")
    else:
        print(f"Max text length: No limit")
    print(f"Error logging:  {'Enabled' if not args.no_error_log else 'Disabled'}")
    print("="*80 + "\n")
    
    # Run preprocessing
    try:
        if args.method == "basic":
            df = load_data(input_path, args.text_column)
            if df is not None:
                if args.sample:
                    df = df.head(args.sample)
                    print(f"\nRunning on {len(df)} reports (test mode)")
                result = basic_preprocess(df[args.text_column].tolist())
                result.to_csv(output_path, index=False)
                print(f"\n✓ Preprocessing complete!")
                print(f"Output saved to: {output_path}")
        
        elif args.method == "llama":
            print(f"\nRunning Llama preprocessing (deterministic, no cropping)...")
            result = preprocess_with_local_llama(
                input_path, 
                output_path, 
                text_column=args.text_column,
                num_samples=args.sample,
                max_text_length=args.max_text_length,
                log_errors=not args.no_error_log
            )
            if result is not None:
                print(f"\n✓ Preprocessing complete!")
        
        elif args.method == "gemini":
            print(f"\nRunning Gemini API preprocessing...")
            result = preprocess_with_gemini_api(
                input_path,
                output_path,
                text_column=args.text_column,
                batch_size=args.batch_size,
                num_samples=args.sample,
                max_text_length=args.max_text_length,
                log_errors=not args.no_error_log
            )
            if result is not None:
                print(f"\n✓ Preprocessing complete!")
    
    except Exception as e:
        print(f"\n✗ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        exit(1)




# # List datasets
# python src/mosaic/preprocessing/preprocessing.py --list-datasets

# # Test Llama with 5 reports
# python src/mosaic/preprocessing/preprocessing.py --dataset dreamachine_DL --method llama --sample 5

# # Full run
# python src/mosaic/preprocessing/preprocessing.py --dataset ganzfeld_GREEN --method gemini

# # With optional length limit
# python src/mosaic/preprocessing/preprocessing.py --dataset dreamachine_DL --method llama --max-text-length 10000

# # Without error log
# python src/mosaic/preprocessing/preprocessing.py --dataset dreamachine_DL --method llama --no-error-log