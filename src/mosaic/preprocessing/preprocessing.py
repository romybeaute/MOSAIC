#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : preprocessing.py
# description     : Unified preprocessing module (Standard + Llama + Gemini)
#                   - Error handling & logging for all methods
#                   - Deterministic Llama (temperature=0, seed=42, no cropping)
#                   - Optional max_text_length filtering
#                   - Auto-config dataset naming
#                   - FIXED: Forces gemini-2.0-flash-lite 
#                   - FIXED: No retry during processing (mark errors, retry later)
# author          : Romy, Beaute (r.beaut@sussex.ac.uk)
# date            : 2026-01-30
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

import string
import importlib

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


# ==============================================================================
# SECTION 0: CONFIGURATION & UTILS
# ==============================================================================

def load_dataset_config(dataset_name):
    """Dynamically load config for the specific dataset."""
    try:
        module_name = f"mosaic.configs.{dataset_name}"
        config_module = importlib.import_module(module_name)
        print(f"Loaded configuration for '{dataset_name}'")
        return config_module.config
    except ImportError:
        print(f"No specific config found for '{dataset_name}'. Using defaults.")
        return None

def count_meaningful_words(text):
    """Counts words excluding punctuation."""
    # Remove punctuation map
    translator = str.maketrans('', '', string.punctuation)
    clean_text = text.translate(translator)
    # Split by whitespace and count
    return len(clean_text.split())



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


def basic_preprocess(ids, texts, dataset_name=None, split_override=None):
    # 1. Load Config
    config = load_dataset_config(dataset_name)
    
    # Defaults if config is missing
    words_to_remove = getattr(config, 'words_to_remove', [])
    patterns_to_remove = getattr(config, 'patterns_to_remove', [])
    min_words = getattr(config, 'min_word_count', 3)
    
    # LOGIC FIX: Prioritize CLI override, fall back to config, fall back to True
    config_split = getattr(config, 'split_sentences', True)
    if split_override is not None:
        do_split = split_override
    else:
        do_split = config_split

    print(f"\nPreprocessing {len(texts)} texts...")
    print(f"Rules: Split={do_split}, Min Words={min_words}")
    print(f"Removing words: {words_to_remove}")

    cleaned_ids = []
    cleaned_texts = []
    
    # --- PHASE 1: ARTIFACT CLEANING (Paragraph Level) ---
    for pid, text in zip(ids, texts):
        if not isinstance(text, str): continue

        # A. Remove Bracketed Content [] and **
        for pattern in patterns_to_remove:
            text = re.sub(pattern, ' ', text)

        # B. Remove Specific Words (Case Insensitive)
        # We use \b boundary to ensure we don't cut words in half
        for word in words_to_remove:
            pattern = re.compile(rf'\b{re.escape(word)}\b', re.IGNORECASE)
            text = pattern.sub(' ', text)

        # C. General cleanup
        text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        
        if text:
            cleaned_ids.append(pid)
            cleaned_texts.append(text)

    # --- PHASE 2: SENTENCE SPLITTING ---
    final_ids = []
    final_texts = []

    if do_split:
        print("Splitting into sentences...")
        tokenizer = PunktSentenceTokenizer()
        for pid, text in zip(cleaned_ids, cleaned_texts):
            sentences = tokenizer.tokenize(text)
            final_ids.extend([pid] * len(sentences))
            final_texts.extend(sentences)
    else:
        final_ids = cleaned_ids
        final_texts = cleaned_texts

    # --- PHASE 3: STRICT FILTERING ---
    output_ids = []
    output_texts = []
    removed_count = 0
    
    for pid, text in zip(final_ids, final_texts):
        # Count words ignoring punctuation
        n_words = count_meaningful_words(text)
        
        if n_words >= min_words:
            output_ids.append(pid)
            output_texts.append(text)
        else:
            # print(f"Removed (too short): '{text}'") # Uncomment to debug
            removed_count += 1

    print(f"Removed short texts (< {min_words} words): {removed_count}")
    print(f"Final count: {len(output_texts)}")
    
    return pd.DataFrame({
        'participant_id': output_ids,
        'cleaned_text': output_texts
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


# =============================================================================
# PRE-FLIGHT CHECK: Analyze reports before processing
# =============================================================================

def preflight_check(texts, model_context_window=16384, chars_per_token=4):
    """
    Analyze texts BEFORE processing to warn about potential issues.
    
    Args:
        texts: List of text strings to analyze
        model_context_window: n_ctx setting (default: 16384)
        chars_per_token: Approximate chars per token (default: 4)
    
    Returns:
        Dictionary with analysis results
    """
    print(f"\n{'='*80}")
    print("PRE-FLIGHT CHECK: Analyzing reports before processing...")
    print(f"{'='*80}\n")
    
    # Calculate stats
    char_counts = [len(t) if isinstance(t, str) else 0 for t in texts]
    token_estimates = [c / chars_per_token for c in char_counts]
    
    total_reports = len(texts)
    max_chars = max(char_counts) if char_counts else 0
    max_tokens = max(token_estimates) if token_estimates else 0
    avg_chars = sum(char_counts) / total_reports if total_reports > 0 else 0
    avg_tokens = sum(token_estimates) / total_reports if total_reports > 0 else 0
    
    # Thresholds (conservative estimates accounting for prompt overhead ~500 tokens)
    safe_tokens = model_context_window * 0.4  # 40% for input, 40% for output, 20% buffer
    warning_tokens = model_context_window * 0.45
    critical_tokens = model_context_window * 0.5
    
    # Categorize reports
    safe_count = sum(1 for t in token_estimates if t <= safe_tokens)
    warning_count = sum(1 for t in token_estimates if safe_tokens < t <= critical_tokens)
    critical_count = sum(1 for t in token_estimates if t > critical_tokens)
    
    # Find problematic reports
    problematic_indices = [i for i, t in enumerate(token_estimates) if t > warning_tokens]
    
    # Print results
    print(f"Model context window:    {model_context_window:,} tokens")
    print(f"Safe input threshold:    ~{int(safe_tokens):,} tokens (~{int(safe_tokens * chars_per_token):,} chars)")
    print(f"")
    print(f"REPORT STATISTICS:")
    print(f"  Total reports:         {total_reports}")
    print(f"  Average length:        {avg_chars:,.0f} chars (~{avg_tokens:,.0f} tokens)")
    print(f"  Longest report:        {max_chars:,} chars (~{max_tokens:,.0f} tokens)")
    print(f"")
    print(f"RISK ASSESSMENT:")
    print(f"  Safe (<{int(safe_tokens)} tokens):       {safe_count:>4} reports ({safe_count/total_reports*100:.1f}%)")
    print(f"  Warning (may truncate):    {warning_count:>4} reports ({warning_count/total_reports*100:.1f}%)")
    print(f"  Critical (likely truncate): {critical_count:>4} reports ({critical_count/total_reports*100:.1f}%)")
    
    # Detailed warnings for problematic reports
    if problematic_indices:
        print(f"\n{'='*80}")
        print(f"REPORTS AT RISK OF TRUNCATION:")
        print(f"{'='*80}")
        for idx in problematic_indices[:10]:  # Show max 10
            chars = char_counts[idx]
            tokens = token_estimates[idx]
            risk = "CRITICAL" if tokens > critical_tokens else "WARNING"
            print(f"  Report {idx}: {chars:,} chars (~{tokens:,.0f} tokens) [{risk}]")
        if len(problematic_indices) > 10:
            print(f"  ... and {len(problematic_indices) - 10} more")
    
    # Recommendation
    print(f"\n{'='*80}")
    if critical_count > 0:
        print(f"RECOMMENDATION: {critical_count} reports may be truncated!")
        print(f"   Options:")
        print(f"   1. Increase n_ctx to 32768 (if your GPU has enough VRAM)")
        print(f"   2. Use --max-text-length {int(safe_tokens * chars_per_token)} to skip long reports")
        print(f"   3. Use Gemini API instead (1M token context)")
    elif warning_count > 0:
        print(f"RECOMMENDATION: {warning_count} reports are borderline.")
        print(f"   Processing should work, but monitor for truncation warnings.")
    else:
        print(f"ALL REPORTS LOOK SAFE! No truncation expected.")
    print(f"{'='*80}\n")
    
    return {
        'total_reports': total_reports,
        'max_chars': max_chars,
        'max_tokens': max_tokens,
        'avg_chars': avg_chars,
        'avg_tokens': avg_tokens,
        'safe_count': safe_count,
        'warning_count': warning_count,
        'critical_count': critical_count,
        'problematic_indices': problematic_indices,
        'model_context_window': model_context_window
    }


def preprocess_with_local_llama(csv_path, output_path,
                                text_column='reflection_answer', num_samples=None,
                                max_text_length=None, log_errors=True, n_ctx=16384):
    """
    Preprocess with Llama + deterministic error handling.
    
    Args:
        csv_path: Path to input CSV
        output_path: Path to output CSV
        text_column: Column name containing text
        num_samples: Limit to N samples (None = all)
        max_text_length: Maximum characters per report (None = no limit, process all)
        log_errors: Save error log to .log file (True recommended)
        n_ctx: Context window size (default: 16384, increase for very long reports)
    """
    if not HAS_LLAMA_CPP or not HAS_HF_HUB:
        raise ImportError("llama-cpp-python and huggingface-hub are required.")
    
    print(f"\n{'='*80}\nLOCAL LLAMA PREPROCESSING\n{'='*80}")
    print(f"Input file: {os.path.basename(csv_path)}")
    print(f"Context window: {n_ctx} tokens")
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
    
    # Run pre-flight check before processing
    preflight_results = preflight_check(
        df_to_process[text_column].tolist(),
        model_context_window=n_ctx
    )
    
    # Ask user to confirm if there are critical reports
    if preflight_results['critical_count'] > 0:
        print("\nWARNING: Some reports may be truncated. Processing will proceed in 5 seconds...")
        time.sleep(5)

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
            n_ctx=n_ctx,  # CONFIGURABLE: default 16384
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
                max_tokens=16384,  # Large enough for most reports, NO CROPPING
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
            
            # Warn if output is significantly shorter than input
            input_words = len(text.split())
            output_words = len(final_text.split())
            retention = output_words / max(input_words, 1)
            
            if retention < 0.5:
                warning_msg = f"Report {idx}: Possible truncation ({output_words}/{input_words} words = {retention:.0%})"
                error_log.append(warning_msg)
                print(f"\n[WARNING] {warning_msg}")
            
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
        print(f"\nError log saved to: {log_path}")
    
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

# Model rate limits for reference
GEMINI_MODEL_LIMITS = {
    # 'gemini-1.5-flash': {'rpm': 15, 'rpd': 1500},   # BEST for free tier
    # 'gemini-1.5-pro': {'rpm': 2, 'rpd': 50},
    'gemini-2.0-flash-lite': {'rpm': 10, 'rpd': 1000},
    'gemini-2.0-flash': {'rpm': 10, 'rpd': 1000},
    'default': {'rpm': 10, 'rpd': 1000},
}


def get_best_gemini_model(preferred_model=None):
    """
    Select Gemini model. FORCES gemini-2.0-flash-lite for best free tier limits.
    
    Args:
        preferred_model: User-specified model (overrides auto-selection)
    
    Returns:
        Model name string
    """
    if not HAS_GEMINI:
        raise ImportError("google-generativeai is required.")
    
    # If user specified a model, use it
    if preferred_model:
        print(f"[OK] Using specified model: {preferred_model}")
        return preferred_model
    
    # FORCE gemini-2.0-flash-lite - it has the best free tier (10 RPM)
    default_model = "gemini-2.0-flash-lite"
    
    # Verify it's available
    try:
        available_models = []
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                available_models.append(m.name)
        
        # Check if our preferred model is available
        for m in available_models:
            if default_model in m:
                print(f"[OK] Selected: {m} (15 RPM - best free tier)")
                return m
        
        # Fallback if 1.5-flash not available
        print(f"[WARNING] {default_model} not available, using first available model")
        return available_models[0] if available_models else default_model
        
    except Exception as e:
        print(f"[WARNING] Could not list models: {e}")
        return default_model


def clean_single_text_with_gemini(text, model, generation_config):
    """
    Process ONE text. NO RETRY - just mark errors for later retry.
    
    Args:
        text: Text to clean
        model: Gemini model instance
        generation_config: Generation config
    
    Returns:
        Cleaned text or error marker
    """
    prompt = f"""Task: Translate and Clean a single text.

Rules:
1. Translate the text into standard British English if it is not already in English.
2. If the text is already in English, correct spelling and grammar errors ONLY.
3. Remove artifacts like '\\n'.
4. Do NOT change the original meaning, punctuation, or structure.
5. Do NOT crop, truncate, summarise, or shorten the text in any way.
6. Your response must contain ONLY the cleaned text - no explanations, no quotes, no markdown.
7. The output should be approximately the same length as the input.

TEXT TO CLEAN:
{text}

CLEANED TEXT:"""

    try:
        response = model.generate_content(prompt, generation_config=generation_config)
        cleaned = response.text.strip()
        
        # Remove potential quotes or markdown the model might add
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1]
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        
        return cleaned
        
    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "quota" in error_str.lower() or "resource" in error_str.lower():
            return "[ERROR: Quota Exceeded]"
        else:
            return f"[ERROR: {type(e).__name__}]"


def preflight_check_gemini(texts, delay_between_texts=2, model_name=None):
    """
    Analyze texts BEFORE Gemini processing to warn about time and quota issues.
    
    Args:
        texts: List of text strings to analyze
        delay_between_texts: Seconds between API calls
        model_name: Model to use (for limit lookup)
    
    Returns:
        Dictionary with analysis results
    """
    print(f"\n{'='*80}")
    print("PRE-FLIGHT CHECK: Gemini API")
    print(f"{'='*80}\n")
    
    # Get model limits
    model_key = 'gemini-2.0-flash-lite' if model_name is None else model_name
    limits = GEMINI_MODEL_LIMITS.get(model_key, GEMINI_MODEL_LIMITS['default'])
    for key in GEMINI_MODEL_LIMITS:
        if key in str(model_key):
            limits = GEMINI_MODEL_LIMITS[key]
            break
    
    FREE_TIER_RPM = limits['rpm']
    FREE_TIER_RPD = limits['rpd']
    
    # Calculate stats
    total_reports = len(texts)
    char_counts = [len(t) if isinstance(t, str) else 0 for t in texts]
    max_chars = max(char_counts) if char_counts else 0
    avg_chars = sum(char_counts) / total_reports if total_reports > 0 else 0
    total_chars = sum(char_counts)
    
    # Time estimates
    base_time_per_text = 2  # ~2 seconds for API call itself
    time_per_text = base_time_per_text + delay_between_texts
    estimated_seconds = total_reports * time_per_text
    estimated_minutes = estimated_seconds / 60
    
    # Calculate effective rate
    effective_rpm = 60 / max(time_per_text, 1)
    
    # Check if safe
    min_safe_delay = max(0, (60 / FREE_TIER_RPM) - base_time_per_text)
    quota_safe = effective_rpm <= FREE_TIER_RPM * 0.9  # 10% margin
    
    # Print results
    print(f"MODEL:")
    print(f"  Selected model:        gemini-2.0-flash-lite (default)")
    print(f"  Rate limit:            {FREE_TIER_RPM} requests/min, {FREE_TIER_RPD}/day")
    print(f"")
    print(f"REPORT STATISTICS:")
    print(f"  Total reports:         {total_reports}")
    print(f"  Total characters:      {total_chars:,}")
    print(f"  Average length:        {avg_chars:,.0f} chars")
    print(f"  Longest report:        {max_chars:,} chars")
    print(f"")
    print(f"API SETTINGS:")
    print(f"  Delay between calls:   {delay_between_texts}s")
    print(f"")
    print(f"TIME ESTIMATES:")
    print(f"  Estimated time:        ~{estimated_minutes:.1f} minutes")
    print(f"")
    
    # Quota warnings
    print(f"{'='*80}")
    print(f"QUOTA ANALYSIS:")
    print(f"{'='*80}")
    print(f"  Model limit:           {FREE_TIER_RPM} requests/minute")
    print(f"  Your effective rate:   ~{effective_rpm:.1f} requests/minute")
    print(f"")
    
    if not quota_safe:
        print(f"  WARNING: With {delay_between_texts}s delay, you may hit quota")
        print(f"")
        print(f"  RECOMMENDATIONS:")
        print(f"    - Use --delay {int(min_safe_delay) + 3} or higher")
        print(f"    - Errors will be marked and can be retried with --retry-failed")
    else:
        print(f"  [OK] Your settings should avoid quota issues")
    
    # Daily limit check
    if total_reports > FREE_TIER_RPD:
        print(f"")
        print(f"  WARNING: {total_reports} reports exceeds daily limit ({FREE_TIER_RPD})")
    
    print(f"\n{'='*80}")
    
    if not quota_safe and total_reports > 20:
        safe_delay = int(min_safe_delay) + 3
        print(f"SUGGESTION: Use --delay {safe_delay} to avoid quota errors")
        print(f"   New estimated time: ~{total_reports * (safe_delay + 2) / 60:.1f} minutes")
    else:
        print(f"[OK] Ready to process {total_reports} reports!")
    
    print(f"{'='*80}\n")
    
    return {
        'total_reports': total_reports,
        'max_chars': max_chars,
        'avg_chars': avg_chars,
        'estimated_minutes': estimated_minutes,
        'quota_safe': quota_safe,
        'model_rpm': FREE_TIER_RPM
    }


def preprocess_with_gemini_api(csv_path, output_path, text_column='reflection_answer', 
                               batch_size=10, num_samples=None, max_text_length=None,
                               log_errors=True, delay_between_texts=2,
                               retry_failed_only=False, model_name=None): 
    """
    Preprocess texts using Gemini API.
    NO RETRY during processing - errors are marked and can be retried later with --retry-failed.
    
    Args:
        csv_path: Input CSV path
        output_path: Output CSV path
        text_column: Column containing text to clean
        batch_size: (DEPRECATED - now processes one at a time)
        num_samples: Limit to N samples (None = all)
        max_text_length: Skip texts longer than this
        log_errors: Save error log
        delay_between_texts: Seconds between API calls (default: 2)
        retry_failed_only: If True and output exists, only reprocess failed rows
        model_name: Specific model to use (default: gemini-2.0-flash-lite)
    """
    if not HAS_GEMINI: raise ImportError("google-generativeai required")
    if HAS_DOTENV: load_dotenv()
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key: raise ValueError("GOOGLE_API_KEY not found.")
    
    genai.configure(api_key=api_key)
    
    # Select model ONCE at the start
    selected_model = get_best_gemini_model(model_name)
    
    print(f"\n{'='*80}\nGEMINI API PREPROCESSING\n{'='*80}")
    print(f"Input file: {os.path.basename(csv_path)}")
    print(f"Model: {selected_model}")
    print(f"Delay between texts: {delay_between_texts}s")
    if max_text_length:
        print(f"Max text length: {max_text_length} characters")
    else:
        print(f"Max text length: No limit (process all reports)")

    # =========================================================================
    # RETRY FAILED ONLY MODE
    # =========================================================================
    if retry_failed_only and os.path.exists(output_path):
        print(f"\nRETRY MODE: Reprocessing only failed rows from existing output")
        
        try:
            existing_df = pd.read_csv(output_path)
        except Exception as e:
            print(f"[ERROR] Could not load existing output: {e}")
            print("Falling back to full processing...")
            retry_failed_only = False
        
        if retry_failed_only:
            # Find rows with errors
            target_col = 'cleaned_reflection'
            if target_col not in existing_df.columns:
                print(f"[ERROR] Column '{target_col}' not found. Running full processing.")
                retry_failed_only = False
            else:
                error_mask = existing_df[target_col].astype(str).str.contains(
                    r'\[ERROR|\[SKIPPED', regex=True, na=False
                )
                failed_indices = existing_df[error_mask].index.tolist()
                
                if not failed_indices:
                    print("[OK] No failed rows found! Nothing to retry.")
                    return existing_df
                
                print(f"Found {len(failed_indices)} failed rows to retry")
                print(f"Failed row indices: {failed_indices[:10]}{'...' if len(failed_indices) > 10 else ''}")
                
                # Setup model
                model = genai.GenerativeModel(selected_model)
                generation_config = genai.types.GenerationConfig(
                    max_output_tokens=8192,
                    temperature=0.1,
                )
                
                # Retry each failed row
                success_count = 0
                still_failed = 0
                for i, idx in enumerate(tqdm(failed_indices, desc="Retrying failed rows")):
                    original_text = existing_df.loc[idx, text_column]
                    
                    if not isinstance(original_text, str) or not original_text.strip():
                        continue
                    
                    cleaned = clean_single_text_with_gemini(
                        original_text, model, generation_config
                    )
                    
                    if not cleaned.startswith("[ERROR"):
                        existing_df.loc[idx, target_col] = cleaned
                        success_count += 1
                    else:
                        existing_df.loc[idx, target_col] = cleaned
                        still_failed += 1
                    
                    # Rate limiting
                    if delay_between_texts > 0 and i < len(failed_indices) - 1:
                        time.sleep(delay_between_texts)
                
                # Save updated file
                existing_df.to_csv(output_path, index=False)
                print(f"\n{'='*80}")
                print(f"RETRY COMPLETE")
                print(f"{'='*80}")
                print(f"Successfully retried: {success_count}/{len(failed_indices)} rows")
                print(f"Still failed: {still_failed} rows")
                print(f"Output saved to: {output_path}")
                return existing_df

    # =========================================================================
    # FULL PROCESSING MODE
    # =========================================================================
    try:
        df = pd.read_csv(csv_path).dropna(subset=[text_column])
    except Exception:
        return None

    df_to_process = df.head(num_samples).copy() if num_samples else df.copy()
    print(f"Processing: {len(df_to_process)} reports")
    
    texts = df_to_process[text_column].tolist()
    
    # Run pre-flight check before processing
    preflight_results = preflight_check_gemini(
        texts,
        delay_between_texts=delay_between_texts,
        model_name=selected_model
    )
    
    # Pause if quota issues expected
    if not preflight_results['quota_safe'] and len(texts) > 20:
        print("WARNING: Quota issues likely. Processing will proceed in 5 seconds...")
        print("   (Errors will be marked - use --retry-failed later)")
        time.sleep(5)
    
    skipped_count = 0
    error_count = 0
    error_log = []
    
    # Setup model
    model = genai.GenerativeModel(selected_model)
    generation_config = genai.types.GenerationConfig(
        max_output_tokens=8192,
        temperature=0.1,
    )
    
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
    
    # Process texts one by one
    all_cleaned = []
    
    print(f"\nProcessing {len(texts)} texts individually...")
    print(f"Estimated time: ~{preflight_results['estimated_minutes']:.1f} minutes")
    
    for i, text in enumerate(tqdm(texts, desc="Gemini API")):
        # Skip already processed error markers from filtering
        if text.startswith("[SKIPPED") or text.startswith("[ERROR"):
            all_cleaned.append(text)
            continue
        
        # Process single text (NO RETRY - just mark errors)
        cleaned = clean_single_text_with_gemini(text, model, generation_config)
        
        if cleaned.startswith("[ERROR"):
            error_count += 1
            error_log.append(f"Report {i}: {cleaned}")
        else:
            # Warn if output is suspiciously shorter than input
            input_words = len(text.split())
            output_words = len(cleaned.split())
            retention = output_words / max(input_words, 1)
            
            if retention < 0.5:
                print(f"\n[WARNING] Text {i}: Possible truncation ({output_words}/{input_words} words = {retention:.0%})")
        
        all_cleaned.append(cleaned)
        
        # Rate limiting between texts
        if delay_between_texts > 0 and i < len(texts) - 1:
            time.sleep(delay_between_texts)

    df_to_process['cleaned_reflection'] = all_cleaned
    df_to_process.to_csv(output_path, index=False)
    
    # Save error log
    if log_errors and error_log:
        log_path = str(output_path).replace('.csv', '_errors.log')
        with open(log_path, 'w') as f:
            f.write(f"Preprocessing Error Log\n")
            f.write(f"{'='*80}\n")
            f.write(f"Dataset: {os.path.basename(csv_path)}\n")
            f.write(f"Method: Gemini API ({selected_model})\n")
            if max_text_length:
                f.write(f"Max text length: {max_text_length} chars\n")
            else:
                f.write(f"Max text length: No limit\n")
            f.write(f"Total reports: {len(texts)}\n")
            f.write(f"Skipped (too long): {skipped_count}\n")
            f.write(f"Errors (quota/other): {error_count}\n")
            f.write(f"Successfully processed: {len(texts) - skipped_count - error_count}\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"DETAILS:\n")
            for err in error_log:
                f.write(f"{err}\n")
            f.write(f"\nTo retry failed rows, run:\n")
            f.write(f"python preprocessing.py --dataset DATASET --method gemini --retry-failed\n")
        print(f"\nError log saved to: {log_path}")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total reports: {len(texts)}")
    print(f"Successfully processed: {len(texts) - skipped_count - error_count}")
    print(f"Errors: {error_count}")
    print(f"Skipped (too long): {skipped_count}")
    print(f"Output saved to: {output_path}")
    
    if error_count > 0:
        print(f"\nTo retry failed rows:")
        print(f"python preprocessing.py --dataset YOUR_DATASET --method gemini --retry-failed --delay 5")
    
    print(f"{'='*80}\n")
    
    return df_to_process

# =============================================================================
# SECTION 4: UTILITY FUNCTIONS
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
  
  # Gemini API preprocessing (uses gemini-2.0-flash-lite by default)
  python preprocessing.py --dataset innerspeech --method gemini
  
  # Gemini with slower rate limiting (recommended for free tier)
  python preprocessing.py --dataset innerspeech --method gemini --delay 5
  
  # RETRY FAILED ROWS ONLY (after first run completes)
  python preprocessing.py --dataset ganzfeld_GREEN --method gemini --retry-failed
  
  # PRE-FLIGHT CHECK: Analyze reports before processing (no processing)
  python preprocessing.py --dataset dreamachine_DL --preflight
  python preprocessing.py --dataset dreamachine_DL --preflight --method gemini
  
  # With optional length limit (skip reports > 10000 chars)
  python preprocessing.py --dataset dreamachine_DL --method llama --max-text-length 10000
  
  # List available datasets
  python preprocessing.py --list-datasets

Available Gemini models (free tier):
  gemini-2.0-flash      10 RPM, 1000/day
  gemini-2.0-flash-lite 10 RPM, 1000/day
        """
    )
    
    parser.add_argument(
        "--dataset",
        required=False,
        help="Dataset name (e.g., dreamachine_DL, MPE, innerspeech)"
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Disable sentence splitting (keep full paragraphs)"
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
        "--preflight",
        action="store_true",
        help="Run pre-flight check only (analyze reports without processing)"
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=16384,
        help="Context window size for Llama (default: 16384)"
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
    parser.add_argument(
        "--delay",
        type=int,
        default=2,
        help="Seconds to wait between API calls (default: 2, use 5+ for free tier)"
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Only reprocess failed rows from existing output file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Gemini model to use (default: gemini-2.0-flash-lite)"
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
                print(f"  - {name}")
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
    
    # Handle --preflight (analyze only, no processing)
    if args.preflight:
        print("\n" + "="*80)
        print("PRE-FLIGHT CHECK MODE (no processing)")
        print("="*80)
        print(f"Dataset:        {args.dataset}")
        print(f"Input:          {input_path}")
        if args.method == "llama":
            print(f"Context window: {args.n_ctx} tokens")
        print("="*80)
        
        try:
            df = load_data(input_path, args.text_column)
            if df is not None:
                if args.sample:
                    df = df.head(args.sample)
                
                # Run appropriate preflight check based on method
                if args.method == "gemini":
                    preflight_check_gemini(
                        df[args.text_column].tolist(),
                        delay_between_texts=args.delay,
                        model_name=args.model
                    )
                else:
                    # Default to Llama preflight (also useful for basic)
                    preflight_check(
                        df[args.text_column].tolist(),
                        model_context_window=args.n_ctx
                    )
        except Exception as e:
            print(f"Error: {e}")
            exit(1)
        exit(0)
    
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
    if args.method == "gemini":
        print(f"API delay:      {args.delay}s between texts")
        print(f"Retry failed:   {'YES - only reprocessing errors' if args.retry_failed else 'No - full processing'}")
    if args.method == "llama":
        print(f"Context window: {args.n_ctx} tokens")
    print("="*80 + "\n")
    
    # Run preprocessing
    try:
        if args.method == "basic":
            df = load_data(input_path, args.text_column)
            if df is not None:
                if args.sample:
                    df = df.head(args.sample)
                    print(f"\nRunning on {len(df)} reports (test mode)")
                
                # Check for participant_id
                if 'participant_id' not in df.columns:
                    print("Warning: 'participant_id' column not found. Creating dummy IDs.")
                    df['participant_id'] = range(len(df))
                
                # CALL THE FUNCTION WITHOUT MANUAL OVERRIDES
                # The function will now look up args.dataset in your config file
                result = basic_preprocess(
                    ids=df['participant_id'].tolist(),
                    texts=df[args.text_column].tolist(),
                    dataset_name=args.dataset,
                    split_override=(not args.no_split) 
                )
                
                result.to_csv(output_path, index=False)
                print(f"\n[OK] Preprocessing complete!")
                print(f"Output saved to: {output_path}")
        
        elif args.method == "llama":
            print(f"\nRunning Llama preprocessing (deterministic, no cropping)...")
            result = preprocess_with_local_llama(
                input_path, 
                output_path, 
                text_column=args.text_column,
                num_samples=args.sample,
                max_text_length=args.max_text_length,
                log_errors=not args.no_error_log,
                n_ctx=args.n_ctx
            )
            if result is not None:
                print(f"\n[OK] Preprocessing complete!")
        
        elif args.method == "gemini":
            if args.retry_failed:
                print(f"\nRETRY MODE: Reprocessing only failed rows...")
            else:
                print(f"\nRunning Gemini API preprocessing...")
            result = preprocess_with_gemini_api(
                input_path,
                output_path,
                text_column=args.text_column,
                num_samples=args.sample,
                max_text_length=args.max_text_length,
                log_errors=not args.no_error_log,
                delay_between_texts=args.delay,
                retry_failed_only=args.retry_failed,
                model_name=args.model
            )
            if result is not None:
                print(f"\n[OK] Preprocessing complete!")
    
    except Exception as e:
        print(f"\n[ERROR] Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        exit(1)



#example usage:
# # Process (errors will be marked, not retried)
# python src/mosaic/preprocessing/preprocessing.py --dataset ganzfeld_GREEN --method gemini
# python src/mosaic/preprocessing/preprocessing.py --dataset ganzfeld_RED --method llama --sample 5

# # After it finishes, retry any failed rows
# python src/mosaic/preprocessing/preprocessing.py --dataset ganzfeld_GREEN --method gemini --retry-failed --delay 7
# ``

# python preprocessing.py --dataset 5MeO_naturalistic --method basic --no-split