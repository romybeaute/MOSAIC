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
    print("📋 PRE-FLIGHT CHECK: Analyzing reports before processing...")
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
    print(f"  ✅ Safe (<{int(safe_tokens)} tokens):       {safe_count:>4} reports ({safe_count/total_reports*100:.1f}%)")
    print(f"  ⚠️  Warning (may truncate):    {warning_count:>4} reports ({warning_count/total_reports*100:.1f}%)")
    print(f"  🔴 Critical (likely truncate): {critical_count:>4} reports ({critical_count/total_reports*100:.1f}%)")
    
    # Detailed warnings for problematic reports
    if problematic_indices:
        print(f"\n{'='*80}")
        print(f"⚠️  REPORTS AT RISK OF TRUNCATION:")
        print(f"{'='*80}")
        for idx in problematic_indices[:10]:  # Show max 10
            chars = char_counts[idx]
            tokens = token_estimates[idx]
            risk = "🔴 CRITICAL" if tokens > critical_tokens else "⚠️  WARNING"
            print(f"  Report {idx}: {chars:,} chars (~{tokens:,.0f} tokens) {risk}")
        if len(problematic_indices) > 10:
            print(f"  ... and {len(problematic_indices) - 10} more")
    
    # Recommendation
    print(f"\n{'='*80}")
    if critical_count > 0:
        print(f"🔴 RECOMMENDATION: {critical_count} reports may be truncated!")
        print(f"   Options:")
        print(f"   1. Increase n_ctx to 32768 (if your GPU has enough VRAM)")
        print(f"   2. Use --max-text-length {int(safe_tokens * chars_per_token)} to skip long reports")
        print(f"   3. Use Gemini API instead (1M token context)")
    elif warning_count > 0:
        print(f"⚠️  RECOMMENDATION: {warning_count} reports are borderline.")
        print(f"   Processing should work, but monitor for truncation warnings.")
    else:
        print(f"✅ ALL REPORTS LOOK SAFE! No truncation expected.")
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
    
    # PATCHED: Run pre-flight check before processing
    preflight_results = preflight_check(
        df_to_process[text_column].tolist(),
        model_context_window=n_ctx
    )
    
    # Ask user to confirm if there are critical reports
    if preflight_results['critical_count'] > 0:
        print("\n⚠️  Some reports may be truncated. Continue anyway? (Processing will proceed in 5 seconds...)")
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
            
            # PATCHED: Warn if output is significantly shorter than input
            input_words = len(text.split())
            output_words = len(final_text.split())
            retention = output_words / max(input_words, 1)
            
            if retention < 0.5:
                warning_msg = f"Report {idx}: Possible truncation ({output_words}/{input_words} words = {retention:.0%})"
                error_log.append(warning_msg)
                print(f"\n⚠️  {warning_msg}")
            
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


def clean_single_text_with_gemini(text, model, generation_config, max_retries=3, retry_delay=60):
    """
    PATCHED v2: Process ONE text at a time WITH RETRY LOGIC.
    
    Args:
        text: Text to clean
        model: Gemini model instance
        generation_config: Generation config
        max_retries: Number of retries on quota error (default: 3)
        retry_delay: Seconds to wait before retry (default: 60)
    
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

    for attempt in range(max_retries + 1):
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
            is_quota_error = "429" in error_str or "quota" in error_str.lower() or "resource" in error_str.lower()
            
            if is_quota_error and attempt < max_retries:
                print(f"\n⏳ Quota exceeded. Waiting {retry_delay}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(retry_delay)
                continue
            elif is_quota_error:
                print(f"\n❌ Quota exceeded after {max_retries} retries. Skipping this text.")
                return "[ERROR: Quota Exceeded - Max Retries]"
            else:
                return f"[ERROR: {type(e).__name__}]"
    
    return "[ERROR: Unknown]"


def clean_batch_with_gemini(texts, model_name=None, delay_between_texts=2, max_retries=3, retry_delay=60):
    """
    PATCHED v2: Processes texts ONE BY ONE with RETRY LOGIC and RATE LIMITING.
    
    Args:
        texts: List of texts to clean
        model_name: Gemini model name (auto-detected if None)
        delay_between_texts: Seconds to wait between each text (default: 2)
        max_retries: Number of retries on quota error (default: 3)
        retry_delay: Seconds to wait before retry (default: 60)
    
    Returns:
        List of cleaned texts
    """
    if not HAS_GEMINI:
        raise ImportError("google-generativeai is required.")
    
    if model_name is None:
        model_name = get_best_gemini_model()
    
    model = genai.GenerativeModel(model_name)
    
    # PATCHED: Explicit generation config with high output token limit
    generation_config = genai.types.GenerationConfig(
        max_output_tokens=8192,  # Explicit high limit
        temperature=0.1,         # Low for deterministic cleaning
    )
    
    cleaned_texts = []
    
    for i, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            cleaned_texts.append("")
            continue
            
        # Skip already-errored texts
        if text.startswith("[SKIPPED") or text.startswith("[ERROR"):
            cleaned_texts.append(text)
            continue
        
        # Clean text with retry logic
        cleaned = clean_single_text_with_gemini(
            text, model, generation_config, 
            max_retries=max_retries, 
            retry_delay=retry_delay
        )
        
        # PATCHED: Warn if output is suspiciously shorter than input
        input_words = len(text.split())
        output_words = len(cleaned.split())
        retention = output_words / max(input_words, 1)
        
        if retention < 0.5 and not cleaned.startswith("[ERROR"):
            print(f"\n⚠️  Text {i}: Possible truncation ({output_words}/{input_words} words = {retention:.0%})")
        
        cleaned_texts.append(cleaned)
        
        # Rate limiting between texts
        if delay_between_texts > 0 and i < len(texts) - 1:
            time.sleep(delay_between_texts)
    
    return cleaned_texts


def preflight_check_gemini(texts, delay_between_texts=2, max_retries=3, retry_delay=60):
    """
    Analyze texts BEFORE Gemini processing to warn about time and quota issues.
    
    Args:
        texts: List of text strings to analyze
        delay_between_texts: Seconds between API calls
        max_retries: Max retries per text
        retry_delay: Seconds to wait on retry
    
    Returns:
        Dictionary with analysis results
    """
    print(f"\n{'='*80}")
    print("📋 PRE-FLIGHT CHECK: Gemini API")
    print(f"{'='*80}\n")
    
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
    
    # Quota estimates (free tier: ~15 requests/minute, ~1500/day)
    FREE_TIER_RPM = 15  # requests per minute
    FREE_TIER_RPD = 1500  # requests per day
    
    texts_before_quota_hit = FREE_TIER_RPM  # Will hit after ~15 texts at full speed
    if delay_between_texts >= 4:
        # With 4+ second delay, we're under 15 RPM
        texts_before_quota_hit = total_reports  # Should complete without quota issues
    
    # Worst case: every text needs max retries
    worst_case_seconds = total_reports * (time_per_text + (max_retries * retry_delay))
    worst_case_minutes = worst_case_seconds / 60
    
    # Print results
    print(f"REPORT STATISTICS:")
    print(f"  Total reports:         {total_reports}")
    print(f"  Total characters:      {total_chars:,}")
    print(f"  Average length:        {avg_chars:,.0f} chars")
    print(f"  Longest report:        {max_chars:,} chars")
    print(f"")
    print(f"API SETTINGS:")
    print(f"  Delay between calls:   {delay_between_texts}s")
    print(f"  Max retries per text:  {max_retries}")
    print(f"  Retry delay:           {retry_delay}s")
    print(f"")
    print(f"TIME ESTIMATES:")
    print(f"  Best case:             ~{estimated_minutes:.1f} minutes")
    print(f"  Worst case (retries):  ~{worst_case_minutes:.1f} minutes")
    print(f"")
    
    # Quota warnings
    print(f"{'='*80}")
    print(f"QUOTA ANALYSIS (Free Tier):")
    print(f"{'='*80}")
    print(f"  Free tier limit:       ~{FREE_TIER_RPM} requests/minute")
    print(f"  Your effective rate:   ~{60/max(time_per_text, 1):.1f} requests/minute")
    print(f"")
    
    if delay_between_texts < 4:
        print(f"  ⚠️  WARNING: With {delay_between_texts}s delay, you may hit quota after ~{texts_before_quota_hit} texts")
        print(f"")
        print(f"  RECOMMENDATIONS:")
        print(f"    • Use --delay 5 or higher for free tier")
        print(f"    • Or expect retries (script will wait {retry_delay}s and retry)")
        quota_safe = False
    else:
        print(f"  ✅ Your delay ({delay_between_texts}s) should avoid most quota issues")
        quota_safe = True
    
    # Check for very long reports (might be slow)
    very_long_threshold = 10000  # 10K chars
    very_long_count = sum(1 for c in char_counts if c > very_long_threshold)
    if very_long_count > 0:
        print(f"")
        print(f"  ℹ️  {very_long_count} reports are >10K chars (may take longer to process)")
    
    print(f"\n{'='*80}")
    
    # Final recommendation
    if not quota_safe and total_reports > 20:
        print(f"💡 SUGGESTION: Consider using --delay 5 to avoid quota errors")
        print(f"   New estimated time: ~{total_reports * 7 / 60:.1f} minutes")
    elif total_reports > 100:
        print(f"💡 SUGGESTION: For {total_reports} reports, consider running overnight")
        print(f"   Or use Llama locally (no API limits): --method llama")
    else:
        print(f"✅ Ready to process {total_reports} reports!")
    
    print(f"{'='*80}\n")
    
    return {
        'total_reports': total_reports,
        'max_chars': max_chars,
        'avg_chars': avg_chars,
        'estimated_minutes': estimated_minutes,
        'worst_case_minutes': worst_case_minutes,
        'quota_safe': quota_safe,
        'very_long_count': very_long_count
    }


def preprocess_with_gemini_api(csv_path, output_path, text_column='reflection_answer', 
                               batch_size=10, num_samples=None, max_text_length=None,
                               log_errors=True, delay_between_texts=2, max_retries=3,
                               retry_delay=60, retry_failed_only=False): 
    """
    Preprocess texts using Gemini API with retry logic.
    
    Args:
        csv_path: Input CSV path
        output_path: Output CSV path
        text_column: Column containing text to clean
        batch_size: (DEPRECATED - now processes one at a time)
        num_samples: Limit to N samples (None = all)
        max_text_length: Skip texts longer than this
        log_errors: Save error log
        delay_between_texts: Seconds between API calls (default: 2)
        max_retries: Retries per text on quota error (default: 3)
        retry_delay: Seconds to wait before retry (default: 60)
        retry_failed_only: If True and output exists, only reprocess failed rows
    """
    if not HAS_GEMINI: raise ImportError("google-generativeai required")
    if HAS_DOTENV: load_dotenv()
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key: raise ValueError("GOOGLE_API_KEY not found.")
    
    genai.configure(api_key=api_key)
    
    print(f"\n{'='*80}\nGEMINI API PREPROCESSING (with retry logic)\n{'='*80}")
    print(f"Input file: {os.path.basename(csv_path)}")
    print(f"Delay between texts: {delay_between_texts}s")
    print(f"Max retries per text: {max_retries} (wait {retry_delay}s each)")
    if max_text_length:
        print(f"Max text length: {max_text_length} characters")
    else:
        print(f"Max text length: No limit (process all reports)")

    # =========================================================================
    # RETRY FAILED ONLY MODE
    # =========================================================================
    if retry_failed_only and os.path.exists(output_path):
        print(f"\n🔄 RETRY MODE: Reprocessing only failed rows from existing output")
        
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
                    print("✅ No failed rows found! Nothing to retry.")
                    return existing_df
                
                print(f"Found {len(failed_indices)} failed rows to retry")
                print(f"Failed row indices: {failed_indices[:10]}{'...' if len(failed_indices) > 10 else ''}")
                
                # Get model
                model_name = get_best_gemini_model()
                model = genai.GenerativeModel(model_name)
                generation_config = genai.types.GenerationConfig(
                    max_output_tokens=8192,
                    temperature=0.1,
                )
                
                # Retry each failed row
                success_count = 0
                for i, idx in enumerate(tqdm(failed_indices, desc="Retrying failed rows")):
                    original_text = existing_df.loc[idx, text_column]
                    
                    if not isinstance(original_text, str) or not original_text.strip():
                        continue
                    
                    cleaned = clean_single_text_with_gemini(
                        original_text, model, generation_config,
                        max_retries=max_retries, retry_delay=retry_delay
                    )
                    
                    if not cleaned.startswith("[ERROR"):
                        existing_df.loc[idx, target_col] = cleaned
                        success_count += 1
                    else:
                        existing_df.loc[idx, target_col] = cleaned
                    
                    # Rate limiting
                    if delay_between_texts > 0 and i < len(failed_indices) - 1:
                        time.sleep(delay_between_texts)
                
                # Save updated file
                existing_df.to_csv(output_path, index=False)
                print(f"\n{'='*80}")
                print(f"RETRY COMPLETE")
                print(f"{'='*80}")
                print(f"Successfully retried: {success_count}/{len(failed_indices)} rows")
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
    
    # PATCHED: Run pre-flight check before processing
    preflight_results = preflight_check_gemini(
        texts,
        delay_between_texts=delay_between_texts,
        max_retries=max_retries,
        retry_delay=retry_delay
    )
    
    # Pause if quota issues expected
    if not preflight_results['quota_safe'] and len(texts) > 20:
        print("⚠️  Quota issues likely. Processing will proceed in 5 seconds...")
        print("   (Use Ctrl+C to cancel and restart with --delay 5)")
        time.sleep(5)
    
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
    
    # PATCHED v2: Process texts with progress bar, rate limiting, and retry logic
    all_cleaned = []
    
    print(f"\nProcessing {len(texts)} texts individually...")
    print(f"⏱️  Estimated time: ~{preflight_results['estimated_minutes']:.1f} minutes")
    
    for i, text in enumerate(tqdm(texts, desc="Gemini API")):
        # Skip already processed error markers from filtering
        if text.startswith("[SKIPPED") or text.startswith("[ERROR"):
            all_cleaned.append(text)
            continue
        
        # Process single text with retry logic
        result = clean_batch_with_gemini(
            [text], 
            delay_between_texts=0,  # Delay handled here, not inside
            max_retries=max_retries,
            retry_delay=retry_delay
        )
        all_cleaned.extend(result)
        
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
  
  # Gemini API preprocessing (with retry logic)
  python preprocessing.py --dataset innerspeech --method gemini
  
  # Gemini with slower rate limiting (for free tier)
  python preprocessing.py --dataset innerspeech --method gemini --delay 5
  
  # RETRY FAILED ROWS ONLY (reprocess errors from previous run)
  python preprocessing.py --dataset ganzfeld_GREEN --method gemini --retry-failed
  
  # Retry with longer wait time
  python preprocessing.py --dataset ganzfeld_GREEN --method gemini --retry-failed --retry-delay 120
  
  # PRE-FLIGHT CHECK: Analyze reports before processing (no processing)
  python preprocessing.py --dataset dreamachine_DL --preflight
  python preprocessing.py --dataset dreamachine_DL --preflight --method gemini --delay 5
  
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
        "--batch-size",
        type=int,
        default=10,
        help="(DEPRECATED) Batch size for Gemini API - now processes one at a time"
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
    # NEW: Retry and rate limiting options
    parser.add_argument(
        "--delay",
        type=int,
        default=2,
        help="Seconds to wait between API calls (default: 2, use 5+ for free tier)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per text on quota error (default: 3)"
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=60,
        help="Seconds to wait before retry on quota error (default: 60)"
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Only reprocess failed rows from existing output file"
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
    
    # Handle --preflight (analyze only, no processing)
    if args.preflight:
        print("\n" + "="*80)
        print("PRE-FLIGHT CHECK MODE (no processing)")
        print("="*80)
        print(f"Dataset:        {args.dataset}")
        print(f"Input:          {input_path}")
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
                        max_retries=args.max_retries,
                        retry_delay=args.retry_delay
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
        print(f"Max retries:    {args.max_retries} (wait {args.retry_delay}s each)")
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
                log_errors=not args.no_error_log,
                n_ctx=args.n_ctx
            )
            if result is not None:
                print(f"\n✓ Preprocessing complete!")
        
        elif args.method == "gemini":
            if args.retry_failed:
                print(f"\n🔄 Running Gemini API preprocessing (RETRY FAILED ONLY)...")
            else:
                print(f"\nRunning Gemini API preprocessing (with retry logic)...")
            result = preprocess_with_gemini_api(
                input_path,
                output_path,
                text_column=args.text_column,
                batch_size=args.batch_size,
                num_samples=args.sample,
                max_text_length=args.max_text_length,
                log_errors=not args.no_error_log,
                delay_between_texts=args.delay,
                max_retries=args.max_retries,
                retry_delay=args.retry_delay,
                retry_failed_only=args.retry_failed
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