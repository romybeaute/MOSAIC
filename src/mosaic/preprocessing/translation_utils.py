#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : translation_utils.py
# description     : Advanced translation & validation utilities for preprocessing.py
#                   - Document-level translation (literal preservation)
#                   - Sentence-aligned translation (1:1 mapping)
#                   - ERROR HANDLING with comprehensive error detection
#                   - ANONYMISATION for PII masking
#                   - MOSAIC integration support
#                   - Works WITH preprocessed data, not redundantly
#                   - Consolidated from local_translator.py + local_translator_sentences.py
# author          : Romy, Beauté (r.beaut@sussex.ac.uk)
# date            : 2026-01-28
# version         : 2.0 (enhanced with anonymisation & error handling)
# ==============================================================================

import os
import json
import argparse
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm

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
    import nltk
    from nltk.data import load as nltk_load
    from nltk.tokenize import PunktSentenceTokenizer
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

# Optional MOSAIC integration
try:
    from mosaic.path_utils import CFG, proc_path
    HAS_MOSAIC = True
except ImportError:
    HAS_MOSAIC = False


# =============================================================================
# SECTION 0: ANONYMISATION (NEW IN v2.0)
# =============================================================================

class PII_PATTERNS:
    """Common PII patterns for anonymisation."""
    # Name patterns (FirstName LastName)
    NAMES = r'\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
    
    # Email patterns
    EMAILS = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,}'
    
    # Phone patterns (various formats)
    PHONES = r'(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\+\d{10,}'
    
    # Dates (YYYY-MM-DD, DD/MM/YYYY, MM/DD/YYYY)
    DATES = r'\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}\.\d{1,2}\.\d{2,4})\b'
    
    # ID/numeric patterns (8+ digits)
    IDS = r'\b\d{8,}\b'
    
    # URLs
    URLS = r'https?://[^\s]+'
    
    # Medical/sensitive keywords (for flagging)
    MEDICAL_KEYWORDS = [
        'diagnosis', 'patient', 'hospital', 'doctor', 'disease', 'cancer',
        'diabetes', 'medication', 'symptoms', 'treatment', 'surgery', 'clinic'
    ]


def anonymise_text(text: str, 
                   mask_names: bool = True,
                   mask_emails: bool = True,
                   mask_phones: bool = True,
                   mask_dates: bool = True,
                   mask_ids: bool = True,
                   mask_urls: bool = True) -> str:
    """
    Anonymise text by masking PII patterns.
    
    Args:
        text: Text to anonymise
        mask_names: Mask person names (default: True)
        mask_emails: Mask email addresses (default: True)
        mask_phones: Mask phone numbers (default: True)
        mask_dates: Mask dates (default: True)
        mask_ids: Mask numeric IDs (default: True)
        mask_urls: Mask URLs (default: True)
    
    Returns:
        Anonymised text
    """
    if not isinstance(text, str):
        return str(text)
    
    result = text
    
    # Order matters: do URLs first (they might contain other patterns)
    if mask_urls:
        result = re.sub(PII_PATTERNS.URLS, '[URL]', result, flags=re.IGNORECASE)
    
    if mask_emails:
        result = re.sub(PII_PATTERNS.EMAILS, '[EMAIL]', result, flags=re.IGNORECASE)
    
    if mask_phones:
        result = re.sub(PII_PATTERNS.PHONES, '[PHONE]', result)
    
    if mask_dates:
        result = re.sub(PII_PATTERNS.DATES, '[DATE]', result)
    
    if mask_ids:
        result = re.sub(PII_PATTERNS.IDS, '[ID]', result)
    
    if mask_names:
        # Names last to avoid double-masking
        result = re.sub(PII_PATTERNS.NAMES, '[PERSON]', result)
    
    return result


def contains_sensitive_info(text: str) -> Tuple[bool, List[str]]:
    """
    Check if text contains potentially sensitive information.
    
    Returns:
        Tuple of (is_sensitive, list_of_keywords_found)
    """
    if not isinstance(text, str):
        return False, []
    
    text_lower = text.lower()
    found_keywords = []
    
    for keyword in PII_PATTERNS.MEDICAL_KEYWORDS:
        if keyword in text_lower:
            found_keywords.append(keyword)
    
    # Check for patterns
    has_emails = bool(re.search(PII_PATTERNS.EMAILS, text))
    has_phones = bool(re.search(PII_PATTERNS.PHONES, text))
    has_dates = bool(re.search(PII_PATTERNS.DATES, text))
    has_ids = bool(re.search(PII_PATTERNS.IDS, text))
    has_names = bool(re.search(PII_PATTERNS.NAMES, text))
    
    if has_emails:
        found_keywords.append('email_found')
    if has_phones:
        found_keywords.append('phone_found')
    if has_dates:
        found_keywords.append('date_found')
    if has_ids:
        found_keywords.append('id_found')
    if has_names:
        found_keywords.append('name_found')
    
    is_sensitive = len(found_keywords) > 0
    
    return is_sensitive, found_keywords


# =============================================================================
# SECTION 1: ENHANCED ERROR HANDLING (NEW IN v2.0)
# =============================================================================

class ErrorDetector:
    """Detect various types of translation errors."""
    
    ERROR_PATTERNS = {
        'timeout': (r'(?i)timeout|timed out', 'Timeout error'),
        'quota': (r'(?i)quota exceeded|rate limit|too many requests', 'API quota exceeded'),
        'json_error': (r'(?i)invalid json|json parse error|json decode', 'Invalid JSON'),
        'empty': (r'(?i)not applicable \(empty\)', 'Empty input'),
        'failed': (r'(?i)failed|failure', 'Translation failed'),
        'exception': (r'(?i)exception|traceback', 'Exception occurred'),
        'no_output': (r'^$|^\s+$', 'No output generated'),
        'truncated': (r'(?i)truncated|cut off|incomplete', 'Incomplete translation'),
        'api_error': (r'(?i)api error|connection error|network error', 'API/Network error'),
        'generic_error': (r'(?i)error', 'Generic error detected'),
    }
    
    @staticmethod
    def check_for_errors(text: str) -> Tuple[bool, str, str]:
        """
        Check for error patterns in text.
        
        Returns:
            Tuple of (has_error, error_type, error_description)
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Check specific patterns in order of specificity
        for error_key in ['timeout', 'quota', 'json_error', 'empty', 'failed', 
                         'exception', 'no_output', 'truncated', 'api_error']:
            pattern, description = ErrorDetector.ERROR_PATTERNS[error_key]
            if re.search(pattern, text):
                return True, error_key, description
        
        # Fallback to generic error pattern
        pattern, description = ErrorDetector.ERROR_PATTERNS['generic_error']
        if re.search(pattern, text):
            return True, 'generic_error', description
        
        return False, '', 'No errors detected'
    
    @staticmethod
    def check_translation_quality(src_len: int, tgt_len: int,
                                 min_src_len: int = 20,
                                 min_ratio: float = 0.5,
                                 max_ratio: float = 2.0) -> Tuple[bool, str]:
        """
        Check translation quality based on length ratios.
        
        Args:
            src_len: Source text length (words)
            tgt_len: Target text length (words)
            min_src_len: Minimum source length to check
            min_ratio: Minimum target/source ratio (default 0.5 = half length)
            max_ratio: Maximum target/source ratio (default 2.0 = double length)
        
        Returns:
            Tuple of (is_valid, message)
        """
        if src_len < min_src_len:
            return True, "Source text too short to validate"
        
        if tgt_len == 0:
            return False, "Target text is empty"
        
        ratio = tgt_len / src_len
        
        if ratio < min_ratio:
            return False, f"Translation too short (ratio={ratio:.2f}, min={min_ratio})"
        
        if ratio > max_ratio:
            return False, f"Translation too long (ratio={ratio:.2f}, max={max_ratio})"
        
        return True, "OK"


# =============================================================================
# SECTION 2: PUNKT TOKENIZER UTILITIES (Language-Aware Sentence Splitting)
# =============================================================================

_PUNKT_CACHE = {}


def get_punkt_tokenizer(lang: str = "english"):
    """
    Lazily load and cache the NLTK PunktSentenceTokenizer for a given language.
    
    Supports:
      - 'english' (DEFAULT)
      - 'french'
      - 'german', 'spanish', etc., if installed.
    
    Install with: python -m nltk.downloader punkt
    """
    if not HAS_NLTK:
        raise ImportError("NLTK is required. Install with: pip install nltk")
    
    lang = lang.lower()
    if lang not in _PUNKT_CACHE:
        try:
            tokenizer = nltk_load(f"tokenizers/punkt/{lang}.pickle")
        except LookupError as e:
            raise RuntimeError(
                f"NLTK Punkt model for language '{lang}' not found.\n"
                f"Run: python -m nltk.downloader punkt"
            ) from e
        _PUNKT_CACHE[lang] = tokenizer
    return _PUNKT_CACHE[lang]


def sentence_split(text: str, sent_lang: str = "english") -> List[str]:
    """
    Sentence splitter using NLTK Punkt.
    More robust than regex, handles abbreviations like 'Dr.', 'M. X', etc.
    
    Args:
        text: Input text to split
        sent_lang: Language code ('english', 'french', etc.)
    
    Returns:
        List of sentences
    """
    if not isinstance(text, str) or not text.strip():
        return []
    
    tokenizer = get_punkt_tokenizer(sent_lang)
    sentences = tokenizer.tokenize(text.strip())
    return [s.strip() for s in sentences if s.strip()]


# =============================================================================
# SECTION 3: JSON EXTRACTION & OUTPUT CLEANING
# =============================================================================

def _extract_json_array(text: str):
    """
    Best-effort extraction of a JSON array from model output.
    Looks for first '[' and last ']', tries to parse that slice.
    
    Raises:
        ValueError: If no valid JSON array found
    """
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON array brackets found in model output.")
    candidate = text[start: end + 1]
    return json.loads(candidate)


def strip_translation_output(text: str) -> str:
    """
    Post-process model output to remove boilerplate preambles.
    
    Removes patterns like:
      - 'Here is the translation:'
      - 'Here is the translation of the text into English:'
      - 'Translation:'
      - Surrounding quotes
    
    Returns only the translated content.
    """
    if not isinstance(text, str):
        return text

    s = text.strip()

    # Common preambles (regex patterns)
    patterns = [
        r'^Here is the translation of the text into English:\s*',
        r'^Here is the translation of the text:\s*',
        r'^Here is the translation:\s*',
        r'^Here is the translation in English:\s*',
        r'^The translation is:\s*',
        r'^Translation:\s*',
    ]
    for pat in patterns:
        new_s = re.sub(pat, '', s, flags=re.IGNORECASE)
        if new_s != s:
            s = new_s.strip()

    # Strip wrapping quotes if present
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == '"' and s[-1] == '"')):
        s = s[1:-1].strip()

    return s


# =============================================================================
# SECTION 4: CLEANING HELPERS (JSON-ARRAY BASED)
# =============================================================================

CLEAN_PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Please act as a data cleaning expert. Your task is to clean each of the following numbered texts.

Follow these rules precisely:
1.  For each text, correct spelling mistakes, fix grammar, and remove artifacts like '\\n'.
2.  Do NOT change the original meaning or remove punctuation.
3.  Return the result as a single, valid JSON array of strings.
4.  The JSON array must have exactly {n_texts} elements, where each string is a cleaned version of the corresponding input text.
5.  Do not include the numbers or any other commentary in your output, only the JSON array.<|eot_id|><|start_header_id|>user<|end_header_id|>

Here are the texts to clean:

{text_block}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


def clean_texts_with_llm(texts: List[str], llm: Llama, 
                         prompt_template: str = CLEAN_PROMPT_TEMPLATE) -> List[str]:
    """
    Clean a batch of texts using LLM with JSON-array output.
    
    Args:
        texts: List of texts to clean
        llm: Llama instance
        prompt_template: Custom prompt template (optional)
    
    Returns:
        List of cleaned texts (same length as input)
    """
    if not texts:
        return []

    text_block = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    prompt = prompt_template.format(text_block=text_block, n_texts=len(texts))

    response = llm(
        prompt=prompt,
        max_tokens=1024,
        stop=["<|eot_id|>"],
        echo=False,
        temperature=0,
        top_p=1.0,
    )
    raw = response["choices"][0]["text"].strip()

    try:
        cleaned = _extract_json_array(raw)
        if not isinstance(cleaned, list) or len(cleaned) != len(texts):
            raise ValueError("JSON array length mismatch.")
        cleaned = [str(x) for x in cleaned]
        return cleaned
    except Exception as e:
        print(f"\n[WARN] Could not parse JSON cleaning output. Error: {e}")
        print("[WARN] Returning original texts for this batch.")
        return texts


# =============================================================================
# SECTION 5: SENTENCE-ALIGNED TRANSLATION
# =============================================================================

SENTENCE_TRANSLATION_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a literal, sentence-aligned translator. Translate each input sentence into English with NO summarising, NO omissions, and NO merging or splitting. Preserve all information and sentence boundaries.
Return ONLY a valid JSON array of strings with exactly the same number of elements as the input sentences. Do not add any commentary, explanations, or labels.<|eot_id|><|start_header_id|>user<|end_header_id|>
Translate each of the following {n} sentences into English. Return ONLY a JSON array with {n} strings, one per sentence, in the same order.

Sentences:
{numbered_sentences}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


def _format_numbered_sentences(sents: List[str]) -> str:
    """Format sentences as numbered list for prompt."""
    return "\n".join(f"{i+1}. {s}" for i, s in enumerate(sents))


def translate_sentences_batch(
    sents: List[str],
    llm: Llama,
    max_ctx: int,
    safety_margin: int = 64,
) -> List[str]:
    """
    Translate a batch of sentences with strict 1:1 mapping.
    
    Args:
        sents: List of sentences to translate
        llm: Llama instance
        max_ctx: Context window size
        safety_margin: Safety margin for token calculation
    
    Returns:
        List of translated sentences (same length as input)
    
    Raises:
        RuntimeError: If prompt too long for context
        ValueError: If JSON array size mismatch
    """
    numbered = _format_numbered_sentences(sents)
    prompt = SENTENCE_TRANSLATION_PROMPT.format(n=len(sents), numbered_sentences=numbered)

    prompt_tokens = len(llm.tokenize(prompt.encode("utf-8")))
    avail_for_gen = max_ctx - prompt_tokens - safety_margin
    if avail_for_gen <= 0:
        raise RuntimeError(
            f"Prompt too long for context window (tokens={prompt_tokens}, max={max_ctx}). "
            f"Reduce batch size."
        )

    resp = llm(
        prompt=prompt,
        max_tokens=avail_for_gen,
        stop=["<|eot_id|>"],
        echo=False,
        temperature=0,
        top_p=1.0,
    )
    raw = resp["choices"][0]["text"].strip()

    out = _extract_json_array(raw)
    if not isinstance(out, list) or len(out) != len(sents):
        raise ValueError(
            f"Model returned invalid JSON array: expected {len(sents)} items, "
            f"got {type(out)} / {len(out) if isinstance(out, list) else 'NA'}."
        )
    return [str(x).strip() for x in out]


def translate_text_sentence_aligned(
    text: str,
    llm: Llama,
    max_ctx: int,
    target_batch_size: int = 6,
    safety_margin: int = 64,
    sent_lang: str = "english",
) -> str:
    """
    Strict literal translation with sentence-level alignment.
    
    Process:
      1. Split text into sentences (NLTK Punkt)
      2. Translate in small batches
      3. Dynamically shrink batch size if it exceeds context
      4. Reassemble sentences in order
    
    Args:
        text: Text to translate
        llm: Llama instance
        max_ctx: Context window size
        target_batch_size: Preferred batch size (6 is a good default)
        safety_margin: Token buffer
        sent_lang: Language for sentence tokenization ('english', 'french', etc.)
    
    Returns:
        Translated text with preserved sentence boundaries
    """
    if not isinstance(text, str) or not text.strip():
        return "Not applicable (empty)"

    sents = sentence_split(text, sent_lang=sent_lang)
    if not sents:
        return ""

    results: List[str] = []
    i = 0
    N = len(sents)

    while i < N:
        batch_end = min(i + target_batch_size, N)
        while True:
            batch = sents[i:batch_end]
            try:
                translated_batch = translate_sentences_batch(
                    batch, llm, max_ctx, safety_margin
                )
                results.extend(translated_batch)
                i = batch_end
                break
            except RuntimeError as e:
                if batch_end - i <= 1:
                    raise RuntimeError(f"Even single sentence too long: {e}") from e
                batch_end = i + max(1, (batch_end - i) // 2)

    return " ".join(results)


# =============================================================================
# SECTION 6: DOCUMENT-LEVEL TRANSLATION (LITERAL)
# =============================================================================

DOCUMENT_TRANSLATION_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert, literal translator. Your task is to translate the user's text into English accurately, preserving ALL information, level of detail, and structure.

Very important rules:
- Do NOT summarise or shorten the text.
- Do NOT omit any sentences, phrases, or details.
- Keep roughly one English sentence for each original sentence whenever possible.
- Do NOT add explanations, commentary, or meta-text.
- The translation must be approximately the same length as the source (similar number of words).
- If you are unsure, err on the side of being more verbose and literal.
- Output ONLY the translated English text and nothing else.<|eot_id|><|start_header_id|>user<|end_header_id|>

Translate the following text to English. Remember: do not shorten or summarise; preserve every detail:

"{text_to_translate}"<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


def translate_text_document_level(
    text: str,
    llm: Llama,
    max_ctx: int,
    safety_margin: int = 64,
) -> str:
    """
    Full-document translation (single pass, literal preservation).
    
    Args:
        text: Text to translate
        llm: Llama instance
        max_ctx: Context window size
        safety_margin: Token buffer
    
    Returns:
        Translated text
    
    Raises:
        RuntimeError: If text too long for context window
    """
    if not isinstance(text, str) or not text.strip():
        return "Not applicable (empty)"

    prompt = DOCUMENT_TRANSLATION_PROMPT.format(text_to_translate=text)
    
    prompt_tokens = len(llm.tokenize(prompt.encode("utf-8")))
    avail_for_gen = max_ctx - prompt_tokens - safety_margin
    if avail_for_gen <= 0:
        raise RuntimeError(
            f"Text too long for context window (tokens={prompt_tokens}, max={max_ctx})"
        )

    response = llm(
        prompt=prompt,
        max_tokens=avail_for_gen,
        temperature=0.0,
        top_p=0.0,
        top_k=0,
        repeat_penalty=1.0,
        stop=["<|eot_id|>", "\n\nTranslation:"],
        echo=False,
    )
    raw_out = response["choices"][0]["text"].strip()
    translated_text = strip_translation_output(raw_out)
    
    return translated_text


# =============================================================================
# SECTION 7: ERROR LOGGING & QUALITY CHECKS (ENHANCED IN v2.0)
# =============================================================================

def log_error(row_idx: int, 
              text_preview: str, 
              error: str, 
              task: str,
              error_type: str = "unknown",
              anonymise: bool = True) -> dict:
    """
    Create an error log entry with optional anonymisation.
    
    Args:
        row_idx: Row index
        text_preview: Text preview (will be anonymised if anonymise=True)
        error: Error message
        task: Task type (translate/preprocess/both)
        error_type: Error category (timeout, quota, etc.)
        anonymise: Whether to anonymise text preview (default: True)
    
    Returns:
        Error log entry dict
    """
    preview = str(text_preview)[:200]
    if anonymise:
        preview = anonymise_text(preview)
    
    return {
        "row_index": row_idx,
        "error_type": error_type,
        "text_preview": preview,
        "error": error,
        "task": task,
        "timestamp": datetime.now().isoformat(),
    }


def save_error_log(error_log: List[dict], output_path: Path):
    """
    Save error log to CSV with timestamp.
    
    Args:
        error_log: List of error log entries
        output_path: Output CSV path
    """
    if not error_log:
        return
    
    error_log_path = output_path.with_name(output_path.stem + "_errors.csv")
    error_df = pd.DataFrame(error_log)
    error_df.to_csv(error_log_path, index=False)
    print(
        f"\n[INFO] {len(error_log)} row(s) logged as problematic. "
        f"Details saved to {error_log_path}"
    )


# =============================================================================
# SECTION 8: DATA QUALITY & STATISTICS (ENHANCED IN v2.0)
# =============================================================================

def word_count(text) -> int:
    """Simple whitespace-based word count."""
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    return len(text.split())


def detect_outliers_by_length(df: pd.DataFrame, 
                              text_col: str,
                              method: str = "iqr",
                              threshold: float = 1.5) -> Dict:
    """
    Detect outlier texts by length (sentence count).
    
    Uses IQR method similar to prepare_data.ipynb.
    
    Args:
        df: DataFrame with texts
        text_col: Column name with texts
        method: Detection method ("iqr" or "zscore")
        threshold: IQR multiplier or z-score threshold
    
    Returns:
        Dict with outlier info
    """
    if text_col not in df.columns:
        return {"error": f"Column '{text_col}' not found"}
    
    # Count sentences for each text
    sentence_counts = df[text_col].apply(lambda x: len(sentence_split(str(x))))
    
    if method == "iqr":
        Q1 = np.percentile(sentence_counts, 25)
        Q3 = np.percentile(sentence_counts, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outlier_mask = (sentence_counts < lower_bound) | (sentence_counts > upper_bound)
    
    elif method == "zscore":
        from scipy import stats
        z_scores = np.abs(stats.zscore(sentence_counts))
        outlier_mask = z_scores > threshold
    
    else:
        return {"error": f"Unknown method: {method}"}
    
    outlier_indices = np.where(outlier_mask)[0]
    
    return {
        "method": method,
        "total_texts": len(df),
        "num_outliers": len(outlier_indices),
        "outlier_indices": outlier_indices.tolist(),
        "sentence_counts": sentence_counts[outlier_mask].to_dict() if len(outlier_indices) > 0 else {},
        "stats": {
            "mean": sentence_counts.mean(),
            "median": sentence_counts.median(),
            "min": sentence_counts.min(),
            "max": sentence_counts.max(),
            "Q1": float(Q1) if method == "iqr" else None,
            "Q3": float(Q3) if method == "iqr" else None,
        }
    }


def check_data_quality(df: pd.DataFrame,
                      source_col: str = "reflection_answer",
                      target_col: Optional[str] = None) -> Dict:
    """
    Check overall data quality.
    
    Args:
        df: DataFrame to check
        source_col: Source text column
        target_col: Optional target text column (for translations)
    
    Returns:
        Quality report dict
    """
    report = {
        "total_rows": len(df),
        "source_col": source_col,
        "target_col": target_col,
    }
    
    # Source text stats
    if source_col in df.columns:
        src_lengths = df[source_col].apply(word_count)
        src_has_pii = df[source_col].apply(lambda x: contains_sensitive_info(str(x))[0])
        
        report["source"] = {
            "total": len(df),
            "empty": (src_lengths == 0).sum(),
            "avg_length": src_lengths.mean(),
            "min_length": src_lengths.min(),
            "max_length": src_lengths.max(),
            "with_pii": src_has_pii.sum(),
        }
    
    # Target text stats (if provided)
    if target_col and target_col in df.columns:
        tgt_lengths = df[target_col].apply(word_count)
        has_errors = df[target_col].apply(lambda x: ErrorDetector.check_for_errors(str(x))[0])
        
        report["target"] = {
            "total": len(df),
            "empty": (tgt_lengths == 0).sum(),
            "with_errors": has_errors.sum(),
            "avg_length": tgt_lengths.mean(),
            "min_length": tgt_lengths.min(),
            "max_length": tgt_lengths.max(),
        }
        
        # Length ratios
        valid_pairs = (src_lengths > 0) & (tgt_lengths > 0)
        if valid_pairs.sum() > 0:
            ratios = tgt_lengths[valid_pairs] / src_lengths[valid_pairs]
            report["translation"] = {
                "avg_ratio": ratios.mean(),
                "median_ratio": ratios.median(),
                "min_ratio": ratios.min(),
                "max_ratio": ratios.max(),
            }
    
    return report


def compare_translations(df: pd.DataFrame, 
                        source_col: str = "reflection_answer",
                        target_col: str = "phen_report_english",
                        num_samples: int = 5,
                        anonymise: bool = True):
    """
    Display side-by-side comparison of source and target translations.
    
    Args:
        df: DataFrame with translations
        source_col: Name of source text column
        target_col: Name of translated text column
        num_samples: Number of samples to display
        anonymise: Whether to anonymise text in output (default: True)
    """
    if source_col not in df.columns or target_col not in df.columns:
        print(f"[ERROR] Columns '{source_col}' or '{target_col}' not found.")
        return
    
    n_samples = min(num_samples, len(df))
    sample = df[[source_col, target_col]].sample(n_samples, random_state=0)

    print(f"\n{'='*80}")
    print(f"TRANSLATION SAMPLES ({n_samples} of {len(df)})")
    if anonymise:
        print("(Anonymised)")
    print(f"{'='*80}")
    
    for i, (row_idx, row) in enumerate(sample.iterrows(), start=1):
        src = row[source_col]
        tgt = row[target_col]
        
        if anonymise:
            src = anonymise_text(src)
            tgt = anonymise_text(tgt)
        
        src_wc = word_count(src)
        tgt_wc = word_count(tgt)
        
        # Check for errors
        has_error, error_type, error_desc = ErrorDetector.check_for_errors(tgt)
        error_msg = f" [ERROR: {error_desc}]" if has_error else ""

        print(f"\n----- Sample {i} (row {row_idx}) -----")
        print(f"[LENGTH] Source: {src_wc} words | Target: {tgt_wc} words", end="")
        if src_wc > 0:
            print(f" | Ratio: {tgt_wc/src_wc:.2f}")
        else:
            print()
        print(f"{error_msg}")
        print(f"\n[SOURCE]\n{src}")
        print(f"\n[TARGET]\n{tgt}")
        print("-" * 80)


def show_translation_stats(df: pd.DataFrame,
                          source_col: str = "reflection_answer",
                          target_col: str = "phen_report_english"):
    """
    Display overall translation statistics with error detection.
    
    Args:
        df: DataFrame with translations
        source_col: Name of source text column
        target_col: Name of translated text column
    """
    if source_col not in df.columns or target_col not in df.columns:
        print(f"[ERROR] Columns '{source_col}' or '{target_col}' not found.")
        return
    
    src_wc_all = df[source_col].apply(word_count)
    tgt_wc_all = df[target_col].apply(word_count)
    ratio_all = tgt_wc_all.divide(src_wc_all.replace({0: pd.NA}))

    print(f"\n{'='*80}")
    print("TRANSLATION STATISTICS")
    print(f"{'='*80}")
    print(f"Total documents: {len(df)}")
    print(f"Avg source length (words):  {src_wc_all.mean():.1f}")
    print(f"Avg target length (words):  {tgt_wc_all.mean():.1f}")
    
    valid_ratio = ratio_all.dropna()
    if len(valid_ratio) > 0:
        print(f"Avg TGT/SRC ratio:          {valid_ratio.mean():.2f}")
        print(f"Median TGT/SRC ratio:       {valid_ratio.median():.2f}")
    else:
        print("No valid TGT/SRC ratio (all source lengths were zero).")

    # Error detection (ENHANCED in v2.0)
    print(f"\n{'='*80}")
    print("ERROR DETECTION")
    print(f"{'='*80}")
    
    error_mask = df[target_col].apply(lambda x: ErrorDetector.check_for_errors(str(x))[0])
    num_errors = error_mask.sum()
    
    if num_errors > 0:
        print(f"Rows with errors: {num_errors} ({100*num_errors/len(df):.1f}%)")
        
        # Show error types
        error_types = {}
        for idx, row in df[error_mask].iterrows():
            has_error, error_type, _ = ErrorDetector.check_for_errors(str(row[target_col]))
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        print("\nError breakdown:")
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {error_type}: {count}")
    else:
        print("No errors detected ✓")
    
    # Quality check (ENHANCED in v2.0)
    print(f"\n{'='*80}")
    print("QUALITY CHECK")
    print(f"{'='*80}")
    
    empty_tgt = (tgt_wc_all == 0).sum()
    if empty_tgt > 0:
        print(f"Empty translations: {empty_tgt}")
    
    # Length ratio issues
    quality_issues = 0
    for idx, (src_len, tgt_len) in enumerate(zip(src_wc_all, tgt_wc_all)):
        is_valid, msg = ErrorDetector.check_translation_quality(src_len, tgt_len)
        if not is_valid:
            quality_issues += 1
    
    if quality_issues > 0:
        print(f"Suspicious length ratios: {quality_issues}")
    else:
        print("Length ratios look good ✓")


# =============================================================================
# SECTION 9: LOAD PREPROCESSED DATA (NEW IN v2.0)
# =============================================================================

def load_preprocessed_data(data_path: Optional[str] = None,
                          dataset: Optional[str] = None,
                          relative_path: Optional[str] = None,
                          text_column: str = "reflection_answer") -> pd.DataFrame:
    """
    Load preprocessed data from file or MOSAIC path.
    
    Three ways to specify the path:
    1. Full path: data_path="/Users/rb666/Projects/MOSAIC/DATA/preprocessed/file.csv"
    2. MOSAIC path: dataset="MPE", relative_path="preprocessed/file.csv"
    3. Interactive: Specify both dataset and filename
    
    Args:
        data_path: Full file path (preferred method)
        dataset: MOSAIC dataset name (used with relative_path)
        relative_path: Relative path within dataset (used with dataset)
        text_column: Text column name to verify
    
    Returns:
        Loaded DataFrame
    
    Raises:
        FileNotFoundError: If file not found
        ImportError: If MOSAIC not available and needed
    """
    # Determine the actual path
    if data_path:
        actual_path = Path(data_path)
    elif dataset and relative_path:
        if not HAS_MOSAIC:
            raise ImportError("MOSAIC is required when using dataset parameter")
        actual_path = proc_path(dataset, relative_path)
    else:
        raise ValueError(
            "Must provide either 'data_path' OR both 'dataset' and 'relative_path'"
        )
    
    actual_path = Path(actual_path)
    
    if not actual_path.exists():
        raise FileNotFoundError(f"Data file not found: {actual_path}")
    
    print(f"Loading preprocessed data from: {actual_path}")
    df = pd.read_csv(actual_path)
    print(f"Loaded {len(df)} rows")
    
    if text_column not in df.columns:
        print(f"[WARN] Column '{text_column}' not found. Available columns: {list(df.columns)}")
    
    return df


# =============================================================================
# SECTION 10: CLI FOR STANDALONE USAGE
# =============================================================================

def main():
    """Standalone CLI for translation utilities."""
    parser = argparse.ArgumentParser(
        description="Advanced translation utilities for preprocessed data"
    )
    
    parser.add_argument(
        "command",
        choices=["validate", "stats", "detect_outliers", "check_quality", "anonymise_preview"],
        help="Command to run"
    )
    
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to preprocessed CSV file"
    )
    
    parser.add_argument(
        "--source-col",
        default="reflection_answer",
        help="Source text column name (default: reflection_answer)"
    )
    
    parser.add_argument(
        "--target-col",
        default="phen_report_english",
        help="Target text column name (default: phen_report_english)"
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to display (for 'validate' command)"
    )
    
    parser.add_argument(
        "--anonymise",
        action="store_true",
        default=True,
        help="Anonymise text in output (default: True)"
    )
    
    parser.add_argument(
        "--no-anonymise",
        action="store_true",
        help="Don't anonymise (for testing only)"
    )
    
    args = parser.parse_args()
    
    anonymise = not args.no_anonymise

    # Load CSV
    try:
        df = pd.read_csv(args.input_csv)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {args.input_csv}")
        return
    except Exception as e:
        print(f"[ERROR] Could not load CSV: {e}")
        return

    # Run command
    if args.command == "validate":
        compare_translations(df, args.source_col, args.target_col, args.num_samples, anonymise)
    
    elif args.command == "stats":
        show_translation_stats(df, args.source_col, args.target_col)
    
    elif args.command == "detect_outliers":
        outliers = detect_outliers_by_length(df, args.source_col)
        print(f"\nOutlier Detection Results:")
        for key, value in outliers.items():
            print(f"  {key}: {value}")
    
    elif args.command == "check_quality":
        quality = check_data_quality(df, args.source_col, args.target_col if args.target_col in df.columns else None)
        print(f"\nData Quality Report:")
        import json
        print(json.dumps(quality, indent=2))
    
    elif args.command == "anonymise_preview":
        sample = df[[args.source_col]].head(3)
        print(f"\nAnonymisation Examples (first 3 rows):")
        for idx, row in sample.iterrows():
            original = str(row[args.source_col])[:100]
            anonymised = anonymise_text(original)
            print(f"\nRow {idx}:")
            print(f"  ORIGINAL:   {original}")
            print(f"  ANONYMISED: {anonymised}")


if __name__ == "__main__":
    main()

# =============================================================================
# EXAMPLE USAGE (in Python)
# =============================================================================
#
# from translation_utils import (
#     load_preprocessed_data,
#     translate_text_sentence_aligned,
#     anonymise_text,
#     show_translation_stats,
#     compare_translations,
#     check_data_quality,
#     detect_outliers_by_length
# )
# from llama_cpp import Llama
# from huggingface_hub import hf_hub_download
# from tqdm import tqdm
# 
# # 1. Load preprocessed data
# df = load_preprocessed_data(
#     data_path="/Users/rb666/Projects/MOSAIC/DATA/preprocessed/MPE_cleaned_llama_5_test.csv"
# )
# 
# # 2. Check data quality before translation
# quality = check_data_quality(df)
# print(quality)
# 
# # 3. Detect outliers (like prepare_data.ipynb)
# outliers = detect_outliers_by_length(df, "reflection_answer")
# print(f"Found {outliers['num_outliers']} outliers")
# 
# # 4. Load model
# model_path = hf_hub_download(
#     repo_id='NousResearch/Meta-Llama-3-8B-Instruct-GGUF',
#     filename='Meta-Llama-3-8B-Instruct-Q4_K_M.gguf'
# )
# llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=8192)
# max_ctx = llm.n_ctx()
# 
# # 5. Translate with anonymised error logs
# translations = []
# for text in tqdm(df["reflection_answer"]):
#     trans = translate_text_sentence_aligned(text, llm, max_ctx)
#     translations.append(trans)
# 
# df["phen_report_english"] = translations
# df.to_csv("translated.csv", index=False)
# 
# # 6. Validate with anonymisation
# compare_translations(df, anonymise=True)
# show_translation_stats(df)
#
# =============================================================================
# EXAMPLE USAGE (CLI)
# =============================================================================
#
# # Check data quality
# python translation_utils.py check_quality --input-csv data.csv
#
# # Detect outliers (replaces prepare_data.ipynb)
# python translation_utils.py detect_outliers --input-csv data.csv
#
# # Validate translations (with anonymisation)
# python translation_utils.py validate --input-csv data_translated.csv --num-samples 5
#
# # Show statistics
# python translation_utils.py stats --input-csv data_translated.csv
#
# # Preview anonymisation
# python translation_utils.py anonymise_preview --input-csv data.csv
#
