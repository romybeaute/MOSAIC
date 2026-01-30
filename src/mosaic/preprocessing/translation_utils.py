#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : translation_utils.py [UPDATED v3.1 - PATCHED]
# description     : Post-preprocessing utilities (validation, anonymisation, error detection)
#                   - Works WITH preprocessed data from preprocessing.py
#                   - Does NOT duplicate sentence splitting or text cleaning
#                   - Provides: validation, anonymisation, error detection, PII masking
#                   - INCLUDES: Word count comparison, Truncation diagnostics
#                   - Reads preprocessing error logs and outputs stats
#                   - Lists preprocessed datasets and files
#                   - PATCHED: Auto-detects project root (works from any directory)
# author          : Romy, Beauté (r.beaut@sussex.ac.uk)
# date            : 2026-01-29
# version         : 3.1 (patched with auto-path detection)
# ==============================================================================

import os
import json
import argparse
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import glob

import pandas as pd
import numpy as np
from tqdm import tqdm


# =============================================================================
# PATCHED: AUTO-DETECT PROJECT ROOT
# =============================================================================

def get_project_root() -> Path:
    """
    Find the project root by looking for the DATA directory.
    Works whether you run from MOSAIC/, src/, or src/mosaic/preprocessing/
    """
    # First try: relative to this script file
    current = Path(__file__).resolve().parent
    
    for _ in range(10):  # Max 10 levels up
        if (current / "DATA").exists():
            return current
        current = current.parent
    
    # Second try: relative to current working directory
    cwd = Path.cwd()
    for _ in range(10):
        if (cwd / "DATA").exists():
            return cwd
        cwd = cwd.parent
    
    # Fallback: assume current directory
    return Path.cwd()


# Set project root globally
PROJECT_ROOT = get_project_root()
DEFAULT_DATA_DIR = PROJECT_ROOT / "DATA"


def resolve_data_path(path_str: str) -> Path:
    """
    Resolve a data path, handling both absolute and relative paths.
    Relative paths are resolved from PROJECT_ROOT.
    """
    path = Path(path_str)
    
    # If absolute path, use as-is
    if path.is_absolute():
        return path
    
    # If relative path, try from project root first
    project_path = PROJECT_ROOT / path
    if project_path.exists():
        return project_path
    
    # Try from current directory
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    
    # Return project root version (will fail with clear error)
    return project_path


# =============================================================================
# SECTION 0: DATASET INVENTORY
# =============================================================================

def list_preprocessed_datasets(data_dir: str = None) -> Dict[str, Dict]:
    """
    List all preprocessed datasets and their files.
    
    Args:
        data_dir: Path to DATA directory (auto-detected if None)
    
    Returns:
        Dictionary mapping dataset_name -> {files, methods, stats}
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    
    preprocessed_dir = Path(data_dir) / "preprocessed"
    
    if not preprocessed_dir.exists():
        # Try resolving from project root
        preprocessed_dir = PROJECT_ROOT / "DATA" / "preprocessed"
    
    if not preprocessed_dir.exists():
        return {"error": f"Directory not found: {preprocessed_dir}"}
    
    datasets = {}
    
    # Find all CSV files
    csv_files = sorted(preprocessed_dir.glob("*.csv"))
    
    for csv_file in csv_files:
        filename = csv_file.name
        
        # Parse filename: {dataset_name}_{method}_{sample}.csv
        # Examples:
        # - dreamachine_DL_preprocessed.csv
        # - dreamachine_DL_cleaned_llama_sample5.csv
        # - innerspeech_cleaned_API.csv
        
        dataset_name = None
        method = None
        sample_size = None
        
        # Try to parse the filename
        if "_preprocessed" in filename:
            dataset_name = filename.replace("_preprocessed.csv", "")
            method = "basic"
        elif "_cleaned_llama" in filename:
            base = filename.replace("_cleaned_llama.csv", "")
            if "_sample" in base:
                parts = base.split("_sample")
                dataset_name = parts[0]
                try:
                    sample_size = int(parts[1])
                except (ValueError, IndexError):
                    sample_size = None
            else:
                dataset_name = base
            method = "llama"
        elif "_cleaned_API" in filename:
            base = filename.replace("_cleaned_API.csv", "")
            if "_sample" in base:
                parts = base.split("_sample")
                dataset_name = parts[0]
                try:
                    sample_size = int(parts[1])
                except (ValueError, IndexError):
                    sample_size = None
            else:
                dataset_name = base
            method = "gemini"
        
        if not dataset_name:
            continue
        
        # Initialize dataset entry if not exists
        if dataset_name not in datasets:
            datasets[dataset_name] = {
                "methods": {},
                "files": [],
                "total_reports": 0,
                "total_size_mb": 0
            }
        
        # Add file info
        file_size_mb = csv_file.stat().st_size / (1024 * 1024)
        
        file_info = {
            "filename": filename,
            "method": method,
            "sample_size": sample_size,
            "size_mb": round(file_size_mb, 2),
            "path": str(csv_file)
        }
        
        datasets[dataset_name]["files"].append(file_info)
        datasets[dataset_name]["total_size_mb"] += file_size_mb
        
        # Track methods
        if method not in datasets[dataset_name]["methods"]:
            datasets[dataset_name]["methods"][method] = []
        datasets[dataset_name]["methods"][method].append(file_info)
        
        # Try to get row count
        try:
            df = pd.read_csv(csv_file)
            num_rows = len(df)
            if sample_size is None:
                datasets[dataset_name]["total_reports"] = num_rows
        except:
            pass
    
    return datasets


def print_dataset_inventory(data_dir: str = "DATA"):
    """
    Print a formatted inventory of preprocessed datasets.
    
    Args:
        data_dir: Path to DATA directory
    """
    datasets = list_preprocessed_datasets(data_dir)
    
    if "error" in datasets:
        print(f"[ERROR] {datasets['error']}")
        return
    
    if not datasets:
        print("No preprocessed datasets found")
        return
    
    print(f"\n{'='*100}")
    print("PREPROCESSED DATASETS INVENTORY")
    print(f"{'='*100}\n")
    
    for dataset_name in sorted(datasets.keys()):
        info = datasets[dataset_name]
        
        print(f"📊 {dataset_name}")
        print(f"   Methods: {', '.join(info['methods'].keys())}")
        print(f"   Reports: {info['total_reports']}")
        print(f"   Total size: {round(info['total_size_mb'], 2)} MB")
        print(f"   Files:")
        
        for file_info in sorted(info["files"], key=lambda x: x["filename"]):
            label = "sample" if file_info["sample_size"] else "full"
            print(f"      • {file_info['filename']:<50} ({file_info['size_mb']:.2f} MB) [{label}]")
        
        print()
    
    print(f"{'='*100}\n")
    print(f"Total datasets: {len(datasets)}")
    print(f"Location: {Path(data_dir) / 'preprocessed'}\n")


def find_preprocessed_file(filename_or_dataset: str, data_dir: str = "DATA") -> Optional[Path]:
    """
    Find a preprocessed file by partial name or dataset name.
    
    Args:
        filename_or_dataset: Full filename, partial name, or dataset name
        data_dir: Path to DATA directory
    
    Returns:
        Full path to file if found, None otherwise
    """
    preprocessed_dir = Path(data_dir) / "preprocessed"
    
    if not preprocessed_dir.exists():
        return None
    
    # Try exact match first
    candidate = preprocessed_dir / filename_or_dataset
    if candidate.exists():
        return candidate
    
    # Try partial match
    matches = list(preprocessed_dir.glob(f"*{filename_or_dataset}*.csv"))
    
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        print(f"[WARNING] Multiple matches found:")
        for m in matches:
            print(f"  - {m.name}")
        return matches[0]  # Return first match
    
    return None


# =============================================================================
# SECTION 1: ANONYMISATION (PII MASKING)
# =============================================================================

class PII_PATTERNS:
    """Common PII patterns for anonymisation."""
    NAMES = r'\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
    EMAILS = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,}'
    PHONES = r'(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\+\d{10,}'
    DATES = r'\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}\.\d{1,2}\.\d{2,4})\b'
    IDS = r'\b\d{8,}\b'
    URLS = r'https?://[^\s]+'
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
# SECTION 2: ERROR DETECTION (NO REDUNDANCY WITH preprocessing.py)
# =============================================================================

class ErrorDetector:
    """Detect translation errors in LLM output."""
    
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
        
        for error_key in ['timeout', 'quota', 'json_error', 'empty', 'failed', 
                         'exception', 'no_output', 'truncated', 'api_error']:
            pattern, description = ErrorDetector.ERROR_PATTERNS[error_key]
            if re.search(pattern, text):
                return True, error_key, description
        
        return False, '', 'No errors detected'


# =============================================================================
# SECTION 3: WORD COUNT & CHARACTER COMPARISON (NEW)
# =============================================================================

def compare_word_counts(csv_path: str, 
                        source_col: str = "reflection_answer",
                        target_col: str = "cleaned_reflection",
                        show_samples: int = 5) -> pd.DataFrame:
    """
    Compare word and character counts between original and cleaned texts.
    
    Args:
        csv_path: Path to preprocessed CSV
        source_col: Name of source text column
        target_col: Name of cleaned text column
        show_samples: Number of problematic samples to display
    
    Returns:
        DataFrame with comparison statistics
    """
    print(f"\n{'='*100}")
    print("WORD COUNT & CHARACTER COMPARISON")
    print(f"{'='*100}\n")
    
    # PATCHED: Use path resolver
    resolved_path = resolve_data_path(csv_path)
    if not resolved_path.exists():
        resolved_path = PROJECT_ROOT / "DATA" / "preprocessed" / Path(csv_path).name
    
    # Load data
    try:
        df = pd.read_csv(resolved_path)
        print(f"Loaded: {resolved_path}")
    except Exception as e:
        print(f"[ERROR] Could not load CSV: {e}")
        print(f"Tried: {resolved_path}")
        return None
    
    if source_col not in df.columns or target_col not in df.columns:
        print(f"[ERROR] Columns not found. Available: {list(df.columns)}")
        return None
    
    # Count function
    def word_count(text):
        if pd.isna(text) or not isinstance(text, str):
            return 0
        return len(text.split())
    
    def char_count(text):
        if pd.isna(text) or not isinstance(text, str):
            return 0
        return len(text)
    
    # Create comparison dataframe
    comparison = pd.DataFrame({
        'source_words': df[source_col].apply(word_count),
        'source_chars': df[source_col].apply(char_count),
        'target_words': df[target_col].apply(word_count),
        'target_chars': df[target_col].apply(char_count),
    })
    
    # Calculate ratios
    comparison['word_ratio'] = comparison['target_words'] / (comparison['source_words'] + 1)
    comparison['char_ratio'] = comparison['target_chars'] / (comparison['source_chars'] + 1)
    comparison['word_loss'] = comparison['source_words'] - comparison['target_words']
    comparison['char_loss'] = comparison['source_chars'] - comparison['target_chars']
    
    # ========================================================================
    # OVERALL STATISTICS
    # ========================================================================
    
    total_source_words = comparison['source_words'].sum()
    total_target_words = comparison['target_words'].sum()
    total_source_chars = comparison['source_chars'].sum()
    total_target_chars = comparison['target_chars'].sum()
    
    word_retention = (total_target_words / total_source_words * 100) if total_source_words > 0 else 0
    char_retention = (total_target_chars / total_source_chars * 100) if total_source_chars > 0 else 0
    
    print("OVERALL STATISTICS")
    print("-" * 100)
    print(f"Total source words:          {total_source_words:,}")
    print(f"Total target words:          {total_target_words:,}")
    print(f"Word retention rate:         {word_retention:.1f}%")
    print(f"Total words lost:            {comparison['word_loss'].sum():,}")
    print()
    print(f"Total source characters:     {total_source_chars:,}")
    print(f"Total target characters:     {total_target_chars:,}")
    print(f"Character retention rate:    {char_retention:.1f}%")
    print(f"Total characters lost:       {comparison['char_loss'].sum():,}")
    
    # ========================================================================
    # STATISTICS PER REPORT
    # ========================================================================
    
    print(f"\n{'='*100}")
    print("PER-REPORT STATISTICS")
    print("-" * 100)
    print(f"Average source words/report: {comparison['source_words'].mean():.1f}")
    print(f"Average target words/report: {comparison['target_words'].mean():.1f}")
    print(f"Average word retention:      {comparison['word_ratio'].mean():.1%}")
    print()
    print(f"Median source words/report:  {comparison['source_words'].median():.0f}")
    print(f"Median target words/report:  {comparison['target_words'].median():.0f}")
    print(f"Median word retention:       {comparison['word_ratio'].median():.1%}")
    
    # ========================================================================
    # TRUNCATION DETECTION
    # ========================================================================
    
    print(f"\n{'='*100}")
    print("TRUNCATION ANALYSIS")
    print("-" * 100)
    
    # Categorize by retention rate
    retained_95 = (comparison['word_ratio'] >= 0.95).sum()
    retained_80_95 = ((comparison['word_ratio'] >= 0.80) & (comparison['word_ratio'] < 0.95)).sum()
    retained_50_80 = ((comparison['word_ratio'] >= 0.50) & (comparison['word_ratio'] < 0.80)).sum()
    retained_below_50 = (comparison['word_ratio'] < 0.50).sum()
    
    print(f"Excellent (95-100% retained):    {retained_95:>5} rows ({retained_95/len(comparison)*100:>5.1f}%)")
    print(f"Good (80-95% retained):          {retained_80_95:>5} rows ({retained_80_95/len(comparison)*100:>5.1f}%)")
    print(f"Poor (50-80% retained):          {retained_50_80:>5} rows ({retained_50_80/len(comparison)*100:>5.1f}%)")
    print(f"Critical (<50% retained):        {retained_below_50:>5} rows ({retained_below_50/len(comparison)*100:>5.1f}%)")
    
    # Calculate impact
    truncation_rate = (len(comparison) - retained_95) / len(comparison) * 100
    
    if truncation_rate > 50:
        print(f"\n🔴 WARNING: {truncation_rate:.1f}% of reports show some truncation!")
    elif truncation_rate > 10:
        print(f"\n🟡 CAUTION: {truncation_rate:.1f}% of reports show truncation.")
    else:
        print(f"\n✅ Good news: Only {truncation_rate:.1f}% of reports show truncation.")
    
    # ========================================================================
    # PROBLEMATIC CASES
    # ========================================================================
    
    if retained_below_50 > 0 or retained_50_80 > 0:
        print(f"\n{'='*100}")
        print(f"CRITICAL CASES: {show_samples} most problematic examples")
        print("-" * 100)
        
        problematic = comparison[comparison['word_ratio'] < 0.80].sort_values('word_ratio').head(show_samples)
        
        for idx, (row_idx, row) in enumerate(problematic.iterrows(), 1):
            print(f"\n[{idx}] Row {row_idx}: {row['word_ratio']:.1%} retention")
            print(f"    Source: {row['source_words']:.0f} words, {row['source_chars']:.0f} chars")
            print(f"    Target: {row['target_words']:.0f} words, {row['target_chars']:.0f} chars")
            print(f"    Loss:   {row['word_loss']:.0f} words, {row['char_loss']:.0f} chars")
    
    print(f"\n{'='*100}\n")
    
    return comparison


def create_word_count_summary_table(csv_path: str,
                                    source_col: str = "reflection_answer",
                                    target_col: str = "cleaned_reflection") -> pd.DataFrame:
    """
    Create a summary table for word counts.
    
    Returns:
        Summary DataFrame suitable for export
    """
    df = pd.read_csv(csv_path)
    
    def word_count(text):
        if pd.isna(text) or not isinstance(text, str):
            return 0
        return len(text.split())
    
    summary = pd.DataFrame({
        'Row_Index': range(len(df)),
        'Source_Words': df[source_col].apply(word_count),
        'Cleaned_Words': df[target_col].apply(word_count),
    })
    
    summary['Word_Ratio'] = summary['Cleaned_Words'] / (summary['Source_Words'] + 1)
    summary['Word_Loss'] = summary['Source_Words'] - summary['Cleaned_Words']
    summary['Status'] = summary['Word_Ratio'].apply(lambda x: 
        'CRITICAL' if x < 0.5 else 'POOR' if x < 0.8 else 'GOOD' if x >= 0.95 else 'OK'
    )
    
    return summary


def export_word_count_report(csv_path: str, output_path: str = None,
                             source_col: str = "reflection_answer",
                             target_col: str = "cleaned_reflection") -> str:
    """
    Export word count comparison to CSV for further analysis.
    
    Returns:
        Path to exported file
    """
    if output_path is None:
        base = Path(csv_path).stem
        output_path = f"{base}_word_count_analysis.csv"
    
    summary = create_word_count_summary_table(csv_path, source_col, target_col)
    summary.to_csv(output_path, index=False)
    
    print(f"\n✓ Word count report saved to: {output_path}")
    
    return output_path


# =============================================================================
# SECTION 4: TRUNCATION DIAGNOSTICS (NEW)
# =============================================================================

class TruncationDiagnostic:
    """Comprehensive diagnostic for data truncation issues."""
    
    def __init__(self, csv_path: str, source_col: str = "reflection_answer",
                 target_col: str = "cleaned_reflection"):
        self.csv_path = csv_path
        self.source_col = source_col
        self.target_col = target_col
        
        # PATCHED: Use path resolver
        resolved_path = resolve_data_path(csv_path)
        if not resolved_path.exists():
            # Try in DATA/preprocessed/
            resolved_path = PROJECT_ROOT / "DATA" / "preprocessed" / Path(csv_path).name
        
        try:
            self.df = pd.read_csv(resolved_path)
            self.csv_path = str(resolved_path)  # Store resolved path
            self.valid = True
        except Exception as e:
            print(f"[ERROR] Could not load CSV: {e}")
            print(f"Tried: {resolved_path}")
            print(f"Project root: {PROJECT_ROOT}")
            self.valid = False
    
    def word_count(self, text):
        if pd.isna(text) or not isinstance(text, str):
            return 0
        return len(text.split())
    
    def char_count(self, text):
        if pd.isna(text) or not isinstance(text, str):
            return 0
        return len(text)
    
    # ========================================================================
    # DETECTION 1: Error Markers
    # ========================================================================
    
    def check_error_markers(self) -> Dict:
        """Detect error patterns that indicate failed processing."""
        if not self.valid:
            return {}
        
        error_patterns = {
            '[ERROR': 'LLM processing error',
            '[SKIPPED': 'Text was skipped (too long)',
            'Error: Mismatch': 'Gemini API returned mismatched count',
            'Quota Exceeded': 'API quota exceeded',
            'Connection error': 'Network error',
            'Timeout': 'Timeout error',
        }
        
        results = {}
        
        for pattern, description in error_patterns.items():
            count = self.df[self.target_col].astype(str).str.contains(
                re.escape(pattern), case=False, na=False
            ).sum()
            
            if count > 0:
                results[pattern] = {
                    'count': count,
                    'percentage': count / len(self.df) * 100,
                    'description': description
                }
        
        return results
    
    # ========================================================================
    # DETECTION 2: Token Count Analysis
    # ========================================================================
    
    def check_token_limits(self) -> Dict:
        """Check if texts are hitting LLM token limits."""
        if not self.valid:
            return {}
        
        # Rough estimate: ~4 characters = 1 token (conservative)
        TOKEN_RATIO = 4
        MAX_TOKENS_LLAMA = 8192  # From preprocessing.py line 225
        MAX_CHARS_LLAMA = MAX_TOKENS_LLAMA * TOKEN_RATIO
        
        self.df['source_chars'] = self.df[self.source_col].apply(self.char_count)
        self.df['source_tokens_est'] = self.df['source_chars'] / TOKEN_RATIO
        
        at_risk = self.df[self.df['source_tokens_est'] > MAX_TOKENS_LLAMA * 0.9]
        
        return {
            'max_tokens_llama': MAX_TOKENS_LLAMA,
            'max_chars_llama': MAX_CHARS_LLAMA,
            'texts_at_risk': len(at_risk),
            'at_risk_percentage': len(at_risk) / len(self.df) * 100,
            'max_source_chars': self.df['source_chars'].max(),
            'max_source_tokens_est': self.df['source_tokens_est'].max(),
            'texts_exceeding_limit': len(self.df[self.df['source_tokens_est'] > MAX_TOKENS_LLAMA])
        }
    
    # ========================================================================
    # DETECTION 3: Retention Rate Analysis
    # ========================================================================
    
    def check_retention_rates(self) -> Dict:
        """Analyze how much data is retained after processing."""
        if not self.valid:
            return {}
        
        self.df['source_words'] = self.df[self.source_col].apply(self.word_count)
        self.df['target_words'] = self.df[self.target_col].apply(self.word_count)
        self.df['word_ratio'] = self.df['target_words'] / (self.df['source_words'] + 1)
        
        total_source_words = self.df['source_words'].sum()
        total_target_words = self.df['target_words'].sum()
        overall_retention = (total_target_words / total_source_words * 100) if total_source_words > 0 else 0
        
        # Categorize
        excellent = (self.df['word_ratio'] >= 0.95).sum()
        good = ((self.df['word_ratio'] >= 0.80) & (self.df['word_ratio'] < 0.95)).sum()
        poor = ((self.df['word_ratio'] >= 0.50) & (self.df['word_ratio'] < 0.80)).sum()
        critical = (self.df['word_ratio'] < 0.50).sum()
        
        return {
            'overall_retention_pct': overall_retention,
            'total_words_lost': total_source_words - total_target_words,
            'excellent_95_100': excellent,
            'good_80_95': good,
            'poor_50_80': poor,
            'critical_below_50': critical,
            'any_data_loss': total_target_words < total_source_words
        }
    
    # ========================================================================
    # DETECTION 4: Pattern Analysis
    # ========================================================================
    
    def check_patterns(self) -> Dict:
        """Detect patterns that indicate specific types of truncation."""
        if not self.valid:
            return {}
        
        patterns = {
            'ends_with_ellipsis': self.df[self.target_col].astype(str).str.endswith('...').sum(),
            'ends_with_dots': self.df[self.target_col].astype(str).str.endswith('..').sum(),
            'ends_with_incomplete_word': sum(
                self.df[self.target_col].astype(str).apply(
                    lambda x: x.rstrip()[-1] == '-' if len(x) > 0 else False
                )
            ),
            'contains_truncation_marker': self.df[self.target_col].astype(str).str.contains(
                'truncat|cut off|incomplete', case=False, na=False
            ).sum(),
        }
        
        return patterns
    
    # ========================================================================
    # DETECTION 5: Batch Processing Issues (Gemini)
    # ========================================================================
    
    def check_batch_issues(self) -> Dict:
        """Detect issues specific to batch processing (Gemini)."""
        if not self.valid:
            return {}
        
        # Check for rows with error markers from batch processing
        mismatch_errors = self.df[self.target_col].astype(str).str.contains(
            'Error: Mismatch', na=False
        ).sum()
        
        quota_errors = self.df[self.target_col].astype(str).str.contains(
            'Quota|quota', case=False, na=False
        ).sum()
        
        return {
            'mismatch_errors': mismatch_errors,
            'quota_errors': quota_errors,
            'has_batch_issues': mismatch_errors > 0 or quota_errors > 0
        }
    
    # ========================================================================
    # MAIN DIAGNOSTIC REPORT
    # ========================================================================
    
    def run_full_diagnostic(self):
        """Run all diagnostics and print comprehensive report."""
        if not self.valid:
            print("[ERROR] Could not load data. Aborting diagnostic.")
            return
        
        print(f"\n{'='*100}")
        print("DATA TRUNCATION DIAGNOSTIC REPORT")
        print(f"{'='*100}\n")
        
        print(f"File: {self.csv_path}")
        print(f"Rows: {len(self.df)}")
        print(f"Columns: {list(self.df.columns)}\n")
        
        # ====================================================================
        # 1. ERROR MARKERS
        # ====================================================================
        
        print(f"{'='*100}")
        print("1. ERROR MARKERS (Processed with Errors)")
        print("-" * 100)
        
        error_markers = self.check_error_markers()
        if error_markers:
            print("⚠️  ERRORS FOUND:\n")
            for marker, info in error_markers.items():
                print(f"  • {marker}")
                print(f"    {info['count']} rows ({info['percentage']:.1f}%)")
                print(f"    Meaning: {info['description']}\n")
            
            print("IMPACT: These rows likely contain corrupted or missing data.")
        else:
            print("✅ No error markers detected.\n")
        
        # ====================================================================
        # 2. TOKEN LIMITS
        # ====================================================================
        
        print(f"{'='*100}")
        print("2. TOKEN LIMIT ANALYSIS (LLM max_tokens setting)")
        print("-" * 100)
        
        token_analysis = self.check_token_limits()
        print(f"LLM max_tokens setting: {token_analysis['max_tokens_llama']}")
        print(f"Estimated max chars:    {token_analysis['max_chars_llama']:.0f}\n")
        print(f"Texts at risk (>90% of limit):  {token_analysis['texts_at_risk']} ({token_analysis['at_risk_percentage']:.1f}%)")
        print(f"Texts exceeding limit:          {token_analysis['texts_exceeding_limit']}")
        print(f"Max source text size:           {token_analysis['max_source_chars']:.0f} chars")
        print(f"Estimated max source tokens:    {token_analysis['max_source_tokens_est']:.0f} tokens\n")
        
        if token_analysis['texts_exceeding_limit'] > len(self.df) * 0.1:
            print("🔴 CRITICAL: >10% of texts exceed token limit!")
            print("   Fix: Increase max_tokens in preprocessing.py line 225")
        elif token_analysis['texts_at_risk'] > len(self.df) * 0.2:
            print("🟡 CAUTION: >20% of texts are at risk of truncation")
            print("   Consider increasing max_tokens")
        else:
            print("✅ Token limits seem adequate")
        
        print()
        
        # ====================================================================
        # 3. RETENTION RATES
        # ====================================================================
        
        print(f"{'='*100}")
        print("3. DATA RETENTION ANALYSIS")
        print("-" * 100)
        
        retention = self.check_retention_rates()
        print(f"Overall word retention rate:  {retention['overall_retention_pct']:.1f}%")
        print(f"Total words lost:            {retention['total_words_lost']:,}\n")
        
        print("Distribution of retention rates:")
        print(f"  Excellent (95-100%):  {retention['excellent_95_100']:>5} rows ({retention['excellent_95_100']/len(self.df)*100:>5.1f}%)")
        print(f"  Good (80-95%):        {retention['good_80_95']:>5} rows ({retention['good_80_95']/len(self.df)*100:>5.1f}%)")
        print(f"  Poor (50-80%):        {retention['poor_50_80']:>5} rows ({retention['poor_50_80']/len(self.df)*100:>5.1f}%)")
        print(f"  Critical (<50%):      {retention['critical_below_50']:>5} rows ({retention['critical_below_50']/len(self.df)*100:>5.1f}%)\n")
        
        if retention['overall_retention_pct'] < 90:
            print(f"🔴 CRITICAL: Only {retention['overall_retention_pct']:.1f}% retention - significant data loss!")
        elif retention['overall_retention_pct'] < 95:
            print(f"🟡 CAUTION: {retention['overall_retention_pct']:.1f}% retention - some data loss")
        else:
            print(f"✅ Excellent retention rate: {retention['overall_retention_pct']:.1f}%")
        
        print()
        
        # ====================================================================
        # 4. PATTERNS
        # ====================================================================
        
        print(f"{'='*100}")
        print("4. TRUNCATION PATTERNS")
        print("-" * 100)
        
        patterns = self.check_patterns()
        print(f"Texts ending with '...':           {patterns['ends_with_ellipsis']}")
        print(f"Texts ending with '..':            {patterns['ends_with_dots']}")
        print(f"Texts ending with '-' (cut word):  {patterns['ends_with_incomplete_word']}")
        print(f"Contains 'truncat/incomplete':      {patterns['contains_truncation_marker']}\n")
        
        if sum(patterns.values()) > 0:
            print("⚠️  FOUND TRUNCATION INDICATORS")
            print("   The LLM may be cutting off at the max_tokens limit")
        else:
            print("✅ No obvious truncation patterns detected")
        
        print()
        
        # ====================================================================
        # 5. BATCH ISSUES
        # ====================================================================
        
        print(f"{'='*100}")
        print("5. BATCH PROCESSING ISSUES (Gemini API)")
        print("-" * 100)
        
        batch_issues = self.check_batch_issues()
        print(f"Error: Mismatch entries:  {batch_issues['mismatch_errors']}")
        print(f"Quota exceeded entries:   {batch_issues['quota_errors']}\n")
        
        if batch_issues['has_batch_issues']:
            print("🔴 BATCH PROCESSING ERRORS DETECTED")
            print("   Fix: Check batch size in preprocessing.py line 408-412")
        else:
            print("✅ No batch processing errors detected")
        
        print()
        
        # ====================================================================
        # FINAL RECOMMENDATIONS
        # ====================================================================
        
        print(f"{'='*100}")
        print("RECOMMENDATIONS")
        print("-" * 100)
        
        issues_found = []
        
        if token_analysis['texts_exceeding_limit'] > 0:
            issues_found.append("1. INCREASE max_tokens in preprocessing.py")
            issues_found.append("   Current: 8192 (line 225)")
            issues_found.append("   Try: 16384 or 32768")
        
        if batch_issues['has_batch_issues']:
            issues_found.append("\n2. FIX BATCH PROCESSING")
            issues_found.append("   Check reduce batch_size in preprocessing.py")
            issues_found.append("   Current: batch_size=10 (line 625)")
        
        if retention['overall_retention_pct'] < 90:
            issues_found.append("\n3. INVESTIGATE GEMINI API")
            issues_found.append("   Your API responses may be getting truncated")
            issues_found.append("   Add JSON response validation")
        
        if not issues_found:
            issues_found.append("✅ No major issues found!")
            issues_found.append("   Your data truncation may be intentional cleaning,")
            issues_found.append("   or due to external factors outside the code.")
        
        print("\n".join(issues_found))
        print(f"\n{'='*100}\n")
    
    # ========================================================================
    # EXPORT DETAILED REPORT
    # ========================================================================
    
    def export_detailed_report(self, output_csv: str = None) -> pd.DataFrame:
        """Export detailed per-row analysis to CSV."""
        if not self.valid:
            return None
        
        self.df['source_words'] = self.df[self.source_col].apply(self.word_count)
        self.df['target_words'] = self.df[self.target_col].apply(self.word_count)
        self.df['source_chars'] = self.df[self.source_col].apply(self.char_count)
        self.df['target_chars'] = self.df[self.target_col].apply(self.char_count)
        self.df['word_ratio'] = self.df['target_words'] / (self.df['source_words'] + 1)
        self.df['has_errors'] = self.df[self.target_col].astype(str).str.contains(
            '[ERROR|[SKIPPED|Mismatch', regex=True, na=False
        )
        
        export_cols = [
            'source_words', 'target_words', 'word_ratio',
            'source_chars', 'target_chars',
            'has_errors', self.source_col, self.target_col
        ]
        
        export_df = self.df[export_cols].copy()
        
        if output_csv is None:
            output_csv = Path(self.csv_path).stem + "_diagnostic.csv"
        
        export_df.to_csv(output_csv, index=True)
        print(f"\n✓ Detailed diagnostic report saved to: {output_csv}")
        
        return export_df


# =============================================================================
# SECTION 5: DATA VALIDATION & STATS
# =============================================================================

def load_preprocessed_data(data_path: str) -> pd.DataFrame:
    """
    Load preprocessed CSV file.
    
    Args:
        data_path: Path to preprocessed CSV file (can be relative or absolute)
    
    Returns:
        Loaded DataFrame
    """
    # Try to find the file using the path resolver
    path = Path(data_path)
    
    if not path.exists():
        # Try resolving from project root
        path = resolve_data_path(data_path)
    
    if not path.exists():
        # Try in DATA/preprocessed/ from project root
        alt_path = PROJECT_ROOT / "DATA" / "preprocessed" / Path(data_path).name
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(
                f"Data file not found: {data_path}\n"
                f"Tried:\n"
                f"  - {data_path}\n"
                f"  - {resolve_data_path(data_path)}\n"
                f"  - {alt_path}\n"
                f"Project root: {PROJECT_ROOT}"
            )
    
    print(f"Loading preprocessed data from: {path}")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows")
    
    return df


def read_preprocessing_log(log_path: str) -> Dict:
    """
    Parse preprocessing error log file and extract statistics.
    
    Args:
        log_path: Path to .log file from preprocessing.py
    
    Returns:
        Dictionary with error statistics
    """
    # Try to find the file
    path = Path(log_path)
    
    if not path.exists():
        path = resolve_data_path(log_path)
    
    if not path.exists():
        # Try in DATA/preprocessed/ from project root
        alt_path = PROJECT_ROOT / "DATA" / "preprocessed" / Path(log_path).name
        if alt_path.exists():
            path = alt_path
        else:
            return {"error": f"Log file not found: {log_path}"}
    
    stats = {
        "log_file": str(path),
        "errors": [],
        "total_reports": 0,
        "successful": 0,
        "errors_count": 0,
        "skipped_count": 0
    }
    
    with open(path, 'r') as f:
        content = f.read()
        
        # Parse header stats
        for line in content.split('\n'):
            if 'Total reports:' in line:
                try:
                    stats['total_reports'] = int(line.split(':')[1].strip())
                except:
                    pass
            elif 'Successfully processed:' in line:
                try:
                    stats['successful'] = int(line.split(':')[1].strip())
                except:
                    pass
            elif 'Errors:' in line and 'Error' not in line:
                try:
                    stats['errors_count'] = int(line.split(':')[1].strip())
                except:
                    pass
            elif 'Skipped' in line:
                try:
                    stats['skipped_count'] = int(line.split(':')[1].strip())
                except:
                    pass
        
        # Extract error details
        in_details = False
        for line in content.split('\n'):
            if 'ERROR DETAILS:' in line or 'DETAILS:' in line:
                in_details = True
                continue
            if in_details and line.startswith('Report'):
                stats['errors'].append(line)
    
    return stats


def show_preprocessing_stats(csv_path: str, log_path: Optional[str] = None):
    """
    Show main statistics from preprocessed data and error log.
    
    Args:
        csv_path: Path to preprocessed CSV
        log_path: Path to error log (optional, auto-detected if not provided)
    """
    print(f"\n{'='*80}")
    print("PREPROCESSING STATISTICS")
    print(f"{'='*80}\n")
    
    # Load CSV
    try:
        df = load_preprocessed_data(csv_path)
    except Exception as e:
        print(f"[ERROR] {e}")
        return
    
    csv_path_obj = Path(csv_path)
    
    # Count rows and sentences
    num_rows = len(df)
    total_sentences = 0
    
    # Assume 'sentences' column exists (from basic_preprocess)
    if 'sentences' in df.columns:
        total_sentences = len(df[df['sentences'].notna()])
        print(f"Reports (rows):        {num_rows}")
        print(f"Sentences (total):     {total_sentences}")
        print(f"Avg sentences/report:  {total_sentences / num_rows:.1f}" if num_rows > 0 else 0)
    else:
        # If no 'sentences' column, just show rows
        print(f"Reports (rows):        {num_rows}")
        print(f"Note: No 'sentences' column found (run basic_preprocess first)")
    
    # Look for error log
    if log_path is None:
        # Auto-detect log file
        log_files = list(csv_path_obj.parent.glob(csv_path_obj.stem + "*_errors.log"))
        if log_files:
            log_path = str(log_files[0])
    
    # Read log if available
    if log_path and Path(log_path).exists():
        print(f"\n{'='*80}")
        print("ERROR LOG SUMMARY")
        print(f"{'='*80}\n")
        
        log_stats = read_preprocessing_log(log_path)
        print(f"Total reports processed:  {log_stats['total_reports']}")
        print(f"Successfully processed:   {log_stats['successful']}")
        print(f"Errors:                   {log_stats['errors_count']}")
        print(f"Skipped (too long):       {log_stats['skipped_count']}")
        
        if log_stats['errors']:
            print(f"\nError Details:")
            for error in log_stats['errors'][:5]:  # Show first 5 errors
                print(f"  {error}")
            if len(log_stats['errors']) > 5:
                print(f"  ... and {len(log_stats['errors']) - 5} more errors")
        
        print(f"\nFull log: {log_path}")
    else:
        print(f"\nNo error log found")
    
    print(f"\n{'='*80}\n")


def compare_source_and_translated(df, source_col: str = "reflection_answer", 
                                   target_col: str = "cleaned_reflection",
                                   num_samples: int = 5, anonymise: bool = True):
    """
    Display side-by-side comparison of source and translated texts.
    
    Args:
        df: DataFrame with source and target columns
        source_col: Name of source column
        target_col: Name of target/cleaned column
        num_samples: Number of samples to display (default: 5)
        anonymise: Whether to mask PII in output
    """
    if source_col not in df.columns or target_col not in df.columns:
        print(f"[ERROR] Columns not found. Available: {list(df.columns)}")
        return
    
    count = min(num_samples, len(df))
    samples = np.random.choice(len(df), count, replace=False)
    
    print(f"\n{'='*100}")
    print(f"COMPARISON: {source_col} → {target_col} ({num_samples} examples)")
    print(f"{'='*100}\n")
    
    for i in samples:
        source = str(df.iloc[i][source_col])
        target = str(df.iloc[i][target_col])
        
        if anonymise:
            source = anonymise_text(source)
            target = anonymise_text(target)
        
        print(f"[Row {i}]")
        print(f"SOURCE ({len(source)} chars):\n{source[:300]}")
        if len(source) > 300:
            print("...")
        print()
        print(f"TARGET ({len(target)} chars):\n{target[:300]}")
        if len(target) > 300:
            print("...")
        print("-" * 100 + "\n")


# =============================================================================
# SECTION 6: CLI FOR STANDALONE USAGE
# =============================================================================

def main():
    """Standalone CLI for translation utilities."""
    parser = argparse.ArgumentParser(
        description="Post-preprocessing utilities v3.0 (validation, anonymisation, diagnostics)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all preprocessed datasets
  python translation_utils.py list
  
  # Show preprocessing statistics and error log
  python translation_utils.py stats --input-csv dreamachine_DL_cleaned_llama.csv
  
  # Compare source and translated texts (5 examples)
  python translation_utils.py compare --input-csv dreamachine_DL_cleaned_llama.csv
  
  # Compare word counts (CRITICAL for detecting truncation)
  python translation_utils.py wordcount --input-csv dreamachine_DL_cleaned_llama.csv
  
  # Export word count report to CSV
  python translation_utils.py wordcount --input-csv dreamachine_DL_cleaned_llama.csv --export
  
  # Full truncation diagnostic
  python translation_utils.py diagnostic --input-csv dreamachine_DL_cleaned_llama.csv
  
  # Full diagnostic + export report
  python translation_utils.py diagnostic --input-csv dreamachine_DL_cleaned_llama.csv --export
  
  # Preview anonymisation
  python translation_utils.py anonymise-preview --input-csv dreamachine_DL_cleaned_llama.csv
        """
    )
    
    parser.add_argument(
        "command",
        choices=["list", "stats", "compare", "wordcount", "diagnostic", "anonymise-preview"],
        help="Command to run"
    )
    
    parser.add_argument(
        "--input-csv",
        default=None,
        help="Path to preprocessed CSV file (relative or absolute)"
    )
    
    parser.add_argument(
        "--source-col",
        default="reflection_answer",
        help="Source text column name (default: reflection_answer)"
    )
    
    parser.add_argument(
        "--target-col",
        default="cleaned_reflection",
        help="Target text column name (default: cleaned_reflection)"
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to display (default: 5)"
    )
    
    parser.add_argument(
        "--no-anonymise",
        action="store_true",
        help="Don't anonymise text in output (for testing only)"
    )
    
    parser.add_argument(
        "--log-file",
        default=None,
        help="Path to error log file (auto-detected if not provided)"
    )
    
    parser.add_argument(
        "--data-dir",
        default="DATA",
        help="Path to DATA directory (default: DATA)"
    )
    
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export results to CSV file"
    )
    
    args = parser.parse_args()
    
    # Handle 'list' command (doesn't need input file)
    if args.command == "list":
        print_dataset_inventory(args.data_dir)
        return
    
    # Other commands require input file
    if not args.input_csv:
        print("[ERROR] --input-csv is required for this command")
        parser.print_help()
        return
    
    # Load CSV (for most commands)
    if args.command != "diagnostic":
        try:
            df = load_preprocessed_data(args.input_csv)
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            return
        except Exception as e:
            print(f"[ERROR] Could not load CSV: {e}")
            return
    
    # Run command
    if args.command == "stats":
        show_preprocessing_stats(args.input_csv, args.log_file)
    
    elif args.command == "compare":
        compare_source_and_translated(
            df, 
            source_col=args.source_col,
            target_col=args.target_col,
            num_samples=args.num_samples,
            anonymise=not args.no_anonymise
        )
    
    elif args.command == "wordcount":
        compare_word_counts(
            args.input_csv,
            source_col=args.source_col,
            target_col=args.target_col,
            show_samples=min(5, args.num_samples)
        )
        
        if args.export:
            export_word_count_report(
                args.input_csv,
                source_col=args.source_col,
                target_col=args.target_col
            )
    
    elif args.command == "diagnostic":
        diagnostic = TruncationDiagnostic(
            args.input_csv,
            source_col=args.source_col,
            target_col=args.target_col
        )
        diagnostic.run_full_diagnostic()
        
        if args.export:
            diagnostic.export_detailed_report()
    
    elif args.command == "anonymise-preview":
        sample = df[[args.source_col]].head(args.num_samples)
        print(f"\n{'='*100}")
        print(f"ANONYMISATION PREVIEW ({args.num_samples} samples)")
        print(f"{'='*100}\n")
        
        for idx, row in sample.iterrows():
            original = str(row[args.source_col])
            anonymised = anonymise_text(original)
            print(f"[Row {idx}]")
            print(f"ORIGINAL:   {original[:150]}")
            print(f"ANONYMISED: {anonymised[:150]}\n")


if __name__ == "__main__":
    main()

# # List all datasets
# python translation_utils.py list

# # Show stats + error log
# python translation_utils.py stats --input-csv filename.csv

# # Compare original vs cleaned (5 examples)
# python translation_utils.py compare --input-csv filename.csv --num-samples 5

# # Word count comparison (CRITICAL for detecting truncation)
# python translation_utils.py wordcount --input-csv filename.csv

# # Export word count analysis
# python translation_utils.py wordcount --input-csv filename.csv --export

# # Full truncation diagnostic
# python translation_utils.py diagnostic --input-csv filename.csv

# # Full diagnostic with export
# python translation_utils.py diagnostic --input-csv filename.csv --export

# # Preview PII anonymization
# python translation_utils.py anonymise-preview --input-csv filename.csv --num-samples 10



# python translation_utils.py compare --input-csv ganzfeld_GREEN_cleaned_llama_sample5.csv

# python src/mosaic/preprocessing/translation_utils.py diagnostic --input-csv ganzfeld_GREEN_cleaned_llama.csv