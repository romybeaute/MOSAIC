#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : translation_utils.py
# description     : Post-preprocessing utilities (validation, anonymisation, error detection)
#                   - Works WITH preprocessed data from preprocessing.py
#                   - Does NOT duplicate sentence splitting or text cleaning
#                   - Provides: validation, anonymisation, error detection, PII masking
#                   - Reads preprocessing error logs and outputs stats
#                   - Lists preprocessed datasets and files
# author          : Romy, Beauté (r.beaut@sussex.ac.uk)
# date            : 2026-01-29
# version         : 2.2 (added dataset inventory)
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
# SECTION 0: DATASET INVENTORY
# =============================================================================

def list_preprocessed_datasets(data_dir: str = "DATA") -> Dict[str, Dict]:
    """
    List all preprocessed datasets and their files.
    
    Args:
        data_dir: Path to DATA directory
    
    Returns:
        Dictionary mapping dataset_name -> {files, methods, stats}
    """
    preprocessed_dir = Path(data_dir) / "preprocessed"
    
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
# SECTION 3: DATA VALIDATION & STATS
# =============================================================================

def load_preprocessed_data(data_path: str) -> pd.DataFrame:
    """
    Load preprocessed CSV file.
    
    Args:
        data_path: Path to preprocessed CSV file (can be relative or absolute)
    
    Returns:
        Loaded DataFrame
    """
    # Try to find the file
    path = Path(data_path)
    
    if not path.exists():
        # Try in DATA/preprocessed/
        alt_path = Path("DATA") / "preprocessed" / data_path
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(f"Data file not found: {data_path}\nTried: {alt_path}")
    
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
        # Try in DATA/preprocessed/
        alt_path = Path("DATA") / "preprocessed" / log_path
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
                                   num_samples: int = 3, anonymise: bool = True):
    """
    Display side-by-side comparison of source and translated texts.
    
    Args:
        df: DataFrame with source and target columns
        source_col: Name of source column
        target_col: Name of target/cleaned column
        num_samples: Number of samples to display
        anonymise: Whether to mask PII in output
    """
    if source_col not in df.columns or target_col not in df.columns:
        print(f"[ERROR] Columns not found. Available: {list(df.columns)}")
        return
    
    count = min(num_samples, len(df))
    samples = np.random.choice(len(df), count, replace=False)
    
    print(f"\n{'='*80}")
    print(f"COMPARISON: {source_col} → {target_col}")
    print(f"{'='*80}\n")
    
    for i in samples:
        source = str(df.iloc[i][source_col])
        target = str(df.iloc[i][target_col])
        
        if anonymise:
            source = anonymise_text(source)
            target = anonymise_text(target)
        
        print(f"[Row {i}]")
        print(f"SOURCE ({len(source)} chars):\n{source[:200]}...\n")
        print(f"TARGET ({len(target)} chars):\n{target[:200]}...\n")
        print("-" * 80 + "\n")


# =============================================================================
# SECTION 4: CLI FOR STANDALONE USAGE
# =============================================================================

def main():
    """Standalone CLI for translation utilities."""
    parser = argparse.ArgumentParser(
        description="Post-preprocessing utilities (validation, anonymisation, error detection)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all preprocessed datasets
  python translation_utils.py list
  
  # Show preprocessing statistics and error log
  python translation_utils.py stats --input-csv dreamachine_DL_cleaned_llama.csv
  
  # Show stats with full path
  python translation_utils.py stats --input-csv DATA/preprocessed/dreamachine_DL_cleaned_llama.csv
  
  # Compare source and translated texts (with anonymisation)
  python translation_utils.py compare --input-csv dreamachine_DL_cleaned_llama.csv
  
  # Preview anonymisation
  python translation_utils.py anonymise-preview --input-csv dreamachine_DL_cleaned_llama.csv
        """
    )
    
    parser.add_argument(
        "command",
        choices=["list", "stats", "compare", "anonymise-preview"],
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
        default=3,
        help="Number of samples to display (default: 3)"
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
    
    # Load CSV
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
    
    elif args.command == "anonymise-preview":
        sample = df[[args.source_col]].head(args.num_samples)
        print(f"\n{'='*80}")
        print(f"ANONYMISATION PREVIEW")
        print(f"{'='*80}\n")
        
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
#python src/mosaic/preprocessing/translation_utils.py stats --input-csv innerspeech_cleaned_API_sample5.csv

# # Compare original vs cleaned
# python translation_utils.py compare --input-csv filename.csv

# # Preview PII anonymization
# python translation_utils.py anonymise-preview --input-csv filename.csv