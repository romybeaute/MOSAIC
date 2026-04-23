#!/usr/bin/env python
# coding=utf-8
import pandas as pd
import numpy as np
import argparse
import os
import sys
import json
import re
from huggingface_hub import hf_hub_download

# Fail fast if libraries are missing
try:
    from llama_cpp import Llama
except ImportError:
    print("Error: llama-cpp-python not installed.")
    sys.exit(1)

# --- CONFIGURATION ---
REPO_ID = "Bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
FILENAME = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

class LlamaCleaner:
    def __init__(self):
        # Load model once per worker
        model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,      
            n_ctx=8192,           
            verbose=False,
            chat_format="llama-3"
        )

    def process_text(self, text):
        if not isinstance(text, str) or len(text.strip()) < 2:
            return ""

        input_len = len(text)
        est_tokens = input_len // 2.5 
        max_tokens = max(512, int(est_tokens * 2.0))

        system_msg = (
            "You are a text rectification engine. You do not speak. You do not converse.\n"
            "INSTRUCTIONS:\n"
            "1. If the text is in English: Fix grammar/spelling ONLY. DO NOT rephrase. Keep original style.\n"
            "2. If the text is NOT in English: Translate to British English.\n"
            "3. CRITICAL: Do NOT summarise. Retain every single sentence.\n"
            "4. OUTPUT RULE: Return valid JSON ONLY: {\"cleaned\": \"...\"}. No 'Here is the text', no preamble."
        )

        user_msg = f"TEXT: {text}"

        try:
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=max_tokens
            )
            
            raw_content = response['choices'][0]['message']['content']
            
            try:
                data = json.loads(raw_content)
                return data.get("cleaned", "")
            except json.JSONDecodeError:
                match = re.search(r'"cleaned"\s*:\s*"(.*?)"', raw_content, re.DOTALL)
                if match:
                    return match.group(1)
                else:
                    return raw_content.strip()

        except Exception as e:
            return f"[ERROR] {str(e)}"

def run_chunk(input_csv, output_dir, start_idx, end_idx, job_id):
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"Error reading input CSV: {e}")
        return

    if start_idx >= len(df):
        return
    
    # Slice the dataframe
    subset = df.iloc[start_idx:min(end_idx, len(df))].copy()
    print(f"Job {job_id}: Processing rows {start_idx} to {end_idx} ({len(subset)} rows)")
    
    # Ensure output dir exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define filenames
    final_file = os.path.join(output_dir, f"chunk_{job_id:04d}.csv")
    temp_file = os.path.join(output_dir, f"chunk_{job_id:04d}_PARTIAL.csv")

    cleaner = LlamaCleaner()
    
    # We loop through index to update the DataFrame in-place
    # This allows us to save the DataFrame at any point
    for i, idx in enumerate(subset.index):
        txt = subset.at[idx, 'sentence']
        cleaned = cleaner.process_text(txt)
        subset.at[idx, 'cleaned_sentence'] = cleaned
        
        # SAVE EVERY 50 ROWS (Crash Protection)
        if (i + 1) % 50 == 0:
            print(f"Job {job_id}: Processed {i + 1}/{len(subset)} - Saving partial...")
            subset.to_csv(temp_file, index=False)
            
    # Final Save
    subset.to_csv(final_file, index=False)
    
    # Remove temp file if successful
    if os.path.exists(temp_file):
        os.remove(temp_file)
        
    print(f"Job {job_id}: Completed. Saved to {final_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--job_id", type=int, required=True)
    args = parser.parse_args()
    
    run_chunk(args.input, args.output_dir, args.start, args.end, args.job_id)