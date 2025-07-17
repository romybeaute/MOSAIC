# compare_translators_final.py

import pandas as pd
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import os
import math
import re

# --- SCRIPT CONFIGURATION ---
# Ensure these paths are correct for your setup
DATASET_NAME = "NDE"
BOX_DIR = os.path.join(os.path.expanduser("~"), "Library", "CloudStorage", "Box-Box", "TMDATA")
DATA_DIR = os.path.join(BOX_DIR, DATASET_NAME)
INPUT_CSV_PATH = os.path.join(DATA_DIR, f"{DATASET_NAME}_reports_with_language.csv")

# --- LANGUAGE CONFIGURATION ---
# The column name that identifies the language of each report.
LANGUAGE_COLUMN = 'Language' 
# List of languages to translate. Reports in other languages will be ignored.
# Based on your data: ['French' 'Flemish/Dutch' 'English']
# LANGUAGES_TO_TRANSLATE = ['French', 'Flemish/Dutch'] 
LANGUAGES_TO_TRANSLATE = ['Flemish/Dutch'] 

# --- MODEL AND PARAMETERS ---
MODEL_CHOICE = 'llama'
# Max context size of the model (in tokens). Llama3-8B is 8192, but we use 4096 to be safe.
MODEL_CONTEXT_SIZE = 4096
# Maximum number of tokens to generate for the translation.
MAX_GENERATION_TOKENS = 1500

# --- NEW: EXECUTION MODE ---
# Set to True to run a test on a small sample, False to run on the full dataset.
RUN_IN_TEST_MODE = True
TEST_SAMPLE_SIZE = 10 # Number of reports to translate in test mode.


def clean_translation(text: str) -> str:
    """
    Cleans the translated text by removing common preambles and wrapping quotes.
    """
    preambles = [
        "Here is the translation of the text into English:", "Here is the translation of the text:",
        "Here's the translation in English:", "Here is the translation:", "The translation is:",
        "Translation:"
    ]
    
    cleaned_text = text.strip()

    for preamble in preambles:
        if cleaned_text.lower().startswith(preamble.lower()):
            cleaned_text = cleaned_text[len(preamble):].lstrip(' :')
            break
    
    if cleaned_text.startswith('"') and cleaned_text.endswith('"'):
        cleaned_text = cleaned_text[1:-1]
    
    return cleaned_text.strip()


def translate_text_with_chunking(
    text: str, llm: Llama, model_config: dict, 
    max_context_tokens: int, max_generation_tokens: int
) -> str:
    """
    Translates a text. If the text is too long, it splits it into chunks,
    translates each chunk, and reassembles the result.
    """
    def create_prompt(content):
        return model_config['prompt_template'].format(text_to_translate=content)

    prompt_template_tokens = len(llm.tokenize(create_prompt("").encode('utf-8', errors='ignore')))
    safe_input_token_limit = max_context_tokens - max_generation_tokens - prompt_template_tokens - 50 # Safety margin

    text_tokens = llm.tokenize(text.encode('utf-8', errors='ignore'))

    if len(text_tokens) <= safe_input_token_limit:
        # Text is short enough to translate in one go
        try:
            prompt = create_prompt(text)
            response = llm(prompt=prompt, max_tokens=max_generation_tokens, stop=["<|eot_id|>", "\n\nTranslation:"], echo=False)
            return clean_translation(response['choices'][0]['text'])
        except Exception as e:
            print(f"\nError translating short text: {e}")
            return "Translation Error"
    
    # --- Chunking logic for long texts ---
    print(f"\nText too long ({len(text_tokens)} tokens), chunking...")
    
    if "\n\n" in text:
        sentences = text.split('\n\n')
    else:
        sentences = re.split(r'(?<=[.!?])\s+', text)

    translated_chunks = []
    current_chunk = ""

    for sentence in sentences:
        next_chunk = f"{current_chunk}\n\n{sentence}" if current_chunk else sentence
        
        if len(llm.tokenize(next_chunk.encode('utf-8', errors='ignore'))) > safe_input_token_limit:
            if current_chunk:
                try:
                    prompt = create_prompt(current_chunk)
                    response = llm(prompt=prompt, max_tokens=max_generation_tokens, stop=["<|eot_id|>", "\n\nTranslation:"], echo=False)
                    translated_chunks.append(clean_translation(response['choices'][0]['text']))
                except Exception as e:
                    print(f"\nError translating chunk: {e}")
                    translated_chunks.append("[Chunk Translation Error]")
            
            if len(llm.tokenize(sentence.encode('utf-8', errors='ignore'))) > safe_input_token_limit:
                 print(f"Warning: A single sentence/paragraph is too long and will be truncated.")
                 truncated_tokens = llm.tokenize(sentence.encode('utf-8', errors='ignore'))[:safe_input_token_limit]
                 current_chunk = llm.detokenize(truncated_tokens).decode('utf-8', errors='ignore')
            else:
                current_chunk = sentence
        else:
            current_chunk = next_chunk
    
    if current_chunk:
        try:
            prompt = create_prompt(current_chunk)
            response = llm(prompt=prompt, max_tokens=max_generation_tokens, stop=["<|eot_id|>", "\n\nTranslation:"], echo=False)
            translated_chunks.append(clean_translation(response['choices'][0]['text']))
        except Exception as e:
            print(f"\nError translating final chunk: {e}")
            translated_chunks.append("[Chunk Translation Error]")

    return " ".join(translated_chunks)


def run_secure_local_translation(
    csv_path: str, output_path: str, model_config: dict, 
    text_column: str = 'report', batch_size: int = 10, is_test_run: bool = False, sample_size: int = 10
):
    """
    Translates a CSV column to English in batches, with resume capability,
    language filtering, and long-text handling.
    """
    print(f"--- Starting Translation with Model: {model_config['name']} ---")

    try:
        df_full = pd.read_csv(csv_path)
        print(f"Successfully loaded {len(df_full)} rows from {csv_path}")
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        return
    
    if LANGUAGE_COLUMN not in df_full.columns:
        print(f"Error: Language column '{LANGUAGE_COLUMN}' not found in the CSV.")
        return

    # --- Resume Logic ---
    if os.path.exists(output_path):
        print(f"Output file found at {output_path}. Resuming translation.")
        df_output = pd.read_csv(output_path)
        if 'report_english' not in df_output.columns:
            df_output['report_english'] = pd.NA
    else:
        print("No existing output file. Starting a new translation.")
        df_output = df_full.copy()
        df_output['report_english'] = pd.NA
    
    untranslated_mask = df_output['report_english'].isna() | (df_output['report_english'] == "")
    language_mask = df_output[LANGUAGE_COLUMN].str.lower().isin([lang.lower() for lang in LANGUAGES_TO_TRANSLATE])
    combined_mask = untranslated_mask & language_mask
    indices_to_translate = df_output[combined_mask].index
    
    # --- Test Mode Sampling ---
    if is_test_run:
        print(f"\n--- TEST MODE: Selecting a random sample of {sample_size} reports. ---")
        if len(indices_to_translate) > sample_size:
            indices_to_translate = indices_to_translate.to_series().sample(sample_size, random_state=42).index
        else:
            print(f"Fewer than {sample_size} reports need translation. Processing all {len(indices_to_translate)} available reports.")

    if indices_to_translate.empty:
        print("Translation already complete for all target reports. No action required.")
        return
        
    print(f"{len(indices_to_translate)} reports ({', '.join(LANGUAGES_TO_TRANSLATE)}) remaining to be translated.")

    # --- Model Loading ---
    print(f"Downloading model '{model_config['filename']}' (if not cached)...")
    model_path = hf_hub_download(repo_id=model_config['repo'], filename=model_config['filename'])
    print("Loading model with Metal GPU acceleration...")
    llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=MODEL_CONTEXT_SIZE, verbose=False)
    print("Model loaded successfully.")

    # --- Batch Translation ---
    num_batches = math.ceil(len(indices_to_translate) / batch_size)
    
    for i in tqdm(range(num_batches), desc="Total Batch Progress"):
        batch_indices = indices_to_translate[i * batch_size : (i + 1) * batch_size]
        
        translations_batch = []
        for index in tqdm(batch_indices, desc=f"Translating Batch {i+1}/{num_batches}", leave=False):
            text = df_output.loc[index, text_column]
            if not isinstance(text, str) or not text.strip():
                translations_batch.append("Not Applicable (empty)")
                continue
            
            translated_text = translate_text_with_chunking(
                text, llm, model_config, MODEL_CONTEXT_SIZE, MAX_GENERATION_TOKENS
            )
            translations_batch.append(translated_text)
        
        df_output.loc[batch_indices, 'report_english'] = translations_batch
        
        df_output.to_csv(output_path, index=False)
        print(f"\nBatch {i+1}/{num_batches} complete. Progress saved to {output_path}.")

    print(f"\n--- Translation Complete ---")
    print(f"Translated data saved to {output_path}")

# --- SCRIPT EXECUTION ---
if __name__ == '__main__':
    
    models = {
        'llama': {
            'name': 'Llama-3-8B-Instruct',
            'repo': 'NousResearch/Meta-Llama-3-8B-Instruct-GGUF',
            'filename': 'Meta-Llama-3-8B-Instruct-Q4_K_M.gguf',
            'prompt_template': """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert translator. Your task is to translate the user's text into English accurately. Do not add any commentary, notes, or explanations. Only return the translated text.<|eot_id|><|start_header_id|>user<|end_header_id|>
Translate the following text to English:
"{text_to_translate}"<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        },
        'nllb': {
            'name': 'NLLB-200-Distilled',
            'repo': 'Qwen/nllb-200-distilled-600M-GGUF', 
            'filename': 'nllb-200-distilled-600M.Q4_0.gguf', 
            'prompt_template': "Translate the following text to English:\n\n{text_to_translate}\n\nTranslation:"
        }
    }

    selected_model_config = models.get(MODEL_CHOICE)
    if not selected_model_config:
        raise ValueError(f"Invalid model choice. Please choose from {list(models.keys())}")

    output_dir = os.path.join(DATA_DIR, 'translations')
    os.makedirs(output_dir, exist_ok=True)
    
    run_mode = "test_sample" if RUN_IN_TEST_MODE else "full_dataset"
    output_filename = f'{DATASET_NAME}_translated_{MODEL_CHOICE}_{run_mode}.csv'
    OUTPUT_CSV_PATH = os.path.join(output_dir, output_filename)

    print(f"\n>>> Preparing to run translation with model '{MODEL_CHOICE}' in '{run_mode}' mode. <<<")
    run_secure_local_translation(
        csv_path=INPUT_CSV_PATH,
        output_path=OUTPUT_CSV_PATH,
        model_config=selected_model_config,
        batch_size=5,
        is_test_run=RUN_IN_TEST_MODE,
        sample_size=TEST_SAMPLE_SIZE
    )
    print("\n--- SCRIPT EXECUTION FINISHED ---")
