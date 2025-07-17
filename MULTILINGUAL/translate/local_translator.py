# compare_translators.py

import pandas as pd
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import os
import time

def secure_local_translation(csv_path: str, output_path: str, model_config: dict, text_column: str = 'phen_report', num_samples: int = None):
    """
    Translates a CSV column to English using a user-selected local model,
    optimised for Mac with Apple Silicon (Metal).

    Args:
        csv_path (str): Path to the input CSV file.
        output_path (str): Path to save the new CSV file with translations.
        model_config (dict): A dictionary containing model repo, filename, and prompt template.
        text_column (str): The name of the column containing text to be translated.
        num_samples (int, optional): The number of rows to translate for a test run. Defaults to all.
    """
    print(f"--- Starting Translation with Model: {model_config['name']} ---")

    # load and prepare data
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded {len(df)} rows from {csv_path}")
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        return

    df_to_process = df.head(num_samples).copy() if num_samples else df
    print(f"Processing {len(df_to_process)} reports for translation.")

    # #set up the selected local model
    # model_dir = "/Users/rbeaute/Projects/MOSAIC/MULTILINGUAL/models"
    # os.makedirs(model_dir, exist_ok=True)
    
    print(f"Downloading model '{model_config['filename']}' (if not cached)...")
    model_path = hf_hub_download(
        repo_id=model_config['repo'],
        filename=model_config['filename']
    )
    print(f"Model path: {model_path}")

    # load the model with Metal GPU acceleration
    print("Loading model with Metal GPU acceleration...")
    # llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=4096, verbose=False)
    llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=4096, verbose=False)
    print("Model loaded successfully.")

    # translate the text column using the appropriate prompt
    translations = []
    texts_to_translate = df_to_process[text_column].tolist()
    prompt_template = model_config['prompt_template']
    
    print(f"Starting translation...")
    for text in tqdm(texts_to_translate, desc=f"Translating with {model_config['name']}"):
        if not isinstance(text, str) or not text.strip():
            translations.append("Not applicable (empty)")
            continue
        
        try:
            prompt = prompt_template.format(text_to_translate=text)
            response = llm(prompt=prompt, max_tokens=1024, stop=["<|eot_id|>", "\n\nTranslation:"], echo=False)
            translated_text = response['choices'][0]['text'].strip()
            translations.append(translated_text)
        except Exception as e:
            print(f"\nCould not translate text: '{text[:50]}...'. Error: {e}")
            translations.append("Error during translation")

    # add translations to the DataFrame and save
    df_to_process['phen_report_english'] = translations
    df_to_process.to_csv(output_path, index=False)
    
    print(f"\n--- Translation Complete ---")
    print(f"Translated data saved to {output_path}")
    print(df_to_process[[text_column, 'phen_report_english']].head())


# --- SCRIPT CONFIGURATION AND EXECUTION ---
if __name__ == '__main__':
    
    MODEL_CHOICE = 'llama' 

    # --- Model configurations ---
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

    if MODEL_CHOICE not in models:
        raise ValueError(f"Invalid model choice. Please choose from {list(models.keys())}")

    selected_model_config = models[MODEL_CHOICE]
    

    INPUT_CSV_PATH = '/Users/rbeaute/Projects/MOSAIC/DATA/multilingual/MPE_dataset.csv'
    output_filename = f'MPE_dataset_translated_{MODEL_CHOICE}.csv'
    OUTPUT_CSV_PATH = os.path.join('/Users/rbeaute/Projects/MOSAIC/DATA/multilingual', output_filename)

    # --- CHOOSE HOW MANY REPORTS TO TRANSLATE ---
    # To test with a small batch, set num_samples to a number (e.g., 10).
    # To run on the full dataset, set num_samples to None.
    
    # # --- To run a test on a small subset ---
    # print(f"\n>>> Preparing to run a TEST with the '{MODEL_CHOICE}' model on 10 samples. <<<")
    # secure_local_translation(
    #     csv_path=INPUT_CSV_PATH,
    #     output_path=OUTPUT_CSV_PATH.replace('.csv', '_10_samples.csv'),
    #     model_config=selected_model_config,
    #     num_samples=10
    # )
    # print("\n--- TEST RUN COMPLETE ---")
    
    # --- To run on the full dataset ---
    # After testing, comment out the block above and uncomment this one.
    print(f"\n>>> Preparing to run on the FULL DATASET with the '{MODEL_CHOICE}' model. This may take time. <<<")
    secure_local_translation(
        csv_path=INPUT_CSV_PATH,
        output_path=OUTPUT_CSV_PATH,
        model_config=selected_model_config,
        num_samples=None
    )
    print("\n--- FULL DATASET RUN COMPLETE ---")