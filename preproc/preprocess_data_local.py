import pandas as pd
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import os

def preprocess_locally(csv_path: str, output_path: str, model_config: dict, text_column: str = 'reflection_answer', num_samples: int = None):
    """
    Cleans a CSV column using a local LLM, optimised for performance
    on machines with GPU support (like Apple Silicon's Metal).

    Args:
        csv_path (str): Path to the input CSV file.
        output_path (str): Path to save the new CSV file with cleaned text.
        model_config (dict): A dictionary containing the model repo, filename, and prompt template.
        text_column (str): The name of the column containing text to be cleaned.
        num_samples (int, optional): The number of rows to process for a test run. Defaults to all.
    """
    print(f"--- Starting Preprocessing for: {os.path.basename(csv_path)} ---")
    print(f"--- Using Model: {model_config['name']} ---")

    # 1. Load and prepare the data
    try:
        df = pd.read_csv(csv_path).dropna(subset=[text_column])
        print(f"Successfully loaded {len(df)} non-empty rows from {csv_path}")
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        return
    except KeyError:
        print(f"Error: The required column '{text_column}' was not found in {csv_path}.")
        return


    df_to_process = df.head(num_samples).copy() if num_samples else df
    print(f"Processing {len(df_to_process)} reports for cleaning.")

    # 2. Set up the local model
    print(f"Downloading model '{model_config['filename']}' (if not already cached)...")
    model_path = hf_hub_download(
        repo_id=model_config['repo'],
        filename=model_config['filename']
    )
    print(f"Model path: {model_path}")

    # Load the model with GPU acceleration (n_gpu_layers=-1 offloads all layers)
    print("Loading model with GPU acceleration...")
    llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=4096, verbose=False)
    print("Model loaded successfully.")

    # 3. Clean the text column using the custom prompt
    cleaned_texts = []
    texts_to_clean = df_to_process[text_column].tolist()
    prompt_template = model_config['prompt_template']
    
    print(f"Starting text cleaning...")
    for text in tqdm(texts_to_clean, desc=f"Cleaning reports"):
        if not isinstance(text, str) or not text.strip():
            cleaned_texts.append("Not applicable (empty)")
            continue
        
        try:
            prompt = prompt_template.format(text_to_clean=text)
            
            response = llm(prompt=prompt, max_tokens=1024, stop=["<|eot_id|>"], echo=False)
            
            # Get the raw output from the model
            raw_output = response['choices'][0]['text'].strip()
            
            # remove the unwanted preamble
            preamble = "Here is the cleaned text:"
            if raw_output.lower().startswith(preamble.lower()):
                cleaned_text = raw_output[len(preamble):].strip()
            else:
                cleaned_text = raw_output
            
            cleaned_texts.append(cleaned_text)
            
        except Exception as e:
            print(f"\\nCould not clean text: '{text[:50]}...'. Error: {e}")
            cleaned_texts.append("Error during cleaning")

    # 4. Add the cleaned text to the DataFrame and save it
    df_to_process['cleaned_reflection'] = cleaned_texts
    df_to_process.to_csv(output_path, index=False)
    
    print(f"\\n--- Preprocessing Complete ---")
    print(f"Cleaned data saved to {output_path}")
    print("--- Sample of Results ---")
    print(df_to_process[[text_column, 'cleaned_reflection']].head())



if __name__ == '__main__':
    
    # --- Model Configuration ---
    model_config = {
        'name': 'Llama-3-8B-Instruct',
        'repo': 'NousResearch/Meta-Llama-3-8B-Instruct-GGUF',
        'filename': 'Meta-Llama-3-8B-Instruct-Q4_K_M.gguf',
        'prompt_template': """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert data cleaner. Your task is to clean the user's text.
Follow these rules precisely:
1. Correct spelling mistakes and fix grammar.
2. Remove artifacts and formatting like '\\n'.
3. Do NOT change the original meaning or punctuation of the text.
4. Your response must contain ONLY the cleaned text, without any introductory phrases or commentary.<|eot_id|><|start_header_id|>user<|end_header_id|>
Clean the following text:

"{text_to_clean}"<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    }

    # --- File Path Configuration ---
    DATA_DIR = os.path.join(os.path.expanduser("~"), "Library", "CloudStorage", "Box-Box", "TMDATA", "dreamachine")
    
    files_to_process = {
        'HS': {'input': 'freeform_HS_SensoryTool_complete.csv', 'output': 'HS_reflections_LOCALcleaned.csv'},
        'DL': {'input': 'freeform_DL_SensoryTool_complete.csv', 'output': 'DL_reflections_LOCALcleaned.csv'}
    }


    NUM_SAMPLES = 5 # Set to None to run on all data

    if NUM_SAMPLES is not None:
        print(f"\\n>>> Preparing to run a TEST on {NUM_SAMPLES} samples. <<<\\n")
    else:
        print(f"\\n>>> Preparing to run on the FULL DATASET. This may take a while. <<<\\n")

    # --- Run the preprocessing for each file ---
    for key, paths in files_to_process.items():
        input_path = os.path.join(DATA_DIR, paths['input'])
        output_path = os.path.join(DATA_DIR, paths['output'])
        
        if NUM_SAMPLES is not None:
            output_path = output_path.replace('.csv', f'_{NUM_SAMPLES}_samples.csv')

        preprocess_locally(
            csv_path=input_path,
            output_path=output_path,
            model_config=model_config,
            num_samples=NUM_SAMPLES
        )
        print(f"\\n--- {key} Dataset Run Complete ---\\n")