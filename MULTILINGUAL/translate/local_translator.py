# local_translator.py

import os
import json
import argparse
from pathlib import Path
import re

import pandas as pd
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from mosaic.path_utils import CFG, proc_path  # MOSAIC helpers


# ---------- PROMPT TEMPLATES ----------

# Cleaning / preprocessing prompt – shared
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


# ---------- LLM HELPERS ----------

def _extract_json_array(text: str):
    """
    Best-effort extraction of a JSON array from the model output.
    Looks for the first '[' and last ']' and tries to json.loads that slice.
    """
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON array brackets found in model output.")
    candidate = text[start: end + 1]
    return json.loads(candidate)


def clean_texts_with_llm(texts, llm, prompt_template: str = CLEAN_PROMPT_TEMPLATE):
    """
    Clean a list of texts using the LLM and the JSON-array cleaning prompt.
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


def strip_translation_output(text: str) -> str:
    """
    Post-process model output to remove boilerplate like:
    - 'Here is the translation:'
    - 'Here is the translation of the text into English:'
    - 'Translation:'
    - surrounding quotes
    so that only the translated content remains.
    """
    if not isinstance(text, str):
        return text

    s = text.strip()

    # Common preambles
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
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == '“' and s[-1] == '”')):
        s = s[1:-1].strip()

    return s


# ---------- MAIN PIPELINE ----------

def secure_local_translation(
    csv_path,
    output_path,
    model_config: dict,
    text_column: str | None = None,
    num_samples: int | None = None,
    task: str = "translate",  # 'translate', 'preprocess', or 'both'
):
    """
    Pipeline that can:
      - preprocess (clean) texts,
      - translate texts,
      - or do both (clean first, then translate).

    Args:
        csv_path (str | Path): Input CSV.
        output_path (str | Path): Output CSV.
        model_config (dict): Model repo/filename and prompt_template for translation.
        text_column (str | None): Column containing the raw text to process.
        num_samples (int | None): Limit rows for test runs.
        task (str): 'translate', 'preprocess', or 'both'.
    """
    csv_path = Path(csv_path)
    output_path = Path(output_path)

    do_preprocess = task in ("preprocess", "both")
    do_translate = task in ("translate", "both")

    print(f"--- Starting pipeline with Model: {model_config['name']} ---")
    print(f"Task        : {task}")
    print(f"Input CSV   : {csv_path}")
    print(f"Output CSV  : {output_path}")
    print(f"Text column : {text_column}")
    if num_samples:
        print(f"num_samples : {num_samples} (test run)")
    else:
        print("num_samples : ALL rows")

    # --- load data ---
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded {len(df)} rows from {csv_path}")
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        return

    df_to_process = df.head(num_samples).copy() if num_samples else df
    print(f"Processing {len(df_to_process)} reports.")

    # collect errors to a log that we can save as CSV
    error_log: list[dict] = []

    # --- choose text column (flexible) ---
    if text_column is None:
        text_column = df_to_process.columns[0]
        print(f"No --text-column provided. Using first column: '{text_column}'")
    elif text_column not in df_to_process.columns:
        fallback = df_to_process.columns[0]
        print(
            f"Column '{text_column}' not found in CSV. "
            f"Falling back to first column: '{fallback}'"
        )
        text_column = fallback
    else:
        print(f"Using specified text column: '{text_column}'")

    # --- download / locate model ---
    print(f"Downloading model '{model_config['filename']}' (if not cached)...")
    model_path = hf_hub_download(
        repo_id=model_config["repo"],
        filename=model_config["filename"],
    )
    print(f"Model path: {model_path}")

    # --- load model (Metal) ---
    print("Loading model with Metal GPU acceleration...")
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=8192,    # larger context: more room for long reports
        n_batch=256,   # batch size for faster GPU eval
        verbose=False,
    )
    print("Model loaded successfully.")

    max_ctx = llm.n_ctx()
    safety_margin = 64  # keep some buffer

    # --- choose source series (raw → maybe cleaned → maybe translated) ---
    source_series = df_to_process[text_column]

    # --- PREPROCESSING STEP (optional) ---
    if do_preprocess:
        print("\n>>> Preprocessing texts (cleaning) <<<")
        cleaned_texts = []

        for text in tqdm(source_series.tolist(), desc="Cleaning texts"):
            if not isinstance(text, str) or not text.strip():
                cleaned_texts.append(text)
                continue

            cleaned = clean_texts_with_llm([text], llm)[0]
            cleaned_texts.append(cleaned)

        clean_col = f"{text_column}_clean"
        df_to_process[clean_col] = cleaned_texts
        source_series = df_to_process[clean_col]
        print(f"Cleaning complete. Stored cleaned texts in column '{clean_col}'.")

    # If only preprocessing is requested, just save and exit
    if do_preprocess and not do_translate:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_to_process.to_csv(output_path, index=False)
        print("\n--- PREPROCESSING ONLY: Complete ---")
        print(f"Saved cleaned data to {output_path}")
        print(df_to_process[[text_column, f"{text_column}_clean"]].head())
        return

    # --- TRANSLATION STEP (optional; uses source_series which may be cleaned) ---
    if do_translate:
        print("\n>>> Translating texts <<<")
        translations = []
        prompt_template = model_config["prompt_template"]

        items = list(source_series.items())  # [(index, text), ...]
        for row_idx, text in tqdm(items, desc=f"Translating with {model_config['name']}"):
            if not isinstance(text, str) or not text.strip():
                translations.append("Not applicable (empty)")
                continue

            try:
                # Build full prompt for this text
                prompt = prompt_template.format(text_to_translate=text)

                # Token count for this specific prompt
                prompt_tokens = len(llm.tokenize(prompt.encode("utf-8")))
                avail_for_gen = max_ctx - prompt_tokens - safety_margin

                if avail_for_gen <= 0:
                    err_msg = (
                        f"Prompt too long for context window "
                        f"(prompt_tokens={prompt_tokens}, max_ctx={max_ctx})"
                    )
                    print(
                        f"\n[ERROR] {err_msg}. "
                        f"Row {row_idx}, preview: '{str(text)[:80]}...'"
                    )
                    translations.append("Error during translation (prompt too long)")
                    error_log.append(
                        {
                            "row_index": row_idx,
                            "text_preview": str(text)[:200],
                            "error": err_msg,
                            "task": task,
                        }
                    )
                    continue

                # Deterministic, literal-oriented decoding
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
                translations.append(translated_text)

                # --- Length sanity check: flag suspiciously short translations ---
                src_len = len(str(text).split())
                tgt_len = len(str(translated_text).split())
                if src_len >= 20 and tgt_len > 0 and tgt_len < 0.8 * src_len:
                    error_log.append(
                        {
                            "row_index": row_idx,
                            "text_preview": str(text)[:200],
                            "error": (
                                f"suspiciously short translation: "
                                f"src_len={src_len}, tgt_len={tgt_len}"
                            ),
                            "task": task,
                        }
                    )

            except Exception as e:
                err_msg = str(e)
                print(
                    f"\n[ERROR] Could not translate row {row_idx} "
                    f"(preview: '{str(text)[:80]}...'). Error: {err_msg}"
                )
                translations.append("Error during translation")
                error_log.append(
                    {
                        "row_index": row_idx,
                        "text_preview": str(text)[:200],
                        "error": err_msg,
                        "task": task,
                    }
                )

        df_to_process["phen_report_english"] = translations

    # --- SAVE OUTPUT ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_to_process.to_csv(output_path, index=False)

    print("\n--- PIPELINE COMPLETE ---")
    print(f"Saved data to {output_path}")

    cols_to_show = [text_column]
    if f"{text_column}_clean" in df_to_process.columns:
        cols_to_show.append(f"{text_column}_clean")
    if "phen_report_english" in df_to_process.columns:
        cols_to_show.append("phen_report_english")

    print(df_to_process[cols_to_show].head())

    # --- SAVE ERROR LOG (if any) ---
    if error_log:
        error_log_path = output_path.with_name(output_path.stem + "_errors.csv")
        error_df = pd.DataFrame(error_log)
        error_df.to_csv(error_log_path, index=False)
        print(
            f"\n[INFO] {len(error_log)} row(s) logged as problematic "
            f"(errors or suspiciously short translations). "
            f"Details saved to {error_log_path}"
        )


def main():
    # --- Model configurations ---
    models = {
        "llama": {
            "name": "Llama-3-8B-Instruct",
            "repo": "NousResearch/Meta-Llama-3-8B-Instruct-GGUF",
            "filename": "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
            "prompt_template": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

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

"{text_to_translate}"<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        },
    }

    parser = argparse.ArgumentParser(
        description="Preprocess (clean) and/or translate a CSV column to English using a local GGUF model (MOSAIC-aware paths)."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Logical dataset name used with mosaic.proc_path, e.g. 'MULTILINGUAL' or 'MPE'.",
    )
    parser.add_argument(
        "--input-csv",
        default="MPE_dataset.csv",
        help="CSV filename (relative to proc_path(dataset)). Default: MPE_dataset.csv",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional output CSV filename. "
             "If omitted, uses '<input_stem>_<task>_<model>.csv'.",
    )
    parser.add_argument(
        "--output-subdir",
        default=None,
        help="Optional subdirectory under proc_path(dataset) for outputs, e.g. 'translated'.",
    )
    parser.add_argument(
        "--text-column",
        default="reflection_answer",
        help=(
            "Name of the text column to process. Default: reflection_answer. "
            "If omitted or not found, the script will use the first column in the CSV."
        ),
    )
    parser.add_argument(
        "--model",
        choices=models.keys(),
        default="llama",
        help="Which local model to use. Only 'llama' (Llama-3-8B-Instruct GGUF) is supported here.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="If set, only process the first N rows (for test runs).",
    )
    parser.add_argument(
        "--task",
        choices=["translate", "preprocess", "both"],
        default="translate",
        help="What to do: 'translate' (only translate), 'preprocess' (clean only), or 'both' (clean then translate).",
    )

    args = parser.parse_args()

    # --- show MOSAIC config roots ---
    print("MOSAIC config:")
    print("  DATA_ROOT =", CFG.get("data_root", "(not set)"))
    print("  BOX_ROOT  =", CFG.get("box_root", "(not set)"))
    print()

    # --- resolve paths via mosaic.proc_path ---
    input_csv_path = proc_path(args.dataset, args.input_csv)

    # default output name
    if args.output_name is not None:
        out_name = args.output_name
    else:
        stem = Path(args.input_csv).stem
        ext = Path(args.input_csv).suffix or ".csv"
        out_name = f"{stem}_{args.task}_{args.model}{ext}"

    if args.output_subdir:
        out_dir = proc_path(args.dataset, args.output_subdir)
    else:
        out_dir = input_csv_path.parent

    output_csv_path = out_dir / out_name

    if not input_csv_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_csv_path}\n"
            f"(dataset='{args.dataset}', input-csv='{args.input_csv}')"
        )

    selected_model_config = models[args.model]

    if args.num_samples:
        print(
            f"\n>>> TEST RUN: task='{args.task}', {args.num_samples} samples, "
            f"model='{args.model}', dataset='{args.dataset}' <<<"
        )
    else:
        print(
            f"\n>>> FULL RUN: task='{args.task}', all samples, "
            f"model='{args.model}', dataset='{args.dataset}' <<<"
        )

    secure_local_translation(
        csv_path=input_csv_path,
        output_path=output_csv_path,
        model_config=selected_model_config,
        text_column=args.text_column,
        num_samples=args.num_samples,
        task=args.task,
    )

    print("\n--- RUN COMPLETE ---")


if __name__ == "__main__":
    main()

    
# python local_translator.py \
#     --dataset nde \
#     --input-csv NDE_reflection_reports.csv \
#     --text-column reflection_answer \
#     --model llama \
#     --task translate \
#     --num-samples 5 \
#     --output-subdir translated
