# local_translator.py

import os
import json
import argparse
from pathlib import Path
import re
from typing import List

import pandas as pd
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from mosaic.path_utils import CFG, proc_path  # MOSAIC helpers

# --- NLTK Punkt sentence tokenizer ---
try:
    import nltk
    from nltk.data import load as nltk_load
except ImportError as e:
    raise ImportError(
        "NLTK is required for sentence tokenization. Install it with:\n"
        "    pip install nltk\n"
        "and download the Punkt models with:\n"
        "    python -m nltk.downloader punkt"
    ) from e

_PUNKT_CACHE = {}


def get_punkt_tokenizer(lang: str = "english"):
    """
    Lazily load and cache the NLTK PunktSentenceTokenizer for a given language.

    Common options:
      - 'english' (DEFAULT)
      - 'french'
      - 'german', 'spanish', etc., if installed.

    Make sure the corresponding Punkt model is installed via:
        python -m nltk.downloader punkt
    which installs multiple languages, including english/french.
    """
    lang = lang.lower()
    if lang not in _PUNKT_CACHE:
        try:
            # NLTK stores these as tokenizers/punkt/<lang>.pickle
            tokenizer = nltk_load(f"tokenizers/punkt/{lang}.pickle")
        except LookupError as e:
            raise RuntimeError(
                f"NLTK Punkt model for language '{lang}' not found.\n"
                f"Run: python -m nltk.downloader punkt"
            ) from e
        _PUNKT_CACHE[lang] = tokenizer
    return _PUNKT_CACHE[lang]


# ---------- PROMPTS ----------

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

# Sentence-aligned translation prompt (strict JSON contract)
SENTENCE_TRANSLATION_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a literal, sentence-aligned translator. Translate each input sentence into English with NO summarising, NO omissions, and NO merging or splitting. Preserve all information and sentence boundaries.
Return ONLY a valid JSON array of strings with exactly the same number of elements as the input sentences. Do not add any commentary, explanations, or labels.<|eot_id|><|start_header_id|>user<|end_header_id|>
Translate each of the following {n} sentences into English. Return ONLY a JSON array with {n} strings, one per sentence, in the same order.

Sentences:
{numbered_sentences}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


# ---------- HELPERS ----------

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


def sentence_split(text: str, sent_lang: str = "english") -> List[str]:
    """
    Sentence splitter using NLTK Punkt.
    - sent_lang controls which language model to use (e.g. 'english', 'french').
    - DEFAULT = 'english'.
    - This is more robust than a simple regex and handles 'Dr. Dupont', 'M. X', etc. better.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    tokenizer = get_punkt_tokenizer(sent_lang)
    sentences = tokenizer.tokenize(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _format_numbered_sentences(sents: List[str]) -> str:
    return "\n".join(f"{i+1}. {s}" for i, s in enumerate(sents))


def translate_sentences_batch(
    sents: List[str],
    llm: Llama,
    max_ctx: int,
    safety_margin: int = 64,
) -> List[str]:
    """
    Translate a small list of sentences using the strict JSON-array prompt.
    Ensures the prompt fits the context; uses all remaining tokens for generation.
    """
    # Build prompt
    numbered = _format_numbered_sentences(sents)
    prompt = SENTENCE_TRANSLATION_PROMPT.format(n=len(sents), numbered_sentences=numbered)

    # Token budget
    prompt_tokens = len(llm.tokenize(prompt.encode("utf-8")))
    avail_for_gen = max_ctx - prompt_tokens - safety_margin
    if avail_for_gen <= 0:
        raise RuntimeError(
            f"Prompt too long for context window (tokens={prompt_tokens}, max={max_ctx}). "
            f"Reduce batch size."
        )

    # Call model
    resp = llm(
        prompt=prompt,
        max_tokens=avail_for_gen,         # use all remaining room; no arbitrary cap
        stop=["<|eot_id|>"],
        echo=False,
        temperature=0,                    # deterministic, reduce paraphrase/summarising
        top_p=1.0,
    )
    raw = resp["choices"][0]["text"].strip()

    # Parse JSON
    out = _extract_json_array(raw)
    if not isinstance(out, list) or len(out) != len(sents):
        raise ValueError(
            f"Model returned invalid JSON array: expected {len(sents)} items, got {type(out)} / "
            f"{len(out) if isinstance(out, list) else 'NA'}."
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
    Strict literal translation:
    - Split into sentences via NLTK Punkt (default 'english').
    - Translate in small batches (JSON array, same length).
    - If a batch is too big for context, shrink it until it fits.
    - Reassemble translated sentences in order.
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
        # Start with target_batch_size, then shrink if needed to fit context
        batch_end = min(i + target_batch_size, N)
        while True:
            batch = sents[i:batch_end]
            try:
                translated = translate_sentences_batch(
                    batch, llm, max_ctx=max_ctx, safety_margin=safety_margin
                )
                results.extend(translated)
                i = batch_end
                break
            except RuntimeError:
                # Context overflow: reduce batch size
                if (batch_end - i) <= 1:
                    # Single sentence doesn't fit: it's truly enormous.
                    # As a last resort, hard-split the sentence into halves and retry.
                    long_sent = batch[0]
                    mid = max(len(long_sent) // 2, 200)
                    sub_a = long_sent[:mid]
                    sub_b = long_sent[mid:]
                    halves = [sub_a, sub_b]
                    try:
                        translated_halves = translate_sentences_batch(
                            halves, llm, max_ctx=max_ctx, safety_margin=safety_margin
                        )
                        results.extend(translated_halves)
                        i = batch_end
                        break
                    except Exception:
                        # Give up and mark it to inspect later
                        results.append(f"[UNTRANSLATED DUE TO CONTEXT] {long_sent[:200]}...")
                        i = batch_end
                        break
                else:
                    # shrink the batch and retry
                    batch_end -= 1
            except Exception:
                # Other errors (JSON parse, etc.) → try shrinking batch once; else log fallback
                if (batch_end - i) > 1:
                    batch_end -= 1
                    continue
                else:
                    # Single sentence failed → return placeholder with original
                    results.append(f"[TRANSLATION ERROR] {sents[i][:200]}...")
                    i += 1
                    break

    # Reassemble with spaces (we preserved sentence boundaries)
    return " ".join(results).strip()


# ---------- MAIN PIPELINE ----------

def secure_local_translation(
    csv_path,
    output_path,
    model_config: dict,
    text_column: str | None = None,
    num_samples: int | None = None,
    task: str = "translate",  # 'translate', 'preprocess', or 'both'
    sent_lang: str = "english",
):
    """
    Pipeline:
      - preprocess (clean) texts,
      - translate texts sentence-aligned (strict, no summarising),
      - or both (clean then translate).
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
    print(f"Sentence tokenizer language: {sent_lang}")
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
    error_log: List[dict] = []

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

    # --- model path ---
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
        n_ctx=4096,     # 4k context; sentence batching avoids overflow
        n_batch=256,
        verbose=False,
    )
    print("Model loaded successfully.")
    max_ctx = llm.n_ctx()
    safety_margin = 64

    # --- choose source series ---
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

    # If only preprocessing is requested, save and exit
    if do_preprocess and not do_translate:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_to_process.to_csv(output_path, index=False)
        print("\n--- PREPROCESSING ONLY: Complete ---")
        print(f"Saved cleaned data to {output_path}")
        print(df_to_process[[text_column, f"{text_column}_clean"]].head())
        return

    # --- TRANSLATION STEP ---
    if do_translate:
        print("\n>>> Translating texts (sentence-aligned) <<<")
        translations = []
        items = list(source_series.items())  # [(index, text), ...]
        for row_idx, text in tqdm(items, desc=f"Translating with {model_config['name']}"):
            if not isinstance(text, str) or not text.strip():
                translations.append("Not applicable (empty)")
                continue
            try:
                translated = translate_text_sentence_aligned(
                    text=text,
                    llm=llm,
                    max_ctx=max_ctx,
                    target_batch_size=5,   # 4–6 is a good sweet spot
                    safety_margin=safety_margin,
                    sent_lang=sent_lang,
                )
                translations.append(translated)
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
        pd.DataFrame(error_log).to_csv(error_log_path, index=False)
        print(f"\n[INFO] {len(error_log)} row(s) logged to {error_log_path}")


def main():
    # --- Model configurations ---
    models = {
        "llama": {
            "name": "Llama-3-8B-Instruct",
            "repo": "NousResearch/Meta-Llama-3-8B-Instruct-GGUF",
            "filename": "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
            # (Prompt is SENTENCE_TRANSLATION_PROMPT above.)
        },
    }

    parser = argparse.ArgumentParser(
        description="Preprocess and/or translate a CSV column to English (local, sentence-aligned, MOSAIC-aware)."
    )
    parser.add_argument("--dataset", required=True, help="Dataset key used with mosaic.proc_path (e.g., 'nde').")
    parser.add_argument("--input-csv", default="MPE_dataset.csv", help="Filename under proc_path(dataset).")
    parser.add_argument(
        "--output-name",
        default=None,
        help="Output filename. "
             "If omitted, uses '<input_stem>_<task>_<model>_sentence.csv'."
    )
    parser.add_argument(
        "--output-subdir",
        default=None,
        help="Subdirectory under proc_path(dataset) for outputs."
    )
    parser.add_argument(
        "--text-column",
        default="reflection_answer",
        help="Column to process; if missing/None, the first column is used."
    )
    parser.add_argument(
        "--model",
        choices=models.keys(),
        default="llama",
        help="Model key (GGUF config)."
    )
    parser.add_argument("--num-samples", type=int, default=None, help="Only process first N rows.")
    parser.add_argument(
        "--task",
        choices=["translate", "preprocess", "both"],
        default="translate",
        help="What to do."
    )
    parser.add_argument(
        "--sent-lang",
        default="english",
        help=(
            "Language code for Punkt sentence tokenizer (e.g. 'english', 'french'). "
            "DEFAULT = 'english'. This only affects how text is split into sentences, not translation."
        ),
    )

    args = parser.parse_args()

    print("MOSAIC config:")
    print("  DATA_ROOT =", CFG.get("data_root", "(not set)"))
    print("  BOX_ROOT  =", CFG.get("box_root", "(not set)"))
    print()

    input_csv_path = proc_path(args.dataset, args.input_csv)

    # --- OUTPUT NAME LOGIC (now with _sentence) ---
    if args.output_name:
        out_name = args.output_name
    else:
        stem = Path(args.input_csv).stem
        ext = Path(args.input_csv).suffix or ".csv"
        # Example: NDE_reflection_reports_translate_llama_sentence.csv
        out_name = f"{stem}_{args.task}_{args.model}_sentence{ext}"

    out_dir = proc_path(args.dataset, args.output_subdir) if args.output_subdir else input_csv_path.parent
    output_csv_path = out_dir / out_name

    if not input_csv_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_csv_path}\n"
            f"(dataset='{args.dataset}', input-csv='{args.input_csv}')"
        )

    selected_model_config = models[args.model]

    print(
        f"\n>>> {'TEST' if args.num_samples else 'FULL'} RUN:"
        f" task='{args.task}', model='{args.model}', dataset='{args.dataset}', "
        f"sent-lang='{args.sent_lang}' <<<"
    )

    secure_local_translation(
        csv_path=input_csv_path,
        output_path=output_csv_path,
        model_config=selected_model_config,
        text_column=args.text_column,
        num_samples=args.num_samples,
        task=args.task,
        sent_lang=args.sent_lang,
    )

    print("\n--- RUN COMPLETE ---")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# EXAMPLE USAGE (copy–paste into your terminal)
#
# Translate first 5 rows, sentence-aligned, using French punkt model
# (but still translating INTO English):
#
#   python local_translator.py \
#       --dataset nde \
#       --input-csv NDE_reflection_reports.csv \
#       --text-column reflection_answer \
#       --model llama \
#       --task translate \
#       --num-samples 5 \
#       --output-subdir translated \
#       --sent-lang french
#
# This will write something like:
#   <DATA_ROOT>/nde/translated/NDE_reflection_reports_translate_llama_sentence.csv
#
# For a full run, drop --num-samples.
# ----------------------------------------------------------------------


# python local_translator_sentences.py \
#     --dataset nde \
#     --input-csv NDE_reflection_reports.csv \
#     --text-column reflection_answer \
#     --model llama \
#     --task translate \
#     --num-samples 5 \
#     --output-subdir translated