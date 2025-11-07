# check_translations.py

import pandas as pd
from mosaic.path_utils import proc_path


def word_count(text) -> int:
    """Simple whitespace-based word count."""
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    return len(text.split())


def main():
    # ------------------------------------------------------------------
    # CONFIG: adjust these for other datasets / runs
    # ------------------------------------------------------------------
    dataset = "nde"  # logical dataset name (used with proc_path)
    check_sentence_translations = True  # set True to check sentence-level translations
    if check_sentence_translations:
        base_name = "NDE_reflection_reports_translate_llama_sentence"
    else:
        base_name = "NDE_reflection_reports_translate_llama"
    # This assumes files live under: <DATA_ROOT>/nde/translated/
    translated_rel = f"translated/{base_name}.csv"
    error_rel = f"translated/{base_name}_errors.csv"
    # ------------------------------------------------------------------

    # 1) Load translated CSV
    csv_path = proc_path(dataset, translated_rel)
    print("Loading translated CSV:", csv_path)
    df = pd.read_csv(csv_path)

    # 2) Identify columns
    text_col = "reflection_answer"
    trans_col = "phen_report_english"
    missing = [c for c in (text_col, trans_col) if c not in df.columns]

    if missing:
        print(
            "\n[WARN] Expected columns not found:",
            ", ".join(missing),
        )
        print("Columns present:", list(df.columns))
    else:
        # 2a) Show a few random reports with clear labels + word counts
        n_samples = min(5, len(df))
        print(f"\n===== SAMPLE OF {n_samples} TRANSLATED REPORTS =====")
        sample = df[[text_col, trans_col]].sample(n_samples, random_state=0)

        for i, (row_idx, row) in enumerate(sample.iterrows(), start=1):
            fr = row[text_col]
            en = row[trans_col]
            fr_wc = word_count(fr)
            en_wc = word_count(en)

            print(f"\n----- Report {i} (row index {row_idx}) -----")
            print(f"[LENGTH] French:  {fr_wc} words")
            print(f"[LENGTH] English: {en_wc} words")
            if fr_wc > 0:
                ratio = en_wc / fr_wc
                print(f"[RATIO]  EN/FR:  {ratio:.2f}")
            print("\n[FRENCH]")
            print(fr)
            print("\n[ENGLISH]")
            print(en)

        # 2b) Overall length stats (optional but useful)
        print("\n===== OVERALL LENGTH STATS =====")
        fr_wc_all = df[text_col].apply(word_count)
        en_wc_all = df[trans_col].apply(word_count)

        # avoid division by zero
        ratio_all = en_wc_all.divide(fr_wc_all.replace({0: pd.NA}))

        print(f"Total reports: {len(df)}")
        print(f"Average French length (words):  {fr_wc_all.mean():.1f}")
        print(f"Average English length (words): {en_wc_all.mean():.1f}")
        valid_ratio = ratio_all.dropna()
        if len(valid_ratio) > 0:
            print(f"Average EN/FR ratio:          {valid_ratio.mean():.2f}")
            print(f"Median EN/FR ratio:           {valid_ratio.median():.2f}")
        else:
            print("No valid EN/FR ratio (all French lengths were zero).")

    # 3) Show rows where translation failed (if any)
    if trans_col in df.columns:
        mask_err = df[trans_col].str.contains("Error during translation", na=False)
        num_err = mask_err.sum()
        print(f"\nRows with 'Error during translation': {num_err}")

        if num_err:
            err_df = df.loc[mask_err, [text_col, trans_col]]
            print("\n===== EXAMPLES OF REPORTS WITH TRANSLATION ERRORS =====")
            for i, (row_idx, row) in enumerate(err_df.head().iterrows(), start=1):
                print(f"\n----- ERROR Report {i} (row index {row_idx}) -----")
                print("[FRENCH]")
                print(row[text_col])
                print("\n[ENGLISH / STATUS]")
                print(row[trans_col])

    # 4) Try to load and show error log CSV (if present)
    err_path = proc_path(dataset, error_rel)
    try:
        err_df = pd.read_csv(err_path)
        print("\nError log file found:", err_path)
        print(f"Total error records: {len(err_df)}")
        print("\nFirst few error records:")
        print(err_df.head().to_string(index=False))
    except FileNotFoundError:
        print(
            "\nNo error log found (no row-level exceptions were recorded, "
            "or you haven't run the latest translator with logging yet)."
        )


if __name__ == "__main__":
    main()
