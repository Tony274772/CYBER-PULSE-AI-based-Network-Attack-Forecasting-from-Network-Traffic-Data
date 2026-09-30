"""
Clean & standardize raw CIC-IDS2017 flow CSVs (GeneratedLabelledFlows / MachineLearningCVE).

Input : a folder containing the 8 raw CSV files (unzipped, flat).
Output: <output_dir>/cic2017_clean.parquet + a data-quality report on stdout.

Usage:
    python clean_cic2017.py --input_dir data/raw/cic2017/csv --output_dir data/processed/cic2017
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm


def clean_one_file(path: str) -> pd.DataFrame:
    # Most CIC-IDS2017 files are UTF-8, but some distributed copies contain
    # Windows-1252 characters (for example 0x96) in text fields.  Do not use
    # ``errors="ignore"`` here: that would silently alter labels/values.
    # Instead, retry with a lossless single-byte encoding when UTF-8 fails.
    try:
        df = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(path, low_memory=False, encoding="cp1252")
            print(f"  {os.path.basename(path)}: read using cp1252 encoding")
        except UnicodeDecodeError:
            # latin-1 maps every byte, so this is a safe final fallback for
            # legacy CSV exports that are neither UTF-8 nor Windows-1252.
            df = pd.read_csv(path, low_memory=False, encoding="latin1")
            print(f"  {os.path.basename(path)}: read using latin1 encoding")

    # 1. strip whitespace from column headers (CIC-IDS2017's well-known gotcha,
    #    e.g. ' Destination Port' with a leading space)
    df.columns = df.columns.str.strip()

    if "Label" not in df.columns:
        raise ValueError(f"{path}: no 'Label' column found after stripping headers — inspect this file")

    # 2. drop embedded duplicate header rows (a known artifact of how the CSVs
    #    were concatenated upstream)
    before = len(df)
    df = df[df["Label"] != "Label"]
    if len(df) != before:
        print(f"  {os.path.basename(path)}: dropped {before - len(df)} embedded header rows")

    # 3. replace inf/-inf with NaN (Flow Bytes/s, Flow Packets/s are Infinity
    #    for zero-duration flows), then drop rows that still have NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna(subset=numeric_cols)
    dropped = before - len(df)
    if dropped:
        print(f"  {os.path.basename(path)}: dropped {dropped} rows with inf/NaN numeric values")

    # 4. parse Timestamp, coercing failures to NaT and dropping them
    if "Timestamp" in df.columns:
        before = len(df)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Timestamp"])
        dropped = before - len(df)
        if dropped:
            print(f"  {os.path.basename(path)}: dropped {dropped} rows with unparsable timestamps")
        # sanity check: documented capture window is July 3-7, 2017
        bad_range = df[(df["Timestamp"] < "2017-06-25") | (df["Timestamp"] > "2017-07-10")]
        if len(bad_range):
            print(f"  WARNING {os.path.basename(path)}: {len(bad_range)} rows fall outside the "
                  f"documented July 3-7 2017 window — double-check the dayfirst assumption on this file")

    df["source_file"] = os.path.basename(path)
    df["dataset"] = "cic2017"
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.csv")))
    if not files:
        sys.exit(f"No CSV files found under {args.input_dir}")

    print(f"Found {len(files)} CSV files")
    frames = [clean_one_file(f) for f in tqdm(files, desc="cleaning")]

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("Timestamp").reset_index(drop=True)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "cic2017_clean.parquet")
    combined.to_parquet(out_path, index=False)

    print("\n=== Data quality report: CIC-IDS2017 ===")
    print(f"Total cleaned rows : {len(combined):,}")
    print(f"Date range         : {combined['Timestamp'].min()} to {combined['Timestamp'].max()}")
    print("Label distribution :")
    print(combined["Label"].value_counts())
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
