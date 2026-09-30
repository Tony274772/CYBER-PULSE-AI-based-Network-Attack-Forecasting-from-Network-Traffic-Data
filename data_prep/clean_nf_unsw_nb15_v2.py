"""
Clean the optional NF-UNSW-NB15-v2 NetFlow CSV. Used only for the cross-dataset
generalization / surprise-AUC test — safe to skip this dataset under time pressure.

Usage:
    python clean_nf_unsw_nb15_v2.py --input_csv data/raw/nf_unsw_nb15_v2/NF-UNSW-NB15-v2.csv \
        --output_dir data/processed/nf_unsw_nb15_v2
"""
import argparse
import os

import numpy as np
import pandas as pd

LABEL_MAP = {
    "Benign": "Benign", "Reconnaissance": "Reconnaissance", "Exploits": "Initial_Access",
    "Backdoor": "Initial_Access", "Shellcode": "Initial_Access", "Analysis": "Initial_Access",
    "Fuzzers": "Initial_Access", "Worms": "Lateral_Movement", "DoS": "Impact_DoS",
    "Generic": "Impact_DoS",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv, low_memory=False)
    df.columns = df.columns.str.strip()

    label_col = "Attack" if "Attack" in df.columns else ("Label" if "Label" in df.columns else None)
    if label_col is None:
        raise ValueError(
            "Could not find an 'Attack' or 'Label' column — open the CSV header and "
            "adjust label_col in this script to match"
        )

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=numeric_cols)

    df["mapped_stage"] = df[label_col].map(LABEL_MAP)
    unmapped = df[df["mapped_stage"].isna()]
    if len(unmapped):
        print(f"WARNING: {len(unmapped)} rows have an unmapped {label_col} value; sample:")
        print(unmapped[label_col].drop_duplicates().head(10).to_string(index=False))

    df["dataset"] = "nf_unsw_nb15_v2"
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "nf_unsw_nb15_v2_clean.parquet")
    df.to_parquet(out_path, index=False)

    print(f"Total cleaned rows: {len(df):,}")
    print(df["mapped_stage"].value_counts())
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
