"""
Clean & label CTU-13 bidirectional NetFlow (.binetflow) files.

Input : the root folder where the extracted CTU-13-Dataset tarball lives
         (recursively globbed for *.binetflow / *.csv — folder names differ
         slightly across mirrors, so nothing is hardcoded).
Output: <output_dir>/ctu13_labeled.parquet          (Benign + mapped-malicious rows)
        <output_dir>/ctu13_background_pool.parquet  (Background rows — unlabeled,
                                                      usable only for Stage-1 pretraining)

Usage:
    python clean_ctu13.py --input_dir data/raw/ctu13 --output_dir data/processed/ctu13
"""
import argparse
import glob
import os
import sys

import pandas as pd
from tqdm import tqdm

# Applied in order; first match wins. Mirrors the ctu13_label_substring_rules
# in mitre_mapping/attack_stage_mapping.json from the build spec.
CTU13_RULES = [
    {"contains": "normal", "stage": "Benign"},
    {"contains": "background", "stage": None},  # excluded from supervised heads
    {"contains_any": ["cc", "botnet"], "excludes": ["scan"], "stage": "Command_And_Control"},
    {"contains_any": ["scan", "portscan"], "stage": "Reconnaissance"},
]


def map_ctu13_label(raw_label: str):
    label = str(raw_label).lower()
    for rule in CTU13_RULES:
        if "contains" in rule and rule["contains"] in label:
            return rule["stage"]
        if "contains_any" in rule and any(s in label for s in rule["contains_any"]):
            if "excludes" in rule and any(s in label for s in rule["excludes"]):
                continue
            return rule["stage"]
    return "UNMATCHED"  # flagged, not silently mis-bucketed


def clean_one_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    expected = {"StartTime", "Dur", "Proto", "SrcAddr", "Sport", "Dir",
                "DstAddr", "Dport", "State", "TotPkts", "TotBytes", "SrcBytes", "Label"}
    missing = expected - set(df.columns)
    if missing:
        print(f"  WARNING {os.path.basename(path)}: missing expected columns {missing} — "
              f"check this file's format before trusting its output")

    df["StartTime"] = pd.to_datetime(df["StartTime"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["StartTime"])
    if len(df) != before:
        print(f"  {os.path.basename(path)}: dropped {before - len(df)} rows with unparsable StartTime")

    df["mapped_stage"] = df["Label"].apply(map_ctu13_label)
    df["scenario_file"] = os.path.basename(path)
    df["dataset"] = "ctu13"
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.binetflow"), recursive=True))
    if not files:
        # some mirrors ship CTU-13 flows as plain .csv instead of .binetflow
        files = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.csv"), recursive=True))
    if not files:
        sys.exit(f"No .binetflow or .csv files found under {args.input_dir}")

    print(f"Found {len(files)} scenario files")
    frames = [clean_one_file(f) for f in tqdm(files, desc="cleaning")]
    combined = pd.concat(frames, ignore_index=True)

    unmatched = combined[combined["mapped_stage"] == "UNMATCHED"]
    if len(unmatched):
        print(f"\nWARNING: {len(unmatched)} rows did not match any CTU-13 label rule. Sample raw labels:")
        print(unmatched["Label"].drop_duplicates().head(10).to_string(index=False))
        print("Extend CTU13_RULES in this script with these substrings if they're a meaningful fraction.")

    background = combined[combined["mapped_stage"].isna()]
    labeled = combined[combined["mapped_stage"].notna() & (combined["mapped_stage"] != "UNMATCHED")]

    os.makedirs(args.output_dir, exist_ok=True)
    labeled_path = os.path.join(args.output_dir, "ctu13_labeled.parquet")
    background_path = os.path.join(args.output_dir, "ctu13_background_pool.parquet")
    labeled.sort_values("StartTime").to_parquet(labeled_path, index=False)
    background.sort_values("StartTime").to_parquet(background_path, index=False)

    print("\n=== Data quality report: CTU-13 ===")
    print(f"Total rows                          : {len(combined):,}")
    print(f"Labeled (Benign/Malicious) rows      : {len(labeled):,}")
    print(f"Background pool rows (Stage-1 only)  : {len(background):,}")
    print("Mapped-stage distribution (labeled):")
    print(labeled["mapped_stage"].value_counts())
    print(f"\nSaved labeled -> {labeled_path}")
    print(f"Saved background pool -> {background_path}")


if __name__ == "__main__":
    main()
