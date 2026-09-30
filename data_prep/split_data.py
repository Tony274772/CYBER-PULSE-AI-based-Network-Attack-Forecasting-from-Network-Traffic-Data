"""
Time-aware train/val/test split for the cleaned datasets. Never splits randomly
by row — that would leak a moment in time (or a botnet run) across splits and
inflate every downstream metric.

CIC-IDS2017: sorted by timestamp, split 70/15/15 *within each source_file*
             (so every split still contains every day's attack types, but no
             window of time is shared across splits).
CTU-13     : split by whole scenario file (one botnet run = one indivisible
             unit) so a model never sees part of a scenario's traffic in train
             and another part of the SAME scenario in test.

Usage:
    python split_data.py --dataset cic2017 --input data/processed/cic2017/cic2017_clean.parquet \
        --output_dir data/processed/cic2017
    python split_data.py --dataset ctu13 --input data/processed/ctu13/ctu13_labeled.parquet \
        --output_dir data/processed/ctu13
"""
import argparse
import os

import pandas as pd


def split_time_chunks(df: pd.DataFrame, time_col: str, group_col: str,
                       train_frac=0.70, val_frac=0.15):
    parts = {"train": [], "val": [], "test": []}
    for _, g in df.groupby(group_col, sort=False):
        g = g.sort_values(time_col)
        n = len(g)
        i1 = int(n * train_frac)
        i2 = int(n * (train_frac + val_frac))
        parts["train"].append(g.iloc[:i1])
        parts["val"].append(g.iloc[i1:i2])
        parts["test"].append(g.iloc[i2:])
    return {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}


def split_by_group(df: pd.DataFrame, group_col: str, train_frac=0.70, val_frac=0.15, seed=42):
    groups = sorted(df[group_col].unique())
    shuffled = pd.Series(groups).sample(frac=1.0, random_state=seed).tolist()
    n = len(shuffled)
    i1 = int(n * train_frac)
    i2 = int(n * (train_frac + val_frac))
    train_groups, val_groups, test_groups = shuffled[:i1], shuffled[i1:i2], shuffled[i2:]
    return {
        "train": df[df[group_col].isin(train_groups)].reset_index(drop=True),
        "val": df[df[group_col].isin(val_groups)].reset_index(drop=True),
        "test": df[df[group_col].isin(test_groups)].reset_index(drop=True),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cic2017", "ctu13", "nf_unsw_nb15_v2"], required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.input)

    if args.dataset == "cic2017":
        splits = split_time_chunks(df, time_col="Timestamp", group_col="source_file")
    elif args.dataset == "ctu13":
        splits = split_by_group(df, group_col="scenario_file")
    else:  # nf_unsw_nb15_v2 has no natural time/group column -> simple row-order chunking
        df = df.reset_index(drop=True)
        n = len(df)
        splits = {
            "train": df.iloc[: int(n * 0.70)],
            "val": df.iloc[int(n * 0.70): int(n * 0.85)],
            "test": df.iloc[int(n * 0.85):],
        }

    os.makedirs(args.output_dir, exist_ok=True)
    for split_name, split_df in splits.items():
        out_path = os.path.join(args.output_dir, f"{args.dataset}_{split_name}.parquet")
        split_df.to_parquet(out_path, index=False)
        print(f"{split_name}: {len(split_df):,} rows -> {out_path}")


if __name__ == "__main__":
    main()
