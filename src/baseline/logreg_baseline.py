"""Logistic Regression baseline (Section 7.7).

One row per (host-pair, window) — the flattened 49-dim edge feature vector,
no graph and no temporal structure — predicting the binary infiltration target
(edge is non-Benign). ``class_weight='balanced'`` handles the heavy class
imbalance. Trained FIRST (Section 8 step 6) as the sanity floor the world model
must beat, and its numbers are written to ``data/processed/eval_results.md``.

This is deliberately the simplest defensible benchmark the brief asks for; it
also produces the feature matrices the XGBoost surrogate (7.8.3) reuses.
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import add_config_args, load_config_from_args
from src.evaluation.metrics import (
    METRICS_TABLE_HEADER,
    binary_metrics,
    format_metrics_row,
)
from src.features.packet_features import PacketFeatureIndex
from src.features.schema import EDGE_FEATURE_COLUMNS
from src.features.windowing import build_edge_table
from src.ingestion import ctu13_loader, flow_csv_loader

_EVAL_RESULTS = "data/processed/eval_results.md"


def _load_tidy(dataset: str, split_path: str, cfg):
    if dataset == "cic2017":
        return flow_csv_loader.load(split_path, cfg.paths.mitre_mapping, spread_minutes=True)
    if dataset == "ctu13":
        return ctu13_loader.load(split_path, drop_background=True)
    raise NotImplementedError(dataset)


def _edge_table_for_split(dataset: str, split: str, cfg, packet_index, capture_start):
    processed = getattr(cfg.paths, f"{dataset}_processed_dir")
    path = os.path.join(processed, f"{dataset}_{split}.parquet")
    if not os.path.exists(path):
        return None
    tidy = _load_tidy(dataset, path, cfg)
    return build_edge_table(tidy, cfg, dataset, packet_index=packet_index,
                            packet_capture_start=capture_start)


def _xy(table: pd.DataFrame):
    X = table[EDGE_FEATURE_COLUMNS].to_numpy(dtype=float)
    y = table["infil"].to_numpy(dtype=int)
    return X, y


def train_and_eval(dataset: str, cfg, max_iter: int = 1000) -> dict:
    """Fit on the train edge table, evaluate on val + test."""
    processed = getattr(cfg.paths, f"{dataset}_processed_dir")
    packet_index = None
    capture_start = None
    pk_path = os.path.join(processed, "cic2017_wednesday_packet_features.parquet")
    if dataset == "cic2017" and os.path.exists(pk_path):
        packet_index = PacketFeatureIndex.from_parquet(pk_path, cfg.data.delta_t_seconds)
        from src.features.build_dataset import _wednesday_capture_start
        capture_start = _wednesday_capture_start(processed)

    train_tbl = _edge_table_for_split(dataset, "train", cfg, packet_index, capture_start)
    if train_tbl is None or train_tbl.empty:
        raise RuntimeError(f"no train edge table for {dataset}")
    X_tr, y_tr = _xy(train_tbl)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(class_weight="balanced", max_iter=max_iter, n_jobs=-1)),
    ])
    clf.fit(X_tr, y_tr)

    results = {"dataset": dataset, "train_rows": int(len(y_tr)),
               "train_pos_rate": float(y_tr.mean()), "splits": {}}
    for split in ("val", "test"):
        tbl = _edge_table_for_split(dataset, split, cfg, packet_index, capture_start)
        if tbl is None or tbl.empty:
            continue
        X, y = _xy(tbl)
        prob = clf.predict_proba(X)[:, 1]
        results["splits"][split] = binary_metrics(y, prob)

    os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)
    model_path = os.path.join(cfg.paths.checkpoint_dir, f"logreg_{dataset}.pkl")
    with open(model_path, "wb") as fh:
        pickle.dump(clf, fh)
    results["model_path"] = model_path
    return results


def _write_eval_results(all_results: list[dict]) -> None:
    os.makedirs(os.path.dirname(_EVAL_RESULTS), exist_ok=True)
    lines = ["# Evaluation Results", "",
             "## Logistic Regression baseline (Section 7.7)", "",
             "Per-edge infiltration detection (one row per host-pair per window, "
             "flattened 49-dim edge vector, `class_weight='balanced'`).", ""]
    for res in all_results:
        lines.append(f"### {res['dataset']} "
                     f"(train rows={res['train_rows']:,}, "
                     f"pos rate={res['train_pos_rate']:.4f})")
        lines.append("")
        lines.append(METRICS_TABLE_HEADER)
        for split, m in res["splits"].items():
            lines.append(format_metrics_row(f"LR ({split})", m))
        lines.append("")
    for out_p in (_EVAL_RESULTS, os.path.join("metrics", "eval_results.md")):
        os.makedirs(os.path.dirname(out_p), exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the LR infiltration baseline.")
    add_config_args(parser)
    parser.add_argument("--datasets", nargs="+", default=["cic2017", "ctu13"],
                        choices=["cic2017", "ctu13"])
    parser.add_argument("--max-iter", type=int, default=1000)
    args = parser.parse_args()
    cfg = load_config_from_args(args)

    all_results = []
    for ds in args.datasets:
        print(f"\n=== training LR baseline: {ds} ===")
        res = train_and_eval(ds, cfg, max_iter=args.max_iter)
        for split, m in res["splits"].items():
            print(f"  {split}: P={m['precision']:.4f} R={m['recall']:.4f} "
                  f"F1={m['f1']:.4f} ROC-AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.4f}")
        all_results.append(res)

    _write_eval_results(all_results)
    print(f"\nwrote {_EVAL_RESULTS}")


if __name__ == "__main__":
    main()
