"""CLI: build the sequence-of-graphs feature store for one dataset (Section 7.4
steps 1-9). Runs the ingestion loader -> windowing -> sharded parquet, per
split, and prints a data-quality report (Section 8 checkpoint 5).

Usage:
    python -m src.features.build_dataset --dataset cic2017 --config configs/default.yaml
    python -m src.features.build_dataset --dataset ctu13
Optional ablation overrides: --delta-t 10 --overlap 0.5 --W 20 --n-max 128
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from src.config import add_config_args, load_config_from_args
from src.features.packet_features import PacketFeatureIndex
from src.features.schema import EDGE_FEATURE_DIM, NODE_FEATURE_DIM
from src.features.windowing import build_windows
from src.ingestion import ctu13_loader, flow_csv_loader

_SPLITS = ["train", "val", "test"]


def _split_path(processed_dir: str, dataset: str, split: str) -> str:
    return os.path.join(processed_dir, f"{dataset}_{split}.parquet")


def _wednesday_capture_start(processed_dir: str) -> float | None:
    """Global Wednesday flow t_min (epoch seconds) = packet-window-0 origin.

    Read from the full clean parquet so the origin is independent of which
    time-split a window falls in. Uses the same Timestamp parse as the loader.
    """
    clean = os.path.join(processed_dir, "cic2017_clean.parquet")
    if not os.path.exists(clean):
        return None
    cols = pd.read_parquet(clean, columns=["Timestamp", "source_file"])
    wed = cols[cols["source_file"].astype(str).str.contains("Wednesday", case=False)]
    if wed.empty:
        return None
    return float(pd.to_datetime(wed["Timestamp"]).astype("int64").min() / 1e9)


def _load_tidy(dataset: str, split_path: str, cfg):
    if dataset == "cic2017":
        return flow_csv_loader.load(split_path, cfg.paths.mitre_mapping, spread_minutes=True)
    if dataset == "ctu13":
        return ctu13_loader.load(split_path, drop_background=True)
    raise NotImplementedError(
        f"dataset '{dataset}' has no timestamped windowing path. "
        "NF-UNSW-NB15-v2 is cross-dataset-eval only (no timestamps/PCAP).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GT-RSSM window feature store.")
    add_config_args(parser)
    parser.add_argument("--dataset", required=True, choices=["cic2017", "ctu13"])
    parser.add_argument("--splits", nargs="+", default=_SPLITS, choices=_SPLITS)
    parser.add_argument("--seqs-per-shard", type=int, default=200)
    parser.add_argument("--background", action="store_true",
                        help="ctu13 only: window the unlabeled Background pool "
                             "into windows/background for the Stage-1 pool.")
    parser.add_argument("--skip-splits", action="store_true",
                        help="do not (re)window train/val/test; useful with "
                             "--background to add only the Stage-1 pool.")
    args = parser.parse_args()

    cfg = load_config_from_args(args)
    processed = getattr(cfg.paths, f"{args.dataset}_processed_dir")

    packet_index = None
    packet_capture_start = None
    if args.dataset == "cic2017" and os.path.exists(
            os.path.join(processed, "cic2017_wednesday_packet_features.parquet")):
        packet_index = PacketFeatureIndex.from_parquet(
            os.path.join(processed, "cic2017_wednesday_packet_features.parquet"),
            cfg.data.delta_t_seconds)
        packet_capture_start = _wednesday_capture_start(processed)

    windows_root = os.path.join(processed, "windows")
    # merge into any existing manifest so a --background-only run keeps prior reports
    manifest_path = os.path.join(windows_root, "manifest.json")
    reports = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                reports = (json.load(fh) or {}).get("reports", {})
        except (json.JSONDecodeError, OSError):
            reports = {}

    splits = [] if args.skip_splits else args.splits
    for split in splits:
        split_path = _split_path(processed, args.dataset, split)
        if not os.path.exists(split_path):
            print(f"[skip] {split}: {split_path} not found")
            continue
        tidy = _load_tidy(args.dataset, split_path, cfg)
        out_dir = os.path.join(windows_root, split)
        report = build_windows(tidy, cfg, args.dataset, out_dir,
                               packet_index=packet_index,
                               packet_capture_start=packet_capture_start,
                               seqs_per_shard=args.seqs_per_shard)
        reports[split] = report
        print(f"\n=== {args.dataset} / {split} ===")
        for k, v in report.items():
            print(f"  {k}: {v}")

    if args.background:
        if args.dataset != "ctu13":
            raise SystemExit("--background is only defined for ctu13")
        pool_path = os.path.join(processed, "ctu13_background_pool.parquet")
        if not os.path.exists(pool_path):
            raise SystemExit(f"background pool not found: {pool_path}")
        tidy = ctu13_loader.load_background(pool_path)
        out_dir = os.path.join(windows_root, "background")
        report = build_windows(tidy, cfg, args.dataset, out_dir,
                               packet_index=None, packet_capture_start=None,
                               seqs_per_shard=args.seqs_per_shard)
        reports["background"] = report
        print(f"\n=== {args.dataset} / background ===")
        for k, v in report.items():
            print(f"  {k}: {v}")

    manifest = {
        "dataset": args.dataset,
        "n_max_nodes": cfg.data.n_max_nodes,
        "edge_feature_dim": EDGE_FEATURE_DIM,
        "node_feature_dim": NODE_FEATURE_DIM,
        "sequence_length_W": cfg.data.sequence_length_W,
        "delta_t_seconds": cfg.data.delta_t_seconds,
        "window_overlap": cfg.data.window_overlap,
        "n_stages": len(cfg.stage_order),
        "stage_order": cfg.stage_order,
        "reports": reports,
    }
    os.makedirs(windows_root, exist_ok=True)
    with open(os.path.join(windows_root, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nwrote manifest -> {os.path.join(windows_root, 'manifest.json')}")


if __name__ == "__main__":
    main()
