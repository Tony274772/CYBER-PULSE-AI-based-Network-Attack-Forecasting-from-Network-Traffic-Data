"""PyTorch dataset over the windowed graph store (Section 7.6.1 data path).

Reads the sharded parquet written by ``features/build_dataset.py`` and rebuilds,
per sequence, the dense tensors the model consumes:
  node_feats      [W, N, d_n]
  edge_feats      [W, N, N, d_e]
  adjacency_mask  [W, N, N]
  node_valid_mask [W, N]
  infil_target    [W]
  stage_target    [W]

Storage is sparse (1-D list columns per window); dense reconstruction happens
here, on the fly, so the on-disk store and RAM stay small. Shards are loaded
lazily with a small LRU cache — each sequence lives entirely within one shard
(the writer flushes only on sequence boundaries), so no sequence is split.
"""
from __future__ import annotations

import glob
import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.features.schema import EDGE_FEATURE_COLUMNS, NODE_FEATURES
from src.features.windowing import reconstruct_dense


class WindowSequenceDataset(Dataset):
    def __init__(self, windows_root: str, split: str, n_max: int, W: int):
        self.split_dir = os.path.join(windows_root, split)
        self.n_max = n_max
        self.W = W
        self.d_n = len(NODE_FEATURES)
        self.d_e = len(EDGE_FEATURE_COLUMNS)
        self._shards = sorted(glob.glob(os.path.join(self.split_dir, "*.parquet")))
        if not self._shards:
            raise FileNotFoundError(f"no shards under {self.split_dir}")

        # global index: (shard_path, seq_id), read only the seq_id column
        self._index: list[tuple[str, int]] = []
        for sp in self._shards:
            seq_ids = pd.read_parquet(sp, columns=["seq_id"])["seq_id"].unique()
            self._index.extend((sp, int(s)) for s in sorted(seq_ids))

    def __len__(self) -> int:
        return len(self._index)

    @lru_cache(maxsize=8)
    def _load_shard(self, shard_path: str) -> dict:
        df = pd.read_parquet(shard_path)
        return {sid: g.sort_values("step") for sid, g in df.groupby("seq_id", sort=False)}

    def __getitem__(self, idx: int) -> dict:
        shard_path, seq_id = self._index[idx]
        seq = self._load_shard(shard_path)[seq_id]

        node = np.zeros((self.W, self.n_max, self.d_n), dtype=np.float32)
        edge = np.zeros((self.W, self.n_max, self.n_max, self.d_e), dtype=np.float32)
        adj = np.zeros((self.W, self.n_max, self.n_max), dtype=np.float32)
        valid = np.zeros((self.W, self.n_max), dtype=np.float32)
        infil = np.zeros((self.W,), dtype=np.float32)
        stage = np.zeros((self.W,), dtype=np.int64)

        for _, row in seq.iterrows():
            w = int(row["step"])
            d = reconstruct_dense(row, self.n_max)
            node[w] = d["node_feats"]
            edge[w] = d["edge_feats"]
            adj[w] = d["adjacency_mask"]
            valid[w] = d["node_valid_mask"]
            infil[w] = d["infil_label"]
            stage[w] = d["stage_label"]

        return {
            "node_feats": torch.from_numpy(node),
            "edge_feats": torch.from_numpy(edge),
            "adjacency_mask": torch.from_numpy(adj),
            "node_valid_mask": torch.from_numpy(valid),
            "infil_target": torch.from_numpy(infil),
            "stage_target": torch.from_numpy(stage),
            "seq_id": seq_id,
        }


def load_manifest(windows_root: str) -> dict:
    with open(os.path.join(windows_root, "manifest.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


def compute_feature_stats(shard_paths: list[str], d_n: int, d_e: int,
                          std_floor: float = 1e-6) -> dict:
    """Per-feature mean/std over the *real* (stored) edges and nodes, streamed
    from the sparse shard columns (no densification). Fit on training shards
    only; consumed by :meth:`GTRSSM.set_feature_stats` to z-score inputs.

    Raw flow/packet features span many orders of magnitude (byte counts, rates),
    so un-normalized MSE reconstruction is numerically hopeless — this is the
    standard input standardization (same idea as the LR baseline's StandardScaler)
    and changes no architectural quantity.
    """
    n_sum = np.zeros(d_n); n_sqsum = np.zeros(d_n); n_cnt = 0
    e_sum = np.zeros(d_e); e_sqsum = np.zeros(d_e); e_cnt = 0
    for sp in shard_paths:
        df = pd.read_parquet(sp, columns=["node_feats_flat", "edge_feats_flat"])
        for arr in df["node_feats_flat"]:
            a = np.asarray(arr, dtype=np.float64).reshape(-1, d_n)
            if a.size:
                n_sum += a.sum(0); n_sqsum += (a * a).sum(0); n_cnt += a.shape[0]
        for arr in df["edge_feats_flat"]:
            a = np.asarray(arr, dtype=np.float64).reshape(-1, d_e)
            if a.size:
                e_sum += a.sum(0); e_sqsum += (a * a).sum(0); e_cnt += a.shape[0]
    n_cnt = max(n_cnt, 1); e_cnt = max(e_cnt, 1)
    node_mean = n_sum / n_cnt
    node_std = np.sqrt(np.maximum(n_sqsum / n_cnt - node_mean ** 2, 0.0))
    edge_mean = e_sum / e_cnt
    edge_std = np.sqrt(np.maximum(e_sqsum / e_cnt - edge_mean ** 2, 0.0))
    return {
        "node_mean": node_mean.tolist(),
        "node_std": np.maximum(node_std, std_floor).tolist(),
        "edge_mean": edge_mean.tolist(),
        "edge_std": np.maximum(edge_std, std_floor).tolist(),
        "n_nodes_seen": int(n_cnt), "n_edges_seen": int(e_cnt),
    }


def stage_class_weights(dataset: WindowSequenceDataset, n_stages: int) -> torch.Tensor:
    """Inverse-frequency class weights over window stage labels (Section 7.5.3
    mitre_head). Computed once from the given (training) split."""
    counts = np.zeros(n_stages, dtype=np.float64)
    for sp in dataset._shards:
        s = pd.read_parquet(sp, columns=["stage_label"])["stage_label"].to_numpy()
        for c in range(n_stages):
            counts[c] += int((s == c).sum())
    counts = np.maximum(counts, 1.0)
    inv = counts.sum() / (n_stages * counts)
    return torch.tensor(inv, dtype=torch.float32)


def sequence_max_stage(dataset: WindowSequenceDataset) -> np.ndarray:
    """Per-sequence most-severe window stage, aligned to ``dataset`` order.

    Used for sequence-level oversampling of rare stages in Stage-2 (7.6.3) —
    the spec's alternative to (forbidden) row-level SMOTE. One value per item
    in ``dataset._index`` order, so it lines up with __getitem__.
    """
    per_shard: dict[str, dict[int, int]] = {}
    for sp in dataset._shards:
        df = pd.read_parquet(sp, columns=["seq_id", "stage_label"])
        per_shard[sp] = df.groupby("seq_id")["stage_label"].max().astype(int).to_dict()
    return np.array([per_shard[sp].get(sid, 0) for sp, sid in dataset._index],
                    dtype=np.int64)

