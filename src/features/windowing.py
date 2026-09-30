"""Windowing and graph construction (Section 7.4) — the core feature module.

Turns a tidy per-flow DataFrame (from the ingestion loaders) into the
sequence-of-graphs training store the RSSM consumes. Implements steps 1-9 of
Section 7.4 exactly, vectorized (no per-flow Python loops; the only Python
iteration is per-window graph assembly, which is unavoidable and cheap).

Documented simplifications (see docs/architecture.md):
  * An event that spans a window boundary is assigned to the window containing
    its START time (step 3).
  * Windowing is run independently per ``source_file`` (per capture day for
    CIC-IDS2017, per scenario for CTU-13). Sequences never cross a file
    boundary, so a sequence's W windows are always Δt-contiguous within one
    capture. Empty windows inside a file's active span are kept (all-padding
    graphs) to preserve the fixed-Δt temporal spacing the RSSM relies on.
  * Only full-length (W-window) sequences are emitted; a trailing remainder
    shorter than W is dropped.
  * Packet features (Wednesday CIC-IDS2017 only) are aligned by common capture
    start: ``packet_window_id = floor((window_start - file_t_min) / Δt)``.

Persistence: one row per (sequence, step) window, storing the graph as sparse
1-D list columns (node_ips, node_feats_flat, edge_src, edge_dst,
edge_feats_flat) rather than dense N_max×N_max×d_e tensors. The PyTorch Dataset
(Section 7.6) rebuilds the dense [N_max, N_max, d_e] tensors, adjacency_mask
and node_valid_mask on the fly. Written in fixed-size shards so the whole
feature store is never held in memory at once (Section 7.4 step 9).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import Config
from src.features.flow_features import _SUM_COLS
from src.features.packet_features import PacketFeatureIndex
from src.features.schema import (
    EDGE_FEATURE_COLUMNS,
    FLOW_AVAIL_FEATURES,
    FLOW_FEATURES,
    NODE_FEATURES,
    PACKET_AVAIL_FEATURE,
    PACKET_FEATURES,
)

_BENIGN = "Benign"


# --------------------------------------------------------------------------- #
# Step 1-2: epochs and window membership                                      #
# --------------------------------------------------------------------------- #
def _epoch_seconds(ts: pd.Series) -> np.ndarray:
    """Datetime column -> float epoch seconds (tz-naive, ns precision)."""
    return ts.astype("int64").to_numpy() / 1e9


def _explode_to_windows(df: pd.DataFrame, t_min: float, stride: float,
                        delta_t: float) -> pd.DataFrame:
    """Step 2-3: replicate each event onto every window whose span contains its
    start time. With 50% overlap each event lands in (at most) 2 windows.

    A window ``t`` spans ``[t_min + t*stride, t_min + t*stride + delta_t)``.
    """
    rel = df["_epoch"].to_numpy() - t_min
    t_hi = np.floor(rel / stride).astype(np.int64)         # latest window it can be in
    n_over = int(np.ceil(delta_t / stride))                # windows overlapping a point
    frames = []
    for k in range(0, n_over + 1):
        wid = t_hi - k
        start = wid * stride
        in_win = (wid >= 0) & (rel >= start) & (rel < start + delta_t)
        idx = np.nonzero(in_win)[0]
        if idx.size == 0:
            continue
        sub = df.iloc[idx].copy()
        sub["window_id"] = wid[idx]
        frames.append(sub)
    if not frames:
        return df.iloc[0:0].assign(window_id=np.array([], dtype=np.int64))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Step 4: per-(window, src, dst) edge aggregation (vectorized across windows)  #
# --------------------------------------------------------------------------- #
_EDGE_KEYS = ["window_id", "src_ip", "dst_ip"]


def _aggregate_edges(ex: pd.DataFrame, sev_rank: dict) -> pd.DataFrame:
    """Aggregate exploded flow rows to one edge row per (window_id, src, dst).

    Same math as :func:`flow_features.aggregate_flows` but with ``window_id``
    added to the group key so every window is aggregated in a single pass.
    Attaches the edge's majority-vote ``mapped_stage`` (step 4).
    """
    g = ex.groupby(_EDGE_KEYS, sort=False)
    agg = g[_SUM_COLS].sum()

    dur = ex["flow_duration_sum"].to_numpy()
    w = np.where(dur > 0, dur, 1e-9)
    tmp = ex[_EDGE_KEYS].copy()
    tmp["_wx"] = ex["iat_mean"].to_numpy() * w
    tmp["_w"] = w
    gw = tmp.groupby(_EDGE_KEYS, sort=False)
    agg["iat_mean"] = gw["_wx"].sum() / gw["_w"].sum()
    agg["iat_std"] = g["iat_std"].mean()
    agg["iat_max"] = g["iat_max"].max()
    agg["iat_min"] = g["iat_min"].min()

    agg["fwd_bwd_byte_ratio"] = agg["tot_fwd_bytes"] / (agg["tot_bwd_bytes"] + 1.0)
    agg["fwd_bwd_pkt_ratio"] = agg["tot_fwd_pkts"] / (agg["tot_bwd_pkts"] + 1.0)
    safe_dur = agg["flow_duration_sum"].replace(0.0, np.nan)
    tot_bytes = agg["tot_fwd_bytes"] + agg["tot_bwd_bytes"]
    tot_pkts = agg["tot_fwd_pkts"] + agg["tot_bwd_pkts"]
    agg["flow_bytes_per_sec"] = (tot_bytes / safe_dur).fillna(0.0)
    agg["flow_pkts_per_sec"] = (tot_pkts / safe_dur).fillna(0.0)

    avail = g[FLOW_AVAIL_FEATURES].max()
    out = agg[FLOW_FEATURES].join(avail)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out["edge_stage"] = _edge_majority_stage(ex, sev_rank)
    return out


def _edge_majority_stage(ex: pd.DataFrame, sev_rank: dict) -> pd.Series:
    """Majority-vote mapped_stage per (window, src, dst); ties -> more severe."""
    cnt = ex.groupby(_EDGE_KEYS + ["mapped_stage"], sort=False).size().reset_index(name="n")
    cnt["_sev"] = cnt["mapped_stage"].map(sev_rank).fillna(0).astype(int)
    cnt = cnt.sort_values(["n", "_sev"], ascending=False).drop_duplicates(_EDGE_KEYS)
    return cnt.set_index(_EDGE_KEYS)["mapped_stage"]


# PLACEHOLDER_A


# --------------------------------------------------------------------------- #
# Step 4 (cont.): packet-feature join (Wednesday CIC-IDS2017 only)            #
# --------------------------------------------------------------------------- #
def _attach_packet_features(edges: pd.DataFrame, packet_index: PacketFeatureIndex | None,
                            file_t_min: float, has_packets: bool,
                            stride: float, packet_capture_start: float | None) -> pd.DataFrame:
    """Left-join the 10-dim packet vector onto each edge, adding the single
    ``packet_features_available`` bit. Windows/edges without PCAP coverage get
    zero-filled packet features and ``packet_features_available = 0``.

    Alignment: the packet parquet's ``window_id`` is relative to the pcap's own
    first packet. Flow windows are aligned to it via the common capture start
    ``packet_capture_start`` — the *global* Wednesday flow t_min (packet window
    0), NOT this split's local t_min (a time-aware split slices out a later part
    of the day). A window's absolute start is ``file_t_min + window_id*stride``,
    so ``packet_wid = floor((file_t_min - packet_capture_start + wid*stride)/Δt)``.
    """
    edges = edges.reset_index()  # window_id, src_ip, dst_ip, <flow...>, edge_stage
    for c in PACKET_FEATURES:
        edges[c] = 0.0
    edges[PACKET_AVAIL_FEATURE] = 0.0

    if has_packets and packet_index is not None:
        origin = packet_capture_start if packet_capture_start is not None else file_t_min
        abs_start = file_t_min + edges["window_id"].to_numpy() * stride
        pkt_wid = np.floor((abs_start - origin) / packet_index.delta_t).astype(int)
        vals = np.zeros((len(edges), len(PACKET_FEATURES)), dtype=float)
        avail = np.zeros(len(edges), dtype=float)
        src = edges["src_ip"].to_numpy()
        dst = edges["dst_ip"].to_numpy()
        for i in range(len(edges)):
            vec = packet_index.get(pkt_wid[i], src[i], dst[i])
            if vec is not None:
                vals[i] = vec
                avail[i] = 1.0
        edges[PACKET_FEATURES] = vals
        edges[PACKET_AVAIL_FEATURE] = avail

    # guarantee finite features regardless of source (packet parquet may carry
    # NaNs for single-packet windows); the model/baseline require finite input
    edges[EDGE_FEATURE_COLUMNS] = (
        edges[EDGE_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0))
    return edges


# --------------------------------------------------------------------------- #
# Step 5: per-(window, ip) node features (Section 6.5)                         #
# --------------------------------------------------------------------------- #
def _node_features(ex: pd.DataFrame) -> pd.DataFrame:
    """One node feature row per (window_id, ip): degree (distinct peers in+out),
    in/out byte volume, distinct dst-port count (ports the node contacted as a
    source). Vectorized via two directional contribution frames."""
    src_c = pd.DataFrame({
        "window_id": ex["window_id"].to_numpy(), "ip": ex["src_ip"].to_numpy(),
        "peer": ex["dst_ip"].to_numpy(),
        "in_b": ex["tot_bwd_bytes"].to_numpy(), "out_b": ex["tot_fwd_bytes"].to_numpy(),
        "port": ex["dst_port"].to_numpy(),
    })
    dst_c = pd.DataFrame({
        "window_id": ex["window_id"].to_numpy(), "ip": ex["dst_ip"].to_numpy(),
        "peer": ex["src_ip"].to_numpy(),
        "in_b": ex["tot_fwd_bytes"].to_numpy(), "out_b": ex["tot_bwd_bytes"].to_numpy(),
        "port": np.nan,  # a node's dst-port count counts only ports it initiated to
    })
    both = pd.concat([src_c, dst_c], ignore_index=True)
    gb = both.groupby(["window_id", "ip"], sort=False)
    node = pd.DataFrame({
        "degree": gb["peer"].nunique(),
        "in_byte_volume": gb["in_b"].sum(),
        "out_byte_volume": gb["out_b"].sum(),
        "distinct_dst_port_count": gb["port"].nunique(),
    })
    return node[NODE_FEATURES].astype(float)


# PLACEHOLDER_B


# --------------------------------------------------------------------------- #
# Step 6-7: assemble one window's graph (sparse) + labels                      #
# --------------------------------------------------------------------------- #
@dataclass
class _Window:
    window_id: int
    window_start: float
    node_ips: list
    node_feats: np.ndarray          # [n, d_n]
    edge_src: list                  # local node indices
    edge_dst: list
    edge_feats: np.ndarray          # [m, d_e]
    infil_label: int
    stage_label: int
    n_pkt_edges: int                # edges with packet coverage (for the report)


def _build_window(window_id: int, window_start: float,
                  edges_w: pd.DataFrame | None, nodes_w: pd.DataFrame | None,
                  n_max: int, sev_rank: dict, inv_sev: dict) -> _Window:
    """Graph assembly for a single window. Empty windows yield an all-padding
    graph (no nodes/edges, Benign, infiltration-negative)."""
    if edges_w is None or edges_w.empty or nodes_w is None or nodes_w.empty:
        return _Window(window_id, window_start, [], np.zeros((0, len(NODE_FEATURES))),
                       [], [], np.zeros((0, len(EDGE_FEATURE_COLUMNS))), 0, 0, 0)

    # Step 6: node cap at N_max by highest degree; drop edges touching a dropped node
    nodes_w = nodes_w.sort_values("degree", ascending=False)
    if len(nodes_w) > n_max:
        nodes_w = nodes_w.iloc[:n_max]
    ips = nodes_w.index.get_level_values("ip").tolist()
    local = {ip: i for i, ip in enumerate(ips)}

    keep = edges_w["src_ip"].isin(local) & edges_w["dst_ip"].isin(local)
    edges_w = edges_w[keep]

    edge_src = [local[s] for s in edges_w["src_ip"]]
    edge_dst = [local[d] for d in edges_w["dst_ip"]]
    edge_feats = edges_w[EDGE_FEATURE_COLUMNS].to_numpy(dtype=float)
    node_feats = nodes_w[NODE_FEATURES].to_numpy(dtype=float)

    # Step 7: window labels from surviving edges
    stages = edges_w["edge_stage"].fillna(_BENIGN)
    sev = stages.map(sev_rank).fillna(0).astype(int)
    max_sev = int(sev.max()) if len(sev) else 0
    stage_label = max_sev
    infil_label = int(max_sev > 0)
    n_pkt = int(edges_w[PACKET_AVAIL_FEATURE].sum())

    return _Window(window_id, window_start, ips, node_feats,
                   edge_src, edge_dst, edge_feats, infil_label, stage_label, n_pkt)


# --------------------------------------------------------------------------- #
# Step 9: sharded parquet writer                                              #
# --------------------------------------------------------------------------- #
class _ShardWriter:
    """Buffers window rows and flushes fixed-size shards (by sequence count)."""

    def __init__(self, out_dir: str, seqs_per_shard: int = 200):
        self.out_dir = out_dir
        self.seqs_per_shard = seqs_per_shard
        os.makedirs(out_dir, exist_ok=True)
        self._rows: list[dict] = []
        self._seqs_in_shard = 0
        self._shard_idx = 0

    def add_sequence(self, seq_id: int, windows: list[_Window]) -> None:
        for step, w in enumerate(windows):
            self._rows.append({
                "seq_id": seq_id, "step": step,
                "window_id": int(w.window_id), "window_start": float(w.window_start),
                "n_nodes": len(w.node_ips), "n_edges": len(w.edge_src),
                "node_ips": list(w.node_ips),
                "node_feats_flat": w.node_feats.reshape(-1).tolist(),
                "edge_src": [int(x) for x in w.edge_src],
                "edge_dst": [int(x) for x in w.edge_dst],
                "edge_feats_flat": w.edge_feats.reshape(-1).tolist(),
                "infil_label": int(w.infil_label), "stage_label": int(w.stage_label),
                "n_pkt_edges": int(w.n_pkt_edges),
            })
        self._seqs_in_shard += 1
        if self._seqs_in_shard >= self.seqs_per_shard:
            self.flush()

    def flush(self) -> None:
        if not self._rows:
            return
        path = os.path.join(self.out_dir, f"seqs_{self._shard_idx:05d}.parquet")
        pd.DataFrame(self._rows).to_parquet(path, index=False)
        self._shard_idx += 1
        self._rows = []
        self._seqs_in_shard = 0


# PLACEHOLDER_C


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #
def _severity_maps(stage_order: list[str]) -> tuple[dict, dict]:
    sev_rank = {s: i for i, s in enumerate(stage_order)}
    inv_sev = {i: s for i, s in enumerate(stage_order)}
    return sev_rank, inv_sev


def _process_file(sub: pd.DataFrame, cfg: Config, stride: float, delta_t: float,
                  sev_rank: dict, inv_sev: dict,
                  packet_index: PacketFeatureIndex | None,
                  has_packets: bool, packet_capture_start: float | None) -> list[_Window]:
    """Run steps 1-7 for one source_file and return its ordered window list."""
    sub = sub.copy()
    sub["_epoch"] = _epoch_seconds(sub["timestamp"])
    t_min = float(sub["_epoch"].min())
    t_max = float(sub["_epoch"].max())
    n_windows = int(np.floor((t_max - t_min) / stride)) + 1

    ex = _explode_to_windows(sub, t_min, stride, delta_t)
    if ex.empty:
        return []

    edges = _aggregate_edges(ex, sev_rank)
    edges = _attach_packet_features(edges, packet_index, t_min, has_packets, stride,
                                    packet_capture_start)
    nodes = _node_features(ex)

    edges_by = {wid: g for wid, g in edges.groupby("window_id", sort=False)}
    nodes_by = {wid: g for wid, g in nodes.groupby(level="window_id", sort=False)}

    win_list: list[_Window] = []
    for wid in range(n_windows):
        start = t_min + wid * stride
        win_list.append(_build_window(wid, start, edges_by.get(wid), nodes_by.get(wid),
                                       cfg.data.n_max_nodes, sev_rank, inv_sev))
    return win_list


def build_windows(tidy: pd.DataFrame, cfg: Config, dataset_name: str, out_dir: str,
                  packet_index: PacketFeatureIndex | None = None,
                  packet_capture_start: float | None = None,
                  seqs_per_shard: int = 200) -> dict:
    """Steps 1-9 end to end for one dataset. Windows per ``source_file``,
    sequences of length W (stride W/2), written to sharded parquet under
    ``out_dir``. Returns a data-quality report dict.

    ``packet_capture_start`` is the global Wednesday flow t_min (epoch seconds)
    used as packet-window origin; only relevant for cic2017.
    """
    stride = cfg.window_stride_seconds
    delta_t = cfg.data.delta_t_seconds
    W = cfg.data.sequence_length_W
    seq_stride = max(1, W // 2)
    sev_rank, inv_sev = _severity_maps(cfg.stage_order)

    writer = _ShardWriter(out_dir, seqs_per_shard=seqs_per_shard)
    seq_id = 0
    n_windows_total = 0
    n_pos_windows = 0
    n_edges_total = 0
    n_pkt_edges_total = 0

    group_col = "source_file" if "source_file" in tidy.columns else None
    groups = tidy.groupby(group_col, sort=False) if group_col else [("_all", tidy)]

    for fname, sub in groups:
        has_packets = (dataset_name == "cic2017" and packet_index is not None
                       and "wednesday" in str(fname).lower())
        win_list = _process_file(sub, cfg, stride, delta_t, sev_rank, inv_sev,
                                 packet_index, has_packets, packet_capture_start)
        if not win_list:
            continue

        # per-file stats over unique windows
        for w in win_list:
            n_windows_total += 1
            n_pos_windows += w.infil_label
            n_edges_total += len(w.edge_src)
            n_pkt_edges_total += w.n_pkt_edges

        # Step 8: overlapping sequences of length W (full-length only)
        n = len(win_list)
        for start in range(0, n - W + 1, seq_stride):
            writer.add_sequence(seq_id, win_list[start:start + W])
            seq_id += 1

    writer.flush()

    return {
        "dataset": dataset_name,
        "n_flows": int(len(tidy)),
        "n_windows": n_windows_total,
        "n_sequences": seq_id,
        "pct_positive_windows": round(100.0 * n_pos_windows / max(n_windows_total, 1), 3),
        "n_edges": n_edges_total,
        "pct_edges_with_packet_coverage": round(
            100.0 * n_pkt_edges_total / max(n_edges_total, 1), 3),
        "out_dir": out_dir,
    }


# --------------------------------------------------------------------------- #
# Dense reconstruction (used by the PyTorch Dataset and the step-5 checkpoint) #
# --------------------------------------------------------------------------- #
def reconstruct_dense(row: dict | pd.Series, n_max: int) -> dict:
    """Rebuild the dense per-window tensors from one stored (sparse) row.

    Returns numpy arrays:
      node_feats     [n_max, d_n]
      edge_feats     [n_max, n_max, d_e]   (0 where no edge)
      adjacency_mask [n_max, n_max] in {0,1}
      node_valid_mask[n_max] in {0,1}
    """
    d_n = len(NODE_FEATURES)
    d_e = len(EDGE_FEATURE_COLUMNS)
    n = int(row["n_nodes"])

    node_feats = np.zeros((n_max, d_n), dtype=np.float32)
    edge_feats = np.zeros((n_max, n_max, d_e), dtype=np.float32)
    adjacency_mask = np.zeros((n_max, n_max), dtype=np.float32)
    node_valid_mask = np.zeros((n_max,), dtype=np.float32)

    # Truncate to n_max if stored window has more nodes (happens when
    # training with --n-max smaller than the windowing N_max). The first
    # n_max nodes are the highest-degree ones (sorted during windowing).
    n_fill = min(n, n_max)
    if n_fill > 0:
        nf = np.asarray(row["node_feats_flat"], dtype=np.float32).reshape(n, d_n)
        node_feats[:n_fill] = nf[:n_fill]
        node_valid_mask[:n_fill] = 1.0

    esrc = np.asarray(row["edge_src"], dtype=np.int64)
    edst = np.asarray(row["edge_dst"], dtype=np.int64)
    if esrc.size:
        ef = np.asarray(row["edge_feats_flat"], dtype=np.float32).reshape(esrc.size, d_e)
        # Filter edges: keep only those whose both endpoints are < n_max
        valid_edges = (esrc < n_max) & (edst < n_max)
        esrc_v = esrc[valid_edges]
        edst_v = edst[valid_edges]
        ef_v = ef[valid_edges]
        if esrc_v.size:
            edge_feats[esrc_v, edst_v] = ef_v
            adjacency_mask[esrc_v, edst_v] = 1.0

    return {
        "node_feats": node_feats,
        "edge_feats": edge_feats,
        "adjacency_mask": adjacency_mask,
        "node_valid_mask": node_valid_mask,
        "infil_label": int(row["infil_label"]),
        "stage_label": int(row["stage_label"]),
    }


# --------------------------------------------------------------------------- #
# Flat per-edge table (LR baseline 7.7, XGBoost surrogate 7.8.3)               #
# --------------------------------------------------------------------------- #
def build_edge_table(tidy: pd.DataFrame, cfg: Config, dataset_name: str,
                     packet_index: PacketFeatureIndex | None = None,
                     packet_capture_start: float | None = None) -> pd.DataFrame:
    """One flattened row per (host-pair, window): the 49-dim edge feature vector
    plus ``edge_stage``, ``stage_label`` (severity index) and ``infil`` (edge is
    non-Benign). No graph, no temporal structure — the tabular view the LR
    baseline and the XGBoost surrogate train on. Reuses the same aggregation and
    packet join as the graph store, so features are identical."""
    stride = cfg.window_stride_seconds
    delta_t = cfg.data.delta_t_seconds
    sev_rank, _ = _severity_maps(cfg.stage_order)

    group_col = "source_file" if "source_file" in tidy.columns else None
    groups = tidy.groupby(group_col, sort=False) if group_col else [("_all", tidy)]

    parts = []
    for fname, sub in groups:
        has_packets = (dataset_name == "cic2017" and packet_index is not None
                       and "wednesday" in str(fname).lower())
        s = sub.copy()
        s["_epoch"] = _epoch_seconds(s["timestamp"])
        t_min = float(s["_epoch"].min())
        ex = _explode_to_windows(s, t_min, stride, delta_t)
        if ex.empty:
            continue
        edges = _aggregate_edges(ex, sev_rank)
        edges = _attach_packet_features(edges, packet_index, t_min, has_packets,
                                        stride, packet_capture_start)
        keep = EDGE_FEATURE_COLUMNS + ["edge_stage"]
        parts.append(edges[keep])

    if not parts:
        return pd.DataFrame(columns=EDGE_FEATURE_COLUMNS + ["edge_stage", "stage_label", "infil"])
    out = pd.concat(parts, ignore_index=True)
    out["edge_stage"] = out["edge_stage"].fillna(_BENIGN)
    out["stage_label"] = out["edge_stage"].map(sev_rank).fillna(0).astype(int)
    out["infil"] = (out["stage_label"] > 0).astype(int)
    return out


