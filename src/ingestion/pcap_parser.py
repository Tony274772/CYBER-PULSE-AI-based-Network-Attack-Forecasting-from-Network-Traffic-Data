"""Ingestion: streaming PCAP parser -> per-(host-pair, window) packet features
(Section 7.2 / 6.4).

Uses ``scapy.utils.PcapReader`` as a streaming iterator — never ``rdpcap``,
which loads the whole capture into RAM. Design is two-pass (identical in
spirit to ``data_prep/extract_packet_features.py``, which already produced the
Wednesday feature parquet for training):

  Pass 1: stream every packet once, buffer compact per-packet records.
  Pass 2: bucket into windows and aggregate with vectorized pandas groupby.

For the training pipeline the Wednesday features are read from the precomputed
parquet; this module is the live path used by ``inference.predict`` when a
user uploads a fresh PCAP to the dashboard.
"""
from __future__ import annotations

import os
import tempfile
from collections import defaultdict

import numpy as np
import pandas as pd

from src.features.schema import PACKET_FEATURES


def _port_scan_scores(dst_ports: pd.Series) -> tuple[float, float]:
    """(sequential_score, randomised_score) over the distinct dst ports one
    src contacted in one window (Section 6.4)."""
    ports = sorted({int(p) for p in dst_ports.dropna()})
    if len(ports) < 2:
        return 0.0, 0.0
    diffs = np.diff(ports)
    sequential_score = float(np.mean(diffs == 1))
    counts = dst_ports.value_counts(normalize=True)
    entropy = float(-(counts * np.log2(counts)).sum())
    max_entropy = np.log2(len(counts)) if len(counts) > 1 else 1.0
    randomised_score = entropy / max_entropy if max_entropy > 0 else 0.0
    return sequential_score, randomised_score


def stream_to_shards(pcap_path: str, shard_dir: str, chunk_size: int = 2_000_000) -> int:
    """Pass 1: stream the pcap once, write per-packet records to parquet shards."""
    from scapy.layers.inet import IP, TCP
    from scapy.utils import PcapReader

    os.makedirs(shard_dir, exist_ok=True)
    seen_seqs: dict = defaultdict(set)
    buffer: list[dict] = []
    shard_idx = 0
    n_packets = 0

    def flush():
        nonlocal buffer, shard_idx
        if not buffer:
            return
        pd.DataFrame(buffer).to_parquet(
            os.path.join(shard_dir, f"shard_{shard_idx:05d}.parquet"), index=False)
        shard_idx += 1
        buffer = []

    with PcapReader(pcap_path) as reader:
        for pkt in reader:
            n_packets += 1
            if IP not in pkt:
                continue
            ip = pkt[IP]
            rec = {
                "timestamp": float(pkt.time),
                "src_ip": ip.src, "dst_ip": ip.dst,
                "ttl": int(ip.ttl),
                "frag": int(bool(ip.flags.MF) or ip.frag > 0),
                "payload_len": len(bytes(ip.payload)) if ip.payload else 0,
                "tcp_window": np.nan, "dst_port": np.nan, "is_retransmission": False,
            }
            if TCP in pkt:
                tcp = pkt[TCP]
                rec["tcp_window"] = int(tcp.window)
                rec["dst_port"] = int(tcp.dport)
                key = (ip.src, ip.dst, tcp.sport, tcp.dport)
                if tcp.seq in seen_seqs[key]:
                    rec["is_retransmission"] = True
                else:
                    seen_seqs[key].add(tcp.seq)
            buffer.append(rec)
            if len(buffer) >= chunk_size:
                flush()
    flush()
    return shard_idx


def aggregate(shard_dir: str, delta_t_seconds: float) -> pd.DataFrame:
    """Pass 2: bucket per-packet records into windows and aggregate (6.4)."""
    shard_files = sorted(os.path.join(shard_dir, f) for f in os.listdir(shard_dir)
                         if f.endswith(".parquet"))
    if not shard_files:
        return pd.DataFrame(columns=["window_id", "src_ip", "dst_ip",
                                     *PACKET_FEATURES, "packet_features_available"])
    df = pd.concat([pd.read_parquet(f) for f in shard_files], ignore_index=True)

    t0 = df["timestamp"].min()
    df["window_id"] = ((df["timestamp"] - t0) // delta_t_seconds).astype(int)

    edge = df.groupby(["window_id", "src_ip", "dst_ip"]).agg(
        ttl_mean=("ttl", "mean"), ttl_var=("ttl", "var"),
        tcp_window_mean=("tcp_window", "mean"), tcp_window_std=("tcp_window", "std"),
        ip_frag_flag_ratio=("frag", "mean"),
        payload_size_mean=("payload_len", "mean"), payload_size_std=("payload_len", "std"),
        retransmission_count=("is_retransmission", "sum"),
    ).reset_index()

    scan_rows = []
    for (window_id, src_ip), g in df.groupby(["window_id", "src_ip"]):
        seq_s, rand_s = _port_scan_scores(g["dst_port"])
        scan_rows.append({"window_id": window_id, "src_ip": src_ip,
                          "port_scan_sequential_score": seq_s,
                          "port_scan_randomised_score": rand_s})
    scan_df = pd.DataFrame(scan_rows)
    edge = edge.merge(scan_df, on=["window_id", "src_ip"], how="left")
    edge = edge.fillna({"ttl_var": 0.0, "tcp_window_std": 0.0, "payload_size_std": 0.0,
                        "tcp_window_mean": 0.0, "port_scan_sequential_score": 0.0,
                        "port_scan_randomised_score": 0.0})
    edge["packet_features_available"] = 1
    return edge


def parse_pcap(pcap_path: str, delta_t_seconds: float = 5.0) -> pd.DataFrame:
    """End-to-end streaming parse -> per-(window_id, src_ip, dst_ip) packet
    feature table. Temporary shard files are cleaned up afterwards."""
    shard_dir = tempfile.mkdtemp(prefix="pcap_shards_")
    try:
        stream_to_shards(pcap_path, shard_dir)
        return aggregate(shard_dir, delta_t_seconds)
    finally:
        import shutil
        shutil.rmtree(shard_dir, ignore_errors=True)
