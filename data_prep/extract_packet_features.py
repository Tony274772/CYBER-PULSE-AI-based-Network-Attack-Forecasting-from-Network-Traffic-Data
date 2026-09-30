"""
Extract packet-level features (TTL variance, TCP window size stats, IP
fragmentation ratio, payload size stats, retransmission counts, port-scan
signature scores) from Wednesday-workingHours.pcap, aggregated per
(src_ip, dst_ip, time_window) — matching Section 6.4 of
GT-RSSM_Build_Instructions.md. This is the step inspect_pcap.py deliberately
did NOT do (that script only sanity-checks readability). Run this once the
sanity check looks right.

Uses the same default window size (delta_t_seconds=5) as configs/default.yaml
in the build spec. If the build agent later changes that config, re-run this
script with --delta_t_seconds matching the new value — the join key
(src_ip, dst_ip, window_id) has to line up with the flow-level windowing the
agent builds on top of the flow-level parquet, or the two feature halves
won't merge correctly.

Two-pass streaming design (never loads the whole multi-GB pcap into RAM):
  Pass 1: stream every packet once with PcapReader, write compact per-packet
          records to sharded parquet files in a temp directory.
  Pass 2: load the shards with pandas, bucket into time windows, and run
          vectorized groupby aggregations (not per-row Python loops).

Usage:
    python extract_packet_features.py --pcap data/raw/cic2017/pcap/Wednesday-workingHours.pcap \
        --output_dir data/processed/cic2017 --delta_t_seconds 5
"""
import argparse
import os
import shutil
import tempfile
from collections import defaultdict

import numpy as np
import pandas as pd
from scapy.layers.inet import IP, TCP
from scapy.utils import PcapReader
from tqdm import tqdm

CHUNK_SIZE = 2_000_000  # packets per shard; lower this if RAM is tight


def stream_to_shards(pcap_path: str, shard_dir: str) -> int:
    """Pass 1: stream the pcap once, write per-packet records to parquet shards."""
    os.makedirs(shard_dir, exist_ok=True)
    seen_seqs = defaultdict(set)  # (src,dst,sport,dport) -> TCP seq numbers already seen

    buffer = []
    shard_idx = 0
    n_packets = 0

    def flush():
        nonlocal buffer, shard_idx
        if not buffer:
            return
        pd.DataFrame(buffer).to_parquet(
            os.path.join(shard_dir, f"shard_{shard_idx:05d}.parquet"), index=False
        )
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
                "src_ip": ip.src,
                "dst_ip": ip.dst,
                "ttl": int(ip.ttl),
                "frag": int(bool(ip.flags.MF) or ip.frag > 0),
                "payload_len": len(bytes(ip.payload)) if ip.payload else 0,
                "tcp_window": np.nan,
                "dst_port": np.nan,
                "is_retransmission": False,
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

            if len(buffer) >= CHUNK_SIZE:
                flush()
            if n_packets % 1_000_000 == 0:
                print(f"  ...{n_packets:,} packets streamed")

    flush()
    print(f"Streamed {n_packets:,} total packets into {shard_idx} shard(s)")
    return shard_idx


def port_scan_scores(dst_ports: pd.Series):
    """Given the destination ports one src contacted in one window, return
    (sequential_score, randomised_score)."""
    ports = sorted(set(int(p) for p in dst_ports.dropna()))
    if len(ports) < 2:
        return 0.0, 0.0
    diffs = np.diff(ports)
    sequential_score = float(np.mean(diffs == 1))
    counts = dst_ports.value_counts(normalize=True)
    entropy = float(-(counts * np.log2(counts)).sum())
    max_entropy = np.log2(len(counts)) if len(counts) > 1 else 1.0
    randomised_score = entropy / max_entropy if max_entropy > 0 else 0.0
    return sequential_score, randomised_score


def aggregate(shard_dir: str, delta_t_seconds: float, output_path: str):
    """Pass 2: load shards, bucket into windows, vectorized aggregation."""
    shard_files = sorted(os.path.join(shard_dir, f) for f in os.listdir(shard_dir))
    df = pd.concat(
        [pd.read_parquet(f) for f in tqdm(shard_files, desc="loading shards")],
        ignore_index=True,
    )

    t0 = df["timestamp"].min()
    df["window_id"] = ((df["timestamp"] - t0) // delta_t_seconds).astype(int)

    edge_features = df.groupby(["window_id", "src_ip", "dst_ip"]).agg(
        ttl_mean=("ttl", "mean"),
        ttl_var=("ttl", "var"),
        tcp_window_mean=("tcp_window", "mean"),
        tcp_window_std=("tcp_window", "std"),
        ip_frag_flag_ratio=("frag", "mean"),
        payload_size_mean=("payload_len", "mean"),
        payload_size_std=("payload_len", "std"),
        retransmission_count=("is_retransmission", "sum"),
    ).reset_index()
    edge_features["packet_features_available"] = 1

    # Port-scan scores are computed per (window_id, src_ip) over the distinct
    # dst ports contacted, then joined back onto every edge row for that pair.
    scan_rows = []
    grouped = df.groupby(["window_id", "src_ip"])
    for (window_id, src_ip), g in tqdm(grouped, desc="port-scan scores", total=grouped.ngroups):
        seq_score, rand_score = port_scan_scores(g["dst_port"])
        scan_rows.append({
            "window_id": window_id, "src_ip": src_ip,
            "port_scan_sequential_score": seq_score,
            "port_scan_randomised_score": rand_score,
        })
    scan_df = pd.DataFrame(scan_rows)

    edge_features = edge_features.merge(scan_df, on=["window_id", "src_ip"], how="left")
    edge_features = edge_features.fillna(
        {"ttl_var": 0.0, "tcp_window_std": 0.0, "payload_size_std": 0.0}
    )

    edge_features.to_parquet(output_path, index=False)
    print("\n=== Packet-level feature report ===")
    print(f"Windows covered              : {edge_features['window_id'].nunique():,}")
    print(f"Edge rows (host-pair, window) : {len(edge_features):,}")
    print(f"Saved to {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--delta_t_seconds", type=float, default=5.0,
                     help="Must match configs/default.yaml's data.delta_t_seconds used by the build agent")
    ap.add_argument("--keep_shards", action="store_true",
                     help="Keep the intermediate per-packet shard files instead of deleting them")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    shard_dir = tempfile.mkdtemp(prefix="pcap_shards_")
    try:
        stream_to_shards(args.pcap, shard_dir)
        out_path = os.path.join(args.output_dir, "cic2017_wednesday_packet_features.parquet")
        aggregate(shard_dir, args.delta_t_seconds, out_path)
    finally:
        if args.keep_shards:
            print(f"Shards kept at {shard_dir}")
        else:
            shutil.rmtree(shard_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
