"""Canonical feature schemas shared across the pipeline.

These column orders are the contract between the windowing/feature code and the
model. Everything downstream indexes features positionally through these lists,
so they must never be reordered without regenerating the feature store.
"""
from __future__ import annotations

# --- 6.3 flow-level feature vector (per host-pair, per window) ---------------
FLOW_FEATURES: list[str] = [
    "flow_duration_sum",
    "tot_fwd_pkts", "tot_bwd_pkts", "tot_fwd_bytes", "tot_bwd_bytes",
    "flag_syn_count", "flag_ack_count", "flag_fin_count",
    "flag_rst_count", "flag_psh_count", "flag_urg_count",
    "iat_mean", "iat_std", "iat_max", "iat_min",
    "fwd_bwd_byte_ratio", "fwd_bwd_pkt_ratio",
    "flow_bytes_per_sec", "flow_pkts_per_sec",
]

# Per-field availability mask for flow features (Section 6.3: CTU-13 cannot
# derive flag counts / IAT stats -> those positions are zero-filled and their
# availability bit set to 0). One bit per FLOW_FEATURES entry.
FLOW_AVAIL_FEATURES: list[str] = [f"{c}__avail" for c in FLOW_FEATURES]

# --- 6.4 packet-level feature vector (per host-pair, per window) -------------
PACKET_FEATURES: list[str] = [
    "ttl_mean", "ttl_var",
    "tcp_window_mean", "tcp_window_std",
    "ip_frag_flag_ratio",
    "payload_size_mean", "payload_size_std",
    "retransmission_count",
    "port_scan_sequential_score", "port_scan_randomised_score",
]

# Single availability bit for the whole packet block (Section 6.1): 1 only for
# Wednesday CIC-IDS2017 edges that were joined to real packet features.
PACKET_AVAIL_FEATURE = "packet_features_available"

# --- 6.5 node-level feature vector (per active IP, per window) ---------------
NODE_FEATURES: list[str] = [
    "degree",
    "in_byte_volume", "out_byte_volume",
    "distinct_dst_port_count",
]

# --- assembled edge feature vector -------------------------------------------
# flow (19) + flow-avail (19) + packet (10) + packet-avail (1) = 49 dims
EDGE_FEATURE_COLUMNS: list[str] = (
    FLOW_FEATURES + FLOW_AVAIL_FEATURES + PACKET_FEATURES + [PACKET_AVAIL_FEATURE]
)
EDGE_FEATURE_DIM = len(EDGE_FEATURE_COLUMNS)   # 49
NODE_FEATURE_DIM = len(NODE_FEATURES)          # 4


def load_stage_order(mapping_path: str = "mitre_mapping/attack_stage_mapping.json") -> list[str]:
    import json
    with open(mapping_path, "r", encoding="utf-8") as fh:
        return json.load(fh)["stage_order"]
