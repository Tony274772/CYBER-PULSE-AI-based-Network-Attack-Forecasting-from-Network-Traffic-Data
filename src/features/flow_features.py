"""Per-window flow-feature aggregation (Section 7.3 / 6.3).

Pure, vectorized functions: given tidy per-flow rows already restricted to one
window, produce one aggregated 6.3 feature row per (src_ip, dst_ip) pair.
Multiple flows for the same host-pair in the same window are combined by
summing counts/bytes, re-deriving ratios and rates, and duration-weighting the
IAT mean (Section 6.3). Never uses per-row Python loops (Section 7.3 / 11).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.schema import FLOW_AVAIL_FEATURES, FLOW_FEATURES

# feature columns aggregated by plain sum
_SUM_COLS = [
    "flow_duration_sum",
    "tot_fwd_pkts", "tot_bwd_pkts", "tot_fwd_bytes", "tot_bwd_bytes",
    "flag_syn_count", "flag_ack_count", "flag_fin_count",
    "flag_rst_count", "flag_psh_count", "flag_urg_count",
]


def aggregate_flows(window_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate tidy flow rows (one window) to one row per (src_ip, dst_ip).

    Returns a DataFrame indexed by [src_ip, dst_ip] with the full FLOW_FEATURES
    + FLOW_AVAIL_FEATURES columns, plus ``__labels`` (list of mapped_stage) used
    by the windowing step for the majority/most-severe vote.
    """
    if window_df.empty:
        return pd.DataFrame(columns=FLOW_FEATURES + FLOW_AVAIL_FEATURES)

    g = window_df.groupby(["src_ip", "dst_ip"], sort=False)

    agg = g[_SUM_COLS].sum()

    # duration-weighted IAT mean; robust max/min/std combinations
    dur = window_df["flow_duration_sum"].to_numpy()
    wdf = window_df.assign(_w=np.where(dur > 0, dur, 1e-9))
    wdf["_wx"] = wdf["iat_mean"] * wdf["_w"]
    gw = wdf.groupby(["src_ip", "dst_ip"], sort=False)
    agg["iat_mean"] = gw["_wx"].sum() / gw["_w"].sum()
    agg["iat_std"] = g["iat_std"].mean()
    agg["iat_max"] = g["iat_max"].max()
    agg["iat_min"] = g["iat_min"].min()

    # re-derived ratios and rates over the aggregated totals
    agg["fwd_bwd_byte_ratio"] = agg["tot_fwd_bytes"] / (agg["tot_bwd_bytes"] + 1.0)
    agg["fwd_bwd_pkt_ratio"] = agg["tot_fwd_pkts"] / (agg["tot_bwd_pkts"] + 1.0)
    safe_dur = agg["flow_duration_sum"].replace(0.0, np.nan)
    tot_bytes = agg["tot_fwd_bytes"] + agg["tot_bwd_bytes"]
    tot_pkts = agg["tot_fwd_pkts"] + agg["tot_bwd_pkts"]
    agg["flow_bytes_per_sec"] = (tot_bytes / safe_dur).fillna(0.0)
    agg["flow_pkts_per_sec"] = (tot_pkts / safe_dur).fillna(0.0)

    # availability bits: available if any contributing flow had it available
    avail = g[FLOW_AVAIL_FEATURES].max()

    out = agg[FLOW_FEATURES].join(avail)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out
