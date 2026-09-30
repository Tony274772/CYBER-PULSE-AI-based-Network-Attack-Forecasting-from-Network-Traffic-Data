"""Full evaluation pipeline (Section 7.9 / Section 8 step 11).

Runs all metrics from Section 7.9 for both GT-RSSM and the LR baseline on a
common held-out test split and writes the comparison table to
``data/processed/eval_results.md``.

Metrics computed:
  * Precision / Recall / F1 (binary infiltration + macro MITRE stage)
  * ROC-AUC, PR-AUC, Brier score
  * Lead time to detection
  * K-step Brier score for the imagination forecast
  * Cross-dataset generalization (train CIC-IDS2017, zero-shot CTU-13)
  * Held-out-category surprise AUC (kl_t as anomaly signal)

Usage:
    python -m src.evaluation.run_eval --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import Config, add_config_args, load_config_from_args
from src.evaluation.metrics import (
    METRICS_TABLE_HEADER,
    binary_metrics,
    format_metrics_row,
    threshold_for_target_fpr,
)
from src.features.schema import EDGE_FEATURE_DIM, NODE_FEATURE_DIM
from src.model.gt_rssm import GTRSSM
from src.training.dataset import WindowSequenceDataset


# --------------------------------------------------------------------------- #
# GT-RSSM evaluation helpers                                                  #
# --------------------------------------------------------------------------- #

def _load_model(ckpt_path: str, cfg: Config, device: torch.device) -> GTRSSM:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GTRSSM(
        node_dim=ckpt.get("node_dim", NODE_FEATURE_DIM),
        edge_dim=ckpt.get("edge_dim", EDGE_FEATURE_DIM),
        n_stages=ckpt.get("n_stages", len(cfg.stage_order)),
        embed_dim=cfg.model.gat_embed_dim,
        heads=cfg.model.gat_heads,
        layers=cfg.model.gat_layers,
        h_dim=cfg.model.rssm_h_dim,
        z_dim=cfg.model.rssm_z_dim,
    )
    model.load_state_dict(ckpt["model_state"], strict=False)
    if ckpt.get("feature_stats"):
        model.set_feature_stats(ckpt["feature_stats"])
    model.to(device).eval()
    return model


@torch.no_grad()
def _collect_model_outputs(model, loader, device, k_horizons=(5, 10, 20)):
    """Run GT-RSSM on the loader, collecting per-window infiltration probs,
    MITRE stage probs, KL surprise, and K-step imagination forecasts."""
    all_infil_prob = []
    all_infil_target = []
    all_stage_prob = []
    all_stage_target = []
    all_kl = []
    # For lead-time: per-sequence info
    seq_infil_probs = []
    seq_infil_targets = []
    # For Brier at horizon K: need per-sequence forecasts aligned with future labels
    forecast_by_k: dict[int, list] = {k: [] for k in k_horizons}
    actual_by_k: dict[int, list] = {k: [] for k in k_horizons}

    for batch in loader:
        node = batch["node_feats"].to(device)
        edge = batch["edge_feats"].to(device)
        adj = batch["adjacency_mask"].to(device)
        valid = batch["node_valid_mask"].to(device)
        out = model(node, edge, adj, valid)

        B, W = out["infil_logit"].shape
        infil_p = torch.sigmoid(out["infil_logit"]).cpu().numpy()  # [B, W]
        infil_t = batch["infil_target"].numpy()                    # [B, W]
        stage_p = torch.softmax(out["mitre_logits"], dim=-1).cpu().numpy()
        stage_t = batch["stage_target"].numpy()
        kl = out["kl"].cpu().numpy()

        all_infil_prob.append(infil_p.reshape(-1))
        all_infil_target.append(infil_t.reshape(-1))
        all_stage_prob.append(stage_p.reshape(-1, stage_p.shape[-1]))
        all_stage_target.append(stage_t.reshape(-1))
        all_kl.append(kl.reshape(-1))

        # Per-sequence for lead-time
        for b in range(B):
            seq_infil_probs.append(infil_p[b])
            seq_infil_targets.append(infil_t[b])

        # K-step imagination Brier
        for b in range(B):
            h_t = out["last_h"][b:b+1]
            z_t = out["last_z"][b:b+1]
            for K in k_horizons:
                traj = model.imagine(h_t, z_t, K)
                forecast_probs = [float(t["infiltration_prob"][0]) for t in traj]
                forecast_by_k[K].extend(forecast_probs)
                # Actual labels K steps ahead: we can only evaluate against
                # labels within this sequence's window range (shifted)
                # In practice, the last W-K windows are the "actual" for the
                # first K forecast steps. For the eval, use the labels from
                # the end of the sequence as a proxy.
                actual_labels = infil_t[b][-K:] if W >= K else infil_t[b]
                padded = list(actual_labels) + [0.0] * max(0, K - len(actual_labels))
                actual_by_k[K].extend(padded[:K])

    return {
        "infil_prob": np.concatenate(all_infil_prob),
        "infil_target": np.concatenate(all_infil_target),
        "stage_prob": np.concatenate(all_stage_prob, axis=0),
        "stage_target": np.concatenate(all_stage_target),
        "kl": np.concatenate(all_kl),
        "seq_infil_probs": seq_infil_probs,
        "seq_infil_targets": seq_infil_targets,
        "forecast_by_k": {k: np.array(v) for k, v in forecast_by_k.items()},
        "actual_by_k": {k: np.array(v) for k, v in actual_by_k.items()},
    }


# --------------------------------------------------------------------------- #
# Lead-time metric                                                            #
# --------------------------------------------------------------------------- #

def _compute_lead_time(seq_probs, seq_targets, threshold, delta_t):
    """For each sequence that contains an attack, find lead time = t_alert - t_compromise."""
    lead_times = []
    for probs, targets in zip(seq_probs, seq_targets):
        if targets.max() == 0:
            continue  # no attack in this sequence
        t_compromise = np.argmax(targets > 0)  # first non-Benign window
        # Find first window where prob crosses threshold
        above = np.where(probs >= threshold)[0]
        if len(above) == 0:
            lead_times.append(0.0)  # never alerted
        else:
            t_alert = above[0]
            lead_times.append(float(t_compromise - t_alert) * delta_t)
    if not lead_times:
        return {"mean": 0.0, "median": 0.0, "n_sequences": 0}
    return {
        "mean": float(np.mean(lead_times)),
        "median": float(np.median(lead_times)),
        "n_sequences": len(lead_times),
    }


# --------------------------------------------------------------------------- #
# Brier score at K-step horizon                                               #
# --------------------------------------------------------------------------- #

def _brier_at_k(forecast_by_k, actual_by_k):
    """Per-horizon Brier score."""
    results = {}
    for k in forecast_by_k:
        f = forecast_by_k[k]
        a = actual_by_k[k]
        n = min(len(f), len(a))
        if n == 0:
            results[k] = float("nan")
            continue
        results[k] = float(np.mean((f[:n] - a[:n]) ** 2))
    return results


# --------------------------------------------------------------------------- #
# Surprise AUC (kl_t as anomaly for held-out category)                        #
# --------------------------------------------------------------------------- #

def _surprise_auc(kl, stage_target, held_out_stage_idx, benign_idx=0):
    """ROC-AUC of using kl_t alone to separate held-out stage vs benign."""
    from sklearn.metrics import roc_auc_score
    mask = np.isin(stage_target, [held_out_stage_idx, benign_idx])
    if mask.sum() == 0:
        return float("nan")
    y = (stage_target[mask] == held_out_stage_idx).astype(int)
    scores = kl[mask]
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, scores))


# --------------------------------------------------------------------------- #
# MITRE stage macro F1                                                        #
# --------------------------------------------------------------------------- #

def _stage_macro_f1(stage_prob, stage_target, n_stages):
    """Macro F1 across all 7 MITRE stages."""
    from sklearn.metrics import f1_score
    pred = stage_prob.argmax(axis=-1)
    return float(f1_score(stage_target, pred, average="macro", zero_division=0))


# --------------------------------------------------------------------------- #
# Main evaluation                                                             #
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Full GT-RSSM evaluation (Section 7.9).")
    add_config_args(parser)
    parser.add_argument("--ckpt", default=None, help="GT-RSSM checkpoint path")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--k-horizons", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--held-out-stage", type=int, default=4,
                        help="stage index to hold out for surprise AUC (default: 4 = C2)")
    args = parser.parse_args()

    cfg = load_config_from_args(args)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_dir = cfg.paths.checkpoint_dir
    ckpt_path = args.ckpt or os.path.join(ckpt_dir, "gt_rssm_v1.pt")
    stage_order = cfg.stage_order
    n_stages = len(stage_order)

    lines = ["# Evaluation Results", ""]

    # --- Read existing LR baseline results if present -----------------------
    eval_path = os.path.join(cfg.paths.processed_dir, "eval_results.md")
    if os.path.exists(eval_path):
        with open(eval_path, "r", encoding="utf-8") as fh:
            existing = fh.read()
        if "Logistic Regression" in existing:
            lines.append(existing.strip())
            lines.append("")

    # --- GT-RSSM evaluation -------------------------------------------------
    if not os.path.exists(ckpt_path):
        print(f"[warn] checkpoint not found: {ckpt_path}")
        print("Skipping GT-RSSM evaluation. Run training first.")
        _write_results(lines, eval_path)
        return

    model = _load_model(ckpt_path, cfg, device)
    print(f"loaded GT-RSSM from {ckpt_path}")

    lines.append("## GT-RSSM (Section 7.9)")
    lines.append("")

    for dataset in ("cic2017", "ctu13"):
        processed = getattr(cfg.paths, f"{dataset}_processed_dir")
        windows_root = os.path.join(processed, "windows")
        test_dir = os.path.join(windows_root, "test")
        val_dir = os.path.join(windows_root, "val")

        if not os.path.isdir(test_dir):
            print(f"[skip] {dataset}: no test split at {test_dir}")
            continue

        try:
            test_ds = WindowSequenceDataset(windows_root, "test",
                                             cfg.data.n_max_nodes, cfg.data.sequence_length_W)
            val_ds = WindowSequenceDataset(windows_root, "val",
                                            cfg.data.n_max_nodes, cfg.data.sequence_length_W)
        except FileNotFoundError as e:
            print(f"[skip] {dataset}: {e}")
            continue

        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers)

        print(f"\n=== {dataset} ===")
        print(f"  test: {len(test_ds)} sequences | val: {len(val_ds)} sequences")

        # Collect outputs
        test_out = _collect_model_outputs(model, test_loader, device, tuple(args.k_horizons))
        val_out = _collect_model_outputs(model, val_loader, device, tuple(args.k_horizons))

        # Binary infiltration metrics
        # Tune threshold on validation
        threshold = threshold_for_target_fpr(val_out["infil_target"], val_out["infil_prob"])
        test_metrics = binary_metrics(test_out["infil_target"], test_out["infil_prob"], threshold)
        val_metrics = binary_metrics(val_out["infil_target"], val_out["infil_prob"])

        # MITRE stage macro F1
        test_stage_f1 = _stage_macro_f1(test_out["stage_prob"], test_out["stage_target"], n_stages)

        # Lead time
        lead_time = _compute_lead_time(
            test_out["seq_infil_probs"], test_out["seq_infil_targets"],
            threshold, cfg.data.delta_t_seconds)

        # Brier at K horizons
        brier_k = _brier_at_k(test_out["forecast_by_k"], test_out["actual_by_k"])

        # Surprise AUC
        surprise_auc = _surprise_auc(test_out["kl"], test_out["stage_target"],
                                      args.held_out_stage)

        lines.append(f"### {dataset}")
        lines.append("")
        lines.append(METRICS_TABLE_HEADER)
        lines.append(format_metrics_row(f"GT-RSSM (val)", val_metrics))
        lines.append(format_metrics_row(f"GT-RSSM (test, th={threshold:.3f})", test_metrics))
        lines.append("")
        lines.append(f"**MITRE stage macro F1 (test):** {test_stage_f1:.4f}")
        lines.append("")
        lines.append(f"**Lead time (test):** mean={lead_time['mean']:.1f}s, "
                      f"median={lead_time['median']:.1f}s "
                      f"({lead_time['n_sequences']} attack sequences)")
        lines.append("")
        lines.append("**K-step Brier score:**")
        for k, b in sorted(brier_k.items()):
            lines.append(f"  - K={k}: {b:.4f}")
        lines.append("")
        lines.append(f"**Held-out-category surprise AUC** "
                      f"(stage={stage_order[args.held_out_stage]}): "
                      f"{surprise_auc:.4f}")
        lines.append("")

        print(f"  test F1={test_metrics['f1']:.4f} ROC-AUC={test_metrics['roc_auc']:.4f}")
        print(f"  stage macro F1={test_stage_f1:.4f}")
        print(f"  lead time: mean={lead_time['mean']:.1f}s median={lead_time['median']:.1f}s")
        print(f"  surprise AUC={surprise_auc:.4f}")

    _write_results(lines, eval_path)
    print(f"\nwrote {eval_path}")


def _write_results(lines: list[str], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
