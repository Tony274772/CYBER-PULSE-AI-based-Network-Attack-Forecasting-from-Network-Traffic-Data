"""Stage-1 unsupervised pretraining (Section 7.6.2).

Trains GT-RSSM's world model on *all* traffic regardless of label — every
window from every dataset (CIC-IDS2017 + CTU-13 labeled splits, benign or not)
plus CTU-13's unlabeled Background pool — using only the self-supervised
objective:

    L = L_reconstruction (MSE on pooled edge features) + β · KL_t

β follows a linear 0->1 ramp over the first ``kl_anneal_fraction`` of total
Stage-1 steps, then holds at 1 (the posterior-collapse fix). No label is used;
the infiltration/MITRE heads are dormant here (λ = 0).

Checkpoint (Section 8, step 7): reconstruction loss decreases and the KL term
neither collapses to ~0 (posterior collapse) nor blows up. Both curves are
written to ``checkpoints/stage1_curves.png`` and ``checkpoints/stage1_log.json``.

Usage:
    python -m src.training.pretrain_stage1 --config configs/default.yaml
    python -m src.training.pretrain_stage1 --epochs 5        # quick run
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.data import ConcatDataset, DataLoader

from src.config import add_config_args, load_config_from_args
from src.model.gt_rssm import GTRSSM
from src.training.dataset import (WindowSequenceDataset, compute_feature_stats)
from src.training.losses import beta_schedule, total_loss

# (dataset, split) pairs pooled for Stage-1. Everything unsupervised.
_STAGE1_SOURCES = [
    ("cic2017", "train"),
    ("ctu13", "train"),
    ("ctu13", "background"),
]


def build_stage1_dataset(cfg) -> ConcatDataset:
    parts = []
    for dataset, split in _STAGE1_SOURCES:
        processed = getattr(cfg.paths, f"{dataset}_processed_dir")
        windows_root = os.path.join(processed, "windows")
        split_dir = os.path.join(windows_root, split)
        if not os.path.isdir(split_dir):
            print(f"[skip] {dataset}/{split}: {split_dir} not present")
            continue
        try:
            ds = WindowSequenceDataset(windows_root, split,
                                       cfg.data.n_max_nodes, cfg.data.sequence_length_W)
        except FileNotFoundError:
            print(f"[skip] {dataset}/{split}: no shards")
            continue
        print(f"[pool] {dataset}/{split}: {len(ds)} sequences")
        parts.append(ds)
    if not parts:
        raise RuntimeError("no Stage-1 sources found; build windows first")
    return ConcatDataset(parts)


def build_model(cfg) -> GTRSSM:
    return GTRSSM(
        node_dim=4,
        edge_dim=49,
        n_stages=len(cfg.stage_order),
        embed_dim=cfg.model.gat_embed_dim,
        heads=cfg.model.gat_heads,
        layers=cfg.model.gat_layers,
        h_dim=cfg.model.rssm_h_dim,
        z_dim=cfg.model.rssm_z_dim,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="GT-RSSM Stage-1 pretraining.")
    add_config_args(parser)
    parser.add_argument("--epochs", type=int, default=None,
                        help="override training.stage1_epochs")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None, help="cuda|cpu (auto if unset)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="cap total optimizer steps (bounded sanity run); "
                             "β anneals over this many steps when set")
    parser.add_argument("--anneal-fraction", type=float, default=None,
                        help="override training.kl_anneal_fraction (Section 11 "
                             "posterior-collapse remedy: raise it)")
    parser.add_argument("--out", default=None, help="checkpoint path (default checkpoints/stage1.pt)")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    cfg = load_config_from_args(args)
    epochs = args.epochs or cfg.training.stage1_epochs
    batch_size = args.batch_size or cfg.training.stage1_batch_size
    lr = args.lr or cfg.training.stage1_lr
    anneal_fraction = args.anneal_fraction or cfg.training.kl_anneal_fraction
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt_dir = cfg.paths.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    out_path = args.out or os.path.join(ckpt_dir, "stage1.pt")

    dataset = build_stage1_dataset(cfg)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=False)
    steps_per_epoch = len(loader)
    total_steps = max(1, steps_per_epoch * epochs)
    if args.max_steps:
        total_steps = args.max_steps           # β anneals within the bounded run
    print(f"stage-1: {len(dataset)} sequences | {steps_per_epoch} steps/epoch "
          f"| {epochs} epochs | {total_steps} total steps | device={device}")

    model = build_model(cfg).to(device)

    # Fit input standardization on the labeled train shards (streamed from the
    # sparse columns) and bake the stats into the model + checkpoint.
    stat_shards = []
    for dataset, split in [("cic2017", "train"), ("ctu13", "train")]:
        processed = getattr(cfg.paths, f"{dataset}_processed_dir")
        d = os.path.join(processed, "windows", split)
        if os.path.isdir(d):
            stat_shards.extend(sorted(__import__("glob").glob(os.path.join(d, "*.parquet"))))
    feature_stats = compute_feature_stats(stat_shards, d_n=4, d_e=49)
    model.set_feature_stats(feature_stats)
    with open(os.path.join(ckpt_dir, "feature_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(feature_stats, fh, indent=2)
    print(f"feature stats: edges seen={feature_stats['n_edges_seen']} "
          f"nodes seen={feature_stats['n_nodes_seen']}")

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    log = {"step": [], "recon": [], "kl": [], "beta": [], "total": [], "epoch": []}
    global_step = 0
    model.train()
    for epoch in range(epochs):
        for batch in loader:
            node = batch["node_feats"].to(device)
            edge = batch["edge_feats"].to(device)
            adj = batch["adjacency_mask"].to(device)
            valid = batch["node_valid_mask"].to(device)

            outputs = model(node, edge, adj, valid)
            beta = beta_schedule(global_step, total_steps, anneal_fraction)
            comps = total_loss(outputs, batch, beta,
                               lambda_infil=0.0, lambda_mitre=0.0)
            loss = comps["total"]

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            optim.step()

            if global_step % args.log_every == 0:
                log["step"].append(global_step)
                log["recon"].append(float(comps["recon"]))
                log["kl"].append(float(comps["kl"]))
                log["beta"].append(float(beta))
                log["total"].append(float(comps["total"]))
                log["epoch"].append(epoch)
                print(f"  e{epoch} s{global_step} beta={beta:.3f} "
                      f"recon={float(comps['recon']):.5f} kl={float(comps['kl']):.5f} "
                      f"total={float(comps['total']):.5f}")
            global_step += 1
            if args.max_steps and global_step >= args.max_steps:
                break
        if args.max_steps and global_step >= args.max_steps:
            break

    torch.save({
        "model_state": model.state_dict(),
        "config": cfg.to_dict(),
        "global_step": global_step,
        "stage": 1,
        "node_dim": 4,
        "edge_dim": 49,
        "n_stages": len(cfg.stage_order),
        "feature_stats": feature_stats,
    }, out_path)
    print(f"\nsaved Stage-1 checkpoint -> {out_path}")

    metrics_dir = getattr(cfg.paths, "metrics_dir", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    for d in (metrics_dir, ckpt_dir):
        log_path = os.path.join(d, "stage1_log.json")
        with open(log_path, "w", encoding="utf-8") as fh:
            json.dump(log, fh, indent=2)
        _plot_curves(log, os.path.join(d, "stage1_curves.png"))
    print(f"saved training log -> {os.path.join(metrics_dir, 'stage1_log.json')}")
    print(f"saved curves -> {os.path.join(metrics_dir, 'stage1_curves.png')}")


def _plot_curves(log: dict, path: str) -> None:
    """Recon + KL vs step (the Section-8 step-7 sanity plot)."""
    if not log["step"]:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not available; skipping curve plot")
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(log["step"], log["recon"], color="tab:blue")
    ax[0].set_title("reconstruction loss"); ax[0].set_xlabel("step"); ax[0].set_ylabel("MSE")
    ax[1].plot(log["step"], log["kl"], color="tab:red")
    ax[1].set_title("KL term (surprise)"); ax[1].set_xlabel("step"); ax[1].set_ylabel("mean KL")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"saved curves -> {path}")


if __name__ == "__main__":
    main()
