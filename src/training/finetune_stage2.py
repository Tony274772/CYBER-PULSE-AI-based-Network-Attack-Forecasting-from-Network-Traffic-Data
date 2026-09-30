"""Stage-2 supervised fine-tuning (Section 7.6.3).

Loads ``stage1.pt``, keeps the pretrained encoder + RSSM + reconstruction head,
and now trains the full objective

    L_total = L_recon + β·KL + λ1·L_infiltration(focal) + λ2·L_mitre(weighted CE)

on the *labeled* Stage-2 mix (CIC-IDS2017 Tue–Fri + Wednesday PCAP-joined
windows + CTU-13 non-Background windows = the cic2017/train + ctu13/train
window splits). β is held at 1 (Stage-1 already annealed it).

Two parameter groups (7.6.3):
  * encoder + RSSM  -> lr 1e-4  (gentle, so Stage-1 dynamics survive)
  * heads           -> lr 1e-3

Class imbalance beyond focal loss / class weights is handled by oversampling
whole **sequences** that contain a rare stage (Command_And_Control,
Exfiltration) — never row-level SMOTE (breaks temporal structure).

Early-stops on validation infiltration F1 and checkpoints the best model to
``checkpoints/gt_rssm_v1.pt`` (Section 8, step 8: must beat the LR baseline).

Usage:
    python -m src.training.finetune_stage2 --config configs/default.yaml
    python -m src.training.finetune_stage2 --epochs 5
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from src.config import add_config_args, load_config_from_args
from src.evaluation.metrics import binary_metrics
from src.model.gt_rssm import GTRSSM
from src.training.dataset import (WindowSequenceDataset, sequence_max_stage,
                                  stage_class_weights)
from src.training.losses import total_loss

# Rare stages to oversample at the sequence level (indices into stage_order).
# stage_order = [Benign, Recon, Initial_Access, Lateral_Movement,
#                Command_And_Control, Exfiltration, Impact_DoS]
_RARE_STAGES = (4, 5)          # Command_And_Control, Exfiltration
_RARE_BOOST = 5.0              # sampling-weight multiplier for rare-stage sequences

_STAGE2_TRAIN = [("cic2017", "train"), ("ctu13", "train")]
_STAGE2_VAL = [("cic2017", "val"), ("ctu13", "val")]


def _pool(cfg, sources) -> tuple[ConcatDataset, list[WindowSequenceDataset]]:
    parts = []
    for dataset, split in sources:
        processed = getattr(cfg.paths, f"{dataset}_processed_dir")
        windows_root = os.path.join(processed, "windows")
        try:
            ds = WindowSequenceDataset(windows_root, split,
                                       cfg.data.n_max_nodes, cfg.data.sequence_length_W)
        except FileNotFoundError:
            print(f"[skip] {dataset}/{split}: no shards")
            continue
        print(f"[pool] {dataset}/{split}: {len(ds)} sequences")
        parts.append(ds)
    if not parts:
        raise RuntimeError("no Stage-2 sources found; build windows first")
    return ConcatDataset(parts), parts


def _oversample_sampler(parts: list[WindowSequenceDataset]) -> WeightedRandomSampler:
    """Sequence-level oversampling weights: rare-stage sequences get boosted."""
    stages = np.concatenate([sequence_max_stage(p) for p in parts])
    weights = np.ones(len(stages), dtype=np.float64)
    weights[np.isin(stages, _RARE_STAGES)] = _RARE_BOOST
    return WeightedRandomSampler(weights=weights.tolist(),
                                 num_samples=len(weights), replacement=True)


def build_model(cfg) -> GTRSSM:
    return GTRSSM(node_dim=4, edge_dim=49, n_stages=len(cfg.stage_order),
                  embed_dim=cfg.model.gat_embed_dim, heads=cfg.model.gat_heads,
                  layers=cfg.model.gat_layers, h_dim=cfg.model.rssm_h_dim,
                  z_dim=cfg.model.rssm_z_dim)


def _param_groups(model: GTRSSM, encoder_lr: float, head_lr: float):
    enc_rssm, heads = [], []
    for name, p in model.named_parameters():
        if name.startswith("encoder.") or name.startswith("rssm."):
            enc_rssm.append(p)
        else:                                   # heads.*
            heads.append(p)
    return [{"params": enc_rssm, "lr": encoder_lr},
            {"params": heads, "lr": head_lr}]


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    probs, targets = [], []
    for batch in loader:
        node = batch["node_feats"].to(device)
        edge = batch["edge_feats"].to(device)
        adj = batch["adjacency_mask"].to(device)
        valid = batch["node_valid_mask"].to(device)
        out = model(node, edge, adj, valid)
        probs.append(torch.sigmoid(out["infil_logit"]).reshape(-1).cpu().numpy())
        targets.append(batch["infil_target"].reshape(-1).numpy())
    y_prob = np.concatenate(probs)
    y_true = np.concatenate(targets)
    return binary_metrics(y_true, y_prob, threshold=0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description="GT-RSSM Stage-2 fine-tuning.")
    add_config_args(parser)
    parser.add_argument("--stage1", default=None, help="path to stage1.pt")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None, help="best-checkpoint path")
    parser.add_argument("--patience", type=int, default=6,
                        help="early-stop patience (epochs w/o val-F1 improvement)")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    cfg = load_config_from_args(args)
    epochs = args.epochs or cfg.training.stage2_epochs
    batch_size = args.batch_size or cfg.training.stage1_batch_size
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_dir = cfg.paths.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    stage1_path = args.stage1 or os.path.join(ckpt_dir, "stage1.pt")
    out_path = args.out or os.path.join(ckpt_dir, "gt_rssm_v1.pt")

    train_ds, train_parts = _pool(cfg, _STAGE2_TRAIN)
    val_ds, _ = _pool(cfg, _STAGE2_VAL)
    sampler = _oversample_sampler(train_parts)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=args.num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=args.num_workers)

    n_stages = len(cfg.stage_order)
    class_weights = stage_class_weights(train_parts[0], n_stages).to(device)
    print(f"stage class weights: {class_weights.tolist()}")

    model = build_model(cfg).to(device)
    feature_stats = None
    if os.path.exists(stage1_path):
        state = torch.load(stage1_path, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(state["model_state"], strict=False)
        feature_stats = state.get("feature_stats")
        if feature_stats:
            model.set_feature_stats(feature_stats)            # keep Stage-1 input scaling
        print(f"loaded {stage1_path} (missing={len(missing)} unexpected={len(unexpected)})")
    else:
        print(f"[warn] {stage1_path} not found; training Stage-2 from scratch")

    optim = torch.optim.Adam(_param_groups(model, cfg.training.stage2_encoder_lr,
                                           cfg.training.stage2_head_lr))

    best_f1 = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    history = []
    global_step = 0
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            node = batch["node_feats"].to(device)
            edge = batch["edge_feats"].to(device)
            adj = batch["adjacency_mask"].to(device)
            valid = batch["node_valid_mask"].to(device)
            batch_t = {"infil_target": batch["infil_target"].to(device),
                       "stage_target": batch["stage_target"].to(device)}

            out = model(node, edge, adj, valid)
            comps = total_loss(out, batch_t, beta=1.0,
                               lambda_infil=cfg.training.lambda_infiltration,
                               lambda_mitre=cfg.training.lambda_mitre,
                               class_weights=class_weights)
            optim.zero_grad()
            comps["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            optim.step()

            if global_step % args.log_every == 0:
                print(f"  e{epoch} s{global_step} total={float(comps['total']):.4f} "
                      f"recon={float(comps['recon']):.4f} kl={float(comps['kl']):.4f} "
                      f"infil={float(comps.get('infil', 0)):.4f} "
                      f"mitre={float(comps.get('mitre', 0)):.4f}")
            global_step += 1

        val = evaluate(model, val_loader, device)
        f1 = val["f1"] if val["f1"] == val["f1"] else -1.0     # NaN guard
        history.append({"epoch": epoch, **val})
        print(f"[val] epoch {epoch}: F1={val['f1']:.4f} P={val['precision']:.4f} "
              f"R={val['recall']:.4f} ROC-AUC={val['roc_auc']:.4f} PR-AUC={val['pr_auc']:.4f}")

        if f1 > best_f1:
            best_f1, best_epoch, epochs_no_improve = f1, epoch, 0
            torch.save({
                "model_state": model.state_dict(),
                "config": cfg.to_dict(),
                "stage": 2,
                "epoch": epoch,
                "val_infil_f1": f1,
                "node_dim": 4, "edge_dim": 49, "n_stages": n_stages,
                "class_weights": class_weights.cpu().tolist(),
                "feature_stats": feature_stats,
            }, out_path)
            print(f"  * new best val F1={f1:.4f} -> saved {out_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"early stop: no val-F1 improvement in {args.patience} epochs")
                break

    with open(os.path.join(ckpt_dir, "stage2_history.json"), "w", encoding="utf-8") as fh:
        json.dump({"best_epoch": best_epoch, "best_val_f1": best_f1,
                   "history": history}, fh, indent=2)
    print(f"\nStage-2 done. best val infiltration F1={best_f1:.4f} @ epoch {best_epoch}")
    print(f"checkpoint -> {out_path}")


if __name__ == "__main__":
    main()
