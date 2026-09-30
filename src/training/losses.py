"""Full training objective (Section 7.6.1).

L_total = L_reconstruction (MSE)
        + β_t · mean(KL_t)
        + λ1 · L_infiltration (focal loss, γ=2)   # Stage 2 only
        + λ2 · L_mitre (class-weighted CE)         # Stage 2 only

β follows a linear 0->1 ramp over the first 10% of Stage-1 steps, then holds at
1 (KL annealing, the posterior-collapse fix). Infiltration/MITRE terms are
gated off in Stage 1 (λ=0) and on in Stage 2.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0,
               alpha: float = 0.25) -> torch.Tensor:
    """Binary focal loss on raw logits (γ=2). Mean over all elements."""
    targets = targets.float()
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * (1.0 - p_t) ** gamma * ce
    return loss.mean()


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def kl_term(kl: torch.Tensor) -> torch.Tensor:
    """Mean per-step KL over the sequence and batch."""
    return kl.mean()


def mitre_ce(logits: torch.Tensor, targets: torch.Tensor,
             class_weights: torch.Tensor | None) -> torch.Tensor:
    """Class-weighted cross-entropy over [B,W,C] logits vs [B,W] targets."""
    C = logits.shape[-1]
    return F.cross_entropy(logits.reshape(-1, C), targets.reshape(-1).long(),
                           weight=class_weights)


def beta_schedule(global_step: int, total_stage1_steps: int,
                  anneal_fraction: float = 0.10) -> float:
    """Linear 0->1 ramp over the first ``anneal_fraction`` of Stage-1 steps."""
    ramp_steps = max(1, int(total_stage1_steps * anneal_fraction))
    return float(min(1.0, global_step / ramp_steps))


def total_loss(outputs: dict, batch: dict, beta: float,
               lambda_infil: float = 0.0, lambda_mitre: float = 0.0,
               class_weights: torch.Tensor | None = None) -> dict:
    """Assemble L_total and return each component (for logging).

    ``lambda_infil``/``lambda_mitre`` = 0 in Stage 1 (unsupervised), 1 (default)
    in Stage 2.
    """
    l_recon = reconstruction_loss(outputs["recon"], outputs["recon_target"])
    l_kl = kl_term(outputs["kl"])
    total = l_recon + beta * l_kl

    comps = {"recon": l_recon, "kl": l_kl}
    if lambda_infil > 0.0:
        l_infil = focal_loss(outputs["infil_logit"], batch["infil_target"])
        total = total + lambda_infil * l_infil
        comps["infil"] = l_infil
    if lambda_mitre > 0.0:
        l_mitre = mitre_ce(outputs["mitre_logits"], batch["stage_target"], class_weights)
        total = total + lambda_mitre * l_mitre
        comps["mitre"] = l_mitre

    comps["total"] = total
    comps["beta"] = torch.as_tensor(beta)
    return comps
