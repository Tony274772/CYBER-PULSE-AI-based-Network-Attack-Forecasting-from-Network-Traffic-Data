"""Dreamer-style RSSM world model (Section 7.5.2).

Diagonal-Gaussian latent variant (not categorical): a deterministic GRU path
``h_t`` plus a stochastic latent ``z_t``. Posterior ``q(z|h,e)`` is used when a
graph embedding ``e_t`` is observed (training, teacher-forced); prior ``p(z|h)``
drives label-free K-step imagination. The analytic Gaussian KL between them is
the surprise/anomaly signal ``kl_t`` — read out directly, never trained against
a target.

Belief state ``s_t = concat(h_t, z_t)`` (dim h_dim + z_dim) feeds every head.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ELU(),
        nn.Linear(hidden, out_dim),
    )


class RSSM(nn.Module):
    def __init__(self, embed_dim: int, h_dim: int = 128, z_dim: int = 32,
                 hidden: int = 128):
        super().__init__()
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.gru = nn.GRUCell(input_size=z_dim, hidden_size=h_dim)
        # posterior sees [h_t, e_t]; prior sees [h_t] only
        self.posterior_net = _mlp(h_dim + embed_dim, hidden, 2 * z_dim)
        self.prior_net = _mlp(h_dim, hidden, 2 * z_dim)

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    @staticmethod
    def _split(params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = params.chunk(2, dim=-1)
        # clamp logvar for numerical stability (avoids KL blow-up / underflow)
        return mu, logvar.clamp(-10.0, 10.0)

    @staticmethod
    def gaussian_kl(mu_q, logvar_q, mu_p, logvar_p) -> torch.Tensor:
        """Analytic KL( N(mu_q,var_q) || N(mu_p,var_p) ), summed over latent dim."""
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)
        kl = 0.5 * (logvar_p - logvar_q
                    + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0)
        return kl.sum(dim=-1)

    def init_state(self, batch: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(batch, self.h_dim, device=device)
        z0 = torch.zeros(batch, self.z_dim, device=device)
        return h0, z0

    # -- steps ---------------------------------------------------------------
    def observe_step(self, h_prev, z_prev, e_t):
        h_t = self.gru(z_prev, h_prev)
        mu_q, logvar_q = self._split(self.posterior_net(torch.cat([h_t, e_t], dim=-1)))
        z_t = self.reparameterize(mu_q, logvar_q)
        mu_p, logvar_p = self._split(self.prior_net(h_t))
        kl_t = self.gaussian_kl(mu_q, logvar_q, mu_p, logvar_p)
        return h_t, z_t, kl_t, (mu_q, logvar_q, mu_p, logvar_p)

    def imagine_step(self, h_prev, z_prev):
        h_t = self.gru(z_prev, h_prev)
        mu_p, logvar_p = self._split(self.prior_net(h_t))
        z_t = self.reparameterize(mu_p, logvar_p)
        return h_t, z_t
