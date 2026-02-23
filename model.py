# model.py
"""
UnfoldNet: T stages of (X-step → U-step → Θ-step), all differentiable.

Each stage has its own learnable step-sizes (β for X-layer, α for Θ-layer).
U-layer has no learnable parameters; it performs the Procrustes projection.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from utils import effective_channel, project_unit_modulus, init_XU
from xlayer import XLayer
from ulayer import ULayer
from thetalayer import ThetaLayer


class UnfoldNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.T   = cfg.T

        self.xlayers = nn.ModuleList([
            XLayer(
                J=cfg.J_x,
                beta_init=cfg.beta_x_init,
                diag_reg=cfg.diag_reg,
                eps=cfg.eps,
                max_beta=cfg.max_beta_x,
            ) for _ in range(cfg.T)
        ])
        self.ulayers     = nn.ModuleList([ULayer() for _ in range(cfg.T)])
        self.thetalayers = nn.ModuleList([
            ThetaLayer(
                J=cfg.J_theta,
                alpha_init=cfg.alpha_theta_init,
                eps=cfg.eps,
                max_alpha=cfg.max_alpha_theta,
            ) for _ in range(cfg.T)
        ])

    def forward(self, H_bu, H_ru, H_br, S, Rd, return_traj: bool = False):
        cfg          = self.cfg
        device, dtype = H_bu.device, H_bu.dtype
        is_batch     = (H_bu.dim() == 3)

        # ── Initialise ────────────────────────────────────────────────────────
        L = H_br.shape[-2]
        if is_batch:
            B     = H_bu.shape[0]
            theta = project_unit_modulus(
                torch.ones(B, L, device=device, dtype=dtype), cfg.eps)
        else:
            theta = project_unit_modulus(
                torch.ones(L, device=device, dtype=dtype), cfg.eps)

        X, U = init_XU(Rd.to(device=device, dtype=dtype), cfg.M, cfg.P0, cfg.eps)
        if is_batch and X.dim() == 2:
            X = X.unsqueeze(0).expand(H_bu.shape[0], *X.shape).contiguous()
            U = U.unsqueeze(0).expand(H_bu.shape[0], *U.shape).contiguous()

        traj = {"pmui": [], "radar": [], "obj": []} if return_traj else None

        # ── Unfolding stages ──────────────────────────────────────────────────
        for t in range(self.T):
            H_eff = effective_channel(H_bu, H_ru, H_br, theta, cfg.eps)
            X     = self.xlayers[t](H_eff, S, U, rho=cfg.rho, P0=cfg.P0, M=cfg.M)
            U     = self.ulayers[t](X, Rd, M=cfg.M)
            theta = self.thetalayers[t](H_bu, H_ru, H_br, X, S, theta)

            if return_traj:
                with torch.no_grad():
                    H_e = effective_channel(H_bu, H_ru, H_br, theta, cfg.eps)
                    p   = float(((H_e @ X - S).abs() ** 2).sum(dim=(-2,-1)).mean().item())
                    r   = float(((X - U).abs() ** 2).sum(dim=(-2,-1)).mean().item())
                    traj["pmui"].append(p)
                    traj["radar"].append(r)
                    traj["obj"].append(cfg.rho * p + (1.0 - cfg.rho) * r)

        if return_traj:
            return X, U, theta, traj
        return X, U, theta
