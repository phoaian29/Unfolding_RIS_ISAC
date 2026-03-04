# train.py
"""
Training loop for UnfoldNet  (Wang et al., IEEE TVT 2021).

Public API
----------
train(cfg, rd_path, ckpt_path, log_every, tag)
    Train one (N,K,M,L,rho) configuration and save checkpoint.
    Returns {"net", "best_val", "hist"}.

train_all_L(base_cfg, rd_path, ckpt_dir, L_range, rho_vals, epochs, log_every)
    Train one model per (L, rho) and save each checkpoint in ckpt_dir.
    Skips combinations whose checkpoint already exists (resume-safe).
    Returns dict: (L, rho) -> ckpt_path.

ckpt_path_for(ckpt_dir, L, rho)
    Canonical checkpoint filename helper (used by simulate.py).
"""
from __future__ import annotations

import copy
import math
import os
from typing import Dict, Any, List, Optional, Tuple

import torch
import numpy as np

from config import Config
from utils import (
    generate_batch, generate_single,
    effective_channel, objective, pmui, radar_mse,
    set_seed,
)
from model import UnfoldNet

DEVICE = "cpu"
DTYPE  = torch.complex64
N_VAL  = 32   # 8 was too few; small fixed val set biases loss low


# ─────────────────────────────────────────────────────────────────────────────
# Sum-rate — paper eq.(4)
# ─────────────────────────────────────────────────────────────────────────────

def _sum_rate_paper(H_eff, X, S, N0):
    if H_eff.dim() == 3:
        H_eff, X, S = H_eff[0], X[0], S[0]
    Y = H_eff @ X
    total = 0.0
    for k in range(Y.shape[0]):
        sig  = float((S[k].abs()**2).mean().real.item())
        mui  = float(((Y[k] - S[k]).abs()**2).mean().real.item())
        sinr = sig / (mui + float(N0) + 1e-30)
        total += math.log2(1.0 + max(sinr, 0.0))
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bad_grad(net):
    for name, p in net.named_parameters():
        if p.grad is None:
            continue
        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
            return name, float(p.grad.abs().nan_to_num(0).max().item())
    return None


def _val_loss(net, cfg, val_set):
    norm = 1.0 / (cfg.M * cfg.K)
    losses, srates = [], []
    net.eval()
    with torch.no_grad():
        for d in val_set:
            X, U, th = net(d["H_bu"], d["H_ru"], d["H_br"], d["S"], d["Rd"])
            H_e = effective_channel(d["H_bu"], d["H_ru"], d["H_br"], th, cfg.eps)
            losses.append(float((norm * objective(H_e, X, d["S"], U, cfg.rho)).item()))
            srates.append(_sum_rate_paper(H_e, X, d["S"], cfg.N0))
    return float(np.mean(losses)), float(np.mean(srates))


def ckpt_path_for(ckpt_dir: str, L: int, rho: float) -> str:
    """Canonical path: ckpt_dir/unfold_L{L}_rho{rho:.2f}.pt"""
    return os.path.join(ckpt_dir, f"unfold_L{L}_rho{rho:.2f}.pt")


def _banner(cfg, tag=""):
    label = f" [{tag}]" if tag else ""
    print("=" * 68)
    print(f"  UnfoldNet Training{label}  (Wang et al., IEEE TVT 2021)")
    print("=" * 68)
    print(f"  N={cfg.N} K={cfg.K} M={cfg.M} L={cfg.L}  "
          f"P0={cfg.P0_dbm}dBm  SNR={cfg.SNR_dB}dB  ρ={cfg.rho}")
    print(f"  T={cfg.T}  J_x={cfg.J_x}  J_θ={cfg.J_theta}  "
          f"batch={cfg.batch_size}  epochs={cfg.epochs}  lr={cfg.lr}")
    print("=" * 68)


# ─────────────────────────────────────────────────────────────────────────────
# Core training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    cfg:       Config,
    rd_path:   str = "Rd.mat",
    ckpt_path: str = "unfold_ckpt.pt",
    log_every: int = 10,
    tag:       str = "",
) -> Dict[str, Any]:
    if cfg.seed is not None:
        set_seed(cfg.seed)

    net = UnfoldNet(cfg).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr,
                            betas=(0.9, 0.999), weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=40, min_lr=1e-7,
    )

    val_set = [generate_single(cfg.N, cfg.K, cfg.M, cfg.L, rd_path, DEVICE, DTYPE)
               for _ in range(N_VAL)]

    norm       = 1.0 / (cfg.M * cfg.K)
    best_val   = float("inf")
    best_state: Optional[dict] = None
    last_ok    = copy.deepcopy(net.state_dict())
    ema_loss   = None
    ema_alpha  = 0.05

    hist: Dict[str, list] = {
        "train_obj": [], "val_obj": [],
        "pmui": [], "radar": [], "sumrate": [],
    }

    _banner(cfg, tag)

    for ep in range(cfg.epochs):
        net.train()
        b = generate_batch(cfg.batch_size, cfg.N, cfg.K, cfg.M, cfg.L,
                           rd_path, DEVICE, DTYPE)
        S, H_bu, H_ru, H_br, Rd = (
            b["S"], b["H_bu"], b["H_ru"], b["H_br"], b["Rd"]
        )

        opt.zero_grad(set_to_none=True)
        X, U, theta = net(H_bu, H_ru, H_br, S, Rd)
        H_eff = effective_channel(H_bu, H_ru, H_br, theta, cfg.eps)
        # U from ULayer is a straight-through estimator: U = U_val + (X - X.detach()).
        # radar_mse(X, U) collapses to ||X.detach() - U_val||^2 = constant → zero gradient.
        # U.detach() == U_val (the actual Procrustes target), so radar_mse(X, U.detach())
        # = ||X - U_val||^2 with gradient 2*(X - U_val) w.r.t. X — the correct signal.
        loss  = norm * objective(H_eff, X, S, U.detach(), cfg.rho)

        if not torch.isfinite(loss):
            for g in opt.param_groups:
                g["lr"] = max(g["lr"] * 0.5, 1e-7)
            net.load_state_dict(last_ok)
            if ep % log_every == 0:
                print(f"  [{ep:04d}] BAD loss → rollback")
            continue

        loss.backward()
        bad = _bad_grad(net)
        if bad is not None:
            for g in opt.param_groups:
                g["lr"] = max(g["lr"] * 0.5, 1e-7)
            net.load_state_dict(last_ok)
            if ep % log_every == 0:
                print(f"  [{ep:04d}] BAD grad '{bad[0]}' → rollback")
            continue

        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip_norm)
        opt.step()
        last_ok = copy.deepcopy(net.state_dict())

        lv = float(loss.item())
        ema_loss = lv if ema_loss is None else (
            (1 - ema_alpha) * ema_loss + ema_alpha * lv
        )

        vl, sr = _val_loss(net, cfg, val_set)
        sched.step(vl)

        if vl < best_val:
            best_val   = vl
            best_state = copy.deepcopy(net.state_dict())

        with torch.no_grad():
            pm = float((norm * pmui(H_eff, X, S)).item())
            rd = float((norm * radar_mse(X, U)).item())

        hist["train_obj"].append(lv)
        hist["val_obj"].append(vl)
        hist["pmui"].append(pm)
        hist["radar"].append(rd)
        hist["sumrate"].append(sr)

        if ep % log_every == 0:
            lr_now = float(opt.param_groups[0]["lr"])
            print(
                f"  [{ep:04d}] train={lv:.3e}(ema={ema_loss:.3e})"
                f" val={vl:.3e} best={best_val:.3e} lr={lr_now:.1e}"
                f" | PMUI={pm:.3e} radar={rd:.3e} R={sr:.2f}bps/Hz"
            )

    # Save checkpoint
    save_state = best_state if best_state is not None else last_ok
    os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)), exist_ok=True)
    torch.save({"state_dict": save_state, "cfg": cfg.__dict__}, ckpt_path)
    print(f"  ✓ Checkpoint → {ckpt_path}  (best_val={best_val:.4e})\n")

    net.load_state_dict(save_state)
    return {"net": net, "best_val": best_val, "hist": hist}


# ─────────────────────────────────────────────────────────────────────────────
# Train all (L, ρ) models for Fig 3b
# ─────────────────────────────────────────────────────────────────────────────

def train_all_L(
    base_cfg:  Config,
    rd_path:   str,
    ckpt_dir:  str,
    L_range:   List[int]   = None,
    rho_vals:  List[float] = None,
    epochs:    int         = 400,
    log_every: int         = 100,
) -> Dict[Tuple[int, float], str]:
    """
    Train one UnfoldNet per (L, rho) pair for Fig 3b.
    Checkpoints saved as:  ckpt_dir/unfold_L{L}_rho{rho:.2f}.pt
    Skips existing checkpoints (resume-safe).
    """
    if L_range  is None: L_range  = [4, 8, 16, 32, 64]
    if rho_vals is None: rho_vals = [0.1, 0.2]

    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_map: Dict[Tuple[int, float], str] = {}
    total = len(L_range) * len(rho_vals)

    print("\n" + "=" * 68)
    print(f"  train_all_L: {total} models  "
          f"L∈{L_range}  ρ∈{rho_vals}  epochs_each={epochs}")
    print("=" * 68 + "\n")

    done = 0
    for L in L_range:
        for rho in rho_vals:
            done += 1
            cp = ckpt_path_for(ckpt_dir, L, rho)

            if os.path.exists(cp):
                print(f"  [{done}/{total}] SKIP  L={L} ρ={rho:.2f}  ({cp})")
                ckpt_map[(L, rho)] = cp
                continue

            print(f"\n  [{done}/{total}] Training  L={L}  ρ={rho:.2f} …")
            cfg_l = Config(**{
                **vars(base_cfg),
                "L":      int(L),
                "rho":    float(rho),
                "epochs": int(epochs),
                # Train at SNR=6dB (paper Fig 3b operating point)
                "SNR_dB": 6.0,
            })
            train(cfg_l, rd_path=rd_path, ckpt_path=cp,
                  log_every=log_every, tag=f"L={L} ρ={rho:.2f}")
            ckpt_map[(L, rho)] = cp

    print("  train_all_L complete.\n")
    return ckpt_map


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = Config()
    train(cfg)