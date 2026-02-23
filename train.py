# train.py
"""
Training loop for UnfoldNet  (Wang et al., IEEE TVT 2021).

Tự động dùng GPU nếu có (CUDA > MPS > CPU) thông qua get_device().
Checkpoint luôn lưu trên CPU để portable giữa các máy.

Public API
----------
train(cfg, rd_path, ckpt_path, log_every, tag)
    Train một model, lưu checkpoint. Returns {"net", "best_val", "hist"}.

train_all_L(base_cfg, rd_path, ckpt_dir, L_range, rho_vals, epochs, log_every)
    Train một model cho mỗi (L, rho). Resume-safe.

ckpt_path_for(ckpt_dir, L, rho)
    Canonical checkpoint path (dùng chung với simulate.py).
"""
from __future__ import annotations

import copy
import os
from typing import Dict, Any, List, Optional, Tuple

import torch
import numpy as np

from config import Config, get_device
from utils import (
    generate_batch, generate_single,
    effective_channel, objective, pmui, radar_mse, sum_rate,
    set_seed,
)
from model import UnfoldNet

DTYPE = torch.complex64
N_VAL = 8          # số validation realisations mỗi epoch


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def ckpt_path_for(ckpt_dir: str, L: int, rho: float) -> str:
    """ckpt_dir/unfold_L{L}_rho{rho:.2f}.pt"""
    return os.path.join(ckpt_dir, f"unfold_L{L}_rho{rho:.2f}.pt")


def _bad_grad(net) -> Optional[Tuple[str, float]]:
    for name, p in net.named_parameters():
        if p.grad is None:
            continue
        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
            return name, float(p.grad.abs().nan_to_num(0).max().item())
    return None


def _val_loss(net, cfg, val_set) -> Tuple[float, float]:
    norm = 1.0 / (cfg.M * cfg.K)
    losses, srates = [], []
    net.eval()
    with torch.no_grad():
        for d in val_set:
            X, U, th = net(d["H_bu"], d["H_ru"], d["H_br"], d["S"], d["Rd"])
            H_e = effective_channel(d["H_bu"], d["H_ru"], d["H_br"], th, cfg.eps)
            losses.append(float((norm * objective(H_e, X, d["S"], U, cfg.rho)).item()))
            srates.append(sum_rate(H_e, X, d["S"], cfg.N0, cfg.M))
    return float(np.mean(losses)), float(np.mean(srates))


def _banner(cfg: Config, device: str, tag: str = "") -> None:
    label = f" [{tag}]" if tag else ""
    gpu_info = ""
    if device == "cuda":
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_info = f"  ({name}, {mem:.1f} GB)"
    print("=" * 68)
    print(f"  UnfoldNet Training{label}  (Wang et al., IEEE TVT 2021)")
    print("=" * 68)
    print(f"  device = {device}{gpu_info}")
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
    """Train một UnfoldNet và lưu checkpoint."""
    device = get_device()

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Model lên GPU
    net = UnfoldNet(cfg).to(device)

    opt = torch.optim.Adam(
        net.parameters(), lr=cfg.lr,
        betas=(0.9, 0.999), weight_decay=cfg.weight_decay,
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=40, min_lr=1e-7,
    )

    # Validation set — sinh trực tiếp trên device
    val_set = [
        generate_single(cfg.N, cfg.K, cfg.M, cfg.L, rd_path, device, DTYPE)
        for _ in range(N_VAL)
    ]

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

    _banner(cfg, device, tag)

    for ep in range(cfg.epochs):
        net.train()

        # Sinh batch trực tiếp trên device
        b = generate_batch(
            cfg.batch_size, cfg.N, cfg.K, cfg.M, cfg.L,
            rd_path, device, DTYPE,
        )
        S, H_bu, H_ru, H_br, Rd = (
            b["S"], b["H_bu"], b["H_ru"], b["H_br"], b["Rd"]
        )

        opt.zero_grad(set_to_none=True)
        X, U, theta = net(H_bu, H_ru, H_br, S, Rd)
        H_eff = effective_channel(H_bu, H_ru, H_br, theta, cfg.eps)
        loss  = norm * objective(H_eff, X, S, U, cfg.rho)

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

        lv       = float(loss.item())
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

    # Lưu checkpoint trên CPU → portable giữa GPU/CPU machines
    save_state = best_state if best_state is not None else last_ok
    cpu_state  = {k: v.cpu() for k, v in save_state.items()}
    os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)), exist_ok=True)
    torch.save({"state_dict": cpu_state, "cfg": cfg.__dict__}, ckpt_path)
    print(f"  ✓ Checkpoint → {ckpt_path}  (best_val={best_val:.4e})\n")

    net.load_state_dict(save_state)
    return {"net": net, "best_val": best_val, "hist": hist}


# ─────────────────────────────────────────────────────────────────────────────
# Train tất cả (L, ρ) models cho Fig 3b
# ─────────────────────────────────────────────────────────────────────────────

def train_all_L(
    base_cfg:  Config,
    rd_path:   str,
    ckpt_dir:  str,
    L_range:   List[int]   = None,
    rho_vals:  List[float] = None,
    epochs:    int         = 800,
    log_every: int         = 100,
) -> Dict[Tuple[int, float], str]:
    """
    Train một model cho mỗi cặp (L, rho).
    Checkpoint: ckpt_dir/unfold_L{L}_rho{rho:.2f}.pt
    Resume-safe: bỏ qua checkpoint đã tồn tại.
    """
    if L_range  is None: L_range  = [4, 8, 16, 32, 64]
    if rho_vals is None: rho_vals = [0.01, 0.1, 0.5]

    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_map: Dict[Tuple[int, float], str] = {}
    total = len(L_range) * len(rho_vals)

    print("\n" + "=" * 68)
    print(f"  train_all_L: {total} models  "
          f"L∈{L_range}  ρ∈{rho_vals}  epochs_each={epochs}")
    print(f"  device = {get_device()}")
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
                "SNR_dB": 6.0,   # paper Fig 3b operating point
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