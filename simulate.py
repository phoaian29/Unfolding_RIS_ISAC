# simulate.py
"""
Generate paper figures by loading pre-trained checkpoints.
Must run  python main.py --mode train  (or  python main.py)  first.

Paper: Wang et al., "Joint Waveform Design and Passive Beamforming for
       RIS-Assisted DFRC System", IEEE TVT 2021.

Figures:
  Fig 2  – Convergence of proposed algorithm
  Fig 3a – Sum rate vs. transmit SNR  (L=16, ρ ∈ {0.01,0.1,0.5})
  Fig 3b – Sum rate vs. L             (SNR=6 dB, pre-trained per-L ckpts)
  Fig 4  – Radar beampattern MSE and shape

SINR — paper eq.(4):
  γ_k = E[|s_kj|²] / (E[|ỹ_kj − s_kj|²] + σ²)
  BPSK: E[|s_kj|²] = 1 (constant).
"""
from __future__ import annotations

import argparse
import math
import os
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import Config
from utils import (
    generate_single, load_Rd,
    effective_channel,
    compute_beampattern, ideal_beampattern_from_Rd,
)
from model import UnfoldNet
from train import ckpt_path_for

DEVICE = "cpu"
DTYPE  = torch.complex64
TARGET_ANGLES_DEG = [-60.0, 0.0, 60.0]


# ─────────────────────────────────────────────────────────────────────────────
# Rd helper
# ─────────────────────────────────────────────────────────────────────────────

def _load_Rd(N, rd_path, device="cpu", dtype=torch.complex64):
    from pathlib import Path
    if Path(rd_path).exists():
        return load_Rd(N, rd_path, device, dtype)
    # Synthesise from paper target angles
    print(f"  [Rd] {rd_path} not found — synthesising from {TARGET_ANGLES_DEG}°")
    n_idx = torch.arange(N, dtype=torch.float32)
    delta = 0.5
    cols  = [torch.exp(1j * 2 * math.pi * delta * n_idx *
                       math.sin(a * math.pi / 180.0))
             for a in TARGET_ANGLES_DEG]
    A  = torch.stack(cols, dim=1).to(dtype)
    Rd = (A @ A.conj().T) / len(TARGET_ANGLES_DEG)
    Rd = Rd / Rd.real.diag().clamp(1e-9).mean()
    return Rd.to(device=device, dtype=dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Network loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_net(cfg, ckpt_path):
    net = UnfoldNet(cfg).to(DEVICE)
    if os.path.exists(ckpt_path):
        pack = torch.load(ckpt_path, map_location=DEVICE)
        net.load_state_dict(pack["state_dict"], strict=True)
    else:
        print(f"  [WARN] {ckpt_path} not found — using random weights")
    net.eval()
    return net


# ─────────────────────────────────────────────────────────────────────────────
# Sum-rate — paper eq.(4)
# ─────────────────────────────────────────────────────────────────────────────

def _sum_rate(H_eff, X, S, N0):
    if H_eff.dim() == 3:
        H_eff, X, S = H_eff[0], X[0], S[0]
    Y = H_eff @ X
    total = 0.0
    for k in range(Y.shape[0]):
        sig  = float((S[k].abs()**2).mean().real.item())   # ≈ 1 for BPSK
        mui  = float(((Y[k] - S[k]).abs()**2).mean().real.item())
        sinr = sig / (mui + float(N0) + 1e-30)
        total += math.log2(1.0 + max(sinr, 0.0))
    return total


# ─────────────────────────────────────────────────────────────────────────────
# No-RIS (trade-off, paper [2]) baseline
# Run the network with H_ru = 0  →  H_eff = H_bu, theta has no effect.
# This exactly solves  min ρ‖H_bu X − S‖² + (1−ρ)‖X−U‖²  without RIS,
# which is the "No RIS [2]" trade-off design from the paper.
# ─────────────────────────────────────────────────────────────────────────────

def _nris_infer(net, cfg, d):
    """Run net with H_ru=0 so H_eff=H_bu (no RIS contribution)."""
    H_ru0 = torch.zeros_like(d["H_ru"])
    with torch.no_grad():
        X, U, th = net(d["H_bu"], H_ru0, d["H_br"], d["S"], d["Rd"])
    # H_eff with zero H_ru is just H_bu
    H_e = effective_channel(d["H_bu"], H_ru0, d["H_br"], th, cfg.eps)
    return X, U, H_e


def _nris_rates(base_cfg, snr_list_db, rho, n_avg, ckpt_dir, rd_path):
    """No-RIS trade-off sum-rates using per-ρ checkpoint."""
    cfg = Config(**{**vars(base_cfg), "rho": float(rho)})
    cp  = ckpt_path_for(ckpt_dir, base_cfg.L, rho)
    net = _load_net(cfg, cp)
    results = []
    for snr_db in snr_list_db:
        N0 = base_cfg.P0 / (10.0 ** (snr_db / 10.0))
        sr = []
        for _ in range(n_avg):
            d = generate_single(base_cfg.N, base_cfg.K, base_cfg.M,
                                base_cfg.L, rd_path, DEVICE, DTYPE)
            _, _, H_e = _nris_infer(net, cfg, d)
            X_nris, _, _ = _nris_infer(net, cfg, d)
            sr.append(_sum_rate(H_e, X_nris, d["S"], N0))
        results.append(float(np.mean(sr)))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 – Convergence
# ─────────────────────────────────────────────────────────────────────────────

def fig2_convergence(cfg, ckpt, rd_path,
                     out="fig2_convergence.png", n_avg=30):
    print("[Fig 2] Convergence …")
    net  = _load_net(cfg, ckpt)
    norm = 1.0 / (cfg.M * cfg.K)

    all_obj, all_pmui, all_rdr = [], [], []
    for _ in range(n_avg):
        d = generate_single(cfg.N, cfg.K, cfg.M, cfg.L, rd_path, DEVICE, DTYPE)
        with torch.no_grad():
            _, _, _, traj = net(d["H_bu"], d["H_ru"], d["H_br"],
                                d["S"], d["Rd"], return_traj=True)
        all_obj.append( [v * norm for v in traj["obj"]])
        all_pmui.append([v * norm for v in traj["pmui"]])
        all_rdr.append( [v * norm for v in traj["radar"]])

    stages = list(range(1, cfg.T + 1))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(stages, np.median(all_obj,  axis=0), "b-o",  ms=6, lw=2.0,
                label="Trade-off (UnfoldNet)")
    ax.semilogy(stages, np.median(all_pmui, axis=0), "r--s", ms=6, lw=1.8,
                label="MUI energy")
    ax.semilogy(stages, np.median(all_rdr,  axis=0), "g:^",  ms=6, lw=1.8,
                label="Beampattern MSE")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Objective value", fontsize=12)
    ax.set_title("Convergence performance of the proposed algorithm", fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.4)
    ax.set_xticks(stages)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3a – Sum rate vs. SNR
# ─────────────────────────────────────────────────────────────────────────────

def fig3a_rate_vs_snr(
    base_cfg, ckpt, rd_path,
    ckpt_dir=None,
    snr_range_db=None,
    rho_vals=(0.01, 0.2),
    n_avg=100,
    out="fig3a_rate_vs_snr.png",
):
    """
    Sum rate vs. SNR  (paper Fig 3a).
    Solid lines = RIS-aided (loads per-ρ ckpt from ckpt_dir when available).
    Dashed lines = No RIS [2] trade-off (same network, H_ru=0).
    """
    if snr_range_db is None:
        snr_range_db = [0, 2, 4, 6, 8, 10]
    print(f"[Fig 3a] Sum rate vs SNR (L={base_cfg.L}, n_avg={n_avg}) …")

    colors  = ["#1565C0", "#E65100", "#2E7D32"]
    markers = ["o", "s", "^"]
    fig, ax = plt.subplots(figsize=(7, 5))

    for idx, rho in enumerate(rho_vals):
        cfg_r = Config(**{**vars(base_cfg), "rho": float(rho)})
        # Prefer per-rho checkpoint from ckpt_dir; fall back to main ckpt
        if ckpt_dir is not None:
            cp = ckpt_path_for(ckpt_dir, base_cfg.L, rho)
            net = _load_net(cfg_r, cp if os.path.exists(cp) else ckpt)
        else:
            net = _load_net(cfg_r, ckpt)

        ris_rates, nris_rates = [], []
        for snr_db in snr_range_db:
            N0 = base_cfg.P0 / (10.0 ** (snr_db / 10.0))
            ris_sr, nris_sr = [], []
            for _ in range(n_avg):
                d = generate_single(base_cfg.N, base_cfg.K, base_cfg.M,
                                    base_cfg.L, rd_path, DEVICE, DTYPE)
                # RIS-aided
                with torch.no_grad():
                    X, _, th = net(d["H_bu"], d["H_ru"], d["H_br"],
                                   d["S"], d["Rd"])
                H_e = effective_channel(d["H_bu"], d["H_ru"], d["H_br"],
                                        th, cfg_r.eps)
                ris_sr.append(_sum_rate(H_e, X, d["S"], N0))
                # No-RIS: same net but H_ru = 0
                X_nr, _, H_e_nr = _nris_infer(net, cfg_r, d)
                nris_sr.append(_sum_rate(H_e_nr, X_nr, d["S"], N0))

            ris_rates.append(float(np.mean(ris_sr)))
            nris_rates.append(float(np.mean(nris_sr)))

        c = colors[idx % len(colors)]
        m = markers[idx % len(markers)]
        ax.plot(snr_range_db, ris_rates,
                color=c, marker=m, ms=6, lw=1.8,
                label=f"RIS-aided, trade-off (ρ={rho})")
        ax.plot(snr_range_db, nris_rates,
                color=c, marker=m, ms=6, lw=1.5, linestyle="--",
                label=f"No RIS [2], trade-off (ρ={rho})")

    ax.set_xlabel("Transmit SNR (dB)", fontsize=12)
    ax.set_ylabel("Sum rate (bps/Hz)", fontsize=12)
    ax.set_title(f"Sum rate vs. Transmit SNR, $L={base_cfg.L}$", fontsize=12)
    ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.35)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3b – Sum rate vs. L  (load pre-trained per-L checkpoints)
# ─────────────────────────────────────────────────────────────────────────────

def fig3b_rate_vs_L(
    base_cfg, ckpt_dir, rd_path,
    L_range=None,
    rho_vals=(0.01, 0.2),
    snr_db=6.0,
    n_avg=50,
    out="fig3b_rate_vs_L.png",
):
    """
    Loads checkpoints from ckpt_dir/unfold_L{L}_rho{rho:.2f}.pt.
    All checkpoints must be produced by train_all_L() during training.
    NO retraining here.
    """
    if L_range is None: L_range = [4, 8, 16, 32, 64]
    print(f"[Fig 3b] Sum rate vs L (SNR={snr_db} dB, n_avg={n_avg}) …")
    N0 = base_cfg.P0 / (10.0 ** (snr_db / 10.0))

    colors  = ["#1565C0", "#E65100", "#C62828"]
    markers = ["o", "s", "^"]
    fig, ax = plt.subplots(figsize=(7, 5))

    for idx, rho in enumerate(rho_vals):
        rates = []
        for L in L_range:
            cp    = ckpt_path_for(ckpt_dir, L, rho)
            cfg_L = Config(**{
                **vars(base_cfg),
                "L":      int(L),
                "rho":    float(rho),
                "SNR_dB": float(snr_db),
            })
            net = _load_net(cfg_L, cp)

            sr_list = []
            for _ in range(n_avg):
                d = generate_single(cfg_L.N, cfg_L.K, cfg_L.M, int(L),
                                    rd_path, DEVICE, DTYPE)
                with torch.no_grad():
                    X, _, th = net(d["H_bu"], d["H_ru"], d["H_br"],
                                   d["S"], d["Rd"])
                H_e = effective_channel(d["H_bu"], d["H_ru"], d["H_br"],
                                        th, cfg_L.eps)
                sr_list.append(_sum_rate(H_e, X, d["S"], N0))
            r = float(np.mean(sr_list))
            rates.append(r)
            print(f"    L={L:2d} ρ={rho:.2f} → {r:.2f} bps/Hz")

        ax.plot(L_range, rates,
                color=colors[idx], marker=markers[idx], ms=6, lw=1.8,
                label=f"RIS-aided, trade-off (ρ={rho})")

    ax.set_xlabel("Number of RIS elements $L$", fontsize=12)
    ax.set_ylabel("Sum rate (bps/Hz)", fontsize=12)
    ax.set_title(f"Sum rate vs. $L$,  SNR = {snr_db} dB", fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.35)
    ax.set_xticks(L_range)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 – Beampattern
# ─────────────────────────────────────────────────────────────────────────────

def fig4_beampattern(
    base_cfg, ckpt, rd_path,
    ckpt_dir=None,
    rho_range=None,
    shape_rhos=(0.01, 0.2),
    n_avg=30,
    out="fig4_beampattern.png",
):
    """
    Paper Fig 4:
      Left  (4b) — MSE of beampattern vs ρ  (RIS-aided X  vs  No-RIS X)
      Right (4a) — Beampattern shape for desired, RIS ρ∈shape_rhos, No-RIS ρ=shape_rhos[0]
    Per-ρ checkpoints from ckpt_dir are preferred; falls back to single ckpt.
    No-RIS beampattern uses network run with H_ru=0 (trade-off without RIS).
    """
    if rho_range is None:
        rho_range = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2]
    print("[Fig 4] Beampattern …")

    Rd_cpu   = _load_Rd(base_cfg.N, rd_path)
    Pt, angs = ideal_beampattern_from_Rd(Rd_cpu, base_cfg.N, n_angles=361)

    def _ckpt_for_rho(rho):
        if ckpt_dir is not None:
            cp = ckpt_path_for(ckpt_dir, base_cfg.L, rho)
            if os.path.exists(cp):
                return cp
        return ckpt

    # ── Fig 4(b): MSE vs ρ ───────────────────────────────────────────────────
    mse_ris, mse_nris = [], []
    for rho in rho_range:
        cfg = Config(**{**vars(base_cfg), "rho": float(rho)})
        net = _load_net(cfg, _ckpt_for_rho(rho))
        mr, mn = [], []
        for _ in range(n_avg):
            d = generate_single(cfg.N, cfg.K, cfg.M, cfg.L, rd_path, DEVICE, DTYPE)
            with torch.no_grad():
                X, _, _ = net(d["H_bu"], d["H_ru"], d["H_br"], d["S"], d["Rd"])
            Pd, _ = compute_beampattern(X, cfg.N, n_angles=361)
            mx = Pd.max(); Pd = Pd / mx if mx > 1e-12 else Pd
            mr.append(float(np.mean((Pd - Pt) ** 2)))
            # No-RIS: H_ru = 0, use X_nris beampattern
            X_nr, _, _ = _nris_infer(net, cfg, d)
            Pn, _ = compute_beampattern(X_nr, cfg.N, n_angles=361)
            mx = Pn.max(); Pn = Pn / mx if mx > 1e-12 else Pn
            mn.append(float(np.mean((Pn - Pt) ** 2)))
        mse_ris.append(float(np.mean(mr)))
        mse_nris.append(float(np.mean(mn)))

    # ── Fig 4(a): beampattern shapes ─────────────────────────────────────────
    # Collect averaged beampatterns for each shape_rho (RIS + No-RIS)
    shape_data = {}   # rho → {"ris": array, "nris": array}
    for rho in shape_rhos:
        cfg = Config(**{**vars(base_cfg), "rho": float(rho)})
        net = _load_net(cfg, _ckpt_for_rho(rho))
        Pd_acc = np.zeros(361); Pn_acc = np.zeros(361)
        for _ in range(n_avg):
            d = generate_single(cfg.N, cfg.K, cfg.M, cfg.L, rd_path, DEVICE, DTYPE)
            with torch.no_grad():
                Xd, _, _ = net(d["H_bu"], d["H_ru"], d["H_br"], d["S"], d["Rd"])
            Pd, _ = compute_beampattern(Xd, cfg.N, n_angles=361)
            Pd_acc += Pd
            Xn, _, _ = _nris_infer(net, cfg, d)
            Pn, _ = compute_beampattern(Xn, cfg.N, n_angles=361)
            Pn_acc += Pn
        Pd_avg = Pd_acc / n_avg; Pd_avg /= max(Pd_avg.max(), 1e-12)
        Pn_avg = Pn_acc / n_avg; Pn_avg /= max(Pn_avg.max(), 1e-12)
        shape_data[rho] = {"ris": Pd_avg, "nris": Pn_avg}

    # ── Layout ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: MSE vs ρ  (Fig 4b)
    ax = axes[0]
    ax.semilogy(rho_range, mse_ris,  "b-o",  ms=6, lw=2.0, label="RIS-assisted")
    ax.semilogy(rho_range, mse_nris, "r--s", ms=6, lw=1.8, label="No RIS [2]")
    ax.set_xlabel(r"Weighting factor $\rho$", fontsize=12)
    ax.set_ylabel("MSE of beampattern", fontsize=12)
    ax.set_title(f"MSE of beampattern,  $N={base_cfg.N}$, $K={base_cfg.K}$",
                 fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.35)

    # Right: beampattern shapes  (Fig 4a)
    ax2 = axes[1]
    ax2.plot(angs, Pt, "k--", lw=2.2, label="Desired beampattern")
    shape_colors = ["#1565C0", "#C62828", "#2E7D32", "#6A1B9A"]
    ci = 0
    for rho in shape_rhos:
        ax2.plot(angs, shape_data[rho]["ris"],
                 color=shape_colors[ci % len(shape_colors)], lw=1.5,
                 label=fr"RIS-aided, trade-off ($\rho$={rho})")
        ci += 1
    # No-RIS only for first (smallest) shape_rho, matching the paper
    rho0 = shape_rhos[0]
    ax2.plot(angs, shape_data[rho0]["nris"],
             color=shape_colors[ci % len(shape_colors)], lw=1.5, linestyle=":",
             label=fr"No RIS [2], trade-off ($\rho$={rho0})")

    ax2.set_xlabel(r"Angle $\phi$ (°)", fontsize=12)
    ax2.set_ylabel("Normalised beampattern", fontsize=12)
    ax2.set_title(f"Radar beampatterns,  $N={base_cfg.N}$, $K={base_cfg.K}$",
                  fontsize=11)
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.35)
    ax2.set_xlim(-90, 90); ax2.set_ylim(-0.05, 1.05)

    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Training history
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_history(hist, out="fig_train_history.png"):
    fig = plt.figure(figsize=(13, 4))
    gs  = gridspec.GridSpec(1, 3, figure=fig)

    def _ema(x, a=0.08):
        s, o = x[0], [x[0]]
        for v in x[1:]: s = (1-a)*s + a*v; o.append(s)
        return o

    ax1 = fig.add_subplot(gs[0])
    ax1.semilogy(hist["train_obj"], "b-", lw=0.5, alpha=0.25)
    ax1.semilogy(_ema(hist["train_obj"]), "b-", lw=1.8, label="Train (EMA)")
    ax1.semilogy(hist["val_obj"], "r-", lw=1.5, label="Val (avg 8 real.)")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Normalised objective")
    ax1.set_title("Training / Validation loss")
    ax1.legend(); ax1.grid(True, which="both", alpha=0.4)

    ax2 = fig.add_subplot(gs[1])
    ax2.semilogy(hist["pmui"],  "r-", lw=1.2, label="MUI energy")
    ax2.semilogy(hist["radar"], "g-", lw=1.2, label="Beampattern MSE")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Normalised metric")
    ax2.set_title("MUI & Beampattern MSE (train batch)")
    ax2.legend(); ax2.grid(True, which="both", alpha=0.4)

    ax3 = fig.add_subplot(gs[2])
    sr   = np.array(hist["sumrate"])
    w    = min(10, len(sr))
    sr_s = np.convolve(sr, np.ones(w)/w, mode="valid")
    ax3.plot(sr, "m-", lw=0.8, alpha=0.4, label="Sum rate (raw)")
    ax3.plot(np.arange(w-1, len(sr)), sr_s, "m-", lw=2.0, label=f"MA-{w}")
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("Sum rate (bps/Hz)")
    ax3.set_title("Achievable sum rate (val)")
    ax3.legend(); ax3.grid(True, alpha=0.4)

    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"  Training history → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI (for running simulate.py standalone)
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rd",       default="Rd.mat")
    p.add_argument("--ckpt",     default="unfold_ckpt.pt")
    p.add_argument("--ckpt_dir", default="ckpts")
    p.add_argument("--out",      default="results")
    p.add_argument("--navg",     type=int,   default=50)
    p.add_argument("--snr_3b",   type=float, default=6.0)
    p.add_argument("--L_range",  type=int,   nargs="+", default=[4,8,16,32,64])
    p.add_argument("--rho_vals", type=float, nargs="+", default=[0.01,0.1,0.5])
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = Config()
    def _p(n): return os.path.join(args.out, n)

    fig2_convergence( cfg,  args.ckpt,     args.rd,  _p("fig2_convergence.png"))
    fig3a_rate_vs_snr(cfg,  args.ckpt,     args.rd,
                      ckpt_dir=args.ckpt_dir,
                      rho_vals=args.rho_vals, n_avg=args.navg,
                      out=_p("fig3a_rate_vs_snr.png"))
    fig3b_rate_vs_L(  cfg,  args.ckpt_dir, args.rd,
                      L_range=args.L_range, rho_vals=args.rho_vals,
                      snr_db=args.snr_3b,  n_avg=args.navg,
                      out=_p("fig3b_rate_vs_L.png"))
    fig4_beampattern( cfg,  args.ckpt,     args.rd,
                      ckpt_dir=args.ckpt_dir,
                      n_avg=args.navg, out=_p("fig4_beampattern.png"))
    print(f"\nAll figures saved to '{args.out}/'")


if __name__ == "__main__":
    main()