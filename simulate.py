# simulate.py
"""
Generate paper figures from pre-trained checkpoints.
Run  python main.py --mode train  first.

Simulation luôn chạy trên CPU (matplotlib, numpy cần CPU tensors).
Checkpoint được load với map_location='cpu' nên GPU-trained models
hoạt động hoàn toàn transparent.

Paper: Wang et al., IEEE TVT 2021  §V Simulation Results.
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
    generate_single, load_Rd, herm,
    effective_channel, objective, pmui, radar_mse, sum_rate,
    compute_beampattern, ideal_beampattern_from_Rd,
)
from model import UnfoldNet
from train import ckpt_path_for

# Simulation chạy CPU — MC loops nhỏ, matplotlib cần CPU arrays
DEVICE = "cpu"
DTYPE  = torch.complex64


# ─────────────────────────────────────────────────────────────────────────────
# Network loader — map_location="cpu" để GPU ckpt load được
# ─────────────────────────────────────────────────────────────────────────────

def _load_net(cfg: Config, ckpt_path: str) -> UnfoldNet:
    net = UnfoldNet(cfg).to(DEVICE)
    if os.path.exists(ckpt_path):
        pack = torch.load(ckpt_path, map_location="cpu")
        net.load_state_dict(pack["state_dict"], strict=True)
    else:
        print(f"  [WARN] Checkpoint not found: {ckpt_path} — using random weights")
    net.eval()
    return net


# ─────────────────────────────────────────────────────────────────────────────
# ZF (No-RIS) baseline
# ─────────────────────────────────────────────────────────────────────────────

def _zf_rate(base_cfg: Config, N0: float, rd_path: str, n_avg: int) -> float:
    sr_list = []
    for _ in range(n_avg):
        d    = generate_single(base_cfg.N, base_cfg.K, base_cfg.M,
                               base_cfg.L, rd_path, DEVICE, DTYPE)
        H_bu = d["H_bu"]; S = d["S"]
        HHh  = H_bu @ herm(H_bu)
        try:
            X_zf = herm(H_bu) @ torch.linalg.solve(HHh, S)
        except Exception:
            X_zf = herm(H_bu) @ S
        nrm  = (X_zf.abs() ** 2).sum().clamp(1e-12).sqrt()
        X_zf = X_zf * math.sqrt(base_cfg.target_power) / nrm
        sr_list.append(sum_rate(H_bu, X_zf, S, N0, base_cfg.M))
    return float(np.mean(sr_list))


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 – Convergence
# ─────────────────────────────────────────────────────────────────────────────

def fig2_convergence(
    cfg: Config, ckpt: str, rd_path: str,
    out: str = "fig2_convergence.png",
    n_avg: int = 30,
) -> None:
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
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.4)
    ax.set_xticks(stages)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3a – Sum rate vs. SNR
# ─────────────────────────────────────────────────────────────────────────────

def fig3a_rate_vs_snr(
    base_cfg: Config, ckpt: str, rd_path: str,
    snr_range_db=None,
    rho_vals=(0.01, 0.1, 0.5),
    n_avg: int = 100,
    out: str = "fig3a_rate_vs_snr.png",
) -> None:
    if snr_range_db is None:
        snr_range_db = [0, 2, 4, 6, 8, 10, 12, 15, 20]
    print(f"[Fig 3a] Sum rate vs SNR (L={base_cfg.L}, n_avg={n_avg}) …")

    colors  = ["#1565C0", "#E65100", "#C62828"]
    markers = ["o", "s", "^"]
    fig, ax = plt.subplots(figsize=(7, 5))

    for idx, rho in enumerate(rho_vals):
        cfg_r = Config(**{**vars(base_cfg), "rho": float(rho)})
        net   = _load_net(cfg_r, ckpt)
        rates = []
        for snr_db in snr_range_db:
            N0 = base_cfg.P0 / (10.0 ** (snr_db / 10.0))
            sr_list = []
            for _ in range(n_avg):
                d = generate_single(base_cfg.N, base_cfg.K, base_cfg.M,
                                    base_cfg.L, rd_path, DEVICE, DTYPE)
                with torch.no_grad():
                    X, U, th = net(d["H_bu"], d["H_ru"], d["H_br"],
                                   d["S"], d["Rd"])
                H_e = effective_channel(d["H_bu"], d["H_ru"], d["H_br"],
                                        th, cfg_r.eps)
                sr_list.append(sum_rate(H_e, X, d["S"], N0, base_cfg.M))
            rates.append(float(np.mean(sr_list)))
        ax.plot(snr_range_db, rates,
                color=colors[idx], marker=markers[idx], ms=6, lw=1.8,
                label=f"RIS Trade-off (ρ={rho})")

    zf_rates = [
        _zf_rate(base_cfg, base_cfg.P0 / (10.0 ** (s / 10.0)), rd_path, n_avg)
        for s in snr_range_db
    ]
    ax.plot(snr_range_db, zf_rates, "k--x", ms=7, lw=1.5, label="No RIS [2]")

    ax.set_xlabel("Transmit SNR (dB)", fontsize=12)
    ax.set_ylabel("Sum rate (bps/Hz)", fontsize=12)
    ax.set_title(f"Sum rate vs. Transmit SNR, $L={base_cfg.L}$", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3b – Sum rate vs. L
# ─────────────────────────────────────────────────────────────────────────────

def fig3b_rate_vs_L(
    base_cfg: Config, ckpt_dir: str, rd_path: str,
    L_range=None,
    rho_vals=(0.01, 0.1, 0.5),
    snr_db: float = 6.0,
    n_avg: int = 50,
    out: str = "fig3b_rate_vs_L.png",
) -> None:
    if L_range is None:
        L_range = [4, 8, 16, 32, 64]
    print(f"[Fig 3b] Sum rate vs L (SNR={snr_db} dB, n_avg={n_avg}) …")

    N0      = base_cfg.P0 / (10.0 ** (snr_db / 10.0))
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
                    X, U, th = net(d["H_bu"], d["H_ru"], d["H_br"],
                                   d["S"], d["Rd"])
                H_e = effective_channel(d["H_bu"], d["H_ru"], d["H_br"],
                                        th, cfg_L.eps)
                sr_list.append(sum_rate(H_e, X, d["S"], N0, cfg_L.M))
            r = float(np.mean(sr_list))
            rates.append(r)
            print(f"    L={L:2d} ρ={rho:.2f} → {r:.2f} bps/Hz")

        ax.plot(L_range, rates,
                color=colors[idx], marker=markers[idx], ms=6, lw=1.8,
                label=f"RIS Trade-off (ρ={rho})")

    zf_val = _zf_rate(base_cfg, N0, rd_path, n_avg * 2)
    ax.axhline(zf_val, color="k", linestyle="--", lw=1.5, label="No RIS [2]")

    ax.set_xlabel("Number of RIS elements $L$", fontsize=12)
    ax.set_ylabel("Sum rate (bps/Hz)", fontsize=12)
    ax.set_title(f"Sum rate vs. $L$,  SNR = {snr_db} dB", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35)
    ax.set_xticks(L_range)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 – Beampattern
# ─────────────────────────────────────────────────────────────────────────────

def fig4_beampattern(
    base_cfg: Config, ckpt: str, rd_path: str,
    rho_range=None,
    n_avg: int = 30,
    out: str = "fig4_beampattern.png",
) -> None:
    if rho_range is None:
        rho_range = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    print("[Fig 4] Beampattern …")

    Rd_cpu   = load_Rd(base_cfg.N, rd_path, device="cpu", dtype=torch.complex64)
    Pt, angs = ideal_beampattern_from_Rd(Rd_cpu, base_cfg.N, n_angles=361)

    mse_ris, mse_nris = [], []
    for rho in rho_range:
        cfg = Config(**{**vars(base_cfg), "rho": float(rho)})
        net = _load_net(cfg, ckpt)
        mr, mn = [], []
        for _ in range(n_avg):
            d = generate_single(cfg.N, cfg.K, cfg.M, cfg.L, rd_path, DEVICE, DTYPE)
            with torch.no_grad():
                X, U, _ = net(d["H_bu"], d["H_ru"], d["H_br"], d["S"], d["Rd"])
            Pd, _ = compute_beampattern(X, cfg.N, n_angles=361)
            mx = Pd.max(); Pd = Pd / mx if mx > 1e-12 else Pd
            mr.append(float(np.mean((Pd - Pt) ** 2)))
            Pu, _ = compute_beampattern(U, cfg.N, n_angles=361)
            mx = Pu.max(); Pu = Pu / mx if mx > 1e-12 else Pu
            mn.append(float(np.mean((Pu - Pt) ** 2)))
        mse_ris.append(np.mean(mr))
        mse_nris.append(np.mean(mn))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.semilogy(rho_range, mse_ris,  "b-o",  ms=6, lw=2.0, label="RIS-assisted (X)")
    ax.semilogy(rho_range, mse_nris, "r--s", ms=6, lw=1.8,
                label="No RIS (U, pure radar)")
    ax.set_xlabel(r"Weighting factor $\rho$", fontsize=12)
    ax.set_ylabel("MSE of beampattern", fontsize=12)
    ax.set_title(f"MSE of beampattern,  $N={base_cfg.N}$, $K={base_cfg.K}$",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.35)

    ax2 = axes[1]
    cfg0 = Config(**{**vars(base_cfg), "rho": 0.01})
    net0 = _load_net(cfg0, ckpt)
    Pd_acc = np.zeros(361)
    Pu_acc = np.zeros(361)
    for _ in range(n_avg):
        d = generate_single(cfg0.N, cfg0.K, cfg0.M, cfg0.L, rd_path, DEVICE, DTYPE)
        with torch.no_grad():
            Xd, Ud, _ = net0(d["H_bu"], d["H_ru"], d["H_br"], d["S"], d["Rd"])
        Pd, _ = compute_beampattern(Xd, cfg0.N, n_angles=361)
        Pu, _ = compute_beampattern(Ud, cfg0.N, n_angles=361)
        Pd_acc += Pd
        Pu_acc += Pu

    Pd_avg = Pd_acc / n_avg; Pd_avg /= max(Pd_avg.max(), 1e-12)
    Pu_avg = Pu_acc / n_avg; Pu_avg /= max(Pu_avg.max(), 1e-12)

    ax2.plot(angs, Pt,     "k--", lw=2.0, label="Desired beampattern")
    ax2.plot(angs, Pd_avg, "b-",  lw=1.5, label=r"RIS Trade-off ($\rho$=0.01)")
    ax2.plot(angs, Pu_avg, "r:",  lw=1.5, label=r"No RIS [2] ($\rho$=0.01)")
    ax2.set_xlabel(r"Angle $\phi$ (°)", fontsize=12)
    ax2.set_ylabel("Normalised beampattern", fontsize=12)
    ax2.set_title(f"Radar beampatterns,  $N={base_cfg.N}$, $K={base_cfg.K}$",
                  fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.35)
    ax2.set_xlim(-90, 90)
    ax2.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Training history
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_history(hist: dict, out: str = "fig_train_history.png") -> None:
    fig = plt.figure(figsize=(13, 4))
    gs  = gridspec.GridSpec(1, 3, figure=fig)

    def _ema(x, a=0.08):
        s, o = x[0], [x[0]]
        for v in x[1:]:
            s = (1 - a) * s + a * v
            o.append(s)
        return o

    ax1 = fig.add_subplot(gs[0])
    ax1.semilogy(hist["train_obj"], "b-", lw=0.5, alpha=0.25)
    ax1.semilogy(_ema(hist["train_obj"]), "b-", lw=1.8, label="Train (EMA)")
    ax1.semilogy(hist["val_obj"], "r-", lw=1.5, label="Val (avg 8 real.)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Normalised objective")
    ax1.set_title("Training / Validation loss")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.4)

    ax2 = fig.add_subplot(gs[1])
    ax2.semilogy(hist["pmui"],  "r-", lw=1.2, label="MUI energy")
    ax2.semilogy(hist["radar"], "g-", lw=1.2, label="Beampattern MSE")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Normalised metric")
    ax2.set_title("MUI & Beampattern MSE (train batch)")
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.4)

    ax3 = fig.add_subplot(gs[2])
    sr   = np.array(hist["sumrate"])
    w    = min(10, len(sr))
    sr_s = np.convolve(sr, np.ones(w) / w, mode="valid")
    ax3.plot(sr, "m-", lw=0.8, alpha=0.4, label="Sum rate (raw)")
    ax3.plot(np.arange(w - 1, len(sr)), sr_s, "m-", lw=2.0, label=f"MA-{w}")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Sum rate (bps/Hz)")
    ax3.set_title("Achievable sum rate (val)")
    ax3.legend()
    ax3.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Training history → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rd",       default="Rd.mat")
    p.add_argument("--ckpt",     default="unfold_ckpt.pt")
    p.add_argument("--ckpt_dir", default="ckpts")
    p.add_argument("--out",      default="results")
    p.add_argument("--navg",     type=int,   default=50)
    p.add_argument("--snr_3b",   type=float, default=6.0)
    p.add_argument("--L_range",  type=int,   nargs="+", default=[4, 8, 16, 32, 64])
    p.add_argument("--rho_vals", type=float, nargs="+", default=[0.01, 0.1, 0.5])
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = Config()
    def _p(n): return os.path.join(args.out, n)

    fig2_convergence( cfg,  args.ckpt,     args.rd,  _p("fig2_convergence.png"))
    fig3a_rate_vs_snr(cfg,  args.ckpt,     args.rd,
                      rho_vals=args.rho_vals, n_avg=args.navg,
                      out=_p("fig3a_rate_vs_snr.png"))
    fig3b_rate_vs_L(  cfg,  args.ckpt_dir, args.rd,
                      L_range=args.L_range, rho_vals=args.rho_vals,
                      snr_db=args.snr_3b,  n_avg=args.navg,
                      out=_p("fig3b_rate_vs_L.png"))
    fig4_beampattern( cfg,  args.ckpt,     args.rd,
                      n_avg=args.navg, out=_p("fig4_beampattern.png"))
    print(f"\nAll figures saved to '{args.out}/'")


if __name__ == "__main__":
    main()