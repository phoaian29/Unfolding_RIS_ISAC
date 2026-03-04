# main.py
"""
Entry point for RIS-DFRC UnfoldNet  (Wang et al., IEEE TVT 2021).

Usage
-----
python main.py                        # full pipeline (train → simulate)
python main.py --mode train           # training only
python main.py --mode simulate        # simulation only (needs checkpoints)
python main.py --mode inference       # single inference

Key arguments
-------------
--epochs    Epochs for main model               (default: Config.epochs=1000)
--l_epochs  Epochs per (L,ρ) model for Fig 3b  (default: 800)
--navg      MC realisations per sim point       (default: 50)
--L_range   L values for Fig 3b                 (default: 4 8 16 32 64)
--rho_vals  ρ values                            (default: 0.01 0.1 0.5)
--snr_3b    SNR (dB) for Fig 3b                 (default: 6.0, paper §V)
--ckpt      Main checkpoint                     (default: unfold_ckpt.pt)
--ckpt_dir  Per-L checkpoint directory          (default: ckpts/)
--out       Output directory for figures        (default: results/)
--rd        Path to Rd.mat                      (default: Rd.mat)
--T         Unfolding stages (overrides Config)
"""
from __future__ import annotations

import argparse
import os

from config import Config
from train import train, train_all_L
from inference import run_inference
from simulate import (
    fig2_convergence,
    fig3a_rate_vs_snr,
    # fig3b_rate_vs_L,   # disabled — training for L=16 only
    fig4_beampattern,
    plot_training_history,
)


def _parse():
    p = argparse.ArgumentParser(description="RIS-DFRC UnfoldNet")
    p.add_argument("--mode",     default="all",
                   choices=["all", "train", "simulate", "inference"])
    p.add_argument("--rd",       default="Rd.mat")
    p.add_argument("--ckpt",     default="unfold_ckpt.pt")
    p.add_argument("--ckpt_dir", default="ckpts",
                   help="Directory for per-L checkpoints (Fig 3b)")
    p.add_argument("--out",      default="results")
    p.add_argument("--epochs",   type=int,   default=None)
    p.add_argument("--l_epochs", type=int,   default=800,
                   help="Training epochs per (L,ρ) model for Fig 3b")
    p.add_argument("--T",        type=int,   default=None)
    p.add_argument("--navg",     type=int,   default=50)
    p.add_argument("--L_range",  type=int,   nargs="+", default=[4, 8, 16, 32, 64])
    p.add_argument("--rho_vals", type=float, nargs="+", default=[0.01, 0.2])
    p.add_argument("--snr_3b",   type=float, default=6.0,
                   help="SNR (dB) for Fig 3b (paper uses 6 dB)")
    return p.parse_args()


def main():
    args = _parse()
    os.makedirs(args.out,      exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Base config
    kw = {}
    if args.epochs: kw["epochs"] = args.epochs
    if args.T:      kw["T"]      = args.T
    base_cfg = Config(**kw)

    def _p(n): return os.path.join(args.out, n)

    # ─── TRAIN ────────────────────────────────────────────────────────────────
    if args.mode in ("all", "train"):

        # Step 1: Main model (L=base_cfg.L, rho=base_cfg.rho)
        print("\n" + "━"*68)
        print("  STEP 1 — Main model training  (L=16)")
        print("━"*68)
        result = train(
            base_cfg,
            rd_path=args.rd,
            ckpt_path=args.ckpt,
            log_every=10,
            tag="main",
        )
        plot_training_history(result["hist"], _p("fig_train_history.png"))

        # Step 2: Per-ρ models at L=16 (needed for Fig 3a and Fig 4)
        print("\n" + "━"*68)
        print("  STEP 2/2 — Per-ρ model training  (L=16)")
        print("━"*68)
        train_all_L(
            base_cfg=base_cfg,
            rd_path=args.rd,
            ckpt_dir=args.ckpt_dir,
            L_range=[base_cfg.L],        # L=16 only
            rho_vals=args.rho_vals,
            epochs=args.l_epochs,
            log_every=100,
        )

    # ─── INFERENCE ────────────────────────────────────────────────────────────
    if args.mode in ("all", "inference"):
        run_inference(base_cfg, ckpt_path=args.ckpt, rd_path=args.rd)

    # ─── SIMULATE ─────────────────────────────────────────────────────────────
    if args.mode in ("all", "simulate"):
        print("\n" + "━"*68)
        print("  SIMULATE — Generating paper figures")
        print("━"*68 + "\n")

        fig2_convergence(
            base_cfg, args.ckpt, args.rd,
            out=_p("fig2_convergence.png"),
        )

        fig3a_rate_vs_snr(
            base_cfg, args.ckpt, args.rd,
            ckpt_dir=args.ckpt_dir,
            rho_vals=args.rho_vals,
            n_avg=args.navg,
            out=_p("fig3a_rate_vs_snr.png"),
        )

        # # Fig 3b — disabled (requires per-L checkpoints, L=16 only mode)
        # fig3b_rate_vs_L(
        #     base_cfg, args.ckpt_dir, args.rd,
        #     L_range=args.L_range,
        #     rho_vals=args.rho_vals,
        #     snr_db=args.snr_3b,
        #     n_avg=args.navg,
        #     out=_p("fig3b_rate_vs_L.png"),
        # )

        fig4_beampattern(
            base_cfg, args.ckpt, args.rd,
            ckpt_dir=args.ckpt_dir,
            n_avg=args.navg,
            out=_p("fig4_beampattern.png"),
        )

        print(f"\n  All figures saved to '{args.out}/'")


if __name__ == "__main__":
    main()