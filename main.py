# main.py
"""
Entry point cho RIS-DFRC UnfoldNet  (Wang et al., IEEE TVT 2021).

Tự động dùng GPU nếu có (CUDA > MPS > CPU).

Usage
-----
python main.py                        # full pipeline: train → simulate
python main.py --mode train           # training only
python main.py --mode simulate        # simulation only (cần checkpoint)
python main.py --mode inference       # single inference

Key arguments
-------------
--epochs    Epochs cho main model               (default: Config.epochs = 1000)
--l_epochs  Epochs mỗi (L,ρ) model cho Fig 3b  (default: 800)
--navg      MC realisations mỗi sim point       (default: 50)
--L_range   Giá trị L cho Fig 3b               (default: 4 8 16 32 64)
--rho_vals  Giá trị ρ                           (default: 0.01 0.1 0.5)
--snr_3b    SNR (dB) cho Fig 3b                 (default: 6.0, paper §V)
--ckpt      Main model checkpoint               (default: unfold_ckpt.pt)
--ckpt_dir  Per-L checkpoint directory          (default: ckpts/)
--out       Output directory cho figures        (default: results/)
--rd        Đường dẫn tới Rd.mat               (default: Rd.mat)
--T         Unfolding stages (override Config)
"""
from __future__ import annotations

import argparse
import os

import torch

from config import Config, get_device
from train import train, train_all_L
from inference import run_inference
from simulate import (
    fig2_convergence,
    fig3a_rate_vs_snr,
    fig3b_rate_vs_L,
    fig4_beampattern,
    plot_training_history,
)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RIS-DFRC UnfoldNet (Wang et al., TVT 2021)")
    p.add_argument("--mode",     default="all",
                   choices=["all", "train", "simulate", "inference"])
    p.add_argument("--rd",       default="Rd.mat")
    p.add_argument("--ckpt",     default="unfold_ckpt.pt")
    p.add_argument("--ckpt_dir", default="ckpts")
    p.add_argument("--out",      default="results")
    p.add_argument("--epochs",   type=int,   default=None)
    p.add_argument("--l_epochs", type=int,   default=800)
    p.add_argument("--T",        type=int,   default=None)
    p.add_argument("--navg",     type=int,   default=50)
    p.add_argument("--L_range",  type=int,   nargs="+", default=[4, 8, 16, 32, 64])
    p.add_argument("--rho_vals", type=float, nargs="+", default=[0.01, 0.2])
    p.add_argument("--snr_3b",   type=float, default=6.0)
    return p.parse_args()


def _print_device_info() -> None:
    device = get_device()
    print("\n" + "━" * 68)
    if device == "cuda":
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"  🚀  GPU detected: {name}  ({mem:.1f} GB VRAM)")
        print(f"      CUDA {torch.version.cuda}  |  PyTorch {torch.__version__}")
    elif device == "mps":
        print(f"  🚀  Apple MPS detected  |  PyTorch {torch.__version__}")
    else:
        print(f"  ⚠️   No GPU found — training on CPU (may be slow)")
        print(f"      PyTorch {torch.__version__}")
    print("━" * 68 + "\n")


def main() -> None:
    args = _parse()
    os.makedirs(args.out,      exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    _print_device_info()

    # Build base config với CLI overrides
    kw: dict = {}
    if args.epochs is not None: kw["epochs"] = args.epochs
    if args.T      is not None: kw["T"]      = args.T
    base_cfg = Config(**kw)

    def _p(name: str) -> str:
        return os.path.join(args.out, name)

    # ─── TRAIN ────────────────────────────────────────────────────────────────
    if args.mode in ("all", "train"):

        print("━" * 68)
        print("  STEP 1/2 — Main model training")
        print("━" * 68)
        result = train(
            base_cfg,
            rd_path=args.rd,
            ckpt_path=args.ckpt,
            log_every=10,
            tag="main",
        )
        plot_training_history(result["hist"], _p("fig_train_history.png"))

        print("\n" + "━" * 68)
        print("  STEP 2/2 — Per-L model training (Fig 3b)")
        print("━" * 68)
        train_all_L(
            base_cfg=base_cfg,
            rd_path=args.rd,
            ckpt_dir=args.ckpt_dir,
            L_range=args.L_range,
            rho_vals=args.rho_vals,
            epochs=args.l_epochs,
            log_every=100,
        )

    # ─── INFERENCE ────────────────────────────────────────────────────────────
    if args.mode in ("all", "inference"):
        run_inference(base_cfg, ckpt_path=args.ckpt, rd_path=args.rd)

    # ─── SIMULATE ─────────────────────────────────────────────────────────────
    if args.mode in ("all", "simulate"):
        print("━" * 68)
        print("  SIMULATE — Generating paper figures")
        print("━" * 68 + "\n")

        fig2_convergence(
            base_cfg, args.ckpt, args.rd,
            out=_p("fig2_convergence.png"),
        )
        fig3a_rate_vs_snr(
            base_cfg, args.ckpt, args.rd,
            rho_vals=args.rho_vals,
            n_avg=args.navg,
            out=_p("fig3a_rate_vs_snr.png"),
        )
        fig3b_rate_vs_L(
            base_cfg, args.ckpt_dir, args.rd,
            L_range=args.L_range,
            rho_vals=args.rho_vals,
            snr_db=args.snr_3b,
            n_avg=args.navg,
            out=_p("fig3b_rate_vs_L.png"),
        )
        fig4_beampattern(
            base_cfg, args.ckpt, args.rd,
            n_avg=args.navg,
            out=_p("fig4_beampattern.png"),
        )
        print(f"\n  All figures saved to '{args.out}/'")


if __name__ == "__main__":
    main()