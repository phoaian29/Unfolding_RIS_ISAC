# inference.py
from __future__ import annotations

import os
import torch

from config import Config, get_device
from utils import generate_single, effective_channel, objective, pmui, radar_mse, sum_rate
from model import UnfoldNet

DTYPE = torch.complex64


def run_inference(
    cfg:       Config,
    ckpt_path: str = "unfold_ckpt.pt",
    rd_path:   str = "Rd.mat",
) -> dict:
    device = get_device()

    if not os.path.exists(ckpt_path):
        print(f"[Inference] Checkpoint not found: {ckpt_path}")
        return {}

    # Checkpoint luôn được lưu trên CPU → map_location đảm bảo load đúng
    ckpt = torch.load(ckpt_path, map_location=device)
    net  = UnfoldNet(cfg).to(device)
    net.load_state_dict(ckpt["state_dict"], strict=True)
    net.eval()

    data = generate_single(cfg.N, cfg.K, cfg.M, cfg.L, rd_path, device, DTYPE)
    S, H_bu, H_ru, H_br, Rd = (
        data["S"], data["H_bu"], data["H_ru"], data["H_br"], data["Rd"]
    )

    with torch.no_grad():
        X, U, theta, traj = net(H_bu, H_ru, H_br, S, Rd, return_traj=True)
        H_eff = effective_channel(H_bu, H_ru, H_br, theta, cfg.eps)

        norm = 1.0 / (cfg.M * cfg.K)
        obj  = float((norm * objective(H_eff, X, S, U, cfg.rho)).item())
        pm   = float((norm * pmui(H_eff, X, S)).item())
        rd   = float((norm * radar_mse(X, U)).item())
        sr   = sum_rate(H_eff, X, S, cfg.N0, cfg.M)
        pw   = float((X.abs() ** 2).sum().item())

    print(f"\n[Inference] device={device}")
    print(f"  Obj={obj:.4e}  PMUI={pm:.4e}  Radar={rd:.4e}")
    print(f"  SumRate={sr:.3f} bps/Hz  ||X||²={pw:.4f}  "
          f"(target={cfg.target_power:.4f})")
    return {
        "X": X, "U": U, "theta": theta, "traj": traj,
        "obj": obj, "pmui": pm, "radar": rd, "sumrate": sr,
    }


if __name__ == "__main__":
    cfg = Config()
    run_inference(cfg)