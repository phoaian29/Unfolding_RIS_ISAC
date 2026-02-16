# inference.py
from __future__ import annotations

import os
import torch
import numpy as np

from config import MatlabScenarioConfig
from utils_complex import set_seed
from dataset import generate_single_scenario
from channels import effective_channel
from model import UnfoldNet
from metrics import pmui, objective_tradeoff, steering_matrix, radar_metrics, sum_rate
from plots import plot_convergence, plot_beampattern, LivePlot


def run_inference(cfg: MatlabScenarioConfig, rd_path: str = "Rd.mat", ckpt_path: str = "unfold_ckpt.pt", live: bool = True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.complex64

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # 1 scenario cố định để vẽ hội tụ giống paper
    data = generate_single_scenario(cfg.N, cfg.K, cfg.M, cfg.L, rd_path=rd_path, device=device, dtype=dtype)
    S = data["S"]
    H_bu = data["H_bu"]
    H_ru = data["H_ru"]
    H_br = data["H_br"]
    Rd = data["Rd"]

    net = UnfoldNet(
        T=cfg.T,                 # IMPORTANT: phải đúng T đã train (ở đây T=100)
        J_theta=cfg.J_theta,
        rho_init=cfg.rho,
        learnable_rho=True,       # để load rho_logits đã học
        learnable_alpha=True,     # để load alpha/eta đã học
        alpha_init=cfg.alpha_theta,
    ).to(device)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {ckpt_path}")

    pack = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(pack["state_dict"], strict=True)
    print(f"Loaded checkpoint: {ckpt_path}")

    net.eval()
    with torch.no_grad():
        X, U, theta, traj = net(H_bu, H_ru, H_br, S, Rd, P0=cfg.P0, M=cfg.M, return_traj=True)

        H_eff = effective_channel(H_bu, H_ru, H_br, theta)
        pmui_val = pmui(H_eff, X, S).item()
        obj_val = objective_tradeoff(H_eff, X, S, U, cfg.rho).item()
        rate_val = sum_rate(H_eff, X, S, sigma2=cfg.N0).item()
        pow_val = (torch.linalg.norm(X) ** 2).item()

    print(
        f"N={cfg.N}, K={cfg.K}, M={cfg.M}, L={cfg.L}, SNR_dB={cfg.SNR_dB}, rho={cfg.rho}, P0={cfg.P0:.6f}"
    )
    print(f"PMUI={pmui_val:.6e}, Obj={obj_val:.6e}, SumRate={rate_val:.4f}, ||X||_F^2={pow_val:.6f} (target {cfg.M*cfg.P0:.6f})")

    # live plot
    if live and traj is not None:
        lp = LivePlot()
        for t in range(len(traj["mui"])):
            lp.update(
                t=t,
                mui=traj["mui"][t],
                powv=traj["pow"][t],
                obj=traj["obj"][t],
                title=f"t={t+1}/{cfg.T} | MUI={traj['mui'][t]:.4f} | ||X||^2={traj['pow'][t]:.4f}"
            )
        lp.close()

    # static convergence
    if traj is not None:
        plot_convergence(traj["mui"], traj["obj"], traj["pow"], target_pow=cfg.M * cfg.P0)

    # beampattern
    angles_deg = np.linspace(cfg.angle_min, cfg.angle_max, cfg.angle_points)
    angles_rad = torch.tensor(np.deg2rad(angles_deg), device=device, dtype=torch.float32)
    A = steering_matrix(cfg.N, angles_rad, delta=cfg.delta, device=device, dtype=dtype)
    _, Pd, Ptar, cov_mse, nmse = radar_metrics(X, Rd, cfg.M, A)

    plot_beampattern(
        angles_deg,
        Pd.detach().cpu().numpy(),
        Ptar.detach().cpu().numpy(),
        title=f"Beampattern | cov_mse={cov_mse.item():.3e}, nmse={nmse.item():.3e}"
    )

    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    # Inference với 100 vòng (T=100)
    cfg = MatlabScenarioConfig(
        N=20, K=4, M=30, L=16,
        SNR_dB=20.0, rho=0.2,
        T=100,            # 100 vòng inference
        J_theta=5,
        alpha_theta=0.1,
        seed=1
    )

    run_inference(cfg, rd_path="Rd.mat", ckpt_path="unfold_ckpt.pt", live=True)
