# main.py
from __future__ import annotations
import torch
import numpy as np
import os
from config import MatlabScenarioConfig
from utils_complex import set_seed
from dataset import generate_single_scenario
from channels import effective_channel
from model import UnfoldNet
from metrics import pmui, objective_tradeoff, steering_matrix, radar_metrics, sum_rate
from plots import plot_convergence, plot_beampattern, LivePlot


def run_once(cfg: MatlabScenarioConfig, rd_path: str = "Rd.mat", live: bool = True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.complex64

    if cfg.seed is not None:
        set_seed(cfg.seed)

    data = generate_single_scenario(cfg.N, cfg.K, cfg.M, cfg.L, rd_path=rd_path, device=device, dtype=dtype)
    S = data["S"]
    H_bu = data["H_bu"]
    H_ru = data["H_ru"]
    H_br = data["H_br"]
    Rd = data["Rd"]

    net = UnfoldNet(
        T=cfg.T,
        J_theta=cfg.J_theta,
        rho_init=cfg.rho,
        learnable_rho=True,       # MUST True to load learned rho
        learnable_alpha=True,     # MUST True to load learned alpha
        alpha_init=cfg.alpha_theta
    ).to(device)

    ckpt = "unfold_ckpt.pt"
    if os.path.exists(ckpt):
        pack = torch.load(ckpt, map_location=device)
        net.load_state_dict(pack["state_dict"], strict=True)
        print(f"Loaded checkpoint: {ckpt}")
    else:
        print("Checkpoint not found, running with init parameters.")

    # run unfolding and collect traj
    X, U, theta, traj = net(H_bu, H_ru, H_br, S, Rd, P0=cfg.P0, M=cfg.M, return_traj=True)

    # final metrics
    H_eff = effective_channel(H_bu, H_ru, H_br, theta)
    mui = pmui(H_eff, X, S).item()
    obj = objective_tradeoff(H_eff, X, S, U, cfg.rho).item()
    rate = sum_rate(H_eff, X, S, sigma2=cfg.N0).item()
    power = (torch.linalg.norm(X) ** 2).item()

    print(
        f"N={cfg.N}, K={cfg.K}, M={cfg.M}, L={cfg.L}, SNR_dB={cfg.SNR_dB}, rho={cfg.rho}, P0={cfg.P0:.6f}"
    )
    print(f"PMUI={mui:.6e}, Obj={obj:.6e}, SumRate={rate:.4f}, ||X||_F^2={power:.6f} (target {cfg.M*cfg.P0:.6f})")

    # live plot stage-by-stage (if you want)
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

    # static plots: convergence + beampattern
    if traj is not None:
        plot_convergence(traj["mui"], traj["obj"], traj["pow"], target_pow=cfg.M * cfg.P0)

    # beampattern
    angles_deg = np.linspace(cfg.angle_min, cfg.angle_max, cfg.angle_points)
    angles_rad = torch.tensor(np.deg2rad(angles_deg), device=device, dtype=torch.float32)
    A = steering_matrix(cfg.N, angles_rad, delta=cfg.delta, device=device, dtype=dtype)
    _, Pd, Ptar, cov_mse, nmse = radar_metrics(X, Rd, cfg.M, A)

    plot_beampattern(angles_deg, Pd.detach().cpu().numpy(), Ptar.detach().cpu().numpy(),
                     title=f"Beampattern | cov_mse={cov_mse.item():.3e}, nmse={nmse.item():.3e}")

    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    cfg = MatlabScenarioConfig(
        N=20, K=4, M=30, L=16,
        SNR_dB=20.0, rho=0.2,
        T=10, J_theta=5, alpha_theta=0.1,
        seed=1
    )
    run_once(cfg, rd_path="Rd.mat", live=True)
