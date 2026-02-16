# train.py
from __future__ import annotations

import copy
import os
import torch
import numpy as np
import math
from config import MatlabScenarioConfig
from utils_complex import set_seed
from dataset import generate_single_scenario
from channels import effective_channel
from model import UnfoldNet
from metrics import pmui, objective_tradeoff

def train_unfoldnet(
    cfg: MatlabScenarioConfig,
    epochs: int = 1000,
    lr: float = 5e-4,  # Lower learning rate for stability
    rd_path: str = "Rd.mat",
    save_ckpt: bool = True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.complex64

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # ===== Model (T phải đúng giữa train & inference) =====
    net = UnfoldNet(
        T=cfg.T,                 # nên để T=100 nếu inference muốn 100 vòng
        J_theta=cfg.J_theta,
        rho_init=cfg.rho,
        learnable_rho=True,       # Enable learnable rho for better adaptation
        learnable_alpha=True,
        alpha_init=cfg.alpha_theta,
    ).to(device)

    # ===== Optimizer param groups (ổn định hơn) =====
    theta_params, rho_params, other_params = [], [], []
    for n, p in net.named_parameters():
        if not p.requires_grad:
            continue
        if "theta_layers" in n and ("eta" in n or "step" in n or "alpha_raw" in n):
            theta_params.append(p)
        elif "rho_logits" in n:
            rho_params.append(p)
        else:
            other_params.append(p)

    optimizer = torch.optim.Adam(
        [
            {"params": other_params, "lr": lr},
            {"params": theta_params, "lr": lr * 0.1},
            {"params": rho_params, "lr": lr * 0.1},
        ],
        betas=(0.9, 0.999),
    )

    # Scheduler: min_lr phải nhỏ hơn lr
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-5
    )

    # ===== Fixed validation scenario (để chọn best ổn định) =====
    val_data = generate_single_scenario(
        cfg.N, cfg.K, cfg.M, cfg.L,
        rd_path=rd_path,
        device=device,
        dtype=dtype
    )
    S_val = val_data["S"]
    H_bu_val = val_data["H_bu"]
    H_ru_val = val_data["H_ru"]
    H_br_val = val_data["H_br"]
    Rd_val = val_data["Rd"]

    best_val_obj = float("inf")
    best_state = None

    train_obj_hist = []
    val_obj_hist = []
    val_pmui_hist = []


    batch_size = 8  # Add batch training for more stable gradients

    for epoch in range(epochs):
        net.train()

        # ===== Train scenario: random each epoch, batched =====
        S_list, H_bu_list, H_ru_list, H_br_list, Rd_list = [], [], [], [], []
        for _ in range(batch_size):
            data = generate_single_scenario(
                cfg.N, cfg.K, cfg.M, cfg.L,
                rd_path=rd_path,
                device=device,
                dtype=dtype
            )
            S_list.append(data["S"])
            H_bu_list.append(data["H_bu"])
            H_ru_list.append(data["H_ru"])
            H_br_list.append(data["H_br"])
            Rd_list.append(data["Rd"])

        S = torch.stack(S_list)
        H_bu = torch.stack(H_bu_list)
        H_ru = torch.stack(H_ru_list)
        H_br = torch.stack(H_br_list)
        Rd = torch.stack(Rd_list)

        optimizer.zero_grad()

        X, U, theta, _ = net(
            H_bu, H_ru, H_br, S, Rd,
            P0=cfg.P0, M=cfg.M,
            return_traj=False
        )
        H_eff = effective_channel(H_bu, H_ru, H_br, theta)

        # Train theo objective (23) để ổn định hơn PMUI
        loss = objective_tradeoff(H_eff, X, S, U, rho=cfg.rho)
        loss.backward()

        # clip để tránh “vọt” lớn
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

        optimizer.step()

        train_obj = float(loss.detach().item())
        train_obj_hist.append(train_obj)

        # ===== Validation (fixed scenario) =====
        net.eval()
        with torch.no_grad():
            Xv, Uv, thetav, _ = net(
                H_bu_val, H_ru_val, H_br_val, S_val, Rd_val,
                P0=cfg.P0, M=cfg.M,
                return_traj=False
            )
            H_eff_val = effective_channel(H_bu_val, H_ru_val, H_br_val, thetav)
            
            val_obj_tensor = objective_tradeoff(H_eff_val, Xv, S_val, Uv, rho=cfg.rho)
            val_pmui_tensor = pmui(H_eff_val, Xv, S_val)
            
            val_obj = float(val_obj_tensor.item())
            val_pmui = float(val_pmui_tensor.item())
            
            # Sửa kiểm tra NaN/Inf đúng cách
            if math.isnan(val_obj) or math.isinf(val_obj):
                print("NaN/Inf detected in validation!")
                print("X norm:", torch.norm(Xv).item())
                print("U norm:", torch.norm(Uv).item())
                print("theta norm:", torch.norm(thetav).item())
                print("H_eff norm:", torch.norm(H_eff_val).item())
                print("||H_eff X||^2:", torch.norm(H_eff_val @ Xv, 'fro').item())
                print("||S||^2:", torch.norm(S_val, 'fro').item())
                print("rho:", net.rhos().detach().cpu().numpy())
                # Có thể thêm pdb.set_trace() nếu cần debug sâu hơn

        val_obj_hist.append(val_obj)
        val_pmui_hist.append(val_pmui)

        # scheduler theo val_obj
        scheduler.step(val_obj)

        # save best theo val objective
        if val_obj < best_val_obj:
            best_val_obj = val_obj
            best_state = copy.deepcopy(net.state_dict())

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"TrainObj={train_obj:.4e} | ValObj={val_obj:.4e} (best={best_val_obj:.4e}) | "
                f"ValPMUI={val_pmui:.4e} | lr={lr_now:.2e}"
            )

    print("Training finished. Best ValObj:", best_val_obj)

    if save_ckpt and best_state is not None:
        torch.save(
            {
                "state_dict": best_state,
                "cfg": {
                    "N": cfg.N, "K": cfg.K, "M": cfg.M, "L": cfg.L,
                    "SNR_dB": cfg.SNR_dB, "rho": cfg.rho,
                    "T": cfg.T, "J_theta": cfg.J_theta, "alpha_theta": cfg.alpha_theta,
                    "P0": cfg.P0, "seed": cfg.seed,
                },
            },
            "unfold_ckpt.pt",
        )
        print("Saved best model to unfold_ckpt.pt")

    # Optional plot
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(train_obj_hist, label="Train Objective")
        plt.plot(val_obj_hist, label="Val Objective")
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Objective")
        plt.grid(True)
        plt.legend()
        plt.title("Training / Validation Objective")
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.plot(val_pmui_hist, label="Val PMUI")
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("PMUI")
        plt.grid(True)
        plt.legend()
        plt.title("Validation PMUI")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Plot skipped:", e)


if __name__ == "__main__":
    # T=100 để inference đúng 100 vòng
    cfg = MatlabScenarioConfig(
        N=20, K=4, M=30, L=16,
        SNR_dB=20.0, rho=0.2,
        T=100,                 # IMPORTANT: giữ cố định giữa train & inference
        J_theta=5,
        alpha_theta=0.1,
        seed=1
    )

    train_unfoldnet(cfg, epochs=300, lr=1e-3, rd_path="Rd.mat", save_ckpt=True)