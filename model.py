# model.py
from __future__ import annotations
import torch
import torch.nn as nn

from channels import effective_channel
from xlayer import XLayer
from ulayer import ULayer
from thetalayer import ThetaLayer
from utils_complex import unit_modulus, init_XU_from_Rd

class UnfoldNet(nn.Module):
    """
    Unfolding network:
      for t in 1..T:
        H_eff(theta)
        X_t = XLayer_t(...)
        U_t = ULayer_t(...)
        theta_t = ThetaLayer_t(...)

    Learnable:
      - rho per stage (optional)
      - alpha inside each ThetaLayer (if learnable_alpha=True)
    """

    def __init__(
        self,
        T: int,
        J_theta: int,
        rho_init: float = 0.2,
        learnable_rho: bool = True,
        learnable_alpha: bool = True,
        alpha_init: float = 0.1,
    ):
        super().__init__()
        self.T = int(T)

        self.xlayers = nn.ModuleList([XLayer(diag_reg=1e-3) for _ in range(self.T)])
        self.ulayers = nn.ModuleList([ULayer() for _ in range(self.T)])
        self.thetalayers = nn.ModuleList(
            [ThetaLayer(J=J_theta, alpha=alpha_init, learnable_alpha=learnable_alpha) for _ in range(self.T)]
        )

        # rho per stage
        if learnable_rho:
            r0 = torch.tensor(rho_init).clamp(1e-4, 1.0 - 1e-4)
            logits0 = torch.log(r0 / (1 - r0))
            self.rho_logits = nn.Parameter(logits0 * torch.ones(self.T))
        else:
            r0 = torch.tensor(rho_init).clamp(1e-4, 1.0 - 1e-4)
            logits0 = torch.log(r0 / (1 - r0))
            self.register_buffer("rho_logits", logits0 * torch.ones(self.T))

        self.learnable_rho = bool(learnable_rho)

    def rhos(self) -> torch.Tensor:
        # SỬA: Clamp logits để tránh NaN/inf
        logits = torch.clamp(self.rho_logits, min=-10.0, max=10.0)
        return torch.sigmoid(logits)

    def forward(
        self,
        H_bu: torch.Tensor,  # (K,N) or (B,K,N)
        H_ru: torch.Tensor,  # (K,L) or (B,K,L)
        H_br: torch.Tensor,  # (L,N) or (B,L,N)
        S: torch.Tensor,     # (K,M) or (B,K,M)
        Rd: torch.Tensor,    # (N,N) or (B,N,N)
        P0: float,
        M: int,
        return_traj: bool = True,
    ):
        device = H_bu.device
        dtype = H_bu.dtype
        is_batch = (H_bu.dim() == 3)

        if not is_batch:
            K, N = H_bu.shape
            L = H_br.shape[0]
            theta = unit_modulus(torch.ones(L, device=device, dtype=dtype))
            # Init U0 theo strict radar (10)/(11) từ Rd.mat
            X, U = init_XU_from_Rd(Rd.to(device=device, dtype=dtype), M=M, P0=P0)
        else:
            B, K, N = H_bu.shape
            L = H_br.shape[1]
            theta = unit_modulus(torch.ones(B, L, device=device, dtype=dtype))
            # Batch init U0 theo strict radar (10)/(11)
            X, U = init_XU_from_Rd(Rd.to(device=device, dtype=dtype), M=M, P0=P0)


        rhos = self.rhos()

        traj = {"mui": [], "obj": [], "pow": [], "theta": []} if return_traj else None

        for t in range(self.T):
            rho_t = rhos[t]  # tensor để rho học được (end-to-end)

            H_eff = effective_channel(H_bu, H_ru, H_br, theta)
            X = self.xlayers[t](H_eff, S, U, rho=rho_t, P0=P0, M=M)
            U = self.ulayers[t](X, Rd, M=M)
            theta = self.thetalayers[t](H_bu, H_ru, H_br, X, S, theta)

            if return_traj:
                # metrics computed after theta updated => consistent
                H_eff2 = effective_channel(H_bu, H_ru, H_br, theta)
                E = H_eff2 @ X - S
                mui = (E.abs() ** 2).sum().item() if not is_batch else (E.abs() ** 2).sum(dim=(-2, -1)).mean().item()
                powv = (torch.linalg.norm(X) ** 2).item() if not is_batch else (torch.linalg.norm(X.reshape(X.shape[0], -1), dim=1) ** 2).mean().item()
                rho_val = float(rho_t.detach().clamp(0.0, 1.0).item())
                obj = (
                    rho_val * ((E.abs() ** 2).sum().item() if not is_batch else (E.abs() ** 2).sum(dim=(-2, -1)).mean().item())
                    + (1 - rho_val) * ((torch.linalg.norm((X - U)) ** 2).item() if not is_batch else (torch.linalg.norm((X - U).reshape(X.shape[0], -1), dim=1) ** 2).mean().item())
                )


                traj["mui"].append(mui)
                traj["obj"].append(obj)
                traj["pow"].append(powv)
                traj["theta"].append(theta.detach().clone())

        return X, U, theta, traj