# xlayer.py
"""
X-step CLOSED-FORM (paper §IV-B-1, eq.24 reformulated):

  min_X  ρ||H̃X - S||_F^2 + (1-ρ)||X - U||_F^2
  s.t.   (1/M)||X||_F^2 = P0

Unconstrained optimum:
  [ρ H̃^H H̃ + (1-ρ+reg) I] X* = ρ H̃^H S + (1-ρ) U

Solved via real-valued 2N×2N block system — GPU-compatible, fully differentiable.

IMPORTANT: lstsq(driver='gelsd') is CPU-only and has been replaced with
torch.linalg.solve + small ridge regularisation, which works on CUDA/MPS/CPU.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import herm, project_power, safe_cplx


class XLayer(nn.Module):
    def __init__(self, J=1, beta_init=0.05, diag_reg=1e-3, eps=1e-9, max_beta=0.5):
        super().__init__()
        self.J        = J
        self.diag_reg = diag_reg
        self.eps      = eps
        self.max_beta = max_beta
        self.reg_raw  = nn.Parameter(torch.tensor(
            math.log(math.expm1(max(diag_reg, 1e-6))), dtype=torch.float32
        ))

    def reg(self) -> float:
        return float(F.softplus(self.reg_raw).clamp(1e-6, 10.0).item())

    def forward(self, H_eff, S, U, rho, P0, M):
        rho    = float(rho)
        orho   = 1.0 - rho
        reg    = float(F.softplus(self.reg_raw).clamp(1e-6, 10.0))
        target = float(M) * float(P0)

        # RHS: ρ H^H S + (1-ρ) U
        rhs = rho * (herm(H_eff) @ S) + orho * U

        # LHS: ρ H^H H + λI,  λ = (1-ρ+reg)
        lam = orho + reg
        A   = rho * (herm(H_eff) @ H_eff)
        n   = A.shape[-1]
        I   = torch.eye(n, device=A.device, dtype=A.dtype)
        if A.dim() == 3:
            I = I.unsqueeze(0)
        A = A + lam * I

        X = _cplx_solve(A, rhs, self.eps)
        X = project_power(X, target, self.eps)
        X = safe_cplx(X, clamp=1e6)
        return X


def _cplx_solve(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Solve AX = B for complex Hermitian PD matrix A via real 2×2 block system:

        [Ar  -Ai] [Xr]   [Br]
        [Ai   Ar] [Xi] = [Bi]

    Uses torch.linalg.solve — works on CUDA, MPS, and CPU.
    (lstsq with driver='gelsd' is CPU-only and is NOT used here.)
    """
    Ar, Ai = A.real, A.imag
    Br, Bi = B.real, B.imag
    N      = B.shape[-2]

    top = torch.cat([Ar, -Ai], dim=-1)
    bot = torch.cat([Ai,  Ar], dim=-1)
    Abl = torch.cat([top, bot], dim=-2)
    Bbl = torch.cat([Br,  Bi], dim=-2)

    # Small diagonal ridge → guarantees non-singular on any device
    I = eps * torch.eye(2 * N, device=A.device, dtype=Ar.dtype)
    if Abl.dim() == 3:
        I = I.unsqueeze(0)

    Xbl = torch.linalg.solve(Abl + I, Bbl)

    if Xbl.dim() == 2:                           # unbatched
        return torch.complex(Xbl[:N],    Xbl[N:])
    return     torch.complex(Xbl[:, :N, :], Xbl[:, N:, :])  # batched