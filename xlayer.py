# xlayer.py
"""
X-step CLOSED-FORM (paper §IV-B-1, eq.24 reformulated):

  min_X  ρ||H̃X - S||_F^2 + (1-ρ)||X - U||_F^2
  s.t.   (1/M)||X||_F^2 = P0

Unconstrained optimum:
  [ρ H̃^H H̃ + (1-ρ+reg) I] X* = ρ H̃^H S + (1-ρ) U

Solved via Cholesky/lstsq, then power-projected.

Why closed-form instead of GD?
  - GD with warm-start X=U makes grad of (X-U) term = 0 at step 0 → 
    the radar term never gets gradient signal → X drifts to minimise
    only PMUI → PMUI stays high because U constraint is ignored.
  - Closed-form solves both terms simultaneously every stage.

Backward through torch.linalg.solve can be unstable for complex.
FIX: use real-valued block representation (2N x 2N real system).
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
        # J kept for API compat but closed-form needs only diag_reg
        self.J = J
        self.diag_reg = diag_reg
        self.eps = eps
        self.max_beta = max_beta
        # Learnable log-regularisation (scalar per stage, real)
        # reg = softplus(reg_raw) + eps → always positive
        self.reg_raw = nn.Parameter(torch.tensor(
            math.log(math.expm1(max(diag_reg, 1e-6))), dtype=torch.float32
        ))

    def reg(self) -> float:
        return float(F.softplus(self.reg_raw).clamp(1e-6, 10.0).item())

    def forward(self, H_eff, S, U, rho, P0, M):
        """
        Solves the X-subproblem in real arithmetic to avoid complex solve NaN.

        We use the identity: for complex A, b → solve via 2x real block system.
        For batch (B,K,N) we solve each independently via bmm + real block.
        """
        rho  = float(rho)
        orho = 1.0 - rho
        reg  = float(F.softplus(self.reg_raw).clamp(1e-6, 10.0))
        target = float(M) * float(P0)

        # ── Solve [ρ H^H H + (1-ρ+reg) I] X = ρ H^H S + (1-ρ) U ──────────
        # RHS
        rhs = rho * (herm(H_eff) @ S) + orho * U           # (B?,N,M) complex

        # LHS matrix A = ρ H^H H + λI,  λ = (1-ρ+reg)
        lam  = orho + reg
        A    = rho * (herm(H_eff) @ H_eff)                  # (B?,N,N) complex
        # Add λI
        n    = A.shape[-1]
        I    = torch.eye(n, device=A.device, dtype=A.dtype)
        if A.dim() == 3:
            I = I.unsqueeze(0)
        A    = A + lam * I

        # Solve via real block system to avoid complex solve backward NaN
        X    = _cplx_solve(A, rhs, self.eps)                # (B?,N,M)

        X = project_power(X, target, self.eps)
        X = safe_cplx(X, clamp=1e6)
        return X


def _cplx_solve(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Solve A X = B for complex Hermitian PD matrix A.
    Uses real 2x2 block trick: [Ar -Ai; Ai Ar] [Xr; Xi] = [Br; Bi]
    This makes backward fully differentiable without complex-solve NaN.

    A: (B?,N,N) complex,  B: (B?,N,M) complex
    Returns X: (B?,N,M) complex
    """
    is_batch = (A.dim() == 3)
    Ar, Ai = A.real, A.imag
    Br, Bi = B.real, B.imag
    N, M   = B.shape[-2], B.shape[-1]

    if not is_batch:
        # Build 2N x 2N real block matrix
        top  = torch.cat([Ar, -Ai], dim=-1)   # (N, 2N)
        bot  = torch.cat([Ai,  Ar], dim=-1)
        Abl  = torch.cat([top, bot], dim=-2)  # (2N, 2N)
        Bbl  = torch.cat([Br, Bi],  dim=-2)   # (2N, M)
        Xbl  = torch.linalg.lstsq(Abl, Bbl, driver="gelsd").solution  # (2N, M)
        return torch.complex(Xbl[:N], Xbl[N:])

    B_sz = A.shape[0]
    top  = torch.cat([Ar, -Ai], dim=-1)       # (B, N, 2N)
    bot  = torch.cat([Ai,  Ar], dim=-1)
    Abl  = torch.cat([top, bot], dim=-2)      # (B, 2N, 2N)
    Bbl  = torch.cat([Br, Bi],  dim=-2)       # (B, 2N, M)

    # torch.linalg.lstsq doesn't support batched; use solve with regularisation
    reg_eye = (eps * torch.eye(2*N, device=A.device, dtype=Ar.dtype)
               .unsqueeze(0).expand(B_sz, 2*N, 2*N))
    Abl_reg = Abl + reg_eye
    Xbl = torch.linalg.solve(Abl_reg, Bbl)   # (B, 2N, M)
    return torch.complex(Xbl[:, :N, :], Xbl[:, N:, :])