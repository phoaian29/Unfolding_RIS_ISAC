# xlayer.py
from __future__ import annotations

import torch
import torch.nn as nn

from utils_complex import herm, cplx_randn  # SỬA: Thêm import cplx_randn nếu cần reset X

class XLayer(nn.Module):
    """
    X-step for trade-off (23):
      X_hat = (rho H^H H + (1-rho)I + reg I)^(-1) (rho H^H S + (1-rho)U)
      then project power: ||X||_F^2 = M*P0

    Supports single + batch.
    Robustified:
      - clamp rho in (0,1) to keep matrix SPD
      - adaptive jitter if solve fails
    """

    def __init__(
        self,
        diag_reg: float = 1e-6,
        eps: float = 1e-10,
        rho_min: float = 1e-4,
        jitter_init: float = 1e-6,
        jitter_max: float = 1e-1,
        jitter_mult: float = 10.0,
        max_tries: int = 6,
    ):
        super().__init__()
        self.diag_reg = float(diag_reg)
        self.eps = float(eps)

        self.rho_min = float(rho_min)

        self.jitter_init = float(jitter_init)
        self.jitter_max = float(jitter_max)
        self.jitter_mult = float(jitter_mult)
        self.max_tries = int(max_tries)

    @staticmethod
    def _project_power(X: torch.Tensor, target_norm: float, eps: float) -> torch.Tensor:
        # project to ||X||_F = target_norm
        if X.dim() == 2:  # (N,M)
            nrm = torch.linalg.norm(X)
            return X * (target_norm / (nrm + eps))
        else:             # (B,N,M)
            B = X.shape[0]
            nrm = torch.linalg.norm(X.reshape(B, -1), dim=1).view(B, 1, 1)
            return X * (target_norm / (nrm + eps))

    def _to_rho_tensor(self, rho: torch.Tensor | float, device: torch.device) -> torch.Tensor:
        if isinstance(rho, torch.Tensor):
            rho_t = rho.to(device=device, dtype=torch.float32)
        else:
            rho_t = torch.tensor(float(rho), device=device, dtype=torch.float32)

        # Clamp rho to keep (1-rho) > 0 => A is SPD in theory
        rho_t = rho_t.clamp(self.rho_min, 1.0 - self.rho_min)
        return rho_t

    def forward(
        self,
        H_eff: torch.Tensor,   # (K,N) or (B,K,N)
        S: torch.Tensor,       # (K,M) or (B,K,M)
        U: torch.Tensor,       # (N,M) or (B,N,M)
        rho: torch.Tensor | float,
        P0: float,
        M: int,
    ) -> torch.Tensor:
        dtype = H_eff.dtype
        device = H_eff.device
        is_batch = (H_eff.dim() == 3)

        rho_t = self._to_rho_tensor(rho, device=device)   # float32 scalar tensor
        one_minus = 1.0 - rho_t

        # Normal equation parts
        Hh = herm(H_eff)     # (N,K) or (B,N,K)
        HhH = Hh @ H_eff     # (N,N) or (B,N,N)
        HhS = Hh @ S         # (N,M) or (B,N,M)

        N = H_eff.shape[-1]

        # Identity with correct shape
        I = torch.eye(N, device=device, dtype=dtype)
        if is_batch:
            I = I.unsqueeze(0).expand(H_eff.shape[0], N, N)

        # Base system
        # A = rho H^H H + (1-rho) I + reg I
        # B = rho H^H S + (1-rho) U
        A0 = rho_t * HhH + one_minus * I + self.diag_reg * I
        B0 = rho_t * HhS + one_minus * U

        # Robust solve with adaptive jitter
        jitter = self.jitter_init
        last_err = None
        X_hat = None

        for _ in range(self.max_tries):
            A = A0 + jitter * I
            try:
                X_hat = torch.linalg.solve(A, B0)
                last_err = None
                break
            except RuntimeError as e:
                last_err = e
                jitter = min(jitter * self.jitter_mult, self.jitter_max)

        if X_hat is None:
            # Final fallback: least-squares solve (more robust for near-singular A)
            # torch.linalg.lstsq supports complex and batch in recent PyTorch versions.
            try:
                X_hat = torch.linalg.lstsq(A0 + jitter * I, B0).solution
            except Exception:
                # If still failing, raise original error
                raise last_err

        # Power projection: ||X||_F^2 = M*P0
        Xr = torch.nan_to_num(X_hat.real, nan=0.0, posinf=0.0, neginf=0.0)
        Xi = torch.nan_to_num(X_hat.imag, nan=0.0, posinf=0.0, neginf=0.0)
        X_hat = torch.complex(Xr, Xi)

        # Power projection
        target = (M * float(P0)) ** 0.5
        X = self._project_power(X_hat, target_norm=target, eps=self.eps)
        
        # SỬA: Bảo vệ nếu X norm quá nhỏ
        final_norm = torch.linalg.norm(X)
        if final_norm < 1e-4 or torch.isnan(final_norm):
            print("Warning: X norm too small after projection → add small perturbation")
            if is_batch:
                X = X + 1e-4 * cplx_randn(X.shape[0], X.shape[1], X.shape[2], device=device, dtype=dtype)
            else:
                X = X + 1e-4 * cplx_randn(X.shape[0], X.shape[1], device=device, dtype=dtype)
            X = self._project_power(X, target_norm=target, eps=self.eps)  # project lại
        
        return X