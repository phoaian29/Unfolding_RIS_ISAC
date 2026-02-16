# ulayer.py
from __future__ import annotations
import torch
import torch.nn as nn

from utils_complex import herm


def safe_cholesky(R: torch.Tensor, jitter: float = 1e-9, max_tries: int = 6) -> torch.Tensor:
    """
    Robust Cholesky for nearly-PSD matrices.
    """
    dtype = R.dtype
    device = R.device
    eye = torch.eye(R.shape[-1], device=device, dtype=dtype)
    if R.dim() == 3:
        eye = eye.unsqueeze(0).expand(R.shape[0], -1, -1)

    for i in range(max_tries):
        try:
            return torch.linalg.cholesky(R + (jitter * (10**i)) * eye)
        except RuntimeError:
            continue
    # last resort: use eig to make PSD
    evals, evecs = torch.linalg.eigh(0.5 * (R + herm(R)))
    evals = torch.clamp(evals, min=0.0)
    R_psd = (evecs * evals.unsqueeze(-2)) @ herm(evecs)
    return torch.linalg.cholesky(R_psd + jitter * eye)


class ULayer(nn.Module):
    """
    U-step projection:
      U = sqrt(M) * F * Z
      where Rd = F F^H, and Z is Procrustes solution from SVD of F^H X.
    """
    def forward(self, X: torch.Tensor, Rd: torch.Tensor, M: int) -> torch.Tensor:
        dtype = X.dtype
        device = X.device

        if X.dim() == 2:
            # single: X (N,M), Rd (N,N)
            F = safe_cholesky(Rd.to(device=device, dtype=dtype))
            Y = herm(F) @ X
            U_svd, _, Vh = torch.linalg.svd(Y, full_matrices=False)
            Z = (U_svd @ Vh).detach()  # Detach SVD output to avoid backward error
            return (M ** 0.5) * (F @ Z)

        # batch: X (B,N,M), Rd (N,N) or (B,N,N)
        B, N, _ = X.shape
        if Rd.dim() == 2:
            Rd_b = Rd.unsqueeze(0).expand(B, N, N)
        else:
            Rd_b = Rd
        F = safe_cholesky(Rd_b.to(device=device, dtype=dtype))
        Y = herm(F) @ X
        U_svd, _, Vh = torch.linalg.svd(Y, full_matrices=False)
        Z = (U_svd @ Vh).detach()  # Detach SVD output to avoid backward error
        return (M ** 0.5) * (F @ Z)
