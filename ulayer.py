# ulayer.py
"""
U-step  (paper §IV-B-2, eq.25-26):

  min_U  ||U - X||_F^2   s.t.  (1/M) U U^H = Rd

Closed-form (eq.26): U = √M · F · Ū V̄^H   where  Ū Σ̄ V̄^H = F^H X

SVD backward on complex tensors can produce NaN when singular values
are near-degenerate.  We run the whole U-step inside no_grad and use
a straight-through estimator for gradient flow:

    return U_value.detach() + (X - X.detach())

This means the loss gradient sees dL/dX, not a potentially-NaN SVD backward.
It is the standard approach for "auxiliary variable projection" in unfolding.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from utils import herm, safe_cholesky


class ULayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X: torch.Tensor, Rd: torch.Tensor, M: int) -> torch.Tensor:
        # Compute projected U value with no gradient tracking
        with torch.no_grad():
            U_val = self._procrustes(X.detach(), Rd, M)

        # Straight-through: value = U_val, gradient path = X
        return U_val + (X - X.detach())

    @staticmethod
    def _procrustes(X: torch.Tensor, Rd: torch.Tensor, M: int) -> torch.Tensor:
        """U = √M · F · (Uu @ Vh)  where  Uu,_,Vh = svd(F^H X)."""
        device, dtype = X.device, X.dtype
        Rd = Rd.to(device=device, dtype=dtype)

        if X.dim() == 2:
            F = safe_cholesky(Rd)                              # (N,N)
            Y = herm(F) @ X                                    # (N,M)
            Uu, _, Vh = torch.linalg.svd(Y, full_matrices=False)
            return math.sqrt(float(M)) * (F @ (Uu @ Vh))      # (N,M)

        B, N, _M = X.shape
        if Rd.dim() == 2:
            Rd = Rd.unsqueeze(0).expand(B, N, N)
        F  = safe_cholesky(Rd)                                 # (B,N,N)
        Y  = herm(F) @ X                                       # (B,N,M)
        Uu, _, Vh = torch.linalg.svd(Y, full_matrices=False)
        return math.sqrt(float(M)) * (F @ (Uu @ Vh))          # (B,N,M)
