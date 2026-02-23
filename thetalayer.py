# thetalayer.py
"""
Θ-step (paper §III-B, Algorithm 1):

  min_θ  θ^H(B⊙C^T)θ + d^T θ + θ^H d*   s.t. |θ_l|=1

Phase parameterisation: θ = exp(jφ)
  dphi = Im{ ∇_θ f ⊙ θ* }   (Riemannian gradient, real)
  φ ← φ - α · dphi
  θ = exp(jφ)  [unit-modulus by construction]
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import herm, project_unit_modulus, safe_cplx


class ThetaLayer(nn.Module):
    def __init__(self, J=3, alpha_init=0.05, eps=1e-9, max_alpha=0.3):
        super().__init__()
        self.J         = J
        self.eps       = eps
        self.max_alpha = max_alpha
        inv_sp = math.log(math.expm1(max(alpha_init, 1e-6)))
        self.alpha_raw = nn.Parameter(torch.full((J,), inv_sp, dtype=torch.float32))

    def alphas(self):
        return F.softplus(self.alpha_raw).clamp(1e-7, self.max_alpha)

    @staticmethod
    def _build_Gd(H_bu, H_ru, H_br, X, S):
        """G = B ⊙ C^T,  d = diag(D)   (paper eq.13-15)"""
        T = H_bu @ X - S
        B = herm(H_ru) @ H_ru
        C = H_br @ (X @ herm(X)) @ herm(H_br)
        D = H_br @ (X @ herm(T)) @ H_ru
        G = B * C.transpose(-2, -1)
        d = torch.diagonal(D, dim1=-2, dim2=-1)
        return safe_cplx(G), safe_cplx(d)

    def forward(self, H_bu, H_ru, H_br, X, S, theta0):
        G, d   = self._build_Gd(H_bu, H_ru, H_br, X, S)
        alphas = self.alphas().to(H_bu.device)

        theta = project_unit_modulus(theta0, self.eps)
        phi   = torch.angle(theta)

        for j in range(self.J):
            theta = torch.polar(torch.ones_like(phi), phi).to(theta0.dtype)
            if theta.dim() == 1:
                eu_grad = 2.0 * (G @ theta + d.conj())
            else:
                eu_grad = 2.0 * ((G @ theta.unsqueeze(-1)).squeeze(-1) + d.conj())
            eu_grad = safe_cplx(eu_grad)
            dphi = (eu_grad * theta.conj()).imag
            dphi = dphi.clamp(-100.0, 100.0)
            phi  = phi - alphas[j] * dphi

        theta = torch.polar(torch.ones_like(phi), phi).to(theta0.dtype)
        return project_unit_modulus(theta, self.eps)
