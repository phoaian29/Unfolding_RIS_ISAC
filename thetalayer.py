# thetalayer.py
from __future__ import annotations
import torch
import torch.nn as nn

from utils_complex import herm, unit_modulus


class ThetaLayer(nn.Module):
    """
    Theta-step (paper-style quadratic form):
      Build B, C, D -> G = B ⊙ C^T, d = diag(D)
      Euclid grad: ∇f = 2(Gθ + d*)
      Riemann grad: grad = ∇f - Re(∇f ⊙ θ*) ⊙ θ
      Retraction: θ <- (θ - α grad) / |θ - α grad|

    Unroll J steps; alpha can be learnable (good for unfolding).
    """

    def __init__(self, J: int = 5, alpha: float = 0.1, learnable_alpha: bool = False, eps: float = 1e-12):
        super().__init__()
        self.J = int(J)
        self.eps = float(eps)

        if learnable_alpha:
            self.alpha_raw = nn.Parameter(torch.full((self.J,), float(alpha)))
        else:
            self.register_buffer("alpha_raw", torch.full((self.J,), float(alpha)))
        self.learnable_alpha = bool(learnable_alpha)

    def alphas(self) -> torch.Tensor:
        # ensure positive + clamp to avoid exploding steps (prevents NaN)
        return torch.nn.functional.softplus(self.alpha_raw).clamp(min=1e-6, max=1.0)


    @staticmethod
    def build_G_d(H_bu, H_ru, H_br, X, S):
        """
        Supports single + batch.
        single:
          H_bu(K,N), H_ru(K,L), H_br(L,N), X(N,M), S(K,M)
        batch:
          H_bu(B,K,N), H_ru(B,K,L), H_br(B,L,N), X(B,N,M), S(B,K,M)
        """
        T = H_bu @ X - S                         # (K,M) or (B,K,M)
        Bm = herm(H_ru) @ H_ru                   # (L,L) or (B,L,L)
        Cm = H_br @ (X @ herm(X)) @ herm(H_br)   # (L,L) or (B,L,L)
        Dm = H_br @ (X @ herm(T)) @ H_ru         # (L,L) or (B,L,L)

        G = Bm * Cm.transpose(-2, -1)            # Hadamard with C^T
        d = torch.diagonal(Dm, dim1=-2, dim2=-1) # (L,) or (B,L)
        return G, d

    @staticmethod
    def riemann_grad(theta, G, d):
        # theta: (L,) or (B,L)
        # G: (L,L) or (B,L,L)
        # d: (L,) or (B,L)
        if theta.dim() == 1:
            nabla = 2.0 * (G @ theta + d.conj())  # (L,)
            proj = torch.real(nabla * theta.conj()) * theta
            return nabla - proj
        else:
            nabla = 2.0 * (G @ theta.unsqueeze(-1)).squeeze(-1) + 2.0 * d.conj()  # (B,L)
            proj = torch.real(nabla * theta.conj()) * theta
            return nabla - proj

    def forward(self, H_bu, H_ru, H_br, X, S, theta0):
        theta = unit_modulus(theta0, eps=self.eps)
        G, d = self.build_G_d(H_bu, H_ru, H_br, X, S)
        alphas = self.alphas()

        for j in range(self.J):
            grad = self.riemann_grad(theta, G, d)
            theta = unit_modulus(theta - alphas[j] * grad, eps=self.eps)

        return theta
