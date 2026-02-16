# metrics.py
from __future__ import annotations
import torch
from utils_complex import herm


def pmui(H_eff: torch.Tensor, X: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    # ||H_eff X - S||_F^2
    E = H_eff @ X - S
    return (E.abs() ** 2).sum()


def objective_tradeoff(H_eff: torch.Tensor, X: torch.Tensor, S: torch.Tensor, U: torch.Tensor, rho: float) -> torch.Tensor:
    # rho||H X - S||^2 + (1-rho)||X-U||^2
    E = H_eff @ X - S
    comm = (E.abs() ** 2).sum()
    rad = (X - U)
    rad_term = (rad.abs() ** 2).sum()
    return rho * comm + (1 - rho) * rad_term


def sum_rate(H_eff: torch.Tensor, X: torch.Tensor, S: torch.Tensor, sigma2: float) -> torch.Tensor:
    # Per-user SINR approx from error power
    Y_hat = H_eff @ X          # (K,M)
    err = Y_hat - S            # (K,M)
    interf_pow = (err.abs() ** 2).mean(dim=1)  # (K,)
    sig_pow = (S.abs() ** 2).mean(dim=1)       # (K,)
    gamma = sig_pow / (interf_pow + sigma2)
    return torch.log2(1 + gamma).sum()


def steering_matrix(N: int, angles_rad: torch.Tensor, delta: float, device, dtype) -> torch.Tensor:
    # a(phi) for ULA
    n = torch.arange(N, device=device).view(N, 1)
    phis = angles_rad.view(1, -1)
    A = torch.exp(1j * 2 * torch.pi * delta * n * torch.sin(phis))
    return A.to(dtype)


def beampattern_from_cov(R: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    # Pd(phi) = a^H R a, for each column of A
    RA = R @ A
    Pd = (A.conj() * RA).sum(dim=0).real
    return Pd


def radar_cov(X: torch.Tensor, M: int) -> torch.Tensor:
    return (X @ herm(X)) / M


def radar_metrics(X: torch.Tensor, Rd: torch.Tensor, M: int, A: torch.Tensor):
    Rx = radar_cov(X, M)
    Pd = beampattern_from_cov(Rx, A)
    Ptar = beampattern_from_cov(Rd, A)
    cov_mse = ((Rx - Rd).abs() ** 2).sum().real
    nmse = ((Pd - Ptar) ** 2).sum() / ((Ptar ** 2).sum() + 1e-12)
    return Rx, Pd, Ptar, cov_mse, nmse
