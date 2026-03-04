# utils.py
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Autograd-safe complex clamp (nan_to_num NOT supported for complex autograd)
# ─────────────────────────────────────────────────────────────────────────────

def safe_cplx(z: torch.Tensor, clamp: float = 1e5) -> torch.Tensor:
    """Clamp real & imag parts to ±clamp.  Fully autograd-compatible."""
    return torch.complex(z.real.clamp(-clamp, clamp),
                         z.imag.clamp(-clamp, clamp))


# ─────────────────────────────────────────────────────────────────────────────
# Basic helpers
# ─────────────────────────────────────────────────────────────────────────────

def herm(A: torch.Tensor) -> torch.Tensor:
    return A.conj().transpose(-2, -1)


def cplx_randn(*shape, device="cpu", dtype=torch.complex64) -> torch.Tensor:
    r = torch.randn(*shape, device=device, dtype=torch.float32)
    i = torch.randn(*shape, device=device, dtype=torch.float32)
    return ((r + 1j * i) / math.sqrt(2.0)).to(dtype)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# Projections  (autograd-safe)
# ─────────────────────────────────────────────────────────────────────────────

def project_unit_modulus(theta: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """θ → exp(j·angle(θ)).  No division → no NaN gradient."""
    phi = torch.angle(theta)
    return torch.polar(torch.ones_like(phi), phi).to(theta.dtype)


def project_power(X: torch.Tensor, target: float, eps: float = 1e-9) -> torch.Tensor:
    """Scale X so ||X||_F^2 = target.  Autograd-safe for complex."""
    target = float(target)
    if X.dim() == 2:
        nrm_sq = (X.abs() ** 2).sum().clamp(min=eps)
        return X * (math.sqrt(target) / nrm_sq.sqrt())
    nrm_sq = (X.abs() ** 2).flatten(1).sum(1).clamp(min=eps)   # (B,)
    scale   = (math.sqrt(target) / nrm_sq.sqrt()).view(-1, 1, 1)
    return X * scale


def safe_cholesky(R: torch.Tensor, jitter: float = 1e-7, tries: int = 8) -> torch.Tensor:
    R = 0.5 * (R + herm(R))
    n = R.shape[-1]
    eye = torch.eye(n, device=R.device, dtype=R.dtype)
    if R.dim() == 3:
        eye = eye.unsqueeze(0)
    for k in range(tries):
        try:
            return torch.linalg.cholesky(R + jitter * (10.0 ** k) * eye)
        except RuntimeError:
            continue
    evals, evecs = torch.linalg.eigh(R)
    evals = evals.clamp(min=0.0)
    R_psd = (evecs * evals.unsqueeze(-2)) @ herm(evecs)
    return torch.linalg.cholesky(R_psd + jitter * eye)


# ─────────────────────────────────────────────────────────────────────────────
# Effective channel
# ─────────────────────────────────────────────────────────────────────────────

def effective_channel(H_bu, H_ru, H_br, theta, eps=1e-9):
    """H_eff = H_bu + H_ru diag(θ) H_br"""
    theta = project_unit_modulus(theta, eps)
    if H_bu.dim() == 2:
        Hr = H_ru * theta.unsqueeze(0)
        return H_bu + Hr @ H_br
    if H_br.dim() == 2:
        H_br = H_br.unsqueeze(0).expand(H_bu.shape[0], *H_br.shape)
    Hr = H_ru * theta.unsqueeze(1)
    return H_bu + Hr @ H_br


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

def bpsk(K: int, M: int, device="cpu", dtype=torch.complex64) -> torch.Tensor:
    bits = (2 * torch.randint(0, 2, (K, M), device=device) - 1).float()
    return bits.to(dtype)


def load_Rd(N: int, path: str = "Rd.mat", device="cpu", dtype=torch.complex64) -> torch.Tensor:
    p = Path(path)
    if p.exists():
        try:
            import scipy.io
            Rd_np = scipy.io.loadmat(str(p))["Rd"]
            return torch.as_tensor(Rd_np, device=device, dtype=dtype)
        except Exception:
            pass
    return torch.eye(N, device=device, dtype=dtype)


def generate_batch(B, N, K, M, L, rd_path="Rd.mat", device="cpu", dtype=torch.complex64):
    S    = bpsk(K, M, device=device, dtype=dtype).unsqueeze(0).expand(B, K, M).contiguous()
    H_bu = cplx_randn(B, K, N, device=device, dtype=dtype)
    H_ru = cplx_randn(B, K, L, device=device, dtype=dtype)
    H_br = cplx_randn(B, L, N, device=device, dtype=dtype)
    Rd   = load_Rd(N, rd_path, device=device, dtype=dtype)
    return dict(S=S, H_bu=H_bu, H_ru=H_ru, H_br=H_br, Rd=Rd)


def generate_single(N, K, M, L, rd_path="Rd.mat", device="cpu", dtype=torch.complex64):
    S    = bpsk(K, M, device=device, dtype=dtype)
    H_bu = cplx_randn(K, N, device=device, dtype=dtype)
    H_ru = cplx_randn(K, L, device=device, dtype=dtype)
    H_br = cplx_randn(L, N, device=device, dtype=dtype)
    Rd   = load_Rd(N, rd_path, device=device, dtype=dtype)
    return dict(S=S, H_bu=H_bu, H_ru=H_ru, H_br=H_br, Rd=Rd)


# ─────────────────────────────────────────────────────────────────────────────
# Initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_XU(Rd, M, P0, eps=1e-9):
    device, dtype = Rd.device, Rd.dtype
    is_batch = (Rd.dim() == 3)
    N = Rd.shape[-1]
    target = float(M) * float(P0)

    with torch.no_grad():
        F = safe_cholesky(Rd)
        if not is_batch:
            G  = cplx_randn(N, M, device=device, dtype=dtype)
            X0 = F @ G
            X0 = project_power(X0, target, eps)
            return X0.clone(), X0.clone()
        B  = Rd.shape[0]
        G  = cplx_randn(B, N, M, device=device, dtype=dtype)
        X0 = F @ G
        X0 = project_power(X0, target, eps)
        return X0.clone(), X0.clone()


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def pmui(H_eff, X, S):
    """||H_eff X - S||_F^2, mean over batch."""
    E   = H_eff @ X - S
    val = (E.abs() ** 2).sum(dim=(-2, -1))
    return val.mean() if val.dim() > 0 else val


def radar_mse(X, U):
    """||X - U||_F^2, mean over batch."""
    D   = X - U
    val = (D.abs() ** 2).sum(dim=(-2, -1))
    return val.mean() if val.dim() > 0 else val


def objective(H_eff, X, S, U, rho):
    """ρ·PMUI + (1-ρ)·||X-U||²  (paper eq.23)."""
    return rho * pmui(H_eff, X, S) + (1.0 - rho) * radar_mse(X, U)


def sum_rate(H_eff, X, S, N0, M):
    """
    Achievable sum-rate per paper eq.(4)-(5).

    SINR_k = E[|s_kj|²] / (E[|ỹ_kj - s_kj|²] + σ²)
           = signal_power / (MUI_power + N0)

    where ỹ_kj = h̃^T_bu,k x_j  is the k-th received sample.

    With BPSK: s_kj ∈ {±1} → E[|s_kj|²] = 1.
    MUI_power = (1/M) Σ_j |ỹ_kj - s_kj|²  (sample estimate).

    H_eff: (K,N) single or (B,K,N) batch
    X:     (N,M) or (B,N,M)
    S:     (K,M) or (B,K,M)
    Returns scalar float.
    """
    # Use single realisation (first of batch if batched)
    if H_eff.dim() == 3:
        H_eff, X, S = H_eff[0], X[0], S[0]

    Y = H_eff @ X   # (K, M)
    K = Y.shape[0]

    # Signal power per user: for BPSK = 1.0; use sample mean for generality
    sig_pwr_per_user = (S.abs() ** 2).mean(dim=1).real   # (K,)

    # MUI: (1/M) Σ_j |y_kj - s_kj|²
    err     = Y - S                                        # (K, M)
    mui_pwr = (err.abs() ** 2).mean(dim=1).real           # (K,)

    total = 0.0
    for k in range(K):
        sp   = float(sig_pwr_per_user[k].item())
        mp   = float(mui_pwr[k].item())
        sinr = sp / (mp + float(N0) + 1e-30)
        total += math.log2(1.0 + max(sinr, 0.0))
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Beampattern helpers  (paper eq.6-7)
# ─────────────────────────────────────────────────────────────────────────────

def compute_beampattern(X: torch.Tensor, N: int, n_angles: int = 181,
                        delta: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pd(φ) = a^H(φ) R_X a(φ),  R_X = (1/M) X X^H   (paper eq.6-7).
    Returns (Pd, angles_deg).
    """
    if X.dim() == 3:
        X = X[0]  # take first in batch
    angles = torch.linspace(-math.pi / 2, math.pi / 2, n_angles)
    n_idx  = torch.arange(N, dtype=torch.float32)
    # Steering matrix A: (N, n_angles)
    A = torch.exp(1j * 2 * math.pi * delta
                  * n_idx.unsqueeze(1)
                  * torch.sin(angles).unsqueeze(0))
    A  = A.to(device=X.device, dtype=X.dtype)
    Rx = (X @ herm(X)) / X.shape[-1]               # (N,N)
    Pd = torch.real(herm(A) @ Rx @ A)               # (n_angles, n_angles)
    Pd = torch.diag(Pd).cpu().numpy()               # (n_angles,)
    return Pd, angles.numpy() * 180.0 / math.pi


def ideal_beampattern_from_Rd(Rd: torch.Tensor, N: int,
                               n_angles: int = 181, delta: float = 0.5
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the target (ideal) beampattern directly from Rd:
      Pd_target(φ) = a^H(φ) Rd a(φ)
    This is what the paper uses as the reference beampattern.
    """
    if Rd.dim() == 3:
        Rd = Rd[0]
    angles = torch.linspace(-math.pi / 2, math.pi / 2, n_angles)
    n_idx  = torch.arange(N, dtype=torch.float32)
    A = torch.exp(1j * 2 * math.pi * delta
                  * n_idx.unsqueeze(1)
                  * torch.sin(angles).unsqueeze(0))
    A  = A.to(device=Rd.device, dtype=Rd.dtype)
    Pd = torch.real(herm(A) @ Rd @ A)
    Pd = torch.diag(Pd).cpu().numpy()
    angles_deg = angles.numpy() * 180.0 / math.pi
    # Normalise to [0, 1]
    mx = Pd.max()
    if mx > 1e-12:
        Pd = Pd / mx
    return Pd, angles_deg