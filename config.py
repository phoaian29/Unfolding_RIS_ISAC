# config.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch


def get_device() -> str:
    """Auto-detect best available compute device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Config:
    # ── System dimensions (paper §V) ──────────────────────────────────────────
    N: int = 20          # BS antennas
    K: int = 4           # single-antenna users
    M: int = 30          # frame length (symbols)
    L: int = 16          # RIS elements

    # ── Power / noise ─────────────────────────────────────────────────────────
    P0_dbm: float = 20.0          # 20 dBm → 0.1 W
    SNR_dB: float = 20.0          # transmit SNR

    # ── Trade-off weight (paper eq.23) ────────────────────────────────────────
    rho: float = 0.5              # ρ ∈ [0,1]; 0 → pure radar, 1 → pure comm

    # ── Unfolding architecture ────────────────────────────────────────────────
    T: int = 10           # number of unfolding stages
    J_x: int = 3          # inner GD steps for X-layer
    J_theta: int = 3      # inner Riemannian steps for Θ-layer

    # ── Learnable step-size init (raw, before softplus) ──────────────────────
    beta_x_init: float = 0.05
    alpha_theta_init: float = 0.05
    max_beta_x: float = 0.5
    max_alpha_theta: float = 0.3

    # ── Regularisation ────────────────────────────────────────────────────────
    diag_reg: float = 1e-3

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size: int = 64
    epochs: int = 1000
    lr: float = 1e-3
    grad_clip_norm: float = 1
    weight_decay: float = 1e-5
    patience_lr: int = 60
    seed: Optional[int] = 42

    # ── Numerics ──────────────────────────────────────────────────────────────
    eps: float = 1e-9

    # ── Derived ───────────────────────────────────────────────────────────────
    @property
    def P0(self) -> float:
        """Power budget in Watts: P0_dBm → W."""
        return 10.0 ** (self.P0_dbm / 10.0) / 1e3

    @property
    def N0(self) -> float:
        """Noise variance: N0 = P0 / SNR_linear."""
        return self.P0 / (10.0 ** (self.SNR_dB / 10.0))

    @property
    def target_power(self) -> float:
        """||X||_F^2 = M * P0  (constraint 23b)."""
        return float(self.M) * self.P0

    @property
    def device(self) -> str:
        """Best available compute device (auto-detected at runtime)."""
        return get_device()