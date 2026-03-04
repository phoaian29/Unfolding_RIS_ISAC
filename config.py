# config.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # ── System dimensions (paper §V) ──────────────────────────────────────────
    N: int = 20          # BS antennas
    K: int = 4           # single-antenna users
    M: int = 30          # frame length (symbols)
    L: int = 16          # RIS elements

    # ── Power / noise ─────────────────────────────────────────────────────────
    P0_dbm: float = 20.0          # 20 dBm → 0.1 W
    SNR_dB: float = 20.0          # transmit SNR (dBm scale, paper §V)

    # ── Trade-off weight (paper eq.23) ────────────────────────────────────────
    rho: float = 0.2              # ρ ∈ [0,1]; 0 → pure radar, 1 → pure comm  (paper Fig 2)

    # ── Unfolding architecture ────────────────────────────────────────────────
    T: int = 10           # number of unfolding stages
    J_x: int = 3          # inner GD steps for X-layer
    J_theta: int = 3      # inner Riemannian steps for Θ-layer

    # ── Learnable step-size init (raw, before softplus) ──────────────────────
    beta_x_init: float = 0.05     # X-layer GD step
    alpha_theta_init: float = 0.05 # Θ-layer Riemannian step
    max_beta_x: float = 0.5
    max_alpha_theta: float = 0.3

    # ── Regularisation ────────────────────────────────────────────────────────
    diag_reg: float = 1e-4        # small ridge on X-step to keep curvature

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size: int = 64   # larger batch reduces MUI gradient variance
    epochs: int = 500
    lr: float = 5e-4
    grad_clip_norm: float = 0.5
    weight_decay: float = 1e-5
    patience_lr: int = 20         # ReduceLROnPlateau patience
    seed: Optional[int] = 42

    # ── Numerics ──────────────────────────────────────────────────────────────
    eps: float = 1e-9             # safe-guard denominator

    # ── Derived ───────────────────────────────────────────────────────────────
    @property
    def P0(self) -> float:
        """Power budget in Watts: P0_dBm → W."""
        return 10.0 ** (self.P0_dbm / 10.0) / 1e3  # 20 dBm → 0.1 W

    @property
    def N0(self) -> float:
        """Noise variance: N0 = P0 / SNR_linear."""
        snr = 10.0 ** (self.SNR_dB / 10.0)
        return self.P0 / snr

    @property
    def target_power(self) -> float:
        """||X||_F^2 = M * P0  (constraint 23b)."""
        return float(self.M) * self.P0
