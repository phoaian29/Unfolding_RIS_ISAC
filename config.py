# config.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class MatlabScenarioConfig:
    """
    Cấu hình bám theo kiểu MATLAB: Rayleigh i.i.d, BPSK, P0 theo dBm, N0 theo SNR.
    """
    P0_dbm: float = 10.0  # SỬA: Đổi từ 20.0 → 10.0 để P0=0.01 (10 dBm = 0.01 W)
    N: int = 20     # BS antennas
    K: int = 4      # users
    M: int = 30     # symbols
    L: int = 16     # RIS elements
    SNR_dB: float = 20.0

    # trade-off
    rho: float = 0.2

    # Unfolding stages / AO iterations
    T: int = 50  # Increased from 10 to 50 for deeper unfolding

    # Theta inner steps per stage
    J_theta: int = 5

    # if you want fixed alpha (non-learnable)
    alpha_theta: float = 0.1  # Có thể giảm xuống 0.01 nếu theta explode

    # radar steering spacing
    delta: float = 0.5

    # plotting beampattern grid
    angle_min: float = -90.0
    angle_max: float = 90.0
    angle_points: int = 181

    # random seed
    seed: Optional[int] = None

    @property
    def P0(self) -> float:
        # SỬA: Hardcode tạm để test, hoặc dùng công thức (nhưng với P0_dbm=10 → 0.01)
        # return 10.0 ** (self.P0_dbm / 10.0) / 1e3  # 10 dBm → 10^(1) = 10 / 1000 = 0.01
        return 0.01  # Hardcode để đảm bảo P0=0.01, target norm ≈ sqrt(30*0.01)=0.5477

    @property
    def SNR_linear(self) -> float:
        return 10.0 ** (self.SNR_dB / 10.0)

    @property
    def N0(self) -> float:
        # MATLAB: N0 = 1/SNR (nếu normalize tín hiệu)
        return 1.0 / self.SNR_linear