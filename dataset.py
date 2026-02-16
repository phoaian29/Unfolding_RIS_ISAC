# dataset.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

import torch

from utils_complex import cplx_randn


def bpsk_symbols(K: int, M: int, device="cpu", dtype=torch.complex64) -> torch.Tensor:
    # {-1, +1}
    s_real = (2 * torch.randint(0, 2, (K, M), device=device) - 1).to(torch.float32)
    return s_real.to(dtype=dtype)


def complex_rayleigh(shape: tuple[int, ...], device="cpu", dtype=torch.complex64) -> torch.Tensor:
    return cplx_randn(*shape, device=device, dtype=dtype)


def load_rd(N: int, rd_path: str | Path = "Rd.mat", device="cpu", dtype=torch.complex64) -> torch.Tensor:
    """
    Load Rd from MATLAB .mat (expects key 'Rd'). Fallback to Identity.
    """
    path = Path(rd_path)
    if path.exists():
        try:
            import scipy.io  # type: ignore
            rd_np = scipy.io.loadmat(path.as_posix())["Rd"]
            return torch.as_tensor(rd_np, device=device, dtype=dtype)
        except Exception:
            pass
    return torch.eye(N, device=device, dtype=dtype)


def generate_single_scenario(
    N: int, K: int, M: int, L: int,
    rd_path: str | Path = "Rd.mat",
    device="cpu",
    dtype=torch.complex64,
) -> Dict[str, Any]:
    """
    MATLAB-like channel generation:
      H_bu: (K,N), H_ru: (K,L), H_br: (L,N) ~ CN(0,1)
      S: (K,M) BPSK
      Rd: (N,N) loaded
    """
    S = bpsk_symbols(K, M, device=device, dtype=dtype)
    H_bu = complex_rayleigh((K, N), device=device, dtype=dtype)
    H_ru = complex_rayleigh((K, L), device=device, dtype=dtype)
    H_br = complex_rayleigh((L, N), device=device, dtype=dtype)
    Rd = load_rd(N, rd_path=rd_path, device=device, dtype=dtype)
    return {"S": S, "H_bu": H_bu, "H_ru": H_ru, "H_br": H_br, "Rd": Rd}


def generate_batch_scenarios(
    B: int, N: int, K: int, M: int, L: int,
    rd_path: str | Path = "Rd.mat",
    device="cpu",
    dtype=torch.complex64,
) -> Dict[str, Any]:
    """
    Batch version for training (B samples).
    """
    S = bpsk_symbols(K, M, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)  # (B,K,M)
    H_bu = complex_rayleigh((B, K, N), device=device, dtype=dtype)
    H_ru = complex_rayleigh((B, K, L), device=device, dtype=dtype)
    H_br = complex_rayleigh((B, L, N), device=device, dtype=dtype)
    Rd = load_rd(N, rd_path=rd_path, device=device, dtype=dtype)  # (N,N)
    return {"S": S, "H_bu": H_bu, "H_ru": H_ru, "H_br": H_br, "Rd": Rd}
