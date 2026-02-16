# channels.py
from __future__ import annotations
import torch


def effective_channel(H_bu: torch.Tensor, H_ru: torch.Tensor, H_br: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    H_eff = H_bu + H_ru * diag(theta) * H_br

    Supports:
      - single sample:
          H_bu (K,N), H_ru (K,L), H_br (L,N), theta (L,) -> (K,N)
      - batch:
          H_bu (B,K,N), H_ru (B,K,L), H_br (B,L,N), theta (B,L) -> (B,K,N)
    """
    # diag(theta)*H_br  == theta[:,None]*H_br  (scale rows)
    ris_part = H_ru @ (theta.unsqueeze(-1) * H_br)
    return H_bu + ris_part
