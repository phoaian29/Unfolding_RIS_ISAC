# utils_complex.py
from __future__ import annotations
import math
import torch

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # deterministic (tùy bạn bật/tắt)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cplx_randn(*shape: int, device="cpu", dtype=torch.complex64) -> torch.Tensor:
    # (randn + j randn)/sqrt(2)
    real = torch.randn(*shape, device=device)
    imag = torch.randn(*shape, device=device)
    return ((real + 1j * imag) / math.sqrt(2.0)).to(dtype)

def herm(A: torch.Tensor) -> torch.Tensor:
    return A.conj().transpose(-2, -1)

def fro2(A: torch.Tensor) -> torch.Tensor:
    return (A.abs() ** 2).sum()

def unit_modulus(theta: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return theta / (theta.abs() + eps)

def safe_cholesky(R: torch.Tensor, jitter: float = 1e-9, max_tries: int = 6) -> torch.Tensor:
    """Cholesky ổn định cho ma trận gần PSD (hỗ trợ complex)."""
    R = 0.5 * (R + herm(R))
    eye = torch.eye(R.shape[-1], device=R.device, dtype=R.dtype)
    if R.dim() == 3:
        eye = eye.unsqueeze(0).expand(R.shape[0], -1, -1)
    for i in range(max_tries):
        try:
            return torch.linalg.cholesky(R + (jitter * (10 ** i)) * eye)
        except RuntimeError:
            continue
    # fallback: làm PSD bằng eigen
    evals, evecs = torch.linalg.eigh(R)
    evals = torch.clamp(evals, min=1e-10)  # SỬA: min=1e-10 để tránh singular
    R_psd = (evecs * evals.unsqueeze(-2)) @ herm(evecs)
    return torch.linalg.cholesky(R_psd + jitter * eye)

def init_XU_from_Rd(Rd: torch.Tensor, M: int, P0: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Init (X0, U0) từ bài strict radar (10)/(11): tạo waveform có công suất ||X||_F^2 = M*P0
    và có covariance bám theo Rd (nếu Rd được chuẩn hoá).
    - Rd: (N,N) hoặc (B,N,N)
    - return: X0, U0 cùng shape (N,M) hoặc (B,N,M)
    """
    device, dtype = Rd.device, Rd.dtype
    is_batch = (Rd.dim() == 3)
    N = Rd.shape[-1]

    F = safe_cholesky(Rd)  # (N,N) hoặc (B,N,N)

    # SỬA: Thêm debug print
    # print("Rd min eigenvalue:", torch.linalg.eigvalsh(Rd).min().item())
    # print("F norm:", torch.norm(F).item())

    # Tạo Q có hàng trực chuẩn: Q Q^H = I_N (khi M >= N)
    # Cách làm: QR trên ma trận (M,N) rồi lấy Q^H -> (N,M)
    if not is_batch:
        G = cplx_randn(M, N, device=device, dtype=dtype)  # (M,N)
        Qm, _ = torch.linalg.qr(G, mode="reduced")        # (M,N)
        Q = herm(Qm)                                     # (N,M)
        X0 = F @ Q                                       # (N,M)
        # SỬA: Debug before scaling
        # print("X0 norm before scaling:", torch.norm(X0).item())
        target = torch.sqrt(torch.tensor(M * float(P0), device=device, dtype=torch.float32))
        # print("Target norm:", target.item())
        nrm = torch.norm(X0) + 1e-12
        scale = target / nrm
        # SỬA: Clamp scale để tránh explode
        if scale > 10.0 or scale < 0.1:
            print(f"Warning: abnormal scale {scale.item():.4f} → clamp to 1.0")
            scale = 1.0
        # print("Scale factor:", scale.item())
        X0 = X0 * scale
        # SỬA: Debug after scaling
        # print("X0 norm after scaling:", torch.norm(X0).item())
        U0 = X0.clone()
        return X0, U0

    # batch (giữ nguyên nhưng thêm debug tương tự nếu cần)
    B = Rd.shape[0]
    G = cplx_randn(B, M, N, device=device, dtype=dtype)   # (B,M,N)
    Qm, _ = torch.linalg.qr(G, mode="reduced")            # (B,M,N)
    Q = herm(Qm)                                          # (B,N,M)
    X0 = F @ Q                                            # (B,N,M)
    target = torch.sqrt(torch.tensor(M * float(P0), device=device, dtype=torch.float32))
    nrm = torch.linalg.norm(X0.reshape(B, -1), dim=1).view(B, 1, 1)
    scale = target / (nrm + 1e-12)
    # SỬA: Clamp scale cho batch
    scale = torch.clamp(scale, min=0.1, max=10.0)
    X0 = X0 * scale
    U0 = X0.clone()
    return X0, U0