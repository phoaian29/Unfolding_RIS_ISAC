"""
Plot the ideal radar beampattern derived from Rd.mat.

Usage:
    python plot_beampattern_Rd.py [--rd Rd.mat] [--out beampattern_Rd.png]
"""
import argparse
import math
import numpy as np
import scipy.io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_Rd(path: str) -> np.ndarray:
    """Load Rd matrix from .mat file; fallback to 3-target synthesis."""
    try:
        mat = scipy.io.loadmat(path)
        return np.array(mat["Rd"], dtype=complex)
    except Exception as e:
        print(f"  [WARN] Could not load {path}: {e}")
        return None


def build_Rd_fallback(N: int, angles_deg=(-60., 0., 60.), delta: float = 0.5) -> np.ndarray:
    """Synthesise Rd from target angles (paper §V default)."""
    n = np.arange(N)
    cols = [np.exp(1j * 2 * math.pi * delta * n * math.sin(a * math.pi / 180))
            for a in angles_deg]
    A  = np.stack(cols, axis=1)
    Rd = (A @ A.conj().T) / len(angles_deg)
    Rd = Rd / np.diag(Rd.real).clip(1e-9).mean()
    return Rd


def compute_beampattern(Rd: np.ndarray, n_angles: int = 361, delta: float = 0.5):
    """Pd(φ) = a^H(φ) Rd a(φ), returned with angles in degrees."""
    N      = Rd.shape[0]
    angles = np.linspace(-math.pi / 2, math.pi / 2, n_angles)
    n_idx  = np.arange(N)
    # A: (N, n_angles)
    A  = np.exp(1j * 2 * math.pi * delta
                * n_idx[:, None] * np.sin(angles)[None, :])
    Pd = np.real(np.einsum("ia,ij,ja->a", A.conj(), Rd, A))
    return Pd, angles * 180.0 / math.pi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rd",  default="Rd.mat",             help="Path to Rd.mat")
    parser.add_argument("--out", default="beampattern_Rd.png", help="Output image path")
    parser.add_argument("--N",   type=int, default=20,         help="Number of antennas (if Rd not loaded)")
    args = parser.parse_args()

    Rd = load_Rd(args.rd)
    if Rd is None:
        print(f"  Falling back to synthetic Rd (N={args.N}, targets: -60°, 0°, 60°)")
        Rd = build_Rd_fallback(args.N)
    else:
        print(f"  Loaded Rd from '{args.rd}', shape: {Rd.shape}")

    N        = Rd.shape[0]
    Pd, angs = compute_beampattern(Rd, n_angles=361)

    # Normalise to 0 dB peak
    Pd_norm = Pd / Pd.max()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: linear scale ──────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(angs, Pd_norm, "b-", lw=2.0)
    ax.fill_between(angs, Pd_norm, alpha=0.12, color="blue")
    ax.set_xlim(-90, 90)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r"Angle $\phi$ (°)", fontsize=13)
    ax.set_ylabel("Normalised beampattern", fontsize=13)
    ax.set_title(f"Ideal beampattern from Rd  ($N={N}$)", fontsize=12)
    ax.grid(True, alpha=0.35)
    ax.set_xticks(range(-90, 91, 15))

    # ── Right: dB scale ──────────────────────────────────────────────────────
    ax2 = axes[1]
    Pd_dB = 10 * np.log10(Pd_norm.clip(1e-6))
    ax2.plot(angs, Pd_dB, "r-", lw=2.0)
    ax2.fill_between(angs, Pd_dB, -60, alpha=0.10, color="red")
    ax2.set_xlim(-90, 90)
    ax2.set_ylim(-60, 5)
    ax2.set_xlabel(r"Angle $\phi$ (°)", fontsize=13)
    ax2.set_ylabel("Beampattern (dB)", fontsize=13)
    ax2.set_title(f"Ideal beampattern from Rd  ($N={N}$, dB)", fontsize=12)
    ax2.grid(True, alpha=0.35)
    ax2.set_xticks(range(-90, 91, 15))

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    plt.close()
    print(f"  → Saved: {args.out}")


if __name__ == "__main__":
    main()
