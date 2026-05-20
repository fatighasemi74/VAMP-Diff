"""
General-purpose utilities: seeding, figure saving, signal helpers,
and global normalisation statistics.
"""

import math
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def seed_all(seed: int = 42):
    """Seed Python, NumPy, and PyTorch (CPU + all GPUs)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def savefig(fig, path: Path):
    """Save a matplotlib figure and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def denorm_window(x_norm_1d: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Invert global normalisation: x_raw = x_norm * std + mean."""
    return x_norm_1d * (std + 1e-8) + mean


def bandpass_fft(x: torch.Tensor, fs: int, f_low: float, f_high: float) -> torch.Tensor:
    """
    Zero-phase FFT bandpass filter.

    Args:
        x:      (B, L) signal tensor
        fs:     sampling frequency in Hz
        f_low:  lower cut-off frequency in Hz
        f_high: upper cut-off frequency in Hz

    Returns:
        Bandpass-filtered signal of the same shape as x.
    """
    Xf    = torch.fft.rfft(x, dim=-1)
    freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / fs).to(x.device)
    mask  = (freqs >= f_low) & (freqs <= f_high)
    Xf    = Xf * mask.unsqueeze(0)
    return torch.fft.irfft(Xf, n=x.shape[-1], dim=-1)


@torch.no_grad()
def compute_global_norm_from_raw_dataset(ds, max_items=None, eps=1e-8):
    """
    Compute global mean and std from a raw (unnormalised) dataset using
    Welford's online algorithm to avoid loading everything into memory.

    Args:
        ds:        Dataset whose __getitem__ returns (x_raw, m, s).
        max_items: If given, use only the first max_items samples.
        eps:       Stability floor for std.

    Returns:
        (mean, std) as Python floats.
    """
    n    = 0
    mean = 0.0
    M2   = 0.0

    N = len(ds) if max_items is None else min(len(ds), max_items)

    for i in range(N):
        x_raw, _, _ = ds[i]
        x = x_raw.reshape(-1).float()

        batch_n    = x.numel()
        batch_mean = x.mean().item()
        batch_var  = x.var(unbiased=False).item()

        if n == 0:
            mean = batch_mean
            M2   = batch_var * batch_n
            n    = batch_n
        else:
            delta  = batch_mean - mean
            new_n  = n + batch_n
            mean   = mean + delta * (batch_n / new_n)
            M2     = M2 + batch_var * batch_n + delta * delta * (n * batch_n / new_n)
            n      = new_n

    var = M2 / max(1, n)
    std = math.sqrt(max(var, eps))
    return float(mean), float(std)
