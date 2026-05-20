"""
Visualisation helpers: reconstruction plots and training curve plots.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from utils.utils import savefig


def plot_recon_pair(x_true: np.ndarray, x_hat: np.ndarray,
                    fs: int, title: str, outpath: Path):
    """
    Plot a single input / reconstruction pair and save to disk.

    Args:
        x_true:  Ground-truth signal (denormalised), shape (L,).
        x_hat:   Reconstructed signal (denormalised), shape (L,).
        fs:      Sampling frequency in Hz (for x-axis in seconds).
        title:   Figure title string.
        outpath: Output file path (parent dirs are created automatically).
    """
    t   = np.arange(len(x_true)) / fs
    fig = plt.figure(figsize=(10, 3))
    plt.plot(t, x_true, label="input")
    plt.plot(t, x_hat, "--", label="reconstruction")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (raw)")
    plt.title(title)
    plt.legend()
    savefig(fig, outpath)


def plot_diff_curves(history: dict, outpath: Path):
    """
    Plot all diffusion / reconstruction loss curves from training history.

    Args:
        history: Dict with keys like 'train_loss', 'val_loss', etc.
        outpath: Output file path.
    """
    fig = plt.figure(figsize=(11, 5))
    keys = [
        "train_loss", "val_loss",
        "train_diff", "val_diff",
        "train_recon", "val_recon",
        "train_spec",  "val_spec",
        "train_deriv", "val_deriv",
    ]
    for key in keys:
        if key in history:
            plt.plot(history[key], label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training curves")
    plt.legend()
    savefig(fig, outpath)


def plot_kl_curves(history: dict, outpath: Path):
    """
    Plot raw and free-bits KL curves from training history.

    Args:
        history: Dict with keys like 'train_kl_raw', 'val_kl_fb', etc.
        outpath: Output file path.
    """
    fig = plt.figure(figsize=(10, 4))
    for key in ["train_kl_raw", "val_kl_raw", "train_kl_fb", "val_kl_fb"]:
        if key in history:
            plt.plot(history[key], label=key)
    plt.xlabel("Epoch")
    plt.ylabel("KL")
    plt.title("KL curves")
    plt.legend()
    savefig(fig, outpath)
