"""
PyTorch Dataset for CapnoBase PPG windows.

Each sample is a fixed-length (WINDOW_SIZE) PPG segment, returned
alongside the mean and std used to normalise it so downstream code
can denormalise for visualisation and metrics.
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.utils import bandpass_fft

from scipy.signal import find_peaks


class CapnoBaseVAE(Dataset):
    """
    Sliding-window PPG dataset built from CapnoBase CSV files.

    Each CSV must contain a 'pleth_y' column (and 'co2_y' for the
    quality check).  Windows that contain fewer than 2 detectable
    PPG peaks are discarded.

    Args:
        data_dir:    Directory containing *_signal.csv files.
        window_size: Number of samples per window (default 3072).
        step_size:   Stride between consecutive windows (default 600).
        fs:          Sampling frequency in Hz (default 300).
        add_noise:   Reserved; currently unused.
        norm_mode:   One of "global", "per_record", or "none".
        global_mean: Required when norm_mode="global".
        global_std:  Required when norm_mode="global".
        eps:         Small constant for numerical stability in normalisation.

    Returns (per item):
        x_norm: Normalised window as a float32 tensor of shape (window_size,).
        m:      Mean used for normalisation (float32 scalar tensor).
        s:      Std  used for normalisation (float32 scalar tensor).
    """

    def __init__(self, data_dir, window_size=3072, step_size=600, fs=300,
                 add_noise=False, norm_mode="global",
                 global_mean=None, global_std=None, eps=1e-8):

        self.X = []
        self.M = []
        self.S = []

        self.norm_mode   = norm_mode
        self.global_mean = global_mean
        self.global_std  = global_std
        self.eps         = eps

        if self.norm_mode == "global":
            assert global_mean is not None and global_std is not None, (
                "norm_mode='global' requires global_mean and global_std."
            )

        files = sorted([f for f in os.listdir(data_dir) if f.endswith("_signal.csv")])

        for file in files:
            try:
                df = pd.read_csv(os.path.join(data_dir, file))
                if not all(col in df.columns for col in ["pleth_y", "co2_y"]):
                    continue

                pleth = df["pleth_y"].values.astype(np.float32)

                # Quality check: require at least 2 detectable peaks
                x_t  = torch.tensor(pleth).unsqueeze(0)
                x_bp = bandpass_fft(x_t, fs, 0.7, 3.0)[0].cpu().numpy()

                peaks, _ = find_peaks(
                    x_bp,
                    distance=int(0.7 * fs),
                    prominence=0.15 * np.std(x_bp),
                    height=np.percentile(x_bp, 60),
                )
                if len(peaks) < 2:
                    continue

                rec_mean = float(pleth.mean())
                rec_std  = float(pleth.std() + 1e-6)

                for i in range(0, len(pleth) - window_size, step_size):
                    seg = pleth[i:i + window_size].copy()

                    if self.norm_mode == "none":
                        m, s    = 0.0, 1.0
                        seg_norm = seg
                    elif self.norm_mode == "global":
                        m = float(self.global_mean)
                        s = float(self.global_std) + self.eps
                        seg_norm = (seg - m) / s
                    elif self.norm_mode == "per_record":
                        m = float(rec_mean)
                        s = float(rec_std) + self.eps
                        seg_norm = (seg - m) / s
                    else:
                        raise ValueError(f"Unknown norm_mode: {self.norm_mode}")

                    self.X.append(seg_norm.astype(np.float32))
                    self.M.append(np.float32(m))
                    self.S.append(np.float32(s))

            except Exception:
                continue

        self.X = np.stack(self.X).astype(np.float32)
        self.M = np.asarray(self.M, dtype=np.float32)
        self.S = np.asarray(self.S, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.M[idx], dtype=torch.float32),
            torch.tensor(self.S[idx], dtype=torch.float32),
        )
