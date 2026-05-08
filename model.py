
# VAE_Diff.py
import os
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from scipy.stats import pearsonr
from torch.utils.data import Dataset, DataLoader


# =========================================================
# Config
# =========================================================
WINDOW_SIZE = 3072
STEP_SIZE = 600
FS = 300
LATENT_NOISE_SCALE = 0.05

MAX_TRAIN_SAMPLES = 6815 #was 5000  

# ---- sequence-latent encoder ----
BASE_CH = 64          # wider than before
LATENT_CH = 256       # sequence latent channels
LATENT_T = WINDOW_SIZE // 4   # 3072 -> 768 after 2 downsamples

# ---- diffusion decoder ----
DIFF_BASE = 64
TIME_DIM = 128
N_RES = 2 #3 #was 2

ENC_RES_PER_STAGE = 2
SKIP_DROPOUT = 0.5
SKIP_GATE_INIT = 0.01 #  summaries start nearly zero  — force decoder toward z
Z_PERTURB_SCALE = 0.01

# ---- train ----
BATCH_SIZE = 32 #was 8
EPOCHS = 200
LR = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

# ---- diffusion schedule ----
DIFF_T = 100
BETA_START = 1e-4
BETA_END = 0.02
SCHEDULE = "linear"     # "linear" or "cosine"

USE_DDIM = True
DDIM_STEPS = 50
DDIM_ETA = 0.0

# ---- prediction target ----
# start with eps first, as discussed
PRED_TARGET = "x0"     # "eps" or "x0"

# ---- losses ----
LAMBDA_DIFF = 1.0
LAMBDA_SPEC = 0.1
LAMBDA_DERIV = 0.1
LAMBDA_RECON = 5.0
LAMBDA_AMP   = 2.0 #try 5.0
LAMBDA_PTP   = 1.0 #try 2.0

# ---- KL ----
ENC_FREEZE_EPOCHS = 20
# BETA_KL_MIN = 1e-7      # tiny floor from epoch 4 onward
WARMUP_EPOCHS = 50
RAMP_EPOCHS = 80
BETA_KL_MAX = 5e-8 # was 3e-5
BETA_KL_MIN    = 1e-8    # ← new: tiny floor applied from epoch 4 onward
BETA_MODE = "floor_then_ramp"   # or "warmup_then_hold" or "zero_then_ramp" or "cyclical" or "floor_then_ramp"
FREE_BITS_NATS = 0.5    # was  0.01

# ---- latent usage ----
USE_MU_FOR_RECON = False   # deterministic recon path for evaluation
SAMPLE_Z_IN_TRAIN = True  # stochastic posterior during training

USE_LATENT_CONDITIONING = False   # ablation: no z
# ---- data ----
TRAIN_DIR = r"\dataverse_files\train_val_test\train"
VAL_DIR   = r"\dataverse_files\train_val_test\val"
TEST_DIR  = r"\dataverse_files\train_val_test\test"

OUT_DIR = Path(r"\full_prior_run")


# =========================================================
# Utilities
# =========================================================
def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def savefig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def denorm_window(x_norm_1d: np.ndarray, mean: float, std: float):
    return x_norm_1d * (std + 1e-8) + mean


def bandpass_fft(x: torch.Tensor, fs: int, f_low: float, f_high: float) -> torch.Tensor:
    """
    Simple FFT bandpass.
    x: (B, L)
    """
    Xf = torch.fft.rfft(x, dim=-1)
    freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / fs).to(x.device)
    mask = (freqs >= f_low) & (freqs <= f_high)
    Xf = Xf * mask.unsqueeze(0)
    x_bp = torch.fft.irfft(Xf, n=x.shape[-1], dim=-1)
    return x_bp


@torch.no_grad()
def compute_global_norm_from_raw_dataset(ds, max_items=None, eps=1e-8):
    n = 0
    mean = 0.0
    M2 = 0.0

    N = len(ds) if max_items is None else min(len(ds), max_items)

    for i in range(N):
        x_raw, _, _ = ds[i]
        x = x_raw.reshape(-1).float()

        batch_n = x.numel()
        batch_mean = x.mean().item()
        batch_var = x.var(unbiased=False).item()

        if n == 0:
            mean = batch_mean
            M2 = batch_var * batch_n
            n = batch_n
        else:
            delta = batch_mean - mean
            new_n = n + batch_n
            mean = mean + delta * (batch_n / new_n)
            M2 = M2 + batch_var * batch_n + delta * delta * (n * batch_n / new_n)
            n = new_n

    var = M2 / max(1, n)
    std = math.sqrt(max(var, eps))
    return float(mean), float(std)


# =========================================================
# Dataset
# =========================================================
class CapnoBaseVAE(Dataset):
    def __init__(self, data_dir, window_size=3072, step_size=200, fs=300,
                 add_noise=False, norm_mode="global",
                 global_mean=None, global_std=None, eps=1e-8):

        self.X = []
        self.M = []
        self.S = []

        self.norm_mode = norm_mode
        self.global_mean = global_mean
        self.global_std = global_std
        self.eps = eps

        if self.norm_mode == "global":
            assert global_mean is not None and global_std is not None, \
                "For norm_mode='global', pass global_mean/global_std."

        files = sorted([f for f in os.listdir(data_dir) if f.endswith("_signal.csv")])

        for file in files:
            try:
                df = pd.read_csv(os.path.join(data_dir, file))
                if not all(col in df.columns for col in ["pleth_y", "co2_y"]):
                    continue

                pleth = df["pleth_y"].values.astype(np.float32)

                # basic quality check
                x_t = torch.tensor(pleth).unsqueeze(0)
                x_bp = bandpass_fft(x_t, fs, 0.7, 3.0)[0].cpu().numpy()

                peaks, _ = find_peaks(
                    x_bp,
                    distance=int(0.7 * fs),
                    prominence=0.15 * np.std(x_bp),
                    height=np.percentile(x_bp, 60)
                )
                if len(peaks) < 2:
                    continue

                rec_mean = float(pleth.mean())
                rec_std = float(pleth.std() + 1e-6)

                for i in range(0, len(pleth) - window_size, step_size):
                    seg = pleth[i:i + window_size].copy()

                    if self.norm_mode == "none":
                        m, s = 0.0, 1.0
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


# =========================================================
# Losses
# =========================================================
def reparam(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_seq(mu, logvar):
    """
    mu, logvar: (B, C, T)
    """
    kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
    kl = kl.sum(dim=(1, 2))
    return kl.mean()


def kl_diag_standard_normal(mu: torch.Tensor, logvar: torch.Tensor):
    kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)  # (B,C,T)
    kl_raw = kl_per_dim.sum(dim=(1, 2)).mean()
    return kl_per_dim, kl_raw


def kl_free_bits(mu: torch.Tensor, logvar: torch.Tensor, free_bits_nats: float):
    kl_per_dim, kl_raw = kl_diag_standard_normal(mu, logvar)
    kl_fb = torch.clamp(kl_per_dim, min=free_bits_nats).sum(dim=(1, 2)).mean()
    return kl_raw, kl_fb


def derivative_loss(x_hat, x):
    dx_hat = x_hat[:, 1:] - x_hat[:, :-1]
    dx = x[:, 1:] - x[:, :-1]
    return F.smooth_l1_loss(dx_hat, dx)


def spectral_loss(x_hat, x):
    X = torch.fft.rfft(x, dim=-1)
    Xh = torch.fft.rfft(x_hat, dim=-1)

    mag = torch.log1p(torch.abs(X))
    mag_h = torch.log1p(torch.abs(Xh))

    return F.smooth_l1_loss(mag_h, mag)


# =========================================================
# Schedules
# =========================================================
def make_beta_schedule(T: int, beta_start=1e-4, beta_end=0.02, kind="linear"):
    if kind == "linear":
        return torch.linspace(beta_start, beta_end, T)
    if kind == "cosine":
        steps = torch.arange(T + 1, dtype=torch.float32)
        s = 0.008
        f = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
        alphas_bar = f / f[0]
        betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
        return betas.clamp(1e-6, 0.999)
    raise ValueError("Unknown schedule kind.")


def beta_linear_warmup_then_hold(epoch: int, beta_max: float, warmup_epochs: int, start: float = 0.0):
    if warmup_epochs <= 0:
        return beta_max
    frac = min(1.0, epoch / warmup_epochs)
    return start + frac * (beta_max - start)

def beta_zero_then_ramp(epoch_idx0, warmup_epochs, ramp_epochs, beta_max):
    if epoch_idx0 < warmup_epochs:
        return 0.0
    t = (epoch_idx0 - warmup_epochs) / max(1, ramp_epochs)
    t = max(0.0, min(1.0, t))
    return beta_max * t

def beta_floor_then_ramp(epoch_idx0, freeze_epochs, warmup_epochs, ramp_epochs, beta_min, beta_max):
    if epoch_idx0 < freeze_epochs:
        return 0.0
    if epoch_idx0 < warmup_epochs:
        return beta_min
    t = (epoch_idx0 - warmup_epochs) / max(1, ramp_epochs)
    t = max(0.0, min(1.0, t))
    # log-space interpolation — avoids shock at ramp start
    log_min = math.log10(beta_min)
    log_max = math.log10(beta_max)
    return 10 ** (log_min + t * (log_max - log_min))



class DiffusionConfig:
    def __init__(self, T=200, beta_start=1e-4, beta_end=0.02, schedule="linear"):
        self.T = T
        self.betas = make_beta_schedule(T, beta_start, beta_end, kind=schedule)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.alphas_bar_prev = torch.cat([torch.ones(1), self.alphas_bar[:-1]], dim=0)

        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - self.alphas_bar)

        self.posterior_var = self.betas * (1.0 - self.alphas_bar_prev) / (1.0 - self.alphas_bar)
        self.posterior_var[0] = 1e-8

    def to(self, device):
        for k, v in self.__dict__.items():
            if torch.is_tensor(v):
                setattr(self, k, v.to(device))
        return self


def extract(a: torch.Tensor, t: torch.Tensor, x_shape):
    out = a.gather(0, t).to(t.device)
    return out.view(-1, *([1] * (len(x_shape) - 1)))


def q_sample(diff: DiffusionConfig, x0, t, noise):
    sqrt_ab = extract(diff.sqrt_alphas_bar, t, x0.shape)
    sqrt_1mab = extract(diff.sqrt_one_minus_alphas_bar, t, x0.shape)
    return sqrt_ab * x0 + sqrt_1mab * noise


def x0_from_eps(diff: DiffusionConfig, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor):
    abar_t = extract(diff.alphas_bar, t, x_t.shape)
    return (x_t - torch.sqrt(1.0 - abar_t) * eps) / torch.sqrt(abar_t)


def eps_from_x0(diff: DiffusionConfig, x_t: torch.Tensor, t: torch.Tensor, x0: torch.Tensor):
    abar_t = extract(diff.alphas_bar, t, x_t.shape)
    return (x_t - torch.sqrt(abar_t) * x0) / torch.sqrt(1.0 - abar_t)


def make_ddim_timesteps(T: int, ddim_steps: int, device):
    if ddim_steps <= 1:
        return torch.tensor([T - 1], device=device, dtype=torch.long)
    ts = torch.linspace(0, T - 1, steps=ddim_steps, device=device)
    ts = torch.round(ts).long()
    ts[0] = 0
    ts[-1] = T - 1
    ts = torch.unique_consecutive(ts)
    return ts



class ResBlockEnc(nn.Module):
    def __init__(self, ch, gn_groups=8, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(gn_groups, ch), ch)
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(gn_groups, ch), ch)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.dropout(h)
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


def make_enc_stage(ch, n_blocks, gn_groups=8, dropout=0.0):
    return nn.Sequential(*[
        ResBlockEnc(ch, gn_groups=gn_groups, dropout=dropout)
        for _ in range(n_blocks)
    ])


class SeqEncoder1D(nn.Module):
    """
    Input:  (B,1,3072)
    Output after 2 downsamples: (B, LATENT_CH, 768)
    Also returns compressed intermediate features if requested.
    """
    def __init__(self, in_ch=1, base=64, latent_ch=256, gn_groups=8):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base, kernel_size=5, padding=2),
            nn.GroupNorm(min(gn_groups, base), base),
            nn.SiLU(),
            make_enc_stage(base, ENC_RES_PER_STAGE, gn_groups=gn_groups, dropout=0.10),
        )

        self.down1 = nn.Sequential(
            nn.Conv1d(base, base * 2, kernel_size=5, stride=2, padding=2),   # 3072 -> 1536
            nn.GroupNorm(min(gn_groups, base * 2), base * 2),
            nn.SiLU(),
            make_enc_stage(base * 2, ENC_RES_PER_STAGE, gn_groups=gn_groups, dropout=0.10),
        )

        self.down2 = nn.Sequential(
            nn.Conv1d(base * 2, base * 4, kernel_size=5, stride=2, padding=2),   # 1536 -> 768
            nn.GroupNorm(min(gn_groups, base * 4), base * 4),
            nn.SiLU(),
            make_enc_stage(base * 4, ENC_RES_PER_STAGE + 1, gn_groups=gn_groups, dropout=0.10),
        )

        feat_ch = base * 4
        self.mu = nn.Conv1d(feat_ch, latent_ch, kernel_size=1)
        self.logvar = nn.Conv1d(feat_ch, latent_ch, kernel_size=1)

    def forward(self, x, return_feats=False):
        h0 = self.stem(x)      # (B, base, 3072)
        h1 = self.down1(h0)    # (B, 2base, 1536)
        h2 = self.down2(h1)    # (B, 4base, 768)

        mu = self.mu(h2)
        logvar = self.logvar(h2)

        if return_feats:
            return mu, logvar, [h0, h1, h2]
        return mu, logvar


# =========================================================
# VampPrior
# =========================================================
class VampPrior(nn.Module):
    """
    Variational Mixture of Posteriors Prior.
    
    K learnable pseudo-inputs (fake PPG waveforms) are passed through
    the encoder to produce K Gaussian distributions. The prior is the
    mixture of these K Gaussians.
    
    Prior: p(z) = (1/K) sum_k N(mu_k, sigma_k^2)
    where mu_k, sigma_k = encoder(u_k)
    
    At generation time:
        1. Pick random pseudo-input u_k
        2. Encode it -> mu_k, sigma_k
        3. Sample z ~ N(mu_k, sigma_k)
        4. Decode z -> PPG
    """

    def __init__(self, K, window_size, encoder):
        super().__init__()
        self.K = K
        self.encoder = encoder
        self.pseudo_inputs = nn.Parameter(
            torch.zeros(K, 1, window_size)
        )



    def log_prob_ds(self, z_ds, mu_k_ds, logvar_k_ds, chunk_size=50):
        """
        Compute log p_vamp(z_ds) using pre-downsampled prior components.
        z_ds:       (B, 256, 32) — downsampled posterior sample
        mu_k_ds:    (K, 256, 32) — downsampled prior means
        logvar_k_ds:(K, 256, 32) — downsampled prior logvars
        """
        K = self.K
        log_p_k_list = []

        for start in range(0, K, chunk_size):
            end = min(start + chunk_size, K)

            mu_chunk     = mu_k_ds[start:end]
            logvar_chunk = logvar_k_ds[start:end]
            var_chunk    = torch.exp(logvar_chunk)

            z_exp  = z_ds.unsqueeze(1).detach()
            mu_exp = mu_chunk.unsqueeze(0)
            lv_exp = logvar_chunk.unsqueeze(0)
            va_exp = var_chunk.unsqueeze(0)

            log_p_per_dim = (
                -0.5 * math.log(2 * math.pi)
                - 0.5 * lv_exp
                - 0.5 * (z_exp - mu_exp).pow(2) / (va_exp + 1e-8)
            )

            log_p_k_list.append(log_p_per_dim.sum(dim=(2, 3)))

            del log_p_per_dim, z_exp, mu_exp, lv_exp, va_exp
            torch.cuda.empty_cache()

        log_p_k    = torch.cat(log_p_k_list, dim=1)
        log_p_vamp = torch.logsumexp(log_p_k, dim=1) - math.log(K)
        return log_p_vamp


    def get_prior_params(self):
        mu_k, logvar_k = self.encoder(
            self.pseudo_inputs, return_feats=False
        )
        return mu_k, logvar_k   # full (K, 256, 768) — no downsampling


    def log_prob(self, z, chunk_size=5):
        """
        Compute log p_vamp(z) in chunks over K to avoid OOM.
        Gradients flow to pseudo_inputs through the encoder.
        Encoder weights are protected by their requires_grad=False during freeze phase.
        """
        mu_k, logvar_k = self.get_prior_params()


        K = self.K
        log_p_k_list = []

        for start in range(0, K, chunk_size):
            end = min(start + chunk_size, K)

            mu_chunk     = mu_k[start:end]
            logvar_chunk = logvar_k[start:end]
            var_chunk    = torch.exp(logvar_chunk)

            z_exp  = z.unsqueeze(1).detach()   # detach z here to save memory
            mu_exp = mu_chunk.unsqueeze(0)
            lv_exp = logvar_chunk.unsqueeze(0)
            va_exp = var_chunk.unsqueeze(0)

            log_p_per_dim = (
                -0.5 * math.log(2 * math.pi)
                - 0.5 * lv_exp
                - 0.5 * (z_exp - mu_exp).pow(2) / (va_exp + 1e-8)
            )

            log_p_k_list.append(log_p_per_dim.sum(dim=(2, 3)))

            del log_p_per_dim, z_exp, mu_exp, lv_exp, va_exp
            torch.cuda.empty_cache()

        log_p_k    = torch.cat(log_p_k_list, dim=1)
        log_p_vamp = torch.logsumexp(log_p_k, dim=1) - math.log(K)
        return log_p_vamp


# =========================================================
# Diffusion decoder conditioned on sequence latent z only
# =========================================================
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=t.device).float() / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


class FiLMResBlock1D(nn.Module):
    def __init__(self, ch, cond_dim, gn_groups=8):  # remove z_spatial_ch param
        super().__init__()
        self.norm1 = nn.GroupNorm(min(gn_groups, ch), ch)
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(gn_groups, ch), ch)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=1)
        self.film = nn.Linear(cond_dim, 2 * ch)  # back to single film

    def forward(self, x, cond):  # remove z_spatial param
        h = self.conv1(F.silu(self.norm1(x)))
        gamma_beta = self.film(cond).unsqueeze(-1)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        h = (1 + gamma) * h + beta
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h

def make_resblocks(n, ch, cond_dim):  # remove z_spatial_ch param
    return nn.ModuleList([FiLMResBlock1D(ch, cond_dim) for _ in range(n)])

class SequenceConditionProjector(nn.Module):
    """
    Convert sequence latent z: (B, Cz, 768)
    into multi-scale conditioning maps for the waveform decoder.
    This uses only z, not encoder intermediate features.
    """
    def __init__(self, latent_ch=128, base=64):
        super().__init__()

        self.proj_l4 = nn.Sequential(
            nn.Conv1d(latent_ch, base * 4, 1),
            nn.SiLU()
        )
        self.proj_l2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),  # 768 -> 1536
            nn.Conv1d(latent_ch, base * 2, 1),
            nn.SiLU()
        )
        self.proj_l1 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="linear", align_corners=False),  # 768 -> 3072
            nn.Conv1d(latent_ch, base, 1),
            nn.SiLU()
        )

        self.gate_l4 = nn.Parameter(torch.tensor(0.1))
        self.gate_l2 = nn.Parameter(torch.tensor(0.1))
        self.gate_l1 = nn.Parameter(torch.tensor(0.1))

    def forward(self, z):
        c4 = self.gate_l4 * self.proj_l4(z)  # (B, 4base, 768)
        c2 = self.gate_l2 * self.proj_l2(z)  # (B, 2base, 1536)
        c1 = self.gate_l1 * self.proj_l1(z)  # (B, base, 3072)
        return c1, c2, c4

class CompressedSummary1D(nn.Module):
    # summaries of intermediate features, not raw skips for decoder.
    def __init__(self, in_ch, out_ch, dropout=0.4, gate_init=0.05):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.tensor(gate_init))

    def forward(self, x):
        x = self.proj(x)
        x = F.silu(self.norm(x))
        x = self.dropout(x)
        return self.gate * x

class DiffusionUNet1D_SeqZ(nn.Module):
    """
    Waveform diffusion decoder conditioned on sequence latent z
    plus compressed encoder summaries (not raw skips).
    """
    def __init__(self, in_ch=1, base=64, time_dim=128, latent_ch=256, n_res=3):
        super().__init__()

        self.time_emb = TimeEmbedding(time_dim)

        self.z_global = nn.Sequential(
            nn.Conv1d(latent_ch, time_dim, 1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.z_seq_proj = SequenceConditionProjector(latent_ch=latent_ch, base=base)

        self.sum0 = CompressedSummary1D(base,     base,     dropout=SKIP_DROPOUT, gate_init=SKIP_GATE_INIT)
        self.sum1 = CompressedSummary1D(base * 2, base * 2, dropout=SKIP_DROPOUT, gate_init=SKIP_GATE_INIT)
        self.sum2 = CompressedSummary1D(base * 4, base * 4, dropout=SKIP_DROPOUT, gate_init=SKIP_GATE_INIT)

        cond_dim = time_dim

        self.in_conv = nn.Conv1d(in_ch, base, 3, padding=1)

        self.down1 = make_resblocks(n_res, base,     cond_dim)
        self.downsample1 = nn.Conv1d(base, base * 2, 4, stride=2, padding=1)

        self.down2 = make_resblocks(n_res, base * 2, cond_dim)
        self.downsample2 = nn.Conv1d(base * 2, base * 4, 4, stride=2, padding=1)

        self.mid = make_resblocks(n_res, base * 4, cond_dim)

        self.upsample2 = nn.ConvTranspose1d(base * 4, base * 2, 4, stride=2, padding=1)
        self.up2 = make_resblocks(n_res, base * 2, cond_dim)

        self.upsample1 = nn.ConvTranspose1d(base * 2, base, 4, stride=2, padding=1)
        self.up1 = make_resblocks(n_res, base, cond_dim)

        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base),
            nn.SiLU(),
            nn.Conv1d(base, 1, 3, padding=1),
        )

    def forward(self, x_t, t, z_seq, enc_summaries=None):
            t_emb = self.time_emb(t)
            
            if USE_LATENT_CONDITIONING:
                z_emb = self.z_global(z_seq).squeeze(-1)
                cond  = t_emb + z_emb
                z_c1, z_c2, z_c4 = self.z_seq_proj(z_seq)
            else:
                cond  = t_emb                        # no z in FiLM
                z_c1 = z_c2 = z_c4 = 0.0            # no spatial z injection

    def forward(self, x_t, t, z_seq, enc_summaries=None):
        t_emb = self.time_emb(t)
        z_emb = self.z_global(z_seq).squeeze(-1)
        cond  = t_emb + z_emb

        # RESTORED: z_seq_proj instead of z_proj_l1/l2/l4
        z_c1, z_c2, z_c4 = self.z_seq_proj(z_seq)

        s0 = s1 = s2 = 0.0
        if enc_summaries is not None:
            h0_enc, h1_enc, h2_enc = enc_summaries
            s0 = self.sum0(h0_enc)
            s1 = self.sum1(h1_enc)
            s2 = self.sum2(h2_enc)

        x = self.in_conv(x_t) + z_c1 + s0

        h1 = x
        # RESTORED: blk(h1, cond) — no z_spatial argument
        for blk in self.down1:
            h1 = blk(h1, cond)

        d1 = self.downsample1(h1) + z_c2 + s1

        h2 = d1
        for blk in self.down2:
            h2 = blk(h2, cond)

        d2 = self.downsample2(h2) + z_c4 + s2

        m = d2
        for blk in self.mid:
            m = blk(m, cond)

        # RESTORED: + z_c2 and + z_c1 (no h1/h2 UNet skips — that stays removed)
        u2 = self.upsample2(m) + z_c2
        for blk in self.up2:
            u2 = blk(u2, cond)

        u1 = self.upsample1(u2) + z_c1
        for blk in self.up1:
            u1 = blk(u1, cond)

        return self.out_conv(u1)



# =========================================================
# Full joint model
# =========================================================
class VAEDiffusion(nn.Module):
    # passes summeries
    def __init__(self, latent_ch=256, enc_base=64, diff_base=64, time_dim=128):
        super().__init__()
        self.encoder = SeqEncoder1D(in_ch=1, base=enc_base, latent_ch=latent_ch)
        self.diffusion = DiffusionUNet1D_SeqZ(
            in_ch=1,
            base=diff_base,
            time_dim=time_dim,
            latent_ch=latent_ch,
            n_res=N_RES
        )

    def encode(self, x0, return_feats=False):
        if return_feats:
            mu, logvar, feats = self.encoder(x0, return_feats=True)
            z = reparam(mu, logvar)
            return z, mu, logvar, feats
        else:
            mu, logvar = self.encoder(x0, return_feats=False)
            z = reparam(mu, logvar)
            return z, mu, logvar

    def encode_mu(self, x0, return_feats=False):
        if return_feats:
            mu, logvar, feats = self.encoder(x0, return_feats=True)
            return mu, logvar, feats
        else:
            mu, logvar = self.encoder(x0, return_feats=False)
            return mu, logvar

    def pred(self, x_t, t, z_seq, enc_summaries=None):
        return self.diffusion(x_t, t, z_seq, enc_summaries=enc_summaries)




# =========================================================
# Sampling
# =========================================================
@torch.no_grad()
def sample_ddim_cond(model: VAEDiffusion,
                     diff: DiffusionConfig,
                     z_seq: torch.Tensor,
                     shape,
                     device,
                     ddim_steps: int = 50,
                     eta: float = 0.0):
    B = shape[0]
    x = torch.randn(shape, device=device)

    ts = make_ddim_timesteps(diff.T, ddim_steps, device=device)
    ts_rev = ts.flip(0)

    for i in range(len(ts_rev)):
        t = int(ts_rev[i])
        t_prev = int(ts_rev[i + 1]) if i + 1 < len(ts_rev) else 0

        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        t_prev_batch = torch.full((B,), t_prev, device=device, dtype=torch.long)

        abar_t = extract(diff.alphas_bar, t_batch, x.shape)
        abar_prev = extract(diff.alphas_bar, t_prev_batch, x.shape)

        pred = model.pred(x, t_batch, z_seq)

        if PRED_TARGET == "eps":
            eps = pred
            x0_hat = x0_from_eps(diff, x, t_batch, eps)
        elif PRED_TARGET == "x0":
            x0_hat = pred
            eps = eps_from_x0(diff, x, t_batch, x0_hat)
        else:
            raise ValueError(f"Unknown PRED_TARGET={PRED_TARGET}")

        if eta == 0.0:
            sigma_t = torch.zeros((), device=device)
        else:
            sigma_t = eta * torch.sqrt((1 - abar_prev) / (1 - abar_t) * (1 - abar_t / abar_prev))

        noise = sigma_t * torch.randn_like(x)
        dir_xt = torch.sqrt(torch.clamp(1.0 - abar_prev - sigma_t ** 2, min=0.0)) * eps
        x = torch.sqrt(abar_prev) * x0_hat + dir_xt + noise

    return x


# =========================================================
# Diagnostics
# =========================================================
@torch.no_grad()
def inspect_loader_stats(model, diff, loader, device, name="set", max_batches=10):
    model.eval()

    x_means, x_stds = [], []
    raw_means, raw_stds = [], []
    m_vals, s_vals = [], []

    mu_means, mu_stds, mu_norms = [], [], []
    logvar_means, logvar_stds = [], []
    z_means, z_stds, z_norms = [], [], []

    pred_means, pred_stds = [], []
    x0hat_means, x0hat_stds = [], []

    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break

        x0, m, s = batch
        x0 = x0.to(device).unsqueeze(1)
        m = m.to(device)
        s = s.to(device)

        x_flat = x0.squeeze(1)
        x_means.append(x_flat.mean().item())
        x_stds.append(x_flat.std().item())

        raw = x_flat * s.unsqueeze(1) + m.unsqueeze(1)
        raw_means.append(raw.mean().item())
        raw_stds.append(raw.std().item())

        m_vals.append(m.mean().item())
        s_vals.append(s.mean().item())

        mu, logvar = model.encode_mu(x0)
        z = reparam(mu, logvar)

        mu_means.append(mu.mean().item())
        mu_stds.append(mu.std().item())
        mu_norms.append(mu.flatten(1).norm(dim=1).mean().item())

        logvar_means.append(logvar.mean().item())
        logvar_stds.append(logvar.std().item())

        z_means.append(z.mean().item())
        z_stds.append(z.std().item())
        z_norms.append(z.flatten(1).norm(dim=1).mean().item())

        B = x0.size(0)
        t = torch.randint(0, diff.T, (B,), device=device)
        noise = torch.randn_like(x0)
        x_t = q_sample(diff, x0, t, noise)

        z_used = z
        pred = model.pred(x_t, t, z_used)
        if PRED_TARGET == "eps":
            x0_hat = x0_from_eps(diff, x_t, t, pred)
        else:
            x0_hat = pred

        pred_means.append(pred.mean().item())
        pred_stds.append(pred.std().item())
        x0hat_means.append(x0_hat.mean().item())
        x0hat_stds.append(x0_hat.std().item())

    print(f"\n===== {name} stats =====")
    print(f"stored m mean      : {np.mean(m_vals):.6f}")
    print(f"stored s mean      : {np.mean(s_vals):.6f}")
    print(f"x_norm mean/std    : {np.mean(x_means):.6f} / {np.mean(x_stds):.6f}")
    print(f"x_raw mean/std     : {np.mean(raw_means):.6f} / {np.mean(raw_stds):.6f}")

    print(f"mu mean/std        : {np.mean(mu_means):.6f} / {np.mean(mu_stds):.6f}")
    print(f"mu norm            : {np.mean(mu_norms):.6f}")
    print(f"logvar mean/std    : {np.mean(logvar_means):.6f} / {np.mean(logvar_stds):.6f}")

    print(f"z mean/std         : {np.mean(z_means):.6f} / {np.mean(z_stds):.6f}")
    print(f"z norm             : {np.mean(z_norms):.6f}")

    print(f"pred mean/std      : {np.mean(pred_means):.6f} / {np.mean(pred_stds):.6f}")
    print(f"x0hat mean/std     : {np.mean(x0hat_means):.6f} / {np.mean(x0hat_stds):.6f}")


@torch.no_grad()
def inspect_mu_vs_z(model, loader, device, name="set", max_batches=10):
    model.eval()

    mu_norms, z_used_norms, z_sampled_norms = [], [], []
    diff_sampled_norms = []

    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break

        x0, _, _ = batch
        x0 = x0.to(device).unsqueeze(1)

        mu, logvar = model.encode_mu(x0)
        # what model ACTUALLY uses
        z_used = mu  

        # what VAE WOULD sample
        z_sampled = reparam(mu, logvar)

        mu_norms.append(mu.flatten(1).norm(dim=1).mean().item())
        z_used_norms.append(z_used.flatten(1).norm(dim=1).mean().item())
        z_sampled_norms.append(z_sampled.flatten(1).norm(dim=1).mean().item())
        diff_sampled_norms.append((z_sampled - mu).flatten(1).norm(dim=1).mean().item())

    print(f"\n===== {name} mu vs z =====")
    print("||mu||:", np.mean(mu_norms))
    print("||z_used||:", np.mean(z_used_norms))        # should match mu now
    print("||z_sampled||:", np.mean(z_sampled_norms))  # hypothetical
    print("||z_sampled - mu||:", np.mean(diff_sampled_norms))


# =========================================================
# Plot helpers
# =========================================================
def plot_recon_pair(x_true, x_hat, fs, title, outpath: Path):
    t = np.arange(len(x_true)) / fs
    fig = plt.figure(figsize=(10, 3))
    plt.plot(t, x_true, label="input")
    plt.plot(t, x_hat, "--", label="reconstruction")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (raw)")
    plt.title(title)
    plt.legend()
    savefig(fig, outpath)


def plot_diff_curves(history: dict, outpath: Path):
    fig = plt.figure(figsize=(11, 5))
    for key in ["train_loss", "val_loss", "train_diff", "val_diff", "train_recon", "val_recon",
                "train_spec", "val_spec", "train_deriv", "val_deriv"]:
        if key in history:
            plt.plot(history[key], label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training curves")
    plt.legend()
    savefig(fig, outpath)


def plot_kl_curves(history: dict, outpath: Path):
    fig = plt.figure(figsize=(10, 4))
    if "train_kl_raw" in history:
        plt.plot(history["train_kl_raw"], label="train_kl_raw")
    if "val_kl_raw" in history:
        plt.plot(history["val_kl_raw"], label="val_kl_raw")
    if "train_kl_fb" in history:
        plt.plot(history["train_kl_fb"], label="train_kl_fb")
    if "val_kl_fb" in history:
        plt.plot(history["val_kl_fb"], label="val_kl_fb")
    plt.xlabel("Epoch")
    plt.ylabel("KL")
    plt.title("KL curves")
    plt.legend()
    savefig(fig, outpath)


# =========================================================
# Train / eval
# =========================================================
def compute_losses(model, diff, x0, beta_kl, vamp_prior=None):
    if SAMPLE_Z_IN_TRAIN:
        z, mu, logvar, feats = model.encode(x0, return_feats=True)
    else:
        mu, logvar, feats = model.encode_mu(x0, return_feats=True)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        z = mu + Z_PERTURB_SCALE * std * eps

    # logvar = torch.clamp(logvar, min=-2.0, max=4.0)
    logvar = torch.clamp(logvar, min=-4.0, max=2.0)

    if model.training:
        z = z + LATENT_NOISE_SCALE * torch.randn_like(z)

    # downsample z, mu, logvar to (B, 256, 32) for VampPrior KL
    # decoder still receives full (B, 256, 768) z below
    z_ds      = F.adaptive_avg_pool1d(z,      32)
    mu_ds     = F.adaptive_avg_pool1d(mu,     32)
    logvar_ds = F.adaptive_avg_pool1d(logvar, 32)


        # ── KL divergence ──────────────────────────────────────────────────────────
    if vamp_prior is not None:
        # log q(z|x) — computed in downsampled space
        log_q = -0.5 * (
            logvar_ds
            + (z_ds - mu_ds).pow(2) / (torch.exp(logvar_ds) + 1e-8)
            + math.log(2 * math.pi)
        ).sum(dim=(1, 2))

        # log p_vamp(z) — prior defined in full space, downsampled for KL
        mu_k, logvar_k   = vamp_prior.get_prior_params()      # (K, 256, 768)
        mu_k_ds          = F.adaptive_avg_pool1d(mu_k,     32) # (K, 256, 32)
        logvar_k_ds      = F.adaptive_avg_pool1d(logvar_k, 32) # (K, 256, 32)
        log_p = vamp_prior.log_prob_ds(z_ds, mu_k_ds, logvar_k_ds, chunk_size=50)

        log_q = torch.clamp(log_q, min=-1e6, max=1e6)
        log_p = torch.clamp(log_p, min=-1e6, max=0.0)

        kl_raw = (log_q - log_p).mean()
        kl_fb  = torch.clamp(kl_raw, min=0.0, max=500000.0)
    else:
        kl_raw, kl_fb = kl_free_bits(mu, logvar, FREE_BITS_NATS)

    # ── diffusion loss ─────────────────────────────────────────────────────────
    B = x0.size(0)
    t = torch.randint(0, diff.T, (B,), device=x0.device, dtype=torch.long)
    noise = torch.randn_like(x0)
    x_t = q_sample(diff, x0, t, noise)

    pred = model.pred(x_t, t, z, enc_summaries=feats)

    if PRED_TARGET == "eps":
        diff_loss = F.mse_loss(pred, noise)
        x0_hat = x0_from_eps(diff, x_t, t, pred)
    elif PRED_TARGET == "x0":
        diff_loss = F.mse_loss(pred, x0)
        x0_hat = pred
    else:
        raise ValueError(f"Unknown PRED_TARGET={PRED_TARGET}")


    # ── auxiliary losses ───────────────────────────────────────────────────────
    x0_1d     = x0.squeeze(1)
    x0_hat_1d = x0_hat.squeeze(1)

    recon_loss = F.smooth_l1_loss(x0_hat_1d, x0_1d)
    spec_loss  = spectral_loss(x0_hat_1d, x0_1d)
    deriv_l    = derivative_loss(x0_hat_1d, x0_1d)

    true_std = x0_1d.std(dim=-1, unbiased=False)
    pred_std = x0_hat_1d.std(dim=-1, unbiased=False)
    amp_loss = F.mse_loss(pred_std, true_std)

    true_ptp = x0_1d.max(dim=-1).values - x0_1d.min(dim=-1).values
    pred_ptp = x0_hat_1d.max(dim=-1).values - x0_hat_1d.min(dim=-1).values
    ptp_loss = F.mse_loss(pred_ptp, true_ptp)

    # ── total loss ─────────────────────────────────────────────────────────────
    loss = (
        LAMBDA_DIFF  * diff_loss
        + LAMBDA_RECON * recon_loss
        + LAMBDA_SPEC  * spec_loss
        + LAMBDA_DERIV * deriv_l
        + LAMBDA_AMP   * amp_loss
        + LAMBDA_PTP   * ptp_loss
        + beta_kl      * kl_fb
    )

    return {
        "loss":    loss,
        "diff":    diff_loss,
        "recon":   recon_loss,
        "spec":    spec_loss,
        "deriv":   deriv_l,
        "amp":     amp_loss,
        "ptp":     ptp_loss,
        "kl_raw":  kl_raw,
        "kl_fb":   kl_fb,
        "x0_hat":  x0_hat,
        "mu":      mu,
        "logvar":  logvar,
        "z":       z,
    }


def train_one_epoch(model, diff, loader, opt, device,
                    beta_kl=0.0, vamp_prior=None, print_every=None):
    model.train()



    total = {
        "loss": 0.0,
        "diff": 0.0,
        "recon": 0.0,
        "spec": 0.0,
        "deriv": 0.0,
        "kl_raw": 0.0,
        "kl_fb": 0.0,
        "amp": 0.0,
        "ptp": 0.0,
    }
    n = 0

    for bi, batch in enumerate(loader):
        if bi % 10 == 0: 
            torch.cuda.empty_cache()   # ← add this line
        x0, _, _ = batch
        x0 = x0.to(device).unsqueeze(1)

        out = compute_losses(model, diff, x0, beta_kl, vamp_prior=vamp_prior)

        bad = False
        for k in ["loss", "diff", "recon", "spec", "deriv", "amp", "ptp", "kl_raw", "kl_fb"]:
            if torch.isnan(out[k]).any() or torch.isinf(out[k]).any():
                bad = True
                break

        if bad:
            print(f"[WARN] NaN/Inf detected in batch {bi}, skipping batch")
            continue

       
        
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()

        if print_every is not None and (bi % print_every == 0):
            print(
                f"[train b{bi:05d}] "
                f"loss={out['loss'].item():.6f} | "
                f"(diff={out['diff'].item():.4f}, recon={out['recon'].item():.4f}) | "
                f"(spec={out['spec'].item():.4f}, deriv={out['deriv'].item():.4f}) | "
                f"(amp={out['amp'].item():.4f}, ptp={out['ptp'].item():.4f}) | "
                f"kl={out['kl_fb'].item():.2f}"
            )
        for k in total:
            total[k] += out[k].item()
        n += 1



    return {k: v / max(1, n) for k, v in total.items()}


@torch.no_grad()
def eval_one_epoch(model, diff, loader, device, beta_kl=0.0, vamp_prior=None, print_every=None):
    model.eval()

    total = {
        "loss": 0.0,
        "diff": 0.0,
        "recon": 0.0,
        "spec": 0.0,
        "deriv": 0.0,
        "kl_raw": 0.0,
        "kl_fb": 0.0,
        "amp": 0.0,
        "ptp": 0.0
    }
    n = 0

    for bi, batch in enumerate(loader):
        x0, _, _ = batch
        x0 = x0.to(device).unsqueeze(1)

        out = compute_losses(model, diff, x0, beta_kl, vamp_prior=vamp_prior)

        bad = False
        for k in ["loss", "diff", "recon", "spec", "deriv", "amp", "ptp", "kl_raw", "kl_fb"]:
            if torch.isnan(out[k]).any() or torch.isinf(out[k]).any():
                bad = True
                break

        if bad:
            print(f"[WARN] NaN/Inf detected in batch {bi}, skipping batch")
            continue

        if print_every is not None and (bi % print_every == 0):
            print(
                f"[train b{bi:05d}] "
                f"loss={out['loss'].item():.6f} | "
                f"(diff={out['diff'].item():.4f}, recon={out['recon'].item():.4f}) | "
                f"(spec={out['spec'].item():.4f}, deriv={out['deriv'].item():.4f}) | "
                f"(amp={out['amp'].item():.4f}, ptp={out['ptp'].item():.4f}) | "
                f"kl={out['kl_fb'].item():.2f}"
            )

        for k in total:
            total[k] += out[k].item()
        n += 1

    return {k: v / max(1, n) for k, v in total.items()}


@torch.no_grad()
def run_validation_recons(model, diff, val_set, device, out_dir: Path, epoch: int, n_show=3, fs=300):
    model.eval()
    idxs = list(range(min(n_show, len(val_set))))

    for i, idx in enumerate(idxs):
        x0, m, s = val_set[idx]
        x0 = x0.unsqueeze(0).unsqueeze(1).to(device)

        if USE_MU_FOR_RECON:
            mu, logvar = model.encode_mu(x0)
            z_seq = mu
        else:
            z_seq, _, _ = model.encode(x0)

        xhat = sample_ddim_cond(
            model, diff, z_seq, x0.shape, device,
            ddim_steps=DDIM_STEPS, eta=DDIM_ETA
        )

        x_true = x0.squeeze().cpu().numpy()
        x_pred = xhat.squeeze().cpu().numpy()

        x_true_dn = denorm_window(x_true, float(m), float(s))
        x_pred_dn = denorm_window(x_pred, float(m), float(s))

        plot_recon_pair(
            x_true_dn,
            x_pred_dn,
            fs=fs,
            title=f"Val example {i} — epoch {epoch}",
            outpath=out_dir / "val_recons" / f"epoch_{epoch:03d}" / f"val_recon_{i:02d}.png"
        )


@torch.no_grad()
def eval_recon_metrics(model, diff, loader, device, max_batches=None):
    model.eval()

    maes, rmses, corrs = [], [], []

    for b, batch in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break

        x0, m, s = batch
        x0 = x0.to(device).unsqueeze(1)
        m = m.to(device).view(-1, 1)
        s = s.to(device).view(-1, 1)

        if USE_MU_FOR_RECON:
            mu, logvar = model.encode_mu(x0)
            z_seq = mu
        else:
            z_seq, _, _ = model.encode(x0)

        x_recon = sample_ddim_cond(
            model, diff, z_seq, x0.shape, device,
            ddim_steps=DDIM_STEPS, eta=DDIM_ETA
        )

        x_dn  = x0.squeeze(1) * s + m
        xr_dn = x_recon.squeeze(1) * s + m

        x_np  = x_dn.detach().cpu().numpy()
        xr_np = xr_dn.detach().cpu().numpy()

        for i in range(x_np.shape[0]):
            a = x_np[i]
            p = xr_np[i]
            mae = np.mean(np.abs(a - p))
            rmse = np.sqrt(np.mean((a - p)**2))
            r = pearsonr(a, p)[0] if np.std(a) > 1e-8 and np.std(p) > 1e-8 else 0.0
            maes.append(mae)
            rmses.append(rmse)
            corrs.append(r)

    return {
        "MAE_mean": float(np.mean(maes)),
        "MAE_std": float(np.std(maes)),
        "RMSE_mean": float(np.mean(rmses)),
        "RMSE_std": float(np.std(rmses)),
        "Corr_mean": float(np.mean(corrs)),
        "Corr_std": float(np.std(corrs)),
    }


@torch.no_grad()
def eval_recon_metrics_extended(model, diff, loader, device, fs=300, max_batches=None):
    model.eval()

    maes, rmses, corrs, signed_errs = [], [], [], []
    ptp_errs = []
    peak_count_diffs = []
    peak_timing_errs_ms = []
    hr_abs_errs = []
    ibi_maes_ms = []

    for b, batch in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break

        x0, m, s = batch
        x0 = x0.to(device).unsqueeze(1)
        m = m.to(device).view(-1, 1)
        s = s.to(device).view(-1, 1)

        # reconstruction path
        if USE_MU_FOR_RECON:
            mu, logvar = model.encode_mu(x0)
            z_seq = mu
        else:
            z_seq, _, _ = model.encode(x0)

        x_recon = sample_ddim_cond(
            model, diff, z_seq, x0.shape, device,
            ddim_steps=DDIM_STEPS, eta=DDIM_ETA
        )

        # denormalize
        x_dn = x0.squeeze(1) * s + m
        xr_dn = x_recon.squeeze(1) * s + m

        x_np = x_dn.detach().cpu().numpy()
        xr_np = xr_dn.detach().cpu().numpy()

        for i in range(x_np.shape[0]):
            a = x_np[i]
            p = xr_np[i]

            mae = np.mean(np.abs(a - p))
            rmse = np.sqrt(np.mean((a - p) ** 2))
            signed = np.mean(p - a)
            corr = pearsonr(a, p)[0] if np.std(a) > 1e-8 and np.std(p) > 1e-8 else 0.0

            maes.append(mae)
            rmses.append(rmse)
            signed_errs.append(signed)
            corrs.append(corr)

            # peak-to-peak amplitude
            a_ptp = np.max(a) - np.min(a)
            p_ptp = np.max(p) - np.min(p)
            ptp_errs.append(abs(p_ptp - a_ptp))

            # bandpass for peak-based metrics
            a_t = torch.tensor(a, dtype=torch.float32).unsqueeze(0)
            p_t = torch.tensor(p, dtype=torch.float32).unsqueeze(0)

            a_bp = bandpass_fft(a_t, fs, 0.7, 3.0)[0].cpu().numpy()
            p_bp = bandpass_fft(p_t, fs, 0.7, 3.0)[0].cpu().numpy()

            true_peaks, _ = find_peaks(
                a_bp,
                distance=int(0.35 * fs),
                prominence=0.1 * np.std(a_bp),
                height=np.percentile(a_bp, 60)
            )
            pred_peaks, _ = find_peaks(
                p_bp,
                distance=int(0.35 * fs),
                prominence=0.1 * np.std(p_bp),
                height=np.percentile(p_bp, 60)
            )

            peak_count_diffs.append(abs(len(true_peaks) - len(pred_peaks)))

            # peak timing
            if len(true_peaks) > 0 and len(pred_peaks) > 0:
                nmatch = min(len(true_peaks), len(pred_peaks))
                tp = true_peaks[:nmatch]
                pp = pred_peaks[:nmatch]
                peak_timing_err_ms = np.mean(np.abs(tp - pp) / fs) * 1000.0
                peak_timing_errs_ms.append(peak_timing_err_ms)
            else:
                peak_timing_errs_ms.append(np.nan)

            # HR and IBI
            if len(true_peaks) >= 2 and len(pred_peaks) >= 2:
                true_ibi = np.diff(true_peaks) / fs
                pred_ibi = np.diff(pred_peaks) / fs

                true_hr = 60.0 / np.mean(true_ibi)
                pred_hr = 60.0 / np.mean(pred_ibi)

                hr_abs_errs.append(abs(true_hr - pred_hr))

                min_len = min(len(true_ibi), len(pred_ibi))
                ibi_mae = np.mean(np.abs(true_ibi[:min_len] - pred_ibi[:min_len])) * 1000.0
                ibi_maes_ms.append(ibi_mae)
            else:
                hr_abs_errs.append(np.nan)
                ibi_maes_ms.append(np.nan)

    def nanmean_safe(x):
        x = np.asarray(x, dtype=np.float64)
        return float(np.nanmean(x)) if np.any(~np.isnan(x)) else float("nan")

    def nanstd_safe(x):
        x = np.asarray(x, dtype=np.float64)
        return float(np.nanstd(x)) if np.any(~np.isnan(x)) else float("nan")

    return {
        "MAE_mean": float(np.mean(maes)),
        "MAE_std": float(np.std(maes)),
        "RMSE_mean": float(np.mean(rmses)),
        "RMSE_std": float(np.std(rmses)),
        "Corr_mean": float(np.mean(corrs)),
        "Corr_std": float(np.std(corrs)),
        "SignedErr_mean": float(np.mean(signed_errs)),
        "SignedErr_std": float(np.std(signed_errs)),
        "PTPError_mean": float(np.mean(ptp_errs)),
        "PTPError_std": float(np.std(ptp_errs)),
        "PeakCountDiff_mean": float(np.mean(peak_count_diffs)),
        "PeakCountDiff_std": float(np.std(peak_count_diffs)),
        "PeakTimingErr_ms_mean": nanmean_safe(peak_timing_errs_ms),
        "PeakTimingErr_ms_std": nanstd_safe(peak_timing_errs_ms),
        "HRAbsErr_bpm_mean": nanmean_safe(hr_abs_errs),
        "HRAbsErr_bpm_std": nanstd_safe(hr_abs_errs),
        "IBI_MAE_ms_mean": nanmean_safe(ibi_maes_ms),
        "IBI_MAE_ms_std": nanstd_safe(ibi_maes_ms),
    }


@torch.no_grad()
def test_z_sensitivity(model, diff, val_set, device, n=3, ddim_steps=20):
    """
    If z is being used, output(mu) and output(random_z) should differ substantially.
    If difference ~ 0, decoder is ignoring z entirely.
    """
    model.eval()
    print("\n===== Z sensitivity test =====")
    for i in range(min(n, len(val_set))):
        x0, _, _ = val_set[i]
        x0 = x0.unsqueeze(0).unsqueeze(1).to(device)
        mu, _ = model.encode_mu(x0)

        x_from_mu   = sample_ddim_cond(model, diff, mu,
                                        x0.shape, device, ddim_steps=ddim_steps)
        z_rand       = torch.randn_like(mu)
        x_from_rand  = sample_ddim_cond(model, diff, z_rand,
                                        x0.shape, device, ddim_steps=ddim_steps)

        diff_val = (x_from_mu - x_from_rand).abs().mean().item()
        mu_range = (x_from_mu.max() - x_from_mu.min()).item()
        print(f"  Sample {i}: |out(mu)-out(z_rand)| = {diff_val:.6f}  |  "
              f"out(mu) range = {mu_range:.6f}  |  "
              f"ratio = {diff_val/max(mu_range,1e-8):.3f}")
    print("  (ratio > 0.3 means z is being used meaningfully)")
# =========================================================
# Main
# =========================================================
def main():
    seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("====================================")
    print(f"Device being used: {device}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print("====================================")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Build raw train set to get global stats
    # ----------------------------
    train_set_raw = CapnoBaseVAE(TRAIN_DIR, window_size=WINDOW_SIZE, step_size=STEP_SIZE,
                                 fs=FS, norm_mode="none", add_noise=False)
    GLOBAL_MEAN, GLOBAL_STD = compute_global_norm_from_raw_dataset(train_set_raw)
    print(f"[GLOBAL NORM] mean={GLOBAL_MEAN:.6f}, std={GLOBAL_STD:.6f}")

    # ----------------------------
    # Rebuild with global norm
    # ----------------------------
    train_set = CapnoBaseVAE(TRAIN_DIR, window_size=WINDOW_SIZE, step_size=STEP_SIZE,
                             fs=FS, norm_mode="global",
                             global_mean=GLOBAL_MEAN, global_std=GLOBAL_STD)
    val_set   = CapnoBaseVAE(VAL_DIR, window_size=WINDOW_SIZE, step_size=STEP_SIZE,
                             fs=FS, norm_mode="global",
                             global_mean=GLOBAL_MEAN, global_std=GLOBAL_STD)
    test_set  = CapnoBaseVAE(TEST_DIR, window_size=WINDOW_SIZE, step_size=STEP_SIZE,
                             fs=FS, norm_mode="global",
                             global_mean=GLOBAL_MEAN, global_std=GLOBAL_STD)

    print(f"[dataset sizes] train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

    # Limit dataset to subset for faster iteration

    if len(train_set) > MAX_TRAIN_SAMPLES:
        indices = list(range(0, len(train_set), len(train_set) // MAX_TRAIN_SAMPLES))[:MAX_TRAIN_SAMPLES]
        train_set = torch.utils.data.Subset(train_set, indices)

    print("[check global-mode m,s consistency]")
    for i in [0, 1, 2, 10, 25]:
        if i < len(train_set):
            x, m, s = train_set[i]
            print(f"idx={i:03d}  m={float(m):.6f}  s={float(s):.6f}  x_mean={x.mean().item():+.4f}  x_std={x.std().item():.4f}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)



    # ----------------------------
    # Model + diffusion config
    # ----------------------------
    model = VAEDiffusion(
        latent_ch=LATENT_CH,
        enc_base=BASE_CH,
        diff_base=DIFF_BASE,
        time_dim=TIME_DIM
    ).to(device)

    diff = DiffusionConfig(
        T=DIFF_T,
        beta_start=BETA_START,
        beta_end=BETA_END,
        schedule=SCHEDULE
    ).to(device)

    print("Model device check:", next(model.parameters()).device)

    # ── VampPrior ──────────────────────────────────────────────────────────────
    VAMP_K = 100 #was 50
    vamp_prior = VampPrior(
        K=VAMP_K,
        window_size=WINDOW_SIZE,
        encoder=model.encoder,
    ).to(device)
    print(f"VampPrior: K={VAMP_K} pseudo-inputs, shape (K, 1, {WINDOW_SIZE})")


    # Load precomputed HR and amplitude features
    all_hrs  = np.load(r"\analysis\all_hrs.npy")
    all_ptps = np.load(r"\analysis\all_ptps.npy")

    # Collect all training signals in order (must match all_hrs/all_ptps order)
    all_signals = []
    for x, m, s in DataLoader(train_set, batch_size=64, shuffle=False):
        for i in range(x.shape[0]):
            sig_raw = x[i].numpy() * float(s[i]) + float(m[i])
            all_signals.append(sig_raw)
    all_signals = np.array(all_signals)

    # Stratified selection by HR bins and amplitude bins
    HR_BINS  = [59, 70, 80, 90, 100, 115, 150]   # 6 HR bins
    AMP_BINS = [2.5, 7.5, 11.0, 15.0, 21.0]      # 4 amplitude bins
    n_hr_bins  = len(HR_BINS) - 1   # 6
    n_amp_bins = len(AMP_BINS) - 1  # 4
    k_per_bin  = max(1, VAMP_K // (n_hr_bins * n_amp_bins))  # ~2 per bin

    selected = []
    for i in range(n_hr_bins):
        for j in range(n_amp_bins):
            hr_mask  = (all_hrs  >= HR_BINS[i])  & (all_hrs  < HR_BINS[i+1])
            amp_mask = (all_ptps >= AMP_BINS[j]) & (all_ptps < AMP_BINS[j+1])
            mask     = hr_mask & amp_mask
            indices  = np.where(mask)[0]
            if len(indices) == 0:
                print(f"  [WARN] No samples in HR=[{HR_BINS[i]},{HR_BINS[i+1]}] AMP=[{AMP_BINS[j]},{AMP_BINS[j+1]}]")
                continue
            chosen = np.random.choice(indices, size=min(k_per_bin, len(indices)), replace=False)
            selected.extend(chosen.tolist())

    # fill remainder randomly if needed
    if len(selected) < VAMP_K:
        remaining = list(set(range(len(all_signals))) - set(selected))
        extra = np.random.choice(remaining, size=VAMP_K - len(selected), replace=False)
        selected.extend(extra.tolist())

    selected = selected[:VAMP_K]
    pseudo   = torch.tensor(
        (all_signals[selected] - GLOBAL_MEAN) / GLOBAL_STD,  # normalize back
        dtype=torch.float32
    ).unsqueeze(1).to(device)  # (K, 1, 3072)

    with torch.no_grad():
        vamp_prior.pseudo_inputs.copy_(pseudo)

    print(f"[INIT] Stratified pseudo-inputs: {VAMP_K} windows")
    print(f"       HR  range: {all_hrs[selected].min():.1f} – {all_hrs[selected].max():.1f} bpm")
    print(f"       PTP range: {all_ptps[selected].min():.1f} – {all_ptps[selected].max():.1f}")     

    # Load best pretrained checkpoint — Corr=0.9996
    FINETUNE_CKPT = Path(r"\best_vae_diffusion.pt")
    if FINETUNE_CKPT.exists():
        ckpt = torch.load(FINETUNE_CKPT, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"[FINETUNE] Loaded checkpoint from {FINETUNE_CKPT}")
    else:
        print("[WARN] No checkpoint found — training from scratch")


    # ----------------------------
    # Freeze encoder at start, then unfreeze
    # ----------------------------
    for p in model.encoder.parameters():
        p.requires_grad = False

    # Optimizer — only include encoder if not frozen
    opt = torch.optim.AdamW([
        {"params": model.diffusion.parameters(), "lr": LR * 0.1},
        {"params": vamp_prior.pseudo_inputs,     "lr": LR * 10},
    ], weight_decay=WEIGHT_DECAY)



    history = {
        "train_loss": [], "train_diff": [], "train_recon": [], "train_spec": [], "train_deriv": [],
        "train_kl_raw": [], "train_kl_fb": [],
        "val_loss": [],   "val_diff": [],   "val_recon": [],   "val_spec": [],   "val_deriv": [],
        "val_kl_raw": [], "val_kl_fb": [], "train_amp" : [], "train_ptp" : [], "val_amp": [], "val_ptp": [],
        "beta": []
    }

    best_val = float("inf")
    best_path = OUT_DIR / "best_vae_diffusion.pt"

    # ----------------------------
    # Train loop
    # ----------------------------
    for ep in range(1, EPOCHS + 1):
        if ep == ENC_FREEZE_EPOCHS + 1:
            print(f"[INFO] Unfreezing encoder at epoch {ep}")
            for p in model.encoder.parameters():
                p.requires_grad = True
            # rebuild optimizer to include encoder
            opt = torch.optim.AdamW([
                {"params": model.diffusion.parameters(), "lr": LR * 0.1},
                {"params": model.encoder.parameters(),   "lr": LR * 0.025},
                {"params": vamp_prior.pseudo_inputs,     "lr": LR * 10},
            ], weight_decay=WEIGHT_DECAY)
            print(f"[INFO] Optimizer rebuilt to include encoder at lr={LR * 0.025:.2e}")

  

        if BETA_MODE == "zero_then_ramp":
            beta = beta_zero_then_ramp(
                ep - 1,
                warmup_epochs=WARMUP_EPOCHS,
                ramp_epochs=RAMP_EPOCHS,
                beta_max=BETA_KL_MAX
            )
        elif BETA_MODE == "warmup_then_hold":
            beta = beta_linear_warmup_then_hold(
                ep - 1,
                beta_max=BETA_KL_MAX,
                warmup_epochs=WARMUP_EPOCHS,
                start=0.0
            )
        elif BETA_MODE == "floor_then_ramp":
            beta = beta_floor_then_ramp(
            ep - 1,
            freeze_epochs=ENC_FREEZE_EPOCHS,
            warmup_epochs=WARMUP_EPOCHS,
            ramp_epochs=RAMP_EPOCHS,
            beta_min=BETA_KL_MIN,
            beta_max=BETA_KL_MAX
        )


        else:
            raise ValueError(f"Unknown BETA_MODE: {BETA_MODE}")
        
        if ep <= ENC_FREEZE_EPOCHS:
            beta = 0.0

        tr = train_one_epoch(model, diff, train_loader, opt, device,
                     beta_kl=beta, vamp_prior=vamp_prior, print_every=None)
        if ep == 1 or ep % 5 == 0:
            va = eval_one_epoch(model, diff, val_loader, device,
                    beta_kl=beta, vamp_prior=vamp_prior, print_every=None)
        else:
                va = {
                "loss": float("nan"),
                "diff": float("nan"),
                "recon": float("nan"),
                "spec": float("nan"),
                "deriv": float("nan"),
                "kl_raw": float("nan"),
                "kl_fb": float("nan"),
                "amp": float("nan"),
                "ptp": float("nan"),
            }

        history["train_loss"].append(tr["loss"])
        history["train_diff"].append(tr["diff"])
        history["train_recon"].append(tr["recon"])
        history["train_spec"].append(tr["spec"])
        history["train_deriv"].append(tr["deriv"])
        history["train_kl_raw"].append(tr["kl_raw"])
        history["train_kl_fb"].append(tr["kl_fb"])
        history["train_amp"].append(tr["amp"])
        history["train_ptp"].append(tr["ptp"])
   

        history["val_loss"].append(va["loss"])
        history["val_diff"].append(va["diff"])
        history["val_recon"].append(va["recon"])
        history["val_spec"].append(va["spec"])
        history["val_deriv"].append(va["deriv"])
        history["val_kl_raw"].append(va["kl_raw"])
        history["val_kl_fb"].append(va["kl_fb"])
        history["val_amp"].append(va["amp"])
        history["val_ptp"].append(va["ptp"])

        history["beta"].append(beta)
        # compute mu_std on a small batch — cheap, no grad needed
        model.eval()
        with torch.no_grad():
            _x, _, _ = next(iter(train_loader))
            _x = _x[:32].to(device).unsqueeze(1)
            _mu, _ = model.encode_mu(_x)
            mu_std_now = _mu.std().item()
        model.train()
        print(
            f"Epoch {ep:03d} | beta={beta:.3e} | mu_std={mu_std_now:.4f} | "
            f"Train: loss={tr['loss']:.4f}, (diff={tr['diff']:.4f}, recon={tr['recon']:.4f}), "
            f"(spec={tr['spec']:.4f}, deriv={tr['deriv']:.4f}), "
            f"(amp={tr['amp']:.4f}, ptp={tr['ptp']:.4f}), kl={tr['kl_fb']:.2f}, | "
            f"Val: loss={va['loss']:.4f}, (diff={va['diff']:.4f}, recon={va['recon']:.4f}), "
            f"(spec={va['spec']:.4f}, deriv={va['deriv']:.4f}), "
            f"(amp={va['amp']:.4f}, ptp={va['ptp']:.4f}), kl={va['kl_fb']:.2f}"
        )
        plot_diff_curves(history, OUT_DIR / "training_curves.png")
        plot_kl_curves(history, OUT_DIR / "kl_curves.png")



        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "latent_ch": LATENT_CH,
                    "enc_base": BASE_CH,
                    "diff_base": DIFF_BASE,
                    "time_dim": TIME_DIM,
                    "global_mean": GLOBAL_MEAN,
                    "global_std": GLOBAL_STD,
                },
                best_path
            )
            # save vamp prior separately
            torch.save(
                {
                    "vamp_state": vamp_prior.state_dict(),
                    "K": VAMP_K,
                },
                OUT_DIR / "best_vamp_prior.pt"
            )
            print(f"[SAVED] VampPrior saved to best_vamp_prior.pt")


    # ----------------------------
    # Save final epoch model AFTER loop ends
    # ----------------------------
    final_path = OUT_DIR / "final_epoch_model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "latent_ch": LATENT_CH,
            "enc_base": BASE_CH,
            "diff_base": DIFF_BASE,
            "time_dim": TIME_DIM,
            "global_mean": GLOBAL_MEAN,
            "global_std": GLOBAL_STD,
        },
        final_path
    )

    torch.save(
        {
            "vamp_state": vamp_prior.state_dict(),
            "K": VAMP_K,
        },
        OUT_DIR / "final_vamp_prior.pt"
    )
    print(f"[SAVED] Final VampPrior saved to final_vamp_prior.pt")

    print(f"[INFO] Final epoch model saved to {final_path}")
    print("\n[CHECK] final_epoch_model.pt encoder stats:")
    fe_ckpt = torch.load(final_path, map_location=device)
    fe_state = fe_ckpt["model_state"]
    mu_w = fe_state["encoder.mu.weight"]
    print(f"  encoder.mu.weight std = {mu_w.std().item():.6f}")

    # Also check mu_std on real data using final model (already in memory)
    model.eval()
    with torch.no_grad():
        diag_batch = next(iter(train_loader))
        x0_diag, _, _ = diag_batch
        x0_diag = x0_diag[:32].to(device).unsqueeze(1)
        mu_diag, _ = model.encode_mu(x0_diag)
        mu_std_final = mu_diag.std().item()
    print(f"[CHECK] final_epoch_model.pt mu_std on real data = {mu_std_final:.4f}")
    if mu_std_final < 0.5:
        print("[WARN]  Encoder COLLAPSED — final_epoch_model.pt not usable for generation")
    elif mu_std_final < 1.0:
        print("[WARN]  Partial collapse — GMM may struggle")
    else:
        print("[OK]    Encoder healthy — ready for GMM prior fitting")

    # ----------------------------
    # Encoder health check — BEST model
    # ----------------------------
    bv_ckpt = torch.load(best_path, map_location=device)
    bv_state = bv_ckpt["model_state"]
    mu_w_best = bv_state["encoder.mu.weight"]
    print(f"\n[CHECK] best_vae_diffusion.pt encoder.mu.weight std = {mu_w_best.std().item():.6f}")

       # ----------------------------
    # Evaluate FINAL epoch model
    # ----------------------------
    print("\n" + "="*60)
    print("EVALUATING: final_epoch_model.pt")
    print("="*60)
    # model is already at final epoch state — no load needed

    test_z_sensitivity(model, diff, val_set, device, n=3)
    inspect_loader_stats(model, diff, train_loader, device, name="TRAIN (final)", max_batches=5)
    inspect_loader_stats(model, diff, val_loader,   device, name="VAL   (final)", max_batches=5)

    test_metrics_final = eval_recon_metrics_extended(
        model, diff, test_loader, device, fs=FS, max_batches=None
    )
    print("\n==== Extended Test Metrics — FINAL epoch model ====")
    for k, v in test_metrics_final.items():
        print(f"  {k}: {v:.6f}")
    pd.DataFrame([test_metrics_final]).to_csv(
        OUT_DIR / "test_recon_metrics_FINAL.csv", index=False
    )

    # ----------------------------
    # Evaluate BEST checkpoint
    # ----------------------------
    print("\n" + "="*60)
    print("EVALUATING: best_vae_diffusion.pt")
    print("="*60)
    model.load_state_dict(bv_ckpt["model_state"])
    print(f"[INFO] Loaded best checkpoint from {best_path}")

    test_z_sensitivity(model, diff, val_set, device, n=3)
    inspect_loader_stats(model, diff, train_loader, device, name="TRAIN (best)", max_batches=5)
    inspect_loader_stats(model, diff, val_loader,   device, name="VAL   (best)", max_batches=5)

    test_metrics_best = eval_recon_metrics_extended(
        model, diff, test_loader, device, fs=FS, max_batches=None
    )
    print("\n==== Extended Test Metrics — BEST checkpoint ====")
    for k, v in test_metrics_best.items():
        print(f"  {k}: {v:.6f}")
    pd.DataFrame([test_metrics_best]).to_csv(
        OUT_DIR / "test_recon_metrics_BEST.csv", index=False
    )

    # ----------------------------
    # Side-by-side summary
    # ----------------------------
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    print(f"{'Metric':<30} {'FINAL':>12} {'BEST':>12}")
    print("-"*56)
    for k in test_metrics_final:
        vf = test_metrics_final[k]
        vb = test_metrics_best[k]
        print(f"  {k:<28} {vf:>12.4f} {vb:>12.4f}")
    print(f"\n  mu_std (real data)           {mu_std_final:>12.4f}        (see best ckpt inspect above)")

    # ----------------------------
    # Save test recon plots for BOTH checkpoints
    # ----------------------------
    # Plots already using best (loaded above) — save best plots
    n_show = 5
    plot_indices = random.sample(range(len(test_set)), n_show)

    for ckpt_tag, ckpt_state in [("best", bv_ckpt), ("final", fe_ckpt)]:
        model.load_state_dict(ckpt_state["model_state"])
        model.eval()

        for plot_idx, ds_idx in enumerate(plot_indices):
            x0, m, s = test_set[ds_idx]
            x0 = x0.unsqueeze(0).unsqueeze(1).to(device)

            if USE_MU_FOR_RECON:
                mu, logvar = model.encode_mu(x0)
                z_seq = mu
            else:
                z_seq, _, _ = model.encode(x0)

            xhat = sample_ddim_cond(
                model, diff, z_seq, x0.shape, device,
                ddim_steps=DDIM_STEPS, eta=DDIM_ETA
            )

            x_true_dn = denorm_window(x0.squeeze().cpu().numpy(), float(m), float(s))
            x_pred_dn = denorm_window(xhat.squeeze().cpu().numpy(), float(m), float(s))

            plot_recon_pair(
                x_true_dn, x_pred_dn, fs=FS,
                title=f"[{ckpt_tag}] Test idx={ds_idx}",
                outpath=OUT_DIR / "test_recons" / f"{ckpt_tag}_recon_{plot_idx:02d}_idx_{ds_idx}.png"
            )

    print("\nDone. Outputs saved to:", OUT_DIR)


if __name__ == "__main__":
    main()