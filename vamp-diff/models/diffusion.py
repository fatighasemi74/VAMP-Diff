"""
Diffusion process: noise schedules, forward diffusion (q_sample),
DDIM timestep construction, and the DDIM conditional sampler.
"""

import math

import torch
import torch.nn.functional as F


# =========================================================
# Noise schedule
# =========================================================

def make_beta_schedule(T: int, beta_start: float = 1e-4,
                       beta_end: float = 0.02,
                       kind: str = "linear") -> torch.Tensor:
    """
    Build a 1-D tensor of betas for T diffusion steps.

    Args:
        T:          Number of diffusion steps.
        beta_start: Starting noise level.
        beta_end:   Ending noise level.
        kind:       "linear" or "cosine".

    Returns:
        betas: Tensor of shape (T,).
    """
    if kind == "linear":
        return torch.linspace(beta_start, beta_end, T)
    if kind == "cosine":
        steps       = torch.arange(T + 1, dtype=torch.float32)
        s           = 0.008
        f           = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
        alphas_bar  = f / f[0]
        betas       = 1 - (alphas_bar[1:] / alphas_bar[:-1])
        return betas.clamp(1e-6, 0.999)
    raise ValueError(f"Unknown schedule kind: {kind!r}")


# =========================================================
# DiffusionConfig
# =========================================================

class DiffusionConfig:
    """
    Pre-computes and caches all diffusion schedule tensors.

    Args:
        T:          Number of diffusion steps.
        beta_start: Starting noise level.
        beta_end:   Ending noise level.
        schedule:   "linear" or "cosine".
    """

    def __init__(self, T: int = 100, beta_start: float = 1e-4,
                 beta_end: float = 0.02, schedule: str = "linear"):
        self.T     = T
        self.betas = make_beta_schedule(T, beta_start, beta_end, kind=schedule)

        self.alphas               = 1.0 - self.betas
        self.alphas_bar           = torch.cumprod(self.alphas, dim=0)
        self.alphas_bar_prev      = torch.cat([torch.ones(1), self.alphas_bar[:-1]], dim=0)

        self.sqrt_alphas_bar          = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - self.alphas_bar)

        self.posterior_var    = (
            self.betas * (1.0 - self.alphas_bar_prev) / (1.0 - self.alphas_bar)
        )
        self.posterior_var[0] = 1e-8

    def to(self, device):
        for k, v in self.__dict__.items():
            if torch.is_tensor(v):
                setattr(self, k, v.to(device))
        return self


# =========================================================
# Helpers
# =========================================================

def extract(a: torch.Tensor, t: torch.Tensor, x_shape) -> torch.Tensor:
    """Gather schedule values at timesteps t and reshape for broadcasting."""
    out = a.gather(0, t).to(t.device)
    return out.view(-1, *([1] * (len(x_shape) - 1)))


def q_sample(diff: DiffusionConfig, x0: torch.Tensor,
             t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Forward diffusion: sample x_t ~ q(x_t | x_0)."""
    sqrt_ab   = extract(diff.sqrt_alphas_bar, t, x0.shape)
    sqrt_1mab = extract(diff.sqrt_one_minus_alphas_bar, t, x0.shape)
    return sqrt_ab * x0 + sqrt_1mab * noise


def x0_from_eps(diff: DiffusionConfig, x_t: torch.Tensor,
                t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """Recover x0 prediction from predicted noise eps."""
    abar_t = extract(diff.alphas_bar, t, x_t.shape)
    return (x_t - torch.sqrt(1.0 - abar_t) * eps) / torch.sqrt(abar_t)


def eps_from_x0(diff: DiffusionConfig, x_t: torch.Tensor,
                t: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    """Recover eps prediction from predicted x0."""
    abar_t = extract(diff.alphas_bar, t, x_t.shape)
    return (x_t - torch.sqrt(abar_t) * x0) / torch.sqrt(1.0 - abar_t)


# =========================================================
# DDIM sampling
# =========================================================

def make_ddim_timesteps(T: int, ddim_steps: int, device) -> torch.Tensor:
    """
    Build a linearly-spaced sub-sequence of T diffusion timesteps for DDIM.

    Args:
        T:          Total diffusion steps.
        ddim_steps: Number of DDIM steps.
        device:     Target device.

    Returns:
        Unique, sorted timestep indices as a LongTensor.
    """
    if ddim_steps <= 1:
        return torch.tensor([T - 1], device=device, dtype=torch.long)
    ts       = torch.linspace(0, T - 1, steps=ddim_steps, device=device)
    ts       = torch.round(ts).long()
    ts[0]    = 0
    ts[-1]   = T - 1
    return torch.unique_consecutive(ts)


@torch.no_grad()
def sample_ddim_cond(model, diff: DiffusionConfig, z_seq: torch.Tensor,
                     shape, device, ddim_steps: int = 50,
                     eta: float = 0.0, pred_target: str = "x0") -> torch.Tensor:
    """
    DDIM conditional sampler.

    Runs reverse diffusion conditioned on a sequence latent z_seq.

    Args:
        model:       VAEDiffusion model (must implement .pred()).
        diff:        DiffusionConfig with pre-computed schedule tensors.
        z_seq:       Sequence latent, shape (B, latent_ch, T_lat).
        shape:       Output shape tuple, e.g. (B, 1, window_size).
        device:      Compute device.
        ddim_steps:  Number of DDIM denoising steps.
        eta:         Stochasticity (0 = deterministic DDIM).
        pred_target: "x0" or "eps" — must match training setting.

    Returns:
        Generated signal tensor of shape `shape`.
    """
    B = shape[0]
    x = torch.randn(shape, device=device)

    ts     = make_ddim_timesteps(diff.T, ddim_steps, device=device)
    ts_rev = ts.flip(0)

    for i in range(len(ts_rev)):
        t      = int(ts_rev[i])
        t_prev = int(ts_rev[i + 1]) if i + 1 < len(ts_rev) else 0

        t_batch      = torch.full((B,), t,      device=device, dtype=torch.long)
        t_prev_batch = torch.full((B,), t_prev, device=device, dtype=torch.long)

        abar_t    = extract(diff.alphas_bar, t_batch,      x.shape)
        abar_prev = extract(diff.alphas_bar, t_prev_batch, x.shape)

        pred = model.pred(x, t_batch, z_seq)

        if pred_target == "eps":
            eps    = pred
            x0_hat = x0_from_eps(diff, x, t_batch, eps)
        elif pred_target == "x0":
            x0_hat = pred
            eps    = eps_from_x0(diff, x, t_batch, x0_hat)
        else:
            raise ValueError(f"Unknown pred_target: {pred_target!r}")

        if eta == 0.0:
            sigma_t = torch.zeros((), device=device)
        else:
            sigma_t = eta * torch.sqrt(
                (1 - abar_prev) / (1 - abar_t) * (1 - abar_t / abar_prev)
            )

        noise   = sigma_t * torch.randn_like(x)
        dir_xt  = torch.sqrt(torch.clamp(1.0 - abar_prev - sigma_t ** 2, min=0.0)) * eps
        x       = torch.sqrt(abar_prev) * x0_hat + dir_xt + noise

    return x
