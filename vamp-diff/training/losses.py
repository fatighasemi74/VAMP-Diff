"""
Loss functions for VAMP-Diff training.

Includes reparameterisation, KL divergence variants (standard normal and
VampPrior), spectral loss, derivative loss, and the combined training loss.
"""

import math

import torch
import torch.nn.functional as F


# =========================================================
# VAE reparameterisation
# =========================================================

def reparam(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Draw z ~ N(mu, exp(logvar)) via the reparameterisation trick."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


# =========================================================
# KL divergence
# =========================================================

def kl_diag_standard_normal(mu: torch.Tensor, logvar: torch.Tensor):
    """
    Per-dimension KL divergence KL[q(z|x) || N(0,I)].

    Returns:
        kl_per_dim: (B, C, T) per-dimension KL values.
        kl_raw:     Scalar mean KL summed over (C, T).
    """
    kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
    kl_raw     = kl_per_dim.sum(dim=(1, 2)).mean()
    return kl_per_dim, kl_raw


def kl_free_bits(mu: torch.Tensor, logvar: torch.Tensor,
                 free_bits_nats: float):
    """
    Free-bits KL: clamp per-dimension KL to a minimum floor before summing.

    Args:
        mu, logvar:      Posterior parameters, each (B, C, T).
        free_bits_nats:  Minimum KL per dimension in nats.

    Returns:
        kl_raw: Scalar unclamped KL.
        kl_fb:  Scalar free-bits KL.
    """
    kl_per_dim, kl_raw = kl_diag_standard_normal(mu, logvar)
    kl_fb = torch.clamp(kl_per_dim, min=free_bits_nats).sum(dim=(1, 2)).mean()
    return kl_raw, kl_fb


# =========================================================
# Auxiliary signal losses
# =========================================================

def spectral_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Log-magnitude spectral loss (smooth L1 on log-amplitude spectra).

    Args:
        x_hat: Predicted signal, (B, L).
        x:     Target signal,    (B, L).
    """
    X    = torch.fft.rfft(x,     dim=-1)
    Xh   = torch.fft.rfft(x_hat, dim=-1)
    mag  = torch.log1p(torch.abs(X))
    magh = torch.log1p(torch.abs(Xh))
    return F.smooth_l1_loss(magh, mag)


def derivative_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    First-order derivative loss (smooth L1 on finite differences).

    Args:
        x_hat: Predicted signal, (B, L).
        x:     Target signal,    (B, L).
    """
    dx_hat = x_hat[:, 1:] - x_hat[:, :-1]
    dx     = x[:, 1:]     - x[:, :-1]
    return F.smooth_l1_loss(dx_hat, dx)


# =========================================================
# Combined training loss
# =========================================================

def compute_losses(model, diff, x0: torch.Tensor, beta_kl: float,
                   vamp_prior=None, cfg=None) -> dict:
    """
    Compute the full VAMP-Diff training loss for a batch.

    Args:
        model:       VAEDiffusion model.
        diff:        DiffusionConfig.
        x0:          Normalised input batch, (B, 1, L).
        beta_kl:     Current KL annealing weight.
        vamp_prior:  VampPrior instance (None → standard normal KL).
        cfg:         Config module.  Defaults to configs.default when None.

    Returns:
        Dict with keys: loss, diff, recon, spec, deriv, amp, ptp,
                         kl_raw, kl_fb, x0_hat, mu, logvar, z.
    """
    if cfg is None:
        import configs.default as cfg

    # ── Encode ─────────────────────────────────────────────────────────
    if cfg.SAMPLE_Z_IN_TRAIN:
        z, mu, logvar, feats = model.encode(x0, return_feats=True)
    else:
        mu, logvar, feats = model.encode_mu(x0, return_feats=True)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        z   = mu + cfg.Z_PERTURB_SCALE * std * eps

    logvar = torch.clamp(logvar, min=-4.0, max=2.0)

    if model.training:
        z = z + cfg.LATENT_NOISE_SCALE * torch.randn_like(z)

    # ── KL divergence ──────────────────────────────────────────────────
    if vamp_prior is not None:
        # Downsample to (B/K, C, 32) for memory-efficient VampPrior KL
        z_ds      = F.adaptive_avg_pool1d(z,      32)
        mu_ds     = F.adaptive_avg_pool1d(mu,     32)
        logvar_ds = F.adaptive_avg_pool1d(logvar, 32)

        log_q = -0.5 * (
            logvar_ds
            + (z_ds - mu_ds).pow(2) / (torch.exp(logvar_ds) + 1e-8)
            + math.log(2 * math.pi)
        ).sum(dim=(1, 2))

        mu_k, logvar_k = vamp_prior.get_prior_params()
        mu_k_ds        = F.adaptive_avg_pool1d(mu_k,     32)
        logvar_k_ds    = F.adaptive_avg_pool1d(logvar_k, 32)
        log_p          = vamp_prior.log_prob_ds(z_ds, mu_k_ds, logvar_k_ds, chunk_size=50)

        log_q  = torch.clamp(log_q, min=-1e6, max=1e6)
        log_p  = torch.clamp(log_p, min=-1e6, max=0.0)
        kl_raw = (log_q - log_p).mean()
        kl_fb  = torch.clamp(kl_raw, min=0.0, max=500000.0)
    else:
        kl_raw, kl_fb = kl_free_bits(mu, logvar, cfg.FREE_BITS_NATS)

    # ── Diffusion loss ─────────────────────────────────────────────────
    from models.diffusion import q_sample, x0_from_eps

    B     = x0.size(0)
    t     = torch.randint(0, diff.T, (B,), device=x0.device, dtype=torch.long)
    noise = torch.randn_like(x0)
    x_t   = q_sample(diff, x0, t, noise)

    pred  = model.pred(x_t, t, z, enc_summaries=feats)

    if cfg.PRED_TARGET == "eps":
        diff_loss = F.mse_loss(pred, noise)
        x0_hat    = x0_from_eps(diff, x_t, t, pred)
    elif cfg.PRED_TARGET == "x0":
        diff_loss = F.mse_loss(pred, x0)
        x0_hat    = pred
    else:
        raise ValueError(f"Unknown PRED_TARGET: {cfg.PRED_TARGET!r}")

    # ── Auxiliary losses ───────────────────────────────────────────────
    x0_1d     = x0.squeeze(1)
    x0_hat_1d = x0_hat.squeeze(1)

    recon_loss = F.smooth_l1_loss(x0_hat_1d, x0_1d)
    spec_loss  = spectral_loss(x0_hat_1d, x0_1d)
    deriv_l    = derivative_loss(x0_hat_1d, x0_1d)

    true_std = x0_1d.std(dim=-1, unbiased=False)
    pred_std = x0_hat_1d.std(dim=-1, unbiased=False)
    amp_loss = F.mse_loss(pred_std, true_std)

    true_ptp = x0_1d.max(dim=-1).values     - x0_1d.min(dim=-1).values
    pred_ptp = x0_hat_1d.max(dim=-1).values - x0_hat_1d.min(dim=-1).values
    ptp_loss = F.mse_loss(pred_ptp, true_ptp)

    # ── Total loss ─────────────────────────────────────────────────────
    loss = (
        cfg.LAMBDA_DIFF  * diff_loss
        + cfg.LAMBDA_RECON * recon_loss
        + cfg.LAMBDA_SPEC  * spec_loss
        + cfg.LAMBDA_DERIV * deriv_l
        + cfg.LAMBDA_AMP   * amp_loss
        + cfg.LAMBDA_PTP   * ptp_loss
        + beta_kl          * kl_fb
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
