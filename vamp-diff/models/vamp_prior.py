"""
VampPrior: Variational Mixture of Posteriors Prior.

K learnable pseudo-inputs (fake PPG waveforms) are passed through the
encoder to produce K Gaussian components. The prior is the uniform
mixture of these K Gaussians.

    p(z) = (1/K) Σ_k  N(z; μ_k, σ_k²)
    where μ_k, σ_k = encoder(u_k)

At generation time:
    1. Pick a pseudo-input u_k (randomly or by index).
    2. Encode it → μ_k, σ_k.
    3. Sample z ~ N(μ_k, σ_k).
    4. Decode z → PPG window.

Reference: Tomczak & Welling, "VAE with a VampPrior", AISTATS 2018.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class VampPrior(nn.Module):
    """
    VampPrior module.

    Args:
        K:           Number of pseudo-inputs (mixture components).
        window_size: Length of each pseudo-input waveform (same as WINDOW_SIZE).
        encoder:     Shared encoder module (SeqEncoder1D); weights are not
                     duplicated — the encoder is called directly.
    """

    def __init__(self, K: int, window_size: int, encoder: nn.Module):
        super().__init__()
        self.K            = K
        self.encoder      = encoder
        self.pseudo_inputs = nn.Parameter(torch.zeros(K, 1, window_size))

    # ------------------------------------------------------------------
    # Prior parameter access
    # ------------------------------------------------------------------

    def get_prior_params(self):
        """
        Pass all pseudo-inputs through the encoder.

        Returns:
            mu_k:     (K, latent_ch, T_lat)
            logvar_k: (K, latent_ch, T_lat)
        """
        mu_k, logvar_k = self.encoder(self.pseudo_inputs, return_feats=False)
        return mu_k, logvar_k

    # ------------------------------------------------------------------
    # log p_vamp(z) — full-resolution latent space
    # ------------------------------------------------------------------

    def log_prob(self, z: torch.Tensor, chunk_size: int = 5) -> torch.Tensor:
        """
        Compute log p_vamp(z) in chunks over K to avoid OOM.

        Gradients flow to pseudo_inputs through the encoder.
        Encoder weights are protected by requires_grad=False during
        the encoder-freeze phase of training.

        Args:
            z:          Posterior sample, (B, latent_ch, T_lat).
            chunk_size: Number of mixture components processed at once.

        Returns:
            log_p_vamp: (B,) log probability under the VampPrior.
        """
        mu_k, logvar_k = self.get_prior_params()

        K              = self.K
        log_p_k_list   = []

        for start in range(0, K, chunk_size):
            end          = min(start + chunk_size, K)
            mu_chunk     = mu_k[start:end]
            logvar_chunk = logvar_k[start:end]
            var_chunk    = torch.exp(logvar_chunk)

            z_exp  = z.unsqueeze(1).detach()
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

    # ------------------------------------------------------------------
    # log p_vamp(z) — downsampled latent space (memory-efficient KL)
    # ------------------------------------------------------------------

    def log_prob_ds(self, z_ds: torch.Tensor, mu_k_ds: torch.Tensor,
                    logvar_k_ds: torch.Tensor,
                    chunk_size: int = 50) -> torch.Tensor:
        """
        Compute log p_vamp(z_ds) using pre-downsampled prior components.

        This avoids re-encoding the pseudo-inputs inside the KL computation
        and allows a larger chunk size for efficiency.

        Args:
            z_ds:        Downsampled posterior sample,  (B, C, T_ds).
            mu_k_ds:     Downsampled prior means,       (K, C, T_ds).
            logvar_k_ds: Downsampled prior log-vars,    (K, C, T_ds).
            chunk_size:  K-components processed at once.

        Returns:
            log_p_vamp: (B,) log probability.
        """
        K            = self.K
        log_p_k_list = []

        for start in range(0, K, chunk_size):
            end          = min(start + chunk_size, K)
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
