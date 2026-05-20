"""
VAEDiffusion: top-level model that combines the sequence encoder and the
diffusion decoder into a single nn.Module.

The encoder maps an input PPG window to a sequence latent distribution
(mu, logvar).  The diffusion decoder denoises x_t conditioned on a
latent sample z drawn from that distribution.
"""

import torch
import torch.nn as nn

from models.encoder import SeqEncoder1D
from models.decoder import DiffusionUNet1D_SeqZ
from training.losses import reparam


class VAEDiffusion(nn.Module):
    """
    Joint VAE + diffusion model for PPG signal generation.

    Args:
        latent_ch:       Latent channel dimension.
        enc_base:        Base channel width for the encoder.
        diff_base:       Base channel width for the diffusion decoder.
        time_dim:        Timestep embedding dimension.
        n_res:           Residual blocks per decoder stage.
        enc_res_per_stage: Residual blocks per encoder stage.
        skip_dropout:    Dropout on encoder summary gates in the decoder.
        skip_gate_init:  Initial value of encoder summary gates.
    """

    def __init__(self, latent_ch: int = 256, enc_base: int = 64,
                 diff_base: int = 64, time_dim: int = 128,
                 n_res: int = 2, enc_res_per_stage: int = 2,
                 skip_dropout: float = 0.5, skip_gate_init: float = 0.01):
        super().__init__()

        self.encoder = SeqEncoder1D(
            in_ch=1,
            base=enc_base,
            latent_ch=latent_ch,
            enc_res_per_stage=enc_res_per_stage,
        )
        self.diffusion = DiffusionUNet1D_SeqZ(
            in_ch=1,
            base=diff_base,
            time_dim=time_dim,
            latent_ch=latent_ch,
            n_res=n_res,
            skip_dropout=skip_dropout,
            skip_gate_init=skip_gate_init,
        )

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, x0: torch.Tensor, return_feats: bool = False):
        """
        Encode x0 and draw a reparameterised latent sample z.

        Args:
            x0:          Input tensor, (B, 1, L).
            return_feats: If True, also return encoder intermediate features.

        Returns:
            z, mu, logvar                      — or —
            z, mu, logvar, feats  (return_feats=True)
        """
        if return_feats:
            mu, logvar, feats = self.encoder(x0, return_feats=True)
            z = reparam(mu, logvar)
            return z, mu, logvar, feats
        else:
            mu, logvar = self.encoder(x0, return_feats=False)
            z = reparam(mu, logvar)
            return z, mu, logvar

    def encode_mu(self, x0: torch.Tensor, return_feats: bool = False):
        """
        Encode x0 and return the posterior mean (deterministic path).

        Args:
            x0:          Input tensor, (B, 1, L).
            return_feats: If True, also return encoder intermediate features.

        Returns:
            mu, logvar                     — or —
            mu, logvar, feats  (return_feats=True)
        """
        if return_feats:
            mu, logvar, feats = self.encoder(x0, return_feats=True)
            return mu, logvar, feats
        else:
            mu, logvar = self.encoder(x0, return_feats=False)
            return mu, logvar

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def pred(self, x_t: torch.Tensor, t: torch.Tensor,
             z_seq: torch.Tensor, enc_summaries=None) -> torch.Tensor:
        """
        Run one forward pass of the diffusion decoder.

        Args:
            x_t:          Noisy input, (B, 1, L).
            t:            Diffusion timestep indices, (B,).
            z_seq:        Sequence latent, (B, latent_ch, T_lat).
            enc_summaries: Optional encoder feature list [h0, h1, h2].

        Returns:
            Predicted signal or noise, (B, 1, L).
        """
        return self.diffusion(x_t, t, z_seq, enc_summaries=enc_summaries)
