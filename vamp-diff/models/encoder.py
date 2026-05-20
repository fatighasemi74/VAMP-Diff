"""
Sequence encoder: maps a raw PPG window (B, 1, 3072) to a sequence
latent distribution (mu, logvar) of shape (B, LATENT_CH, 768).

Two stride-2 downsampling stages give a 4x temporal compression:
    stem  -> (B, base,   3072)
    down1 -> (B, 2*base, 1536)
    down2 -> (B, 4*base,  768) -> mu / logvar
"""

import torch.nn as nn
import torch.nn.functional as F


class ResBlockEnc(nn.Module):
    """1-D residual block for the encoder (no conditioning)."""

    def __init__(self, ch: int, gn_groups: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm1    = nn.GroupNorm(min(gn_groups, ch), ch)
        self.conv1    = nn.Conv1d(ch, ch, 3, padding=1)
        self.norm2    = nn.GroupNorm(min(gn_groups, ch), ch)
        self.conv2    = nn.Conv1d(ch, ch, 3, padding=1)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.dropout(h)
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


def _make_enc_stage(ch: int, n_blocks: int, gn_groups: int = 8,
                    dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(*[
        ResBlockEnc(ch, gn_groups=gn_groups, dropout=dropout)
        for _ in range(n_blocks)
    ])


class SeqEncoder1D(nn.Module):
    """
    1-D convolutional VAE encoder.

    Args:
        in_ch:            Input channels (1 for single-channel PPG).
        base:             Base channel width.
        latent_ch:        Latent channel dimension.
        gn_groups:        GroupNorm group count.
        enc_res_per_stage: Residual blocks per downsampling stage.

    Forward:
        x            : (B, 1, 3072)
        return_feats : if True, also return intermediate feature maps

    Returns:
        mu, logvar          : each (B, latent_ch, 768)
        feats (optional)    : [h0, h1, h2] — used by decoder as summaries
    """

    def __init__(self, in_ch: int = 1, base: int = 64, latent_ch: int = 256,
                 gn_groups: int = 8, enc_res_per_stage: int = 2):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base, kernel_size=5, padding=2),
            nn.GroupNorm(min(gn_groups, base), base),
            nn.SiLU(),
            _make_enc_stage(base, enc_res_per_stage, gn_groups=gn_groups, dropout=0.10),
        )

        self.down1 = nn.Sequential(
            nn.Conv1d(base, base * 2, kernel_size=5, stride=2, padding=2),   # -> 1536
            nn.GroupNorm(min(gn_groups, base * 2), base * 2),
            nn.SiLU(),
            _make_enc_stage(base * 2, enc_res_per_stage, gn_groups=gn_groups, dropout=0.10),
        )

        self.down2 = nn.Sequential(
            nn.Conv1d(base * 2, base * 4, kernel_size=5, stride=2, padding=2),  # -> 768
            nn.GroupNorm(min(gn_groups, base * 4), base * 4),
            nn.SiLU(),
            _make_enc_stage(base * 4, enc_res_per_stage + 1, gn_groups=gn_groups, dropout=0.10),
        )

        feat_ch      = base * 4
        self.mu      = nn.Conv1d(feat_ch, latent_ch, kernel_size=1)
        self.logvar  = nn.Conv1d(feat_ch, latent_ch, kernel_size=1)

    def forward(self, x, return_feats: bool = False):
        h0 = self.stem(x)    # (B, base,   3072)
        h1 = self.down1(h0)  # (B, 2*base, 1536)
        h2 = self.down2(h1)  # (B, 4*base,  768)

        mu     = self.mu(h2)
        logvar = self.logvar(h2)

        if return_feats:
            return mu, logvar, [h0, h1, h2]
        return mu, logvar
