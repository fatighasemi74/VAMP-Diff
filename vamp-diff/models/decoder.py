"""
Diffusion decoder: a 1-D U-Net conditioned on diffusion timestep t and
sequence latent z via FiLM modulation and multi-scale spatial injection.

Optionally receives compressed encoder summaries (not raw skip connections)
to provide mild input-dependent guidance during training.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Time embedding
# =========================================================

class TimeEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding followed by a 2-layer MLP.

    Args:
        dim: Embedding dimension.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half  = self.dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(0, half, device=t.device).float() / half
        )
        args  = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb   = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


# =========================================================
# FiLM residual block
# =========================================================

class FiLMResBlock1D(nn.Module):
    """
    1-D residual block with Feature-wise Linear Modulation (FiLM).

    Conditioning vector `cond` (time + global z) modulates the
    intermediate activations via learned gamma/beta offsets.

    Args:
        ch:       Number of channels.
        cond_dim: Conditioning vector dimension.
        gn_groups: GroupNorm group count.
    """

    def __init__(self, ch: int, cond_dim: int, gn_groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(gn_groups, ch), ch)
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(gn_groups, ch), ch)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=1)
        self.film  = nn.Linear(cond_dim, 2 * ch)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h            = self.conv1(F.silu(self.norm1(x)))
        gamma_beta   = self.film(cond).unsqueeze(-1)
        gamma, beta  = torch.chunk(gamma_beta, 2, dim=1)
        h            = (1 + gamma) * h + beta
        h            = self.conv2(F.silu(self.norm2(h)))
        return x + h


def _make_resblocks(n: int, ch: int, cond_dim: int) -> nn.ModuleList:
    return nn.ModuleList([FiLMResBlock1D(ch, cond_dim) for _ in range(n)])


# =========================================================
# Sequence-latent conditioning projector
# =========================================================

class SequenceConditionProjector(nn.Module):
    """
    Project the sequence latent z (B, latent_ch, 768) into three
    multi-scale spatial conditioning maps matched to the decoder stages.

    Outputs:
        c1: (B, base,     3072)   — full resolution
        c2: (B, 2*base,   1536)   — half resolution
        c4: (B, 4*base,    768)   — quarter resolution
    """

    def __init__(self, latent_ch: int = 256, base: int = 64):
        super().__init__()

        self.proj_l4 = nn.Sequential(
            nn.Conv1d(latent_ch, base * 4, 1), nn.SiLU()
        )
        self.proj_l2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(latent_ch, base * 2, 1), nn.SiLU()
        )
        self.proj_l1 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="linear", align_corners=False),
            nn.Conv1d(latent_ch, base, 1), nn.SiLU()
        )

        self.gate_l4 = nn.Parameter(torch.tensor(0.1))
        self.gate_l2 = nn.Parameter(torch.tensor(0.1))
        self.gate_l1 = nn.Parameter(torch.tensor(0.1))

    def forward(self, z: torch.Tensor):
        c4 = self.gate_l4 * self.proj_l4(z)   # (B, 4*base, 768)
        c2 = self.gate_l2 * self.proj_l2(z)   # (B, 2*base, 1536)
        c1 = self.gate_l1 * self.proj_l1(z)   # (B,   base, 3072)
        return c1, c2, c4


# =========================================================
# Compressed encoder summary (optional input-dependent guidance)
# =========================================================

class CompressedSummary1D(nn.Module):
    """
    Lightweight projection of an encoder intermediate feature map into
    a gated summary that the decoder can optionally attend to.

    The initial gate value is set near zero so the decoder learns to
    rely on z first and only admits encoder information gradually.

    Args:
        in_ch:     Input channel count.
        out_ch:    Output channel count.
        dropout:   Dropout probability.
        gate_init: Initial value of the learnable scalar gate.
    """

    def __init__(self, in_ch: int, out_ch: int,
                 dropout: float = 0.4, gate_init: float = 0.05):
        super().__init__()
        self.proj    = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        self.norm    = nn.GroupNorm(min(8, out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.gate    = nn.Parameter(torch.tensor(gate_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = F.silu(self.norm(x))
        x = self.dropout(x)
        return self.gate * x


# =========================================================
# Diffusion U-Net decoder
# =========================================================

class DiffusionUNet1D_SeqZ(nn.Module):
    """
    1-D U-Net waveform decoder conditioned on:
        - diffusion timestep t  (via sinusoidal embedding + FiLM)
        - sequence latent z     (via global pooled embedding + spatial injection)
        - encoder summaries     (optional gated projections, not raw skip connections)

    Args:
        in_ch:      Input channels (1 for waveform noise).
        base:       Base channel width.
        time_dim:   Timestep embedding dimension.
        latent_ch:  Latent channel dimension.
        n_res:      Residual blocks per U-Net stage.
        skip_dropout: Dropout on encoder summary gates.
        skip_gate_init: Initial gate value for encoder summaries.
    """

    def __init__(self, in_ch: int = 1, base: int = 64, time_dim: int = 128,
                 latent_ch: int = 256, n_res: int = 2,
                 skip_dropout: float = 0.5, skip_gate_init: float = 0.01):
        super().__init__()

        self.time_emb    = TimeEmbedding(time_dim)

        self.z_global    = nn.Sequential(
            nn.Conv1d(latent_ch, time_dim, 1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.z_seq_proj  = SequenceConditionProjector(latent_ch=latent_ch, base=base)

        self.sum0        = CompressedSummary1D(base,     base,     dropout=skip_dropout, gate_init=skip_gate_init)
        self.sum1        = CompressedSummary1D(base * 2, base * 2, dropout=skip_dropout, gate_init=skip_gate_init)
        self.sum2        = CompressedSummary1D(base * 4, base * 4, dropout=skip_dropout, gate_init=skip_gate_init)

        cond_dim         = time_dim

        self.in_conv     = nn.Conv1d(in_ch, base, 3, padding=1)

        self.down1       = _make_resblocks(n_res, base,     cond_dim)
        self.downsample1 = nn.Conv1d(base,     base * 2, 4, stride=2, padding=1)

        self.down2       = _make_resblocks(n_res, base * 2, cond_dim)
        self.downsample2 = nn.Conv1d(base * 2, base * 4, 4, stride=2, padding=1)

        self.mid         = _make_resblocks(n_res, base * 4, cond_dim)

        self.upsample2   = nn.ConvTranspose1d(base * 4, base * 2, 4, stride=2, padding=1)
        self.up2         = _make_resblocks(n_res, base * 2, cond_dim)

        self.upsample1   = nn.ConvTranspose1d(base * 2, base, 4, stride=2, padding=1)
        self.up1         = _make_resblocks(n_res, base, cond_dim)

        self.out_conv    = nn.Sequential(
            nn.GroupNorm(8, base),
            nn.SiLU(),
            nn.Conv1d(base, 1, 3, padding=1),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                z_seq: torch.Tensor, enc_summaries=None) -> torch.Tensor:
        """
        Args:
            x_t:          Noisy input, (B, 1, L).
            t:            Diffusion timestep indices, (B,).
            z_seq:        Sequence latent, (B, latent_ch, T_lat).
            enc_summaries: Optional list [h0, h1, h2] of encoder features.

        Returns:
            Predicted signal or noise, shape (B, 1, L).
        """
        t_emb  = self.time_emb(t)
        z_emb  = self.z_global(z_seq).squeeze(-1)
        cond   = t_emb + z_emb

        z_c1, z_c2, z_c4 = self.z_seq_proj(z_seq)

        s0 = s1 = s2 = 0.0
        if enc_summaries is not None:
            h0_enc, h1_enc, h2_enc = enc_summaries
            s0 = self.sum0(h0_enc)
            s1 = self.sum1(h1_enc)
            s2 = self.sum2(h2_enc)

        x  = self.in_conv(x_t) + z_c1 + s0

        h1 = x
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

        u2 = self.upsample2(m) + z_c2
        for blk in self.up2:
            u2 = blk(u2, cond)

        u1 = self.upsample1(u2) + z_c1
        for blk in self.up1:
            u1 = blk(u1, cond)

        return self.out_conv(u1)
