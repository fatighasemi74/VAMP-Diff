"""
Training and validation loop functions.
"""

import torch
import torch.nn as nn
from pathlib import Path

from training.losses import compute_losses
from models.diffusion import sample_ddim_cond
from utils.utils import denorm_window
from utils.visualization import plot_recon_pair


# =========================================================
# Training epoch
# =========================================================

def train_one_epoch(model, diff, loader, opt, device,
                    beta_kl: float = 0.0, vamp_prior=None,
                    print_every=None, cfg=None) -> dict:
    """
    Run one full training epoch.

    Args:
        model:        VAEDiffusion model.
        diff:         DiffusionConfig.
        loader:       Training DataLoader.
        opt:          Optimizer.
        device:       Compute device.
        beta_kl:      KL weight for this epoch.
        vamp_prior:   VampPrior instance (None → standard normal KL).
        print_every:  Log every N batches (None = silent).
        cfg:          Config module.

    Returns:
        Dict of averaged loss components for this epoch.
    """
    if cfg is None:
        import configs.default as cfg

    model.train()

    total = {k: 0.0 for k in
             ["loss", "diff", "recon", "spec", "deriv",
              "kl_raw", "kl_fb", "amp", "ptp"]}
    n = 0

    for bi, batch in enumerate(loader):
        if bi % 10 == 0:
            torch.cuda.empty_cache()

        x0, _, _ = batch
        x0 = x0.to(device).unsqueeze(1)

        out = compute_losses(model, diff, x0, beta_kl,
                             vamp_prior=vamp_prior, cfg=cfg)

        # Skip batch if NaN/Inf detected
        if any(torch.isnan(out[k]).any() or torch.isinf(out[k]).any()
               for k in ["loss", "diff", "recon", "spec", "deriv",
                         "amp", "ptp", "kl_raw", "kl_fb"]):
            print(f"[WARN] NaN/Inf in batch {bi}, skipping")
            continue

        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        opt.step()

        if print_every is not None and (bi % print_every == 0):
            print(
                f"[train b{bi:05d}] "
                f"loss={out['loss'].item():.6f} | "
                f"diff={out['diff'].item():.4f}  recon={out['recon'].item():.4f} | "
                f"spec={out['spec'].item():.4f}  deriv={out['deriv'].item():.4f} | "
                f"amp={out['amp'].item():.4f}  ptp={out['ptp'].item():.4f} | "
                f"kl={out['kl_fb'].item():.2f}"
            )

        for k in total:
            total[k] += out[k].item()
        n += 1

    return {k: v / max(1, n) for k, v in total.items()}


# =========================================================
# Validation epoch
# =========================================================

@torch.no_grad()
def eval_one_epoch(model, diff, loader, device,
                   beta_kl: float = 0.0, vamp_prior=None,
                   print_every=None, cfg=None) -> dict:
    """
    Run one full validation epoch (no gradient computation).

    Args / Returns: same structure as train_one_epoch.
    """
    if cfg is None:
        import configs.default as cfg

    model.eval()

    total = {k: 0.0 for k in
             ["loss", "diff", "recon", "spec", "deriv",
              "kl_raw", "kl_fb", "amp", "ptp"]}
    n = 0

    for bi, batch in enumerate(loader):
        x0, _, _ = batch
        x0 = x0.to(device).unsqueeze(1)

        out = compute_losses(model, diff, x0, beta_kl,
                             vamp_prior=vamp_prior, cfg=cfg)

        if any(torch.isnan(out[k]).any() or torch.isinf(out[k]).any()
               for k in ["loss", "diff", "recon", "spec", "deriv",
                         "amp", "ptp", "kl_raw", "kl_fb"]):
            print(f"[WARN] NaN/Inf in val batch {bi}, skipping")
            continue

        for k in total:
            total[k] += out[k].item()
        n += 1

    return {k: v / max(1, n) for k, v in total.items()}


# =========================================================
# Validation reconstruction visualisation
# =========================================================

@torch.no_grad()
def run_validation_recons(model, diff, val_set, device,
                          out_dir: Path, epoch: int,
                          n_show: int = 3, cfg=None):
    """
    Generate and save reconstruction plots for a few validation examples.

    Args:
        model:    VAEDiffusion model (eval mode).
        diff:     DiffusionConfig.
        val_set:  Validation Dataset.
        device:   Compute device.
        out_dir:  Root output directory.
        epoch:    Current epoch number (used in filenames).
        n_show:   Number of examples to plot.
        cfg:      Config module.
    """
    if cfg is None:
        import configs.default as cfg

    model.eval()
    idxs = list(range(min(n_show, len(val_set))))

    for i, idx in enumerate(idxs):
        x0, m, s = val_set[idx]
        x0 = x0.unsqueeze(0).unsqueeze(1).to(device)

        if cfg.USE_MU_FOR_RECON:
            mu, _ = model.encode_mu(x0)
            z_seq = mu
        else:
            z_seq, _, _ = model.encode(x0)

        xhat = sample_ddim_cond(
            model, diff, z_seq, x0.shape, device,
            ddim_steps=cfg.DDIM_STEPS, eta=cfg.DDIM_ETA,
            pred_target=cfg.PRED_TARGET,
        )

        x_true_dn = denorm_window(x0.squeeze().cpu().numpy(),   float(m), float(s))
        x_pred_dn = denorm_window(xhat.squeeze().cpu().numpy(), float(m), float(s))

        plot_recon_pair(
            x_true_dn, x_pred_dn,
            fs=cfg.FS,
            title=f"Val example {i} — epoch {epoch}",
            outpath=out_dir / "val_recons" / f"epoch_{epoch:03d}" / f"val_recon_{i:02d}.png",
        )
