"""
Evaluation metrics for VAMP-Diff.

Includes standard waveform reconstruction metrics (MAE, RMSE, Pearson r),
PPG-specific peak metrics (HR error, IBI MAE, peak timing), and diagnostic
utilities (z-sensitivity test, latent stats inspection).
"""

import numpy as np
import torch

from scipy.signal import find_peaks
from scipy.stats import pearsonr

from models.diffusion import sample_ddim_cond
from utils.utils import bandpass_fft, denorm_window


# =========================================================
# Helper utilities
# =========================================================

def _nanmean(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.nanmean(x)) if np.any(~np.isnan(x)) else float("nan")


def _nanstd(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.nanstd(x)) if np.any(~np.isnan(x)) else float("nan")


# =========================================================
# Standard reconstruction metrics
# =========================================================

@torch.no_grad()
def eval_recon_metrics(model, diff, loader, device,
                       max_batches=None, cfg=None) -> dict:
    """
    Compute MAE, RMSE, and Pearson correlation on denormalised signals.

    Args:
        model:       VAEDiffusion model.
        diff:        DiffusionConfig.
        loader:      DataLoader (val or test).
        device:      Compute device.
        max_batches: Cap the number of batches evaluated (None = all).
        cfg:         Config module.

    Returns:
        Dict with keys: MAE_mean, MAE_std, RMSE_mean, RMSE_std,
                         Corr_mean, Corr_std.
    """
    if cfg is None:
        import configs.default as cfg

    model.eval()
    maes, rmses, corrs = [], [], []

    for b, batch in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break

        x0, m, s = batch
        x0 = x0.to(device).unsqueeze(1)
        m  = m.to(device).view(-1, 1)
        s  = s.to(device).view(-1, 1)

        if cfg.USE_MU_FOR_RECON:
            mu, _ = model.encode_mu(x0)
            z_seq = mu
        else:
            z_seq, _, _ = model.encode(x0)

        x_recon = sample_ddim_cond(
            model, diff, z_seq, x0.shape, device,
            ddim_steps=cfg.DDIM_STEPS, eta=cfg.DDIM_ETA,
            pred_target=cfg.PRED_TARGET,
        )

        x_dn  = (x0.squeeze(1) * s + m).detach().cpu().numpy()
        xr_dn = (x_recon.squeeze(1) * s + m).detach().cpu().numpy()

        for i in range(x_dn.shape[0]):
            a, p = x_dn[i], xr_dn[i]
            maes.append(np.mean(np.abs(a - p)))
            rmses.append(np.sqrt(np.mean((a - p) ** 2)))
            r = pearsonr(a, p)[0] if np.std(a) > 1e-8 and np.std(p) > 1e-8 else 0.0
            corrs.append(r)

    return {
        "MAE_mean":  float(np.mean(maes)),
        "MAE_std":   float(np.std(maes)),
        "RMSE_mean": float(np.mean(rmses)),
        "RMSE_std":  float(np.std(rmses)),
        "Corr_mean": float(np.mean(corrs)),
        "Corr_std":  float(np.std(corrs)),
    }


# =========================================================
# Extended PPG metrics
# =========================================================

@torch.no_grad()
def eval_recon_metrics_extended(model, diff, loader, device,
                                fs: int = 300, max_batches=None,
                                cfg=None) -> dict:
    """
    Extended PPG reconstruction metrics including peak-based HR and IBI.

    In addition to MAE / RMSE / Pearson r, this function computes:
        - Peak-to-peak amplitude error
        - Peak count difference
        - Peak timing error (ms)
        - Heart-rate absolute error (bpm)
        - IBI MAE (ms)

    Args:
        model:       VAEDiffusion model.
        diff:        DiffusionConfig.
        loader:      DataLoader (val or test).
        device:      Compute device.
        fs:          Sampling frequency in Hz.
        max_batches: Cap the number of batches (None = all).
        cfg:         Config module.

    Returns:
        Dict of mean/std metric values.
    """
    if cfg is None:
        import configs.default as cfg

    model.eval()

    maes, rmses, corrs, signed_errs = [], [], [], []
    ptp_errs, peak_count_diffs     = [], []
    peak_timing_errs_ms             = []
    hr_abs_errs, ibi_maes_ms        = [], []

    for b, batch in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break

        x0, m, s = batch
        x0 = x0.to(device).unsqueeze(1)
        m  = m.to(device).view(-1, 1)
        s  = s.to(device).view(-1, 1)

        if cfg.USE_MU_FOR_RECON:
            mu, _ = model.encode_mu(x0)
            z_seq = mu
        else:
            z_seq, _, _ = model.encode(x0)

        x_recon = sample_ddim_cond(
            model, diff, z_seq, x0.shape, device,
            ddim_steps=cfg.DDIM_STEPS, eta=cfg.DDIM_ETA,
            pred_target=cfg.PRED_TARGET,
        )

        x_np  = (x0.squeeze(1) * s + m).detach().cpu().numpy()
        xr_np = (x_recon.squeeze(1) * s + m).detach().cpu().numpy()

        for i in range(x_np.shape[0]):
            a, p = x_np[i], xr_np[i]

            maes.append(np.mean(np.abs(a - p)))
            rmses.append(np.sqrt(np.mean((a - p) ** 2)))
            signed_errs.append(np.mean(p - a))
            corr = pearsonr(a, p)[0] if np.std(a) > 1e-8 and np.std(p) > 1e-8 else 0.0
            corrs.append(corr)

            ptp_errs.append(abs((np.max(p) - np.min(p)) - (np.max(a) - np.min(a))))

            # Bandpass before peak detection
            a_bp = bandpass_fft(
                torch.tensor(a).unsqueeze(0), fs, 0.7, 3.0
            )[0].cpu().numpy()
            p_bp = bandpass_fft(
                torch.tensor(p).unsqueeze(0), fs, 0.7, 3.0
            )[0].cpu().numpy()

            kw = dict(distance=int(0.35 * fs), prominence=0.1)
            true_peaks, _ = find_peaks(
                a_bp, height=np.percentile(a_bp, 60),
                prominence=kw["prominence"] * np.std(a_bp),
                distance=kw["distance"],
            )
            pred_peaks, _ = find_peaks(
                p_bp, height=np.percentile(p_bp, 60),
                prominence=kw["prominence"] * np.std(p_bp),
                distance=kw["distance"],
            )

            peak_count_diffs.append(abs(len(true_peaks) - len(pred_peaks)))

            if len(true_peaks) > 0 and len(pred_peaks) > 0:
                n_match = min(len(true_peaks), len(pred_peaks))
                peak_timing_errs_ms.append(
                    np.mean(np.abs(true_peaks[:n_match] - pred_peaks[:n_match]) / fs) * 1000.0
                )
            else:
                peak_timing_errs_ms.append(np.nan)

            if len(true_peaks) >= 2 and len(pred_peaks) >= 2:
                true_ibi = np.diff(true_peaks) / fs
                pred_ibi = np.diff(pred_peaks) / fs
                hr_abs_errs.append(abs(60.0 / np.mean(true_ibi) - 60.0 / np.mean(pred_ibi)))
                min_len = min(len(true_ibi), len(pred_ibi))
                ibi_maes_ms.append(
                    np.mean(np.abs(true_ibi[:min_len] - pred_ibi[:min_len])) * 1000.0
                )
            else:
                hr_abs_errs.append(np.nan)
                ibi_maes_ms.append(np.nan)

    return {
        "MAE_mean":                float(np.mean(maes)),
        "MAE_std":                 float(np.std(maes)),
        "RMSE_mean":               float(np.mean(rmses)),
        "RMSE_std":                float(np.std(rmses)),
        "Corr_mean":               float(np.mean(corrs)),
        "Corr_std":                float(np.std(corrs)),
        "SignedErr_mean":          float(np.mean(signed_errs)),
        "SignedErr_std":           float(np.std(signed_errs)),
        "PTPError_mean":           float(np.mean(ptp_errs)),
        "PTPError_std":            float(np.std(ptp_errs)),
        "PeakCountDiff_mean":      float(np.mean(peak_count_diffs)),
        "PeakCountDiff_std":       float(np.std(peak_count_diffs)),
        "PeakTimingErr_ms_mean":   _nanmean(peak_timing_errs_ms),
        "PeakTimingErr_ms_std":    _nanstd(peak_timing_errs_ms),
        "HRAbsErr_bpm_mean":       _nanmean(hr_abs_errs),
        "HRAbsErr_bpm_std":        _nanstd(hr_abs_errs),
        "IBI_MAE_ms_mean":         _nanmean(ibi_maes_ms),
        "IBI_MAE_ms_std":          _nanstd(ibi_maes_ms),
    }


# =========================================================
# Diagnostic utilities
# =========================================================

@torch.no_grad()
def test_z_sensitivity(model, diff, val_set, device,
                       n: int = 3, ddim_steps: int = 20, cfg=None):
    """
    Check whether the decoder is actually using z.

    Compares decoder output from the posterior mean z=mu against output
    from a random z ~ N(0,I).  A large difference (ratio > 0.3) confirms
    the decoder is conditioning on z meaningfully.

    Args:
        model:      VAEDiffusion model.
        diff:       DiffusionConfig.
        val_set:    Validation Dataset.
        device:     Compute device.
        n:          Number of examples to test.
        ddim_steps: Denoising steps (fewer is fine for diagnostics).
        cfg:        Config module.
    """
    if cfg is None:
        import configs.default as cfg

    model.eval()
    print("\n===== Z sensitivity test =====")

    for i in range(min(n, len(val_set))):
        x0, _, _ = val_set[i]
        x0       = x0.unsqueeze(0).unsqueeze(1).to(device)
        mu, _    = model.encode_mu(x0)

        x_from_mu   = sample_ddim_cond(model, diff, mu, x0.shape, device,
                                        ddim_steps=ddim_steps,
                                        pred_target=cfg.PRED_TARGET)
        z_rand      = torch.randn_like(mu)
        x_from_rand = sample_ddim_cond(model, diff, z_rand, x0.shape, device,
                                        ddim_steps=ddim_steps,
                                        pred_target=cfg.PRED_TARGET)

        diff_val = (x_from_mu - x_from_rand).abs().mean().item()
        mu_range = (x_from_mu.max() - x_from_mu.min()).item()
        print(
            f"  Sample {i}: |out(mu)-out(z_rand)| = {diff_val:.6f}  |  "
            f"out(mu) range = {mu_range:.6f}  |  "
            f"ratio = {diff_val / max(mu_range, 1e-8):.3f}"
        )

    print("  (ratio > 0.3 → z is being used meaningfully)")


@torch.no_grad()
def inspect_loader_stats(model, diff, loader, device,
                         name: str = "set", max_batches: int = 10, cfg=None):
    """
    Print latent-space and prediction statistics for a data loader.

    Useful for sanity-checking encoder health and decoder behaviour
    during / after training.

    Args:
        model:       VAEDiffusion model.
        diff:        DiffusionConfig.
        loader:      DataLoader to inspect.
        device:      Compute device.
        name:        Label printed in the summary.
        max_batches: Number of batches to inspect.
        cfg:         Config module.
    """
    import numpy as np
    from training.losses import reparam
    from models.diffusion import q_sample, x0_from_eps

    if cfg is None:
        import configs.default as cfg

    model.eval()

    x_means, x_stds           = [], []
    mu_means, mu_stds, mu_norms = [], [], []
    logvar_means, logvar_stds = [], []
    z_means, z_stds, z_norms  = [], [], []
    pred_means, pred_stds     = [], []
    x0hat_means, x0hat_stds   = [], []

    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        x0, m, s = batch
        x0 = x0.to(device).unsqueeze(1)

        x_flat = x0.squeeze(1)
        x_means.append(x_flat.mean().item())
        x_stds.append(x_flat.std().item())

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

        B    = x0.size(0)
        t    = torch.randint(0, diff.T, (B,), device=device)
        noise = torch.randn_like(x0)
        x_t  = q_sample(diff, x0, t, noise)
        pred = model.pred(x_t, t, z)

        if cfg.PRED_TARGET == "eps":
            x0_hat = x0_from_eps(diff, x_t, t, pred)
        else:
            x0_hat = pred

        pred_means.append(pred.mean().item())
        pred_stds.append(pred.std().item())
        x0hat_means.append(x0_hat.mean().item())
        x0hat_stds.append(x0_hat.std().item())

    print(f"\n===== {name} stats =====")
    print(f"x_norm mean/std    : {np.mean(x_means):.6f} / {np.mean(x_stds):.6f}")
    print(f"mu mean/std        : {np.mean(mu_means):.6f} / {np.mean(mu_stds):.6f}")
    print(f"mu norm            : {np.mean(mu_norms):.6f}")
    print(f"logvar mean/std    : {np.mean(logvar_means):.6f} / {np.mean(logvar_stds):.6f}")
    print(f"z mean/std         : {np.mean(z_means):.6f} / {np.mean(z_stds):.6f}")
    print(f"z norm             : {np.mean(z_norms):.6f}")
    print(f"pred mean/std      : {np.mean(pred_means):.6f} / {np.mean(pred_stds):.6f}")
    print(f"x0hat mean/std     : {np.mean(x0hat_means):.6f} / {np.mean(x0hat_stds):.6f}")
