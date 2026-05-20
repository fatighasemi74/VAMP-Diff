"""
VAMP-Diff training entry point.

Usage:
    python train.py
    python train.py --config configs/default.py
    python train.py --config configs/default.py --out experiments/my_run

The script:
    1. Loads the config module.
    2. Computes global normalisation statistics from the training set.
    3. Builds train / val / test datasets and data loaders.
    4. Initialises VAEDiffusion, DiffusionConfig, and VampPrior.
    5. Optionally loads a fine-tune checkpoint.
    6. Runs the training loop with KL annealing and encoder-freeze warmup.
    7. Saves best and final checkpoints, then evaluates on the test set.
"""

import argparse
import importlib.util
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.dataset import CapnoBaseVAE
from models import VAEDiffusion, VampPrior, DiffusionConfig
from models.diffusion import sample_ddim_cond
from training.trainer import train_one_epoch, eval_one_epoch, run_validation_recons
from training.kl_schedules import get_beta
from evaluation.metrics import (
    eval_recon_metrics_extended,
    test_z_sensitivity,
    inspect_loader_stats,
)
from utils.utils import seed_all, compute_global_norm_from_raw_dataset, denorm_window
from utils.visualization import plot_recon_pair, plot_diff_curves, plot_kl_curves


# =========================================================
# Config loader
# =========================================================

def load_config(path: str):
    """Load a Python config file as a module."""
    spec   = importlib.util.spec_from_file_location("cfg", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# =========================================================
# Main
# =========================================================

def main(cfg):
    seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    out_dir = Path(cfg.OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # 1. Compute global normalisation statistics
    # ----------------------------------------------------------------
    train_set_raw = CapnoBaseVAE(
        cfg.TRAIN_DIR, window_size=cfg.WINDOW_SIZE,
        step_size=cfg.STEP_SIZE, fs=cfg.FS, norm_mode="none",
    )
    GLOBAL_MEAN, GLOBAL_STD = compute_global_norm_from_raw_dataset(train_set_raw)
    print(f"[GLOBAL NORM] mean={GLOBAL_MEAN:.6f}, std={GLOBAL_STD:.6f}")

    # ----------------------------------------------------------------
    # 2. Build datasets
    # ----------------------------------------------------------------
    norm_kwargs = dict(
        window_size=cfg.WINDOW_SIZE, step_size=cfg.STEP_SIZE,
        fs=cfg.FS, norm_mode="global",
        global_mean=GLOBAL_MEAN, global_std=GLOBAL_STD,
    )
    train_set = CapnoBaseVAE(cfg.TRAIN_DIR, **norm_kwargs)
    val_set   = CapnoBaseVAE(cfg.VAL_DIR,   **norm_kwargs)
    test_set  = CapnoBaseVAE(cfg.TEST_DIR,  **norm_kwargs)

    print(f"[dataset] train={len(train_set)}  val={len(val_set)}  test={len(test_set)}")

    if cfg.MAX_TRAIN_SAMPLES and len(train_set) > cfg.MAX_TRAIN_SAMPLES:
        stride     = len(train_set) // cfg.MAX_TRAIN_SAMPLES
        indices    = list(range(0, len(train_set), stride))[:cfg.MAX_TRAIN_SAMPLES]
        train_set  = torch.utils.data.Subset(train_set, indices)
        print(f"[dataset] training set capped at {len(train_set)} samples")

    train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE,
                              shuffle=True,  num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=cfg.BATCH_SIZE,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=cfg.BATCH_SIZE,
                              shuffle=False, num_workers=0)

    # ----------------------------------------------------------------
    # 3. Build model and diffusion config
    # ----------------------------------------------------------------
    model = VAEDiffusion(
        latent_ch=cfg.LATENT_CH, enc_base=cfg.BASE_CH,
        diff_base=cfg.DIFF_BASE, time_dim=cfg.TIME_DIM,
        n_res=cfg.N_RES, enc_res_per_stage=cfg.ENC_RES_PER_STAGE,
        skip_dropout=cfg.SKIP_DROPOUT, skip_gate_init=cfg.SKIP_GATE_INIT,
    ).to(device)

    diff = DiffusionConfig(
        T=cfg.DIFF_T, beta_start=cfg.BETA_START,
        beta_end=cfg.BETA_END, schedule=cfg.SCHEDULE,
    ).to(device)

    # ----------------------------------------------------------------
    # 4. Build VampPrior and initialise pseudo-inputs
    # ----------------------------------------------------------------
    vamp_prior = VampPrior(
        K=cfg.VAMP_K, window_size=cfg.WINDOW_SIZE,
        encoder=model.encoder,
    ).to(device)

    _init_vamp_pseudo_inputs(vamp_prior, train_set, cfg, GLOBAL_MEAN, GLOBAL_STD, device)

    # ----------------------------------------------------------------
    # 5. Optional fine-tune checkpoint
    # ----------------------------------------------------------------
    if cfg.FINETUNE_CKPT and Path(cfg.FINETUNE_CKPT).exists():
        ckpt = torch.load(cfg.FINETUNE_CKPT, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"[FINETUNE] Loaded checkpoint from {cfg.FINETUNE_CKPT}")
    else:
        print("[INFO] Training from scratch")

    # ----------------------------------------------------------------
    # 6. Freeze encoder initially; build optimizer
    # ----------------------------------------------------------------
    for p in model.encoder.parameters():
        p.requires_grad = False

    opt = torch.optim.AdamW([
        {"params": model.diffusion.parameters(), "lr": cfg.LR * 0.1},
        {"params": vamp_prior.pseudo_inputs,     "lr": cfg.LR * 10},
    ], weight_decay=cfg.WEIGHT_DECAY)

    # ----------------------------------------------------------------
    # 7. Training loop
    # ----------------------------------------------------------------
    history = {
        k: [] for k in [
            "train_loss", "train_diff", "train_recon", "train_spec", "train_deriv",
            "train_kl_raw", "train_kl_fb", "train_amp", "train_ptp",
            "val_loss",   "val_diff",   "val_recon",   "val_spec",   "val_deriv",
            "val_kl_raw", "val_kl_fb",   "val_amp",   "val_ptp",
            "beta",
        ]
    }

    best_val  = float("inf")
    best_path = out_dir / "best_vae_diffusion.pt"

    for ep in range(1, cfg.EPOCHS + 1):

        # Unfreeze encoder after freeze phase
        if ep == cfg.ENC_FREEZE_EPOCHS + 1:
            print(f"[INFO] Unfreezing encoder at epoch {ep}")
            for p in model.encoder.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW([
                {"params": model.diffusion.parameters(), "lr": cfg.LR * 0.1},
                {"params": model.encoder.parameters(),   "lr": cfg.LR * 0.025},
                {"params": vamp_prior.pseudo_inputs,     "lr": cfg.LR * 10},
            ], weight_decay=cfg.WEIGHT_DECAY)

        beta = get_beta(ep - 1, cfg)

        tr = train_one_epoch(model, diff, train_loader, opt, device,
                             beta_kl=beta, vamp_prior=vamp_prior, cfg=cfg)

        # Evaluate every 5 epochs (and epoch 1)
        if ep == 1 or ep % 5 == 0:
            va = eval_one_epoch(model, diff, val_loader, device,
                                beta_kl=beta, vamp_prior=vamp_prior, cfg=cfg)
        else:
            va = {k: float("nan") for k in tr}

        # Track mu_std as a collapse indicator
        model.eval()
        with torch.no_grad():
            _x, _, _ = next(iter(train_loader))
            _x = _x[:32].to(device).unsqueeze(1)
            _mu, _ = model.encode_mu(_x)
            mu_std = _mu.std().item()
        model.train()

        # Logging
        print(
            f"Epoch {ep:03d} | beta={beta:.3e} | mu_std={mu_std:.4f} | "
            f"Train: loss={tr['loss']:.4f} diff={tr['diff']:.4f} "
            f"recon={tr['recon']:.4f} kl={tr['kl_fb']:.2f} | "
            f"Val: loss={va['loss']:.4f} diff={va['diff']:.4f} "
            f"recon={va['recon']:.4f} kl={va['kl_fb']:.2f}"
        )

        # History update
        for key in ["loss", "diff", "recon", "spec", "deriv",
                    "kl_raw", "kl_fb", "amp", "ptp"]:
            history[f"train_{key}"].append(tr[key])
            history[f"val_{key}"].append(va[key])
        history["beta"].append(beta)

        # Training curves
        plot_diff_curves(history, out_dir / "training_curves.png")
        plot_kl_curves(history,   out_dir / "kl_curves.png")

        # Periodic validation reconstructions
        if ep % 10 == 0:
            run_validation_recons(model, diff, val_set, device,
                                  out_dir=out_dir, epoch=ep, cfg=cfg)

        # Save best checkpoint
        if not np.isnan(va["loss"]) and va["loss"] < best_val:
            best_val = va["loss"]
            _save_checkpoint(model, vamp_prior, cfg, GLOBAL_MEAN, GLOBAL_STD,
                             best_path, out_dir / "best_vamp_prior.pt")
            print(f"  [SAVED] best checkpoint (val_loss={best_val:.4f})")

    # ----------------------------------------------------------------
    # 8. Save final checkpoint
    # ----------------------------------------------------------------
    final_path = out_dir / "final_epoch_model.pt"
    _save_checkpoint(model, vamp_prior, cfg, GLOBAL_MEAN, GLOBAL_STD,
                     final_path, out_dir / "final_vamp_prior.pt")
    print(f"[SAVED] Final model → {final_path}")

    # ----------------------------------------------------------------
    # 9. Encoder health check
    # ----------------------------------------------------------------
    _check_encoder_health(model, train_loader, device, "final_epoch_model.pt")

    bv_ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(bv_ckpt["model_state"])
    _check_encoder_health(model, train_loader, device, "best_vae_diffusion.pt")

    # ----------------------------------------------------------------
    # 10. Final evaluation (best and final checkpoints)
    # ----------------------------------------------------------------
    fe_ckpt = torch.load(final_path, map_location=device)

    results = {}
    for tag, state in [("best", bv_ckpt), ("final", fe_ckpt)]:
        model.load_state_dict(state["model_state"])
        print(f"\n{'='*60}\nEVALUATING: {tag}\n{'='*60}")

        test_z_sensitivity(model, diff, val_set, device, cfg=cfg)
        inspect_loader_stats(model, diff, train_loader, device,
                             name=f"TRAIN ({tag})", cfg=cfg)
        inspect_loader_stats(model, diff, val_loader, device,
                             name=f"VAL   ({tag})", cfg=cfg)

        metrics = eval_recon_metrics_extended(
            model, diff, test_loader, device, fs=cfg.FS, cfg=cfg
        )
        results[tag] = metrics

        print(f"\n==== Test Metrics — {tag} ====")
        for k, v in metrics.items():
            print(f"  {k}: {v:.6f}")
        pd.DataFrame([metrics]).to_csv(
            out_dir / f"test_recon_metrics_{tag.upper()}.csv", index=False
        )

        # Save test reconstruction plots
        _save_test_plots(model, diff, test_set, device, tag, out_dir, cfg)

    # Summary comparison
    print(f"\n{'='*60}\nSUMMARY COMPARISON\n{'='*60}")
    print(f"{'Metric':<30} {'FINAL':>12} {'BEST':>12}")
    print("-" * 56)
    for k in results["final"]:
        vf = results["final"][k]
        vb = results["best"][k]
        print(f"  {k:<28} {vf:>12.4f} {vb:>12.4f}")

    print(f"\nDone. Outputs saved to: {out_dir}")


# =========================================================
# Internal helpers
# =========================================================

def _save_checkpoint(model, vamp_prior, cfg, global_mean, global_std,
                     model_path: Path, vamp_path: Path):
    torch.save({
        "model_state": model.state_dict(),
        "latent_ch":   cfg.LATENT_CH,
        "enc_base":    cfg.BASE_CH,
        "diff_base":   cfg.DIFF_BASE,
        "time_dim":    cfg.TIME_DIM,
        "global_mean": global_mean,
        "global_std":  global_std,
    }, model_path)
    torch.save({
        "vamp_state": vamp_prior.state_dict(),
        "K":          cfg.VAMP_K,
    }, vamp_path)


def _check_encoder_health(model, train_loader, device, label: str):
    model.eval()
    with torch.no_grad():
        x0, _, _ = next(iter(train_loader))
        x0 = x0[:32].to(device).unsqueeze(1)
        mu, _ = model.encode_mu(x0)
        mu_std = mu.std().item()
    print(f"\n[CHECK] {label} mu_std on real data = {mu_std:.4f}")
    if mu_std < 0.5:
        print("[WARN]  Encoder COLLAPSED — checkpoint not usable for generation")
    elif mu_std < 1.0:
        print("[WARN]  Partial collapse — GMM may struggle")
    else:
        print("[OK]    Encoder healthy — ready for generation")
    model.train()


def _save_test_plots(model, diff, test_set, device, tag: str,
                     out_dir: Path, cfg, n_show: int = 5):
    model.eval()
    plot_indices = random.sample(range(len(test_set)), min(n_show, len(test_set)))

    for plot_idx, ds_idx in enumerate(plot_indices):
        x0, m, s = test_set[ds_idx]
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
            x_true_dn, x_pred_dn, fs=cfg.FS,
            title=f"[{tag}] Test idx={ds_idx}",
            outpath=(out_dir / "test_recons"
                     / f"{tag}_recon_{plot_idx:02d}_idx_{ds_idx}.png"),
        )


def _init_vamp_pseudo_inputs(vamp_prior, train_set, cfg,
                              global_mean, global_std, device):
    """
    Initialise VampPrior pseudo-inputs.

    If pre-computed HR and amplitude arrays are provided in the config,
    uses stratified sampling to cover the signal distribution.
    Otherwise falls back to leaving pseudo-inputs as zeros.
    """
    if cfg.VAMP_HR_NPY is None or cfg.VAMP_PTP_NPY is None:
        print("[VAMP] No HR/PTP arrays provided — pseudo-inputs initialised to zeros")
        return

    try:
        all_hrs  = np.load(cfg.VAMP_HR_NPY)
        all_ptps = np.load(cfg.VAMP_PTP_NPY)
    except FileNotFoundError as e:
        print(f"[WARN] {e}  — pseudo-inputs initialised to zeros")
        return

    # Collect training signals in order
    all_signals = []
    for x, m, s in DataLoader(train_set, batch_size=64, shuffle=False):
        for i in range(x.shape[0]):
            all_signals.append(x[i].numpy() * float(s[i]) + float(m[i]))
    all_signals = np.array(all_signals)

    HR_BINS  = cfg.HR_BINS
    AMP_BINS = cfg.AMP_BINS
    n_hr     = len(HR_BINS) - 1
    n_amp    = len(AMP_BINS) - 1
    k_per    = max(1, cfg.VAMP_K // (n_hr * n_amp))

    selected = []
    for i in range(n_hr):
        for j in range(n_amp):
            mask    = ((all_hrs  >= HR_BINS[i])  & (all_hrs  < HR_BINS[i + 1]) &
                       (all_ptps >= AMP_BINS[j]) & (all_ptps < AMP_BINS[j + 1]))
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue
            chosen  = np.random.choice(indices, size=min(k_per, len(indices)), replace=False)
            selected.extend(chosen.tolist())

    if len(selected) < cfg.VAMP_K:
        remaining = list(set(range(len(all_signals))) - set(selected))
        extra     = np.random.choice(remaining, size=cfg.VAMP_K - len(selected), replace=False)
        selected.extend(extra.tolist())

    selected = selected[:cfg.VAMP_K]
    pseudo   = torch.tensor(
        (all_signals[selected] - global_mean) / global_std,
        dtype=torch.float32,
    ).unsqueeze(1).to(device)

    with torch.no_grad():
        vamp_prior.pseudo_inputs.copy_(pseudo)

    print(f"[VAMP] Stratified pseudo-inputs: K={cfg.VAMP_K}")
    print(f"       HR  range: {all_hrs[selected].min():.1f}–{all_hrs[selected].max():.1f} bpm")
    print(f"       PTP range: {all_ptps[selected].min():.1f}–{all_ptps[selected].max():.1f}")


# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VAMP-Diff")
    parser.add_argument(
        "--config", type=str, default="configs/default.py",
        help="Path to config Python file (default: configs/default.py)",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Override OUT_DIR from config",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.out is not None:
        cfg.OUT_DIR = args.out

    main(cfg)
