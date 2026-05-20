# VAMP-Diff: PPG Signal Generation with VAE + VampPrior + Diffusion Decoder

VAMP-Diff is a generative model for photoplethysmography (PPG) signals. It combines a 1-D convolutional VAE encoder, a VampPrior for a richer latent space, and a diffusion-based U-Net decoder conditioned on the sequence latent.

## Repository Structure

```
vamp-diff/
├── train.py                  # Main training entry point
├── configs/
│   └── default.py            # All hyperparameters
├── data/
│   └── dataset.py            # CapnoBaseVAE sliding-window dataset
├── models/
│   ├── encoder.py            # SeqEncoder1D (1-D conv VAE encoder)
│   ├── decoder.py            # DiffusionUNet1D_SeqZ (FiLM-conditioned U-Net)
│   ├── vamp_prior.py         # VampPrior (mixture of posteriors prior)
│   ├── diffusion.py          # DiffusionConfig, q_sample, DDIM sampler
│   └── vae_diffusion.py      # VAEDiffusion (top-level joint model)
├── training/
│   ├── losses.py             # compute_losses, KL, spectral, derivative losses
│   ├── kl_schedules.py       # KL beta annealing schedules
│   └── trainer.py            # train_one_epoch, eval_one_epoch
├── evaluation/
│   └── metrics.py            # Reconstruction + PPG-specific metrics
└── utils/
    ├── utils.py              # Seeding, normalisation, bandpass filter
    └── visualization.py      # Training curve and reconstruction plots
```

## Installation

```bash
pip install -r requirements.txt
```

## Data Preparation

The model expects CapnoBase CSV files with `pleth_y` and `co2_y` columns, organised into `train/`, `val/`, and `test/` directories.

Update the data paths in `configs/default.py`:
```python
TRAIN_DIR = "path/to/train"
VAL_DIR   = "path/to/val"
TEST_DIR  = "path/to/test"
OUT_DIR   = Path("experiments/run_001")
```

## Training

```bash
# Using default config
python train.py

# Using a custom config
python train.py --config configs/default.py

# Override output directory
python train.py --out experiments/my_run
```

Training saves:
- `best_vae_diffusion.pt` — best validation checkpoint
- `best_vamp_prior.pt`   — corresponding VampPrior state
- `final_epoch_model.pt` — model at the end of training
- `training_curves.png`, `kl_curves.png` — loss plots
- `test_recon_metrics_BEST.csv`, `test_recon_metrics_FINAL.csv` — evaluation results

## Model Overview

### Encoder
A 2-stage strided 1-D CNN maps a PPG window `(B, 1, 3072)` to a sequence latent distribution `(B, 256, 768)`. Encoder intermediate features are passed to the decoder as compressed summaries.

### VampPrior
K learnable pseudo-inputs are encoded to define K Gaussian mixture components. The prior `p(z) = (1/K) Σ N(μ_k, σ_k²)` is richer than a standard normal, enabling sharper generation. Pseudo-inputs are optionally initialised via stratified sampling over the HR and amplitude distribution of the training set.

### Diffusion Decoder
A 1-D FiLM-conditioned U-Net denoises signals conditioned on the sequence latent z (via global pooling + spatial multi-scale injection) and diffusion timestep t. Compressed encoder summaries provide mild input-dependent guidance during training.

### Training Objective
```
L = λ_diff · L_diffusion
  + λ_recon · L_recon
  + λ_spec  · L_spectral
  + λ_deriv · L_derivative
  + λ_amp   · L_amplitude
  + λ_ptp   · L_peak-to-peak
  + β(t)    · KL[q(z|x) || p_vamp(z)]
```

KL annealing (`floor_then_ramp` schedule) and encoder freezing prevent posterior collapse in early training.

## Configuration

All hyperparameters live in `configs/default.py`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WINDOW_SIZE` | 3072 | Samples per window |
| `FS` | 300 | Sampling frequency (Hz) |
| `LATENT_CH` | 256 | Latent channel dimension |
| `VAMP_K` | 100 | VampPrior components |
| `BETA_MODE` | `floor_then_ramp` | KL annealing schedule |
| `PRED_TARGET` | `x0` | Diffusion prediction target |
| `DDIM_STEPS` | 50 | Denoising steps at inference |

## Citation

If you use this code, please cite:

```bibtex
@article{VAMP_Diff__VampPrior_Latent_Diffusion_for_Photoplethysmography_Modeling,
  title   = {VAMP-Diff: VampPrior Latent Diffusion for Photoplethysmography Modeling},
  author  = {Fatemeh Ghasemi Balouei, Nathan Willemsen, Mahesh Banavar, Bahman Moraffah},
  conference = {Asilomar Conference 2026},
  year    = {2026},
}
```
