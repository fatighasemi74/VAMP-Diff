"""
Default configuration for VAMP-Diff training.
All hyperparameters are defined here and imported throughout the codebase.
Override by passing --config path/to/your_config.py to train.py.
"""

from pathlib import Path

# =========================================================
# Data
# =========================================================
WINDOW_SIZE = 3072
STEP_SIZE   = 600
FS          = 300

TRAIN_DIR = "data/train"
VAL_DIR   = "data/val"
TEST_DIR  = "data/test"

OUT_DIR = Path("experiments/run_001")

MAX_TRAIN_SAMPLES = 6815   # cap training set size; set to None to use all

# =========================================================
# Model architecture
# =========================================================
# Encoder
BASE_CH    = 64    # base channel width
LATENT_CH  = 256   # sequence latent channels
LATENT_T   = WINDOW_SIZE // 4   # 3072 -> 768 after 2 stride-2 downsamples

ENC_RES_PER_STAGE = 2   # residual blocks per encoder stage

# Decoder (diffusion U-Net)
DIFF_BASE  = 64
TIME_DIM   = 128
N_RES      = 2     # residual blocks per decoder stage

# Skip connections from encoder to decoder
SKIP_DROPOUT   = 0.5
SKIP_GATE_INIT = 0.01   # gates start near zero — decoder learns from z first

# Latent conditioning
USE_LATENT_CONDITIONING = True   # set False to ablate z conditioning

# =========================================================
# Training
# =========================================================
BATCH_SIZE   = 32
EPOCHS       = 200
LR           = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 1.0

LATENT_NOISE_SCALE = 0.05   # noise added to z during training
Z_PERTURB_SCALE    = 0.01   # perturbation scale when SAMPLE_Z_IN_TRAIN=False
SAMPLE_Z_IN_TRAIN  = True   # stochastic posterior during training
USE_MU_FOR_RECON   = False  # deterministic (mu) path for evaluation

# =========================================================
# Diffusion schedule
# =========================================================
DIFF_T      = 100
BETA_START  = 1e-4
BETA_END    = 0.02
SCHEDULE    = "linear"   # "linear" or "cosine"

USE_DDIM   = True
DDIM_STEPS = 50
DDIM_ETA   = 0.0

PRED_TARGET = "x0"   # "eps" or "x0"

# =========================================================
# Loss weights
# =========================================================
LAMBDA_DIFF  = 1.0
LAMBDA_SPEC  = 0.1
LAMBDA_DERIV = 0.1
LAMBDA_RECON = 5.0
LAMBDA_AMP   = 2.0
LAMBDA_PTP   = 1.0

# =========================================================
# KL annealing
# =========================================================
ENC_FREEZE_EPOCHS = 20     # encoder frozen for this many epochs at start

BETA_MODE    = "floor_then_ramp"   # "zero_then_ramp" | "warmup_then_hold" | "floor_then_ramp"
WARMUP_EPOCHS = 50
RAMP_EPOCHS   = 80
BETA_KL_MAX   = 5e-8
BETA_KL_MIN   = 1e-8    # floor beta applied after freeze phase

FREE_BITS_NATS = 0.5    # used only when vamp_prior is None

# =========================================================
# VampPrior
# =========================================================
VAMP_K = 100   # number of pseudo-inputs

# Stratified pseudo-input initialisation bins
HR_BINS  = [59, 70, 80, 90, 100, 115, 150]
AMP_BINS = [2.5, 7.5, 11.0, 15.0, 21.0]

# Optional: paths to pre-computed HR/amplitude features for stratified init.
# If None, pseudo-inputs are initialised to zeros.
VAMP_HR_NPY  = None   # e.g. "analysis/all_hrs.npy"
VAMP_PTP_NPY = None   # e.g. "analysis/all_ptps.npy"

# Optional: pre-trained checkpoint to fine-tune from
FINETUNE_CKPT = None   # e.g. "checkpoints/pretrained.pt"
