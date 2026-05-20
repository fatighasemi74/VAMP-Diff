"""
KL annealing schedules for beta_kl.

All schedules take the current epoch index (0-based) and return a scalar
float beta value to weight the KL term in the ELBO.
"""


def beta_linear_warmup_then_hold(epoch: int, beta_max: float,
                                 warmup_epochs: int,
                                 start: float = 0.0) -> float:
    """
    Linearly ramp beta from `start` to `beta_max` over `warmup_epochs`,
    then hold at `beta_max`.

    Args:
        epoch:          Current epoch (0-based).
        beta_max:       Target KL weight.
        warmup_epochs:  Number of epochs to ramp over.
        start:          Initial beta value (default 0.0).
    """
    if warmup_epochs <= 0:
        return beta_max
    frac = min(1.0, epoch / warmup_epochs)
    return start + frac * (beta_max - start)


def beta_zero_then_ramp(epoch: int, warmup_epochs: int,
                        ramp_epochs: int, beta_max: float) -> float:
    """
    Keep beta = 0 for `warmup_epochs`, then linearly ramp to `beta_max`
    over `ramp_epochs`.

    Args:
        epoch:          Current epoch (0-based).
        warmup_epochs:  Number of epochs with beta = 0.
        ramp_epochs:    Number of epochs for the linear ramp.
        beta_max:       Maximum beta value.
    """
    if epoch < warmup_epochs:
        return 0.0
    t = (epoch - warmup_epochs) / max(1, ramp_epochs)
    return beta_max * max(0.0, min(1.0, t))


def beta_floor_then_ramp(epoch: int, freeze_epochs: int,
                         warmup_epochs: int, ramp_epochs: int,
                         beta_min: float, beta_max: float) -> float:
    """
    Three-phase KL schedule:
        1. [0, freeze_epochs)       → beta = 0          (encoder frozen)
        2. [freeze_epochs, warmup)  → beta = beta_min   (tiny floor)
        3. [warmup, warmup+ramp)    → log-space ramp from beta_min → beta_max

    Log-space interpolation avoids a sudden jump at the start of the ramp.

    Args:
        epoch:          Current epoch (0-based).
        freeze_epochs:  Epochs with beta = 0 (encoder frozen phase).
        warmup_epochs:  Epoch at which the ramp starts.
        ramp_epochs:    Number of epochs for the ramp.
        beta_min:       Floor beta (applied between freeze and ramp).
        beta_max:       Maximum beta.
    """
    import math

    if epoch < freeze_epochs:
        return 0.0
    if epoch < warmup_epochs:
        return beta_min

    t = (epoch - warmup_epochs) / max(1, ramp_epochs)
    t = max(0.0, min(1.0, t))

    log_min = math.log10(beta_min)
    log_max = math.log10(beta_max)
    return 10 ** (log_min + t * (log_max - log_min))


def get_beta(epoch: int, cfg) -> float:
    """
    Dispatch to the correct schedule based on cfg.BETA_MODE.

    Also enforces beta = 0 while epoch < cfg.ENC_FREEZE_EPOCHS,
    regardless of the chosen schedule.

    Args:
        epoch: Current epoch (0-based).
        cfg:   Config module (e.g. configs.default).

    Returns:
        beta: Scalar KL weight for this epoch.
    """
    mode = cfg.BETA_MODE

    if mode == "zero_then_ramp":
        beta = beta_zero_then_ramp(
            epoch,
            warmup_epochs=cfg.WARMUP_EPOCHS,
            ramp_epochs=cfg.RAMP_EPOCHS,
            beta_max=cfg.BETA_KL_MAX,
        )
    elif mode == "warmup_then_hold":
        beta = beta_linear_warmup_then_hold(
            epoch,
            beta_max=cfg.BETA_KL_MAX,
            warmup_epochs=cfg.WARMUP_EPOCHS,
            start=0.0,
        )
    elif mode == "floor_then_ramp":
        beta = beta_floor_then_ramp(
            epoch,
            freeze_epochs=cfg.ENC_FREEZE_EPOCHS,
            warmup_epochs=cfg.WARMUP_EPOCHS,
            ramp_epochs=cfg.RAMP_EPOCHS,
            beta_min=cfg.BETA_KL_MIN,
            beta_max=cfg.BETA_KL_MAX,
        )
    else:
        raise ValueError(f"Unknown BETA_MODE: {mode!r}")

    # Hard override during encoder-freeze phase
    if epoch < cfg.ENC_FREEZE_EPOCHS:
        beta = 0.0

    return beta
