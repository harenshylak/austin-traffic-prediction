"""
metrics.py
Evaluation metrics computed on de-normalized (mph) speed predictions.

All functions accept tensors or numpy arrays of shape (B, H, N, 1) and
return a dict keyed by horizon index, e.g. {3: {"mae": ..., "rmse": ..., "mape": ...}}.

MAPE excludes samples where |y_true| < eps to avoid division by near-zero.
"""

import numpy as np
import torch


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def denormalize(
    x: np.ndarray,
    scaler,
    speed_idx: int = 0,
) -> np.ndarray:
    """
    Inverse-scale speed predictions back to mph.

    Args:
        x        : (..., 1) array of normalized speed values
        scaler   : fitted sklearn StandardScaler for traffic (shape F)
        speed_idx: column index of speed in the scaler (default 0)

    Returns:
        array of same shape in mph
    """
    mean  = scaler.mean_[speed_idx]
    scale = scaler.scale_[speed_idx]
    return x * scale + mean


def compute_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    eval_horizons: list[int],
    eps: float = 1.0,
) -> dict[int, dict[str, float]]:
    """
    Compute MAE, RMSE, MAPE at each specified horizon step.

    Args:
        preds          : (B, H, N, 1) de-normalized predictions (mph)
        targets        : (B, H, N, 1) de-normalized ground truth (mph)
        eval_horizons  : list of 1-based horizon indices to evaluate, e.g. [3, 6, 12]
        eps            : minimum |y_true| for MAPE inclusion (mph)

    Returns:
        dict mapping horizon step → {"mae": float, "rmse": float, "mape": float}
    """
    preds   = _to_numpy(preds).squeeze(-1)    # (B, H, N)
    targets = _to_numpy(targets).squeeze(-1)  # (B, H, N)

    results = {}
    for h in eval_horizons:
        h_idx = h - 1  # 0-based
        p = preds[:, h_idx, :]    # (B, N)
        t = targets[:, h_idx, :]  # (B, N)

        err  = p - t
        mae  = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))

        valid = np.abs(t) >= eps
        mape  = float(np.mean(np.abs(err[valid] / t[valid])) * 100) if valid.any() else float("nan")

        results[h] = {"mae": mae, "rmse": rmse, "mape": mape}

    return results


def format_metrics(results: dict[int, dict[str, float]], prefix: str = "") -> str:
    """Return a human-readable table string for logging."""
    header = f"{'Horizon':>8}  {'MAE':>8}  {'RMSE':>8}  {'MAPE':>8}"
    rows = [header, "-" * len(header)]
    for h, m in sorted(results.items()):
        label = f"{h*5:3d} min"
        rows.append(
            f"{label:>8}  {m['mae']:8.3f}  {m['rmse']:8.3f}  {m['mape']:7.2f}%"
        )
    block = "\n".join(rows)
    return f"{prefix}\n{block}" if prefix else block


def aggregate_epoch_metrics(
    all_preds: list[np.ndarray],
    all_targets: list[np.ndarray],
    eval_horizons: list[int],
) -> dict[int, dict[str, float]]:
    """
    Concatenate per-batch arrays and compute metrics over the full epoch.

    Args:
        all_preds   : list of (B, H, N, 1) arrays (de-normalized)
        all_targets : list of (B, H, N, 1) arrays (de-normalized)
    """
    preds   = np.concatenate(all_preds,   axis=0)
    targets = np.concatenate(all_targets, axis=0)
    return compute_metrics(preds, targets, eval_horizons)
