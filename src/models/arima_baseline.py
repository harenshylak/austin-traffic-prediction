"""
arima_baseline.py  —  ARIMA classical baseline
Fits one ARIMA(p,d,q) model per sensor on training data, then evaluates
on the test set using the same sliding-window protocol as the deep learning models.

Evaluation protocol (matches trainer.py):
  - Same T=12 input, H=12 horizon, eval at steps [3, 6, 12]
  - Same test split indices from split_indices.json
  - Metrics de-normalized via scaler_traffic.pkl
  - Results saved to results/arima_<timestamp>.json

Usage:
    python -m src.models.arima_baseline
    python -m src.models.arima_baseline --order 2 1 2
"""

import argparse
import json
import os
import pickle
import time
from datetime import datetime

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from src.training.metrics import aggregate_epoch_metrics, denormalize, format_metrics


def fit_arima_per_sensor(
    train_speed: np.ndarray,
    order: tuple[int, int, int],
) -> list:
    """
    Fit one ARIMA model per sensor on training data.

    Args:
        train_speed : (T_train, N) normalized speed array
        order       : (p, d, q) ARIMA order

    Returns:
        list of N fitted ARIMA results
    """
    N = train_speed.shape[1]
    fitted_models = []
    for n in range(N):
        print(f"  Fitting sensor {n+1}/{N}...", end="\r")
        series = train_speed[:, n].astype(np.float64)
        model  = ARIMA(series, order=order)
        result = model.fit()
        fitted_models.append(result)
    print(f"  Fitted {N} sensors.           ")
    return fitted_models


def rolling_forecast(
    fitted_models: list,
    test_speed: np.ndarray,
    T: int,
    H: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sliding-window H-step forecasts using fitted ARIMA models.

    For each window starting at position i:
      - Condition on train history + test[:i+T]
      - Forecast steps i+T through i+T+H-1

    Args:
        fitted_models : list of N fitted ARIMA results
        test_speed    : (T_test_extended, N) normalized speed (includes train history
                        prepended so ARIMA has enough context)
        T             : input window length
        H             : forecast horizon

    Returns:
        preds   : (n_windows, H, N, 1)
        targets : (n_windows, H, N, 1)
    """
    n_windows = len(test_speed) - T - H + 1
    N = test_speed.shape[1]

    all_preds   = np.zeros((n_windows, H, N, 1), dtype=np.float32)
    all_targets = np.zeros((n_windows, H, N, 1), dtype=np.float32)

    t0 = time.time()
    for i in range(n_windows):
        if i % 500 == 0:
            elapsed = time.time() - t0
            rate    = i / max(elapsed, 1)
            eta     = (n_windows - i) / max(rate, 1e-6)
            print(f"  Window {i:5d}/{n_windows} | {elapsed:.0f}s elapsed | ETA {eta:.0f}s", end="\r")

        t1 = i + T   # first target step
        t2 = t1 + H  # exclusive end of target

        for n, res in enumerate(fitted_models):
            # Extend model with new observations up to t1 and forecast H steps
            history = test_speed[:t1, n].astype(np.float64)
            updated = res.apply(history)
            forecast = updated.forecast(steps=H)
            all_preds[i, :, n, 0]   = forecast.astype(np.float32)
            all_targets[i, :, n, 0] = test_speed[t1:t2, n].astype(np.float32)

    print(f"  Completed {n_windows} windows in {time.time()-t0:.1f}s.          ")
    return all_preds, all_targets


def evaluate(config_path: str = "configs/default.yaml", order: tuple = (2, 1, 2)):
    import yaml

    def _load(p):
        with open(p) as f:
            return yaml.safe_load(f)

    cfg = _load(config_path)

    graph_dir    = cfg["paths"]["graph_dir"]
    T            = cfg["data"]["T"]
    H            = cfg["data"]["H"]
    eval_horizons = cfg["data"]["eval_horizons"]

    # Load data
    print("Loading data...")
    traffic_full = np.load(os.path.join(graph_dir, "traffic.npy"), mmap_mode="r")
    with open(os.path.join(graph_dir, "split_indices.json")) as f:
        splits = json.load(f)
    with open(os.path.join(graph_dir, "scaler_traffic.pkl"), "rb") as f:
        scaler = pickle.load(f)

    # Speed only (feature 0)
    train_s, train_e = splits["train"]
    test_s,  test_e  = splits["test"]

    train_speed = np.array(traffic_full[train_s:train_e + 1, :, 0])  # (T_train, N)

    # For rolling forecast we prepend training data so the model has full history
    # at each test window. Use last 500 train steps as context (saves RAM/time).
    ctx_len     = min(500, len(train_speed))
    context     = train_speed[-ctx_len:]                              # (ctx, N)
    test_speed_raw = np.array(traffic_full[test_s:test_e + 1, :, 0]) # (T_test, N)
    full_series = np.concatenate([context, test_speed_raw], axis=0)   # (ctx+T_test, N)

    # Fit per sensor
    print(f"\nFitting ARIMA{order} per sensor on {len(train_speed):,} training steps...")
    fitted = fit_arima_per_sensor(train_speed, order)

    # Rolling forecast on test windows
    print(f"\nRunning rolling {H}-step forecasts on test set...")
    preds, targets = rolling_forecast(fitted, full_series, T, H)

    # De-normalize and compute metrics
    preds_denorm   = denormalize(preds,   scaler)
    targets_denorm = denormalize(targets, scaler)

    metrics = aggregate_epoch_metrics(
        [preds_denorm], [targets_denorm], eval_horizons
    )

    print("\n" + "=" * 50)
    print(f"TEST RESULTS  (ARIMA{order})")
    print("=" * 50)
    print(format_metrics(metrics))
    print("=" * 50)

    # Save results
    os.makedirs("results", exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "run_id":       run_id,
        "model":        f"arima_{order[0]}_{order[1]}_{order[2]}",
        "arima_order":  list(order),
        "config":       config_path,
        "n_sensors":    train_speed.shape[1],
        "test_metrics": {
            str(h * 5) + "min": {k: round(v, 4) for k, v in m.items()}
            for h, m in metrics.items()
        },
    }
    result_path = os.path.join("results", f"arima_{run_id}.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {result_path}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--order", nargs=3, type=int, default=[2, 1, 2],
                        metavar=("p", "d", "q"),
                        help="ARIMA order (default: 2 1 2)")
    args = parser.parse_args()
    evaluate(config_path=args.config, order=tuple(args.order))
