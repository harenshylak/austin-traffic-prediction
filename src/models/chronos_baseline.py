"""
chronos_baseline.py  —  Amazon Chronos-T5 Foundation Model Baseline
Zero-shot evaluation of Chronos-T5-Base on the Austin traffic test set.

Chronos is a pretrained time-series foundation model from Amazon that can
forecast any univariate time series without fine-tuning. We use it here as
a zero-shot baseline — the model has never seen Austin traffic data.

Protocol (matches all other models):
  - Same test split (Jul–Sep 2021)
  - Same T=12 input steps (60 min of history)
  - Same H=12 forecast steps, evaluated at horizons [3, 6, 12]
  - Per-sensor univariate forecasting (speed only, feature 0)
  - Predictions de-normalized via scaler_traffic.pkl

Chronos variant: amazon/chronos-t5-base
  - 200M parameters
  - Pretrained on a large corpus of diverse time series
  - Returns a probabilistic forecast; we use the median (quantile 0.5)

Usage:
    python -m src.models.chronos_baseline
    python -m src.models.chronos_baseline --batch-windows 256
"""

import argparse
import json
import os
import pickle
import time
from datetime import datetime

import numpy as np
import torch
from chronos import ChronosPipeline

from src.training.metrics import aggregate_epoch_metrics, denormalize, format_metrics


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Chronos does not run efficiently on MPS — always use CPU
    return torch.device("cpu")


def evaluate(
    config_path: str = "configs/default.yaml",
    model_id: str = "amazon/chronos-t5-base",
    batch_windows: int = 256,
    n_eval_windows: int = 200,
):
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    graph_dir     = cfg["paths"]["graph_dir"]
    T             = cfg["data"]["T"]
    H             = cfg["data"]["H"]
    eval_horizons = cfg["data"]["eval_horizons"]

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading data...")
    traffic_full = np.load(os.path.join(graph_dir, "traffic.npy"), mmap_mode="r")
    with open(os.path.join(graph_dir, "split_indices.json")) as f:
        splits = json.load(f)
    with open(os.path.join(graph_dir, "scaler_traffic.pkl"), "rb") as f:
        scaler = pickle.load(f)

    train_s, train_e = splits["train"]
    test_s,  test_e  = splits["test"]

    # Speed only (feature 0), normalized
    train_speed = np.array(traffic_full[train_s : train_e + 1, :, 0])  # (T_train, N)
    test_speed  = np.array(traffic_full[test_s  : test_e  + 1, :, 0])  # (T_test, N)

    N         = train_speed.shape[1]
    n_windows_total = len(test_speed) - T - H + 1

    # Sub-sample evenly for efficiency — 2000 windows is statistically sufficient
    if n_eval_windows and n_eval_windows < n_windows_total:
        step = n_windows_total // n_eval_windows
        eval_indices = list(range(0, n_windows_total, step))[:n_eval_windows]
    else:
        eval_indices = list(range(n_windows_total))
    n_windows = len(eval_indices)
    print(f"  Total test windows: {n_windows_total:,}  |  Evaluating: {n_windows:,}  |  Sensors: {N}  |  Model: {model_id}")

    # ------------------------------------------------------------------
    # Load Chronos
    # ------------------------------------------------------------------
    device = get_device()
    print(f"\nLoading {model_id} on {device}...")
    pipeline = ChronosPipeline.from_pretrained(
        model_id,
        device_map=str(device),
        torch_dtype=torch.float32,
    )
    print("Model loaded.")

    # ------------------------------------------------------------------
    # Context: use last ctx_len train steps prepended to test.
    # 120 steps = 10 hours — enough for Chronos to see daily patterns
    # without overloading MPS memory on long sequences.
    # ------------------------------------------------------------------
    ctx_len   = min(120, len(train_speed))
    context   = np.concatenate([train_speed[-ctx_len:], test_speed], axis=0)  # (ctx+T_test, N)

    # ------------------------------------------------------------------
    # Rolling forecast — iterate over eval_indices in batches
    # ------------------------------------------------------------------
    all_preds   = np.zeros((n_windows, H, N, 1), dtype=np.float32)
    all_targets = np.zeros((n_windows, H, N, 1), dtype=np.float32)

    print(f"\nRunning rolling forecasts (batch_windows={batch_windows}, N={N})...")
    t0 = time.time()

    for b_start in range(0, n_windows, batch_windows):
        b_end      = min(b_start + batch_windows, n_windows)
        batch_size = b_end - b_start
        batch_idx  = eval_indices[b_start:b_end]

        if b_start % 100 == 0:
            elapsed = time.time() - t0
            rate    = (b_start + 1) / max(elapsed, 1)
            eta     = (n_windows - b_start) / max(rate, 1e-6)
            print(f"  Window {b_start:4d}/{n_windows} | "
                  f"{elapsed:.0f}s elapsed | ETA {eta:.0f}s", end="\r")

        # One tensor per (window, sensor): history up to window start
        input_tensors = []
        for i in batch_idx:
            t1 = i + T
            for n in range(N):
                series = context[:t1, n].astype(np.float32)
                input_tensors.append(torch.from_numpy(series))

        # Chronos: (batch_size*N, num_samples, H)
        with torch.no_grad():
            forecast = pipeline.predict(
                inputs=input_tensors,
                prediction_length=H,
                num_samples=5,
                limit_prediction_length=False,
            )

        # Median across samples → (batch_size*N, H)
        median_forecast = np.median(forecast.numpy(), axis=1)
        median_forecast = median_forecast.reshape(batch_size, N, H)  # (batch, N, H)

        for bi, i in enumerate(batch_idx):
            t1 = i + T
            t2 = t1 + H
            all_preds[b_start + bi, :, :, 0]   = median_forecast[bi].T
            all_targets[b_start + bi, :, :, 0] = context[t1:t2, :]

    print(f"\n  Completed {n_windows} windows in {time.time()-t0:.1f}s.")

    # ------------------------------------------------------------------
    # De-normalize and compute metrics
    # ------------------------------------------------------------------
    preds_denorm   = denormalize(all_preds,   scaler)
    targets_denorm = denormalize(all_targets, scaler)

    metrics = aggregate_epoch_metrics(
        [preds_denorm], [targets_denorm], eval_horizons
    )

    print("\n" + "=" * 50)
    print(f"TEST RESULTS  (Chronos-T5-Base, zero-shot)")
    print("=" * 50)
    print(format_metrics(metrics))
    print("=" * 50)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    os.makedirs("results", exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_name = model_id.split("/")[-1].replace("-", "_")
    result = {
        "run_id":       run_id,
        "model":        short_name,
        "model_id":     model_id,
        "zero_shot":    True,
        "config":       config_path,
        "n_sensors":    N,
        "context_steps": ctx_len,
        "test_metrics": {
            str(h * 5) + "min": {k: round(v, 4) for k, v in m.items()}
            for h, m in metrics.items()
        },
    }
    result_path = os.path.join("results", f"chronos_{run_id}.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {result_path}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        default="configs/default.yaml")
    parser.add_argument("--model-id",      default="amazon/chronos-t5-base")
    parser.add_argument("--batch-windows", type=int, default=256,
                        help="Number of windows to process per Chronos call")
    args = parser.parse_args()
    evaluate(
        config_path=args.config,
        model_id=args.model_id,
        batch_windows=args.batch_windows,
    )
