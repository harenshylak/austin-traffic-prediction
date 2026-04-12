"""
trainer.py
Generic training loop compatible with any model that takes a batch dict and
returns (B, H, N, 1) speed predictions.

Features:
  - AdamW + ReduceLROnPlateau
  - Gradient clipping
  - Early stopping on val MAE
  - Best-checkpoint saving
  - Optional wandb logging
  - MPS / CUDA / CPU device selection

Usage:
    python -m src.training.trainer --config configs/experiment/lstm_only.yaml
"""

import argparse
import json
import os
import pickle
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.data.dataset import make_dataloaders
from src.training.metrics import aggregate_epoch_metrics, denormalize, format_metrics


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def merge_configs(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    merged = base.copy()
    for k, v in override.items():
        if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
            merged[k] = merge_configs(merged[k], v)
        else:
            merged[k] = v
    return merged


def resolve_config(config_path: str) -> dict:
    """Load config, resolving _base_ inheritance if present."""
    cfg = load_config(config_path)
    if "_base_" in cfg:
        base_path = os.path.join(os.path.dirname(config_path), cfg.pop("_base_"))
        base = load_config(base_path)
        cfg = merge_configs(base, cfg)
    return cfg


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(cfg: dict, N: int = None) -> nn.Module:
    model_name = cfg["model"].get("name", "lstm_only")
    if model_name == "lstm_only":
        from src.models.lstm_baseline import build_from_config
        return build_from_config(cfg)
    if model_name == "gcn_lstm":
        from src.models.gcn_lstm import build_from_config
        return build_from_config(cfg)
    if model_name == "lstm_context":
        from src.models.lstm_context import build_from_config
        return build_from_config(cfg)
    if model_name == "full_hmstgn":
        from src.models.hmstgn import build_from_config
        if N is None:
            raise ValueError("N (number of sensors) must be provided for full_hmstgn")
        return build_from_config(cfg, N)
    raise ValueError(f"Unknown model name: {model_name!r}. Add it to trainer.build_model().")


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader,
    optimizer,
    scaler_traffic,
    eval_horizons: list[int],
    device: torch.device,
    is_train: bool,
    teacher_forcing_ratio: float = 0.5,
) -> tuple[float, dict]:
    """
    Run one full epoch.

    Returns:
        mean_loss : average Huber loss over all batches
        metrics   : dict from aggregate_epoch_metrics (de-normalized)
    """
    model.train(is_train)

    total_loss = 0.0
    all_preds, all_targets = [], []

    # Detect model type by forward signature
    import inspect
    sig_params     = set(inspect.signature(model.forward).parameters.keys())
    is_hmstgn      = "event_flag" in sig_params               # full HM-STGN
    is_lstm_ctx    = "context" in sig_params and not is_hmstgn # LSTMWithContext
    needs_adj      = "adjacency" in sig_params and not is_hmstgn  # GCN-based

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            traffic    = batch["traffic"].to(device)    # (B, T, N, F)
            target     = batch["target"].to(device)     # (B, H, N, 1)

            if is_hmstgn:
                context    = batch["context"].to(device)     # (B, T, K)
                adjacency  = batch["adjacency"].to(device)   # (B, N, N)
                event_flag = batch["event_flag"].to(device)  # (B, 1)

                out  = model(traffic, context, adjacency, event_flag, target=target)
                loss = out["loss"]
                preds = out["pred_speed"]

            elif is_lstm_ctx:
                context = batch["context"].to(device)        # (B, T, K)
                tfr = teacher_forcing_ratio if is_train else 0.0
                preds = model(traffic, context, target=target, teacher_forcing_ratio=tfr)
                loss = nn.HuberLoss(delta=1.0)(preds, target)

            elif needs_adj:
                adjacency = batch["adjacency"].to(device)
                tfr = teacher_forcing_ratio if is_train else 0.0
                preds = model(traffic, adjacency, target=target, teacher_forcing_ratio=tfr)
                loss = nn.HuberLoss(delta=1.0)(preds, target)

            else:
                tfr = teacher_forcing_ratio if is_train else 0.0
                preds = model(traffic, target=target, teacher_forcing_ratio=tfr)
                loss = nn.HuberLoss(delta=1.0)(preds, target)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += loss.item()

            # De-normalize for metrics
            p_np = denormalize(preds.detach().cpu().numpy(),  scaler_traffic)
            t_np = denormalize(target.detach().cpu().numpy(), scaler_traffic)
            all_preds.append(p_np)
            all_targets.append(t_np)

    mean_loss = total_loss / len(loader)
    metrics   = aggregate_epoch_metrics(all_preds, all_targets, eval_horizons)
    return mean_loss, metrics


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(config_path: str) -> nn.Module:
    cfg    = resolve_config(config_path)
    device = get_device()
    print(f"Device: {device}")

    # Paths
    graph_dir      = cfg["paths"]["graph_dir"]
    checkpoint_dir = cfg["paths"]["checkpoint_dir"]
    results_dir    = "results"
    model_name     = cfg["model"].get("name", "lstm_only")
    ckpt_path      = os.path.join(checkpoint_dir, f"{model_name}_best.pt")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Data
    print("Loading data...")
    loaders = make_dataloaders(config_path)
    with open(os.path.join(graph_dir, "scaler_traffic.pkl"), "rb") as f:
        scaler_traffic = pickle.load(f)

    eval_horizons = cfg["data"]["eval_horizons"]

    # Model (need N = number of sensors for HM-STGN)
    import json as _json
    with open(os.path.join(graph_dir, "sensor_order.json")) as f:
        N = len(_json.load(f))
    model = build_model(cfg, N=N).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_name}  |  Parameters: {n_params:,}")

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg["training"]["lr_scheduler"]["factor"],
        patience=cfg["training"]["lr_scheduler"]["patience"],
    )

    # Training loop
    max_epochs = cfg["training"]["max_epochs"]
    patience   = cfg["training"]["early_stopping_patience"]
    best_val_mae = float("inf")
    patience_counter = 0
    epoch_log = []
    run_start = datetime.now()

    print(f"\nTraining for up to {max_epochs} epochs (early stop patience={patience})...\n")

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        train_loss, train_metrics = run_epoch(
            model, loaders["train"], optimizer, scaler_traffic,
            eval_horizons, device, is_train=True,
        )
        val_loss, val_metrics = run_epoch(
            model, loaders["val"], optimizer, scaler_traffic,
            eval_horizons, device, is_train=False,
        )

        val_mae = val_metrics[eval_horizons[0]]["mae"]  # primary: shortest horizon MAE
        scheduler.step(val_mae)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{max_epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_MAE@{eval_horizons[0]*5}min={val_mae:.3f} mph | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s"
        )

        # Log epoch
        epoch_log.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss, 6),
            "val_mae":    round(val_mae, 4),
            "lr":         optimizer.param_groups[0]["lr"],
            "elapsed_s":  round(elapsed, 1),
        })

        # Checkpoint
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_mae":     val_mae,
                "cfg":         cfg,
            }, ckpt_path)
            print(f"  ✓ New best val MAE: {best_val_mae:.3f} mph — checkpoint saved")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping after {epoch} epochs (no improvement for {patience} epochs).")
                break

    # --- Final evaluation on test set ---
    print(f"\nLoading best checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    _, test_metrics = run_epoch(
        model, loaders["test"], optimizer, scaler_traffic,
        eval_horizons, device, is_train=False,
    )

    print("\n" + "=" * 50)
    print(f"TEST RESULTS  ({model_name})")
    print("=" * 50)
    print(format_metrics(test_metrics))
    print("=" * 50)

    # --- Save results JSON ---
    run_id = run_start.strftime("%Y%m%d_%H%M%S")
    result = {
        "run_id":        run_id,
        "model":         model_name,
        "config":        config_path,
        "best_epoch":    ckpt["epoch"],
        "best_val_mae":  round(best_val_mae, 4),
        "n_params":      n_params,
        "epochs_run":    len(epoch_log),
        "splits":        cfg["data"]["splits"],
        "test_metrics":  {
            str(h * 5) + "min": {k: round(v, 4) for k, v in m.items()}
            for h, m in test_metrics.items()
        },
        "epoch_log":     epoch_log,
    }
    result_path = os.path.join(results_dir, f"{model_name}_{run_id}.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {result_path}")

    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiment/lstm_only.yaml",
        help="Path to experiment config (inherits from default.yaml via _base_)",
    )
    args = parser.parse_args()
    train(args.config)
