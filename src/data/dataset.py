"""
dataset.py
PyTorch Dataset + DataLoader for the preprocessed sliding-window tensors.

Loads memory-mapped numpy arrays from data/graph/ and computes sliding windows
on-the-fly to avoid materialising all windows in RAM.

Each sample is a dict:
  traffic    (T, N, F)  — normalized speed / volume / occupancy
  context    (T, K)     — normalized weather + calendar + event features
  adjacency  (N, N)     — static Gaussian-kernel road-network adjacency
  target     (H, N, 1)  — normalized ground-truth speed (for inverse-scaling later)
  event_flag (1,)       — 1 if any special event is active in the input window

Usage:
    from src.data.dataset import make_dataloaders
    loaders = make_dataloaders("configs/default.yaml")
    for batch in loaders["train"]: ...
"""

import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TrafficDataset(Dataset):
    """
    Sliding-window dataset over pre-processed traffic arrays.

    Args:
        traffic:   (T_split, N, F) float32 array
        context:   (T_split, K)   float32 array
        adj:       (N, N)         float32 array (shared, not split)
        T:         input window length (timesteps)
        H:         prediction horizon (timesteps)
        speed_idx: column index of speed in F dimension (default 0)
    """

    def __init__(
        self,
        traffic: np.ndarray,
        context: np.ndarray,
        adj: np.ndarray,
        T: int,
        H: int,
        K: int | None = None,
        speed_idx: int = 0,
    ):
        self.traffic   = traffic    # (T_split, N, F)
        self.context   = context    # (T_split, K_full)
        self.adj       = torch.from_numpy(adj).float()
        self.T         = T
        self.H         = H
        self.K         = K if K is not None else context.shape[1]  # cols to expose
        self.speed_idx = speed_idx

        # Number of valid windows
        self.n_windows = len(traffic) - T - H + 1
        if self.n_windows <= 0:
            raise ValueError(
                f"Split too short ({len(traffic)} steps) for T={T} + H={H}. "
                "Check split boundaries in config."
            )

        # Pre-compute event_flag per position: 1 if any event_flag > 0 in [i, i+T)
        # Event flag is always at absolute col index 13 (weather=6, calendar=7, then events).
        # If the full context doesn't reach col 13, no events → flags all zero.
        if context.shape[1] > 13:
            event_col = context[:, 13]  # absolute index; safe regardless of K
        else:
            event_col = np.zeros(len(context), dtype=np.float32)
        self._event_flags = self._compute_window_events(event_col, T)

    @staticmethod
    def _compute_window_events(event_col: np.ndarray, T: int) -> np.ndarray:
        """Rolling sum > 0 for window [i, i+T). Shape: (n_windows,)."""
        cum = np.cumsum(np.concatenate([[0], event_col]))
        n = len(event_col) - T + 1
        window_sums = cum[T:n + T] - cum[:n]
        return (window_sums > 0).astype(np.float32)

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        t0 = idx
        t1 = idx + self.T          # exclusive end of input
        t2 = idx + self.T + self.H # exclusive end of target

        traffic_window = torch.from_numpy(self.traffic[t0:t1]).float()           # (T, N, F)
        context_window = torch.from_numpy(self.context[t0:t1, :self.K]).float() # (T, K)
        target_window  = torch.from_numpy(                                 # (H, N, 1)
            self.traffic[t1:t2, :, self.speed_idx : self.speed_idx + 1]
        ).float()
        event_flag = torch.tensor([self._event_flags[idx]], dtype=torch.float32)  # (1,)

        return {
            "traffic":    traffic_window,
            "context":    context_window,
            "adjacency":  self.adj,
            "target":     target_window,
            "event_flag": event_flag,
        }


def _load_split(
    graph_dir: str,
    split_indices: dict,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Memory-map the full arrays and slice the requested split."""
    traffic_full = np.load(os.path.join(graph_dir, "traffic.npy"), mmap_mode="r")
    context_full = np.load(os.path.join(graph_dir, "context.npy"), mmap_mode="r")

    s, e = split_indices[split_name]
    # +1 because we need H extra steps after the split end for the last window's target
    return traffic_full[s : e + 1], context_full[s : e + 1]


def make_dataloaders(
    config_path: str = "configs/default.yaml",
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    """
    Build train / val / test DataLoaders from a config file.

    Returns:
        dict with keys "train", "val", "test"
    """
    import yaml  # lazy import to avoid top-level dep in non-config usage

    def _load(path):
        with open(path) as f:
            return yaml.safe_load(f)

    def _merge(base, override):
        merged = base.copy()
        for k, v in override.items():
            if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
                merged[k] = _merge(merged[k], v)
            else:
                merged[k] = v
        return merged

    cfg = _load(config_path)
    if "_base_" in cfg:
        base_path = os.path.join(os.path.dirname(config_path), cfg.pop("_base_"))
        cfg = _merge(_load(base_path), cfg)

    graph_dir  = cfg["paths"]["graph_dir"]
    T          = cfg["data"]["T"]
    H          = cfg["data"]["H"]
    K          = cfg["data"].get("K", None)   # None = use all context cols
    batch_size = cfg["training"]["batch_size"]

    # Load shared data
    adj = np.load(os.path.join(graph_dir, "adj_matrix.npy")).astype(np.float32)
    with open(os.path.join(graph_dir, "split_indices.json")) as f:
        split_indices = json.load(f)

    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        s, e = split_indices[split]
        # Slice needs extra H rows for the last window's target
        traffic_full = np.load(os.path.join(graph_dir, "traffic.npy"), mmap_mode="r")
        context_full = np.load(os.path.join(graph_dir, "context.npy"), mmap_mode="r")

        # For train/val: target can spill into the next split by at most H steps.
        # We allow this since splits are contiguous and normalization is uniform.
        t_end = min(e + H, len(traffic_full) - 1)
        traffic_split = np.array(traffic_full[s : t_end + 1])
        context_split = np.array(context_full[s : t_end + 1])

        dataset = TrafficDataset(traffic_split, context_split, adj, T, H, K=K)

        shuffle = (split == "train")
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        print(f"{split:5s}: {len(dataset):,} windows, {len(loaders[split]):,} batches "
              f"(batch_size={batch_size})")

    return loaders


if __name__ == "__main__":
    # Quick smoke test
    print("Building DataLoaders...")
    loaders = make_dataloaders()

    for split, loader in loaders.items():
        batch = next(iter(loader))
        print(f"\n{split} batch shapes:")
        for k, v in batch.items():
            print(f"  {k:12s}: {tuple(v.shape)}")
