"""
preprocess.py
Transforms raw CSVs into model-ready arrays following the 7-step pipeline:

  1. Load sensor data, aggregate lanes → intersections, resample to 5-min,
     forward-fill gaps ≤ gap_fill_limit steps; drop sensors above missing_threshold
  2. Forward-fill hourly weather to 5-min resolution
  3. Compute cyclic calendar features (hour, dow, month sin/cos, is_weekend)
  4. Join binary event flags and impact level from events CSV
  5. Normalize with StandardScaler fit on train split only
  6. Chronological split → record index boundaries
  7. Save arrays + metadata to data/processed/

Outputs (data/processed/):
  traffic.npy         (T, N, 3)  float32  speed / volume / occupancy
  context.npy         (T, K)     float32  weather + calendar + event features
  timestamps.npy      (T,)       datetime64[ns]
  adj_matrix.npy      (N, N)     float32  Gaussian-kernel road-network distances
  sensor_order.json   list of int_ids in N-axis order
  split_indices.json  {train/val/test: [start, end]} inclusive index pairs
  scaler_traffic.pkl  sklearn StandardScaler (fitted on train)
  scaler_context.pkl  sklearn StandardScaler (fitted on train)

Usage:
    python src/data/preprocess.py
    python src/data/preprocess.py --config configs/default.yaml
"""

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Step 1 — Sensor data
# ---------------------------------------------------------------------------

def load_sensor_data(
    raw_dir: str,
    full_index: pd.DatetimeIndex,
    gap_fill_limit: int,
    missing_threshold: float,
) -> tuple[np.ndarray, list[int]]:
    """
    Two-step resample: fill gaps at 15-min resolution first, then upsample to 5-min.

    gap_fill_limit: max consecutive missing 15-min readings to forward-fill (e.g. 4 = 1 hour).
    missing_threshold: drop sensor if fraction missing at 15-min level exceeds this after fill.

    Returns:
        traffic: (T, N, 3) float32 — speed, volume, occupancy
        sensor_order: list of int_ids kept (N entries)
    """
    print("Step 1: Loading sensor data...")
    df = pd.read_csv(
        os.path.join(raw_dir, "radar_traffic.csv"),
        parse_dates=["datetime"],
        dtype={"int_id": int, "detid": int},
    )

    # Aggregate detector lanes → intersection level
    agg = (
        df.groupby(["datetime", "int_id"], as_index=False)
        .agg(speed=("speed", "mean"), volume=("volume", "sum"), occupancy=("occupancy", "mean"))
    )

    # Reference 15-min index covering the same span as full_index
    idx_15 = pd.date_range(full_index[0], full_index[-1], freq="15min")

    sensor_ids = sorted(agg["int_id"].unique().tolist())
    T = len(full_index)
    traffic_buf = np.full((T, len(sensor_ids), 3), np.nan, dtype=np.float32)

    kept_indices, kept_ids = [], []
    for i, sid in enumerate(sensor_ids):
        s = (
            agg[agg["int_id"] == sid]
            .set_index("datetime")[["speed", "volume", "occupancy"]]
        )
        s = s[~s.index.duplicated(keep="first")]

        # Step 1a: fill real data gaps at 15-min resolution
        s15 = s.reindex(idx_15).ffill(limit=gap_fill_limit)

        missing_frac = float(s15.isna().any(axis=1).mean())
        if missing_frac > missing_threshold:
            print(f"  Dropping sensor {sid}: {missing_frac:.1%} missing at 15-min "
                  f"(threshold {missing_threshold:.0%})")
            continue

        # Step 1b: upsample 15-min → 5-min (fills the 2 intermediate steps per slot)
        s5 = s15.reindex(full_index).ffill(limit=2)

        # Replace any remaining NaN (edge of dataset) with column median
        for col_idx, col in enumerate(["speed", "volume", "occupancy"]):
            col_data = s5[col].values.astype(np.float32)
            nan_mask = np.isnan(col_data)
            if nan_mask.any():
                col_data[nan_mask] = np.nanmedian(col_data)
            traffic_buf[:, i, col_idx] = col_data

        kept_indices.append(i)
        kept_ids.append(sid)

    if not kept_ids:
        raise RuntimeError("All sensors dropped — lower missing_threshold or check raw data.")

    traffic = traffic_buf[:, kept_indices, :]
    print(f"  Kept {len(kept_ids)}/{len(sensor_ids)} sensors: {kept_ids}")
    print(f"  Traffic array: {traffic.shape}")
    return traffic, kept_ids


# ---------------------------------------------------------------------------
# Step 2 — Weather
# ---------------------------------------------------------------------------

WEATHER_COLS = ["temp_f", "precip_in", "wind_mph", "humidity_pct", "visibility_m", "weather_code"]


def load_weather(raw_dir: str, full_index: pd.DatetimeIndex) -> np.ndarray:
    """Returns (T, 6) float32."""
    print("Step 2: Loading weather data...")
    df = pd.read_csv(
        os.path.join(raw_dir, "weather.csv"),
        index_col=0,
        parse_dates=True,
    )
    df = df[WEATHER_COLS]

    # Forward-fill hourly → 5-min (each hour covers 12 steps; limit=12 prevents
    # propagating across multi-hour outages in the weather data)
    weather = df.reindex(full_index).ffill(limit=12).bfill(limit=12)

    # Remaining NaN: column median, or 0 if entire column is NaN
    for col in WEATHER_COLS:
        col_data = weather[col].values.astype(np.float32)
        nan_mask = np.isnan(col_data)
        if nan_mask.any():
            fill_val = 0.0 if nan_mask.all() else float(np.nanmedian(col_data))
            col_data[nan_mask] = fill_val
            weather[col] = col_data

    arr = weather.values.astype(np.float32)
    print(f"  Weather array: {arr.shape}")
    return arr


# ---------------------------------------------------------------------------
# Step 3 — Calendar features
# ---------------------------------------------------------------------------

def make_calendar_features(full_index: pd.DatetimeIndex) -> np.ndarray:
    """
    Returns (T, 7) float32:
      hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos, is_weekend
    """
    print("Step 3: Computing calendar features...")
    hour  = full_index.hour.values.astype(np.float32)
    dow   = full_index.dayofweek.values.astype(np.float32)
    month = full_index.month.values.astype(np.float32)

    arr = np.stack([
        np.sin(2 * np.pi * hour  / 24),
        np.cos(2 * np.pi * hour  / 24),
        np.sin(2 * np.pi * dow   / 7),
        np.cos(2 * np.pi * dow   / 7),
        np.sin(2 * np.pi * month / 12),
        np.cos(2 * np.pi * month / 12),
        (dow >= 5).astype(np.float32),   # is_weekend
    ], axis=1).astype(np.float32)

    print(f"  Calendar array: {arr.shape}")
    return arr


# ---------------------------------------------------------------------------
# Step 4 — Event features
# ---------------------------------------------------------------------------

def load_event_features(raw_dir: str, full_index: pd.DatetimeIndex) -> np.ndarray:
    """
    Returns (T, 2) float32:
      event_flag (0/1), event_impact (0-3)
    """
    print("Step 4: Loading event features...")
    df = pd.read_csv(os.path.join(raw_dir, "events.csv"), parse_dates=["date"])

    # Per-date: max impact, flag
    by_date = (
        df.groupby("date")
        .agg(event_flag=("impact", "count"), event_impact=("impact", "max"))
        .reset_index()
    )
    by_date["event_flag"] = 1.0
    by_date = by_date.set_index("date")

    dates = full_index.normalize()  # midnight for each 5-min timestamp
    flag   = dates.map(lambda d: by_date.loc[d, "event_flag"]   if d in by_date.index else 0.0)
    impact = dates.map(lambda d: by_date.loc[d, "event_impact"] if d in by_date.index else 0.0)

    arr = np.stack([
        np.array(flag,   dtype=np.float32),
        np.array(impact, dtype=np.float32),
    ], axis=1)

    n_event_steps = int((arr[:, 0] > 0).sum())
    print(f"  Event array: {arr.shape}  ({n_event_steps:,} timesteps with active event)")
    return arr


# ---------------------------------------------------------------------------
# Step 5 — Normalize
# ---------------------------------------------------------------------------

def fit_scalers(
    traffic: np.ndarray,
    context: np.ndarray,
    train_slice: slice,
) -> tuple[StandardScaler, StandardScaler]:
    T, N, F = traffic.shape

    scaler_t = StandardScaler()
    scaler_t.fit(traffic[train_slice].reshape(-1, F))

    scaler_c = StandardScaler()
    scaler_c.fit(context[train_slice])

    return scaler_t, scaler_c


def apply_scalers(
    traffic: np.ndarray,
    context: np.ndarray,
    scaler_t: StandardScaler,
    scaler_c: StandardScaler,
) -> tuple[np.ndarray, np.ndarray]:
    T, N, F = traffic.shape
    traffic_norm = scaler_t.transform(traffic.reshape(-1, F)).reshape(T, N, F).astype(np.float32)
    context_norm = scaler_c.transform(context).astype(np.float32)
    return traffic_norm, context_norm


# ---------------------------------------------------------------------------
# Step 6 — Split indices
# ---------------------------------------------------------------------------

def get_split_indices(
    full_index: pd.DatetimeIndex,
    splits: dict,
) -> dict[str, tuple[int, int]]:
    def _bounds(start_str: str, end_str: str) -> tuple[int, int]:
        start = pd.Timestamp(start_str)
        end   = pd.Timestamp(end_str) + pd.Timedelta(days=1) - pd.Timedelta(minutes=5)
        mask  = (full_index >= start) & (full_index <= end)
        idxs  = np.where(mask)[0]
        return int(idxs[0]), int(idxs[-1])

    result = {
        "train": _bounds(splits["train_start"], splits["train_end"]),
        "val":   _bounds(splits["val_start"],   splits["val_end"]),
        "test":  _bounds(splits["test_start"],  splits["test_end"]),
    }
    for name, (s, e) in result.items():
        print(f"  {name:5s}: indices [{s:6d}, {e:6d}]  "
              f"({full_index[s].date()} → {full_index[e].date()}, {e-s+1:,} steps)")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir       = cfg["paths"]["raw_dir"]
    processed_dir = cfg["paths"]["processed_dir"]   # build_graph outputs (read-only here)
    graph_dir     = cfg["paths"]["graph_dir"]       # preprocess outputs (write here)
    splits        = cfg["data"]["splits"]
    gap_fill_limit    = cfg["data"]["gap_fill_limit"]
    missing_threshold = cfg["data"]["missing_threshold"]

    os.makedirs(graph_dir, exist_ok=True)

    # Full 5-min datetime index spanning all splits
    full_index = pd.date_range(
        start=splits["train_start"],
        end=pd.Timestamp(splits["test_end"]) + pd.Timedelta(hours=23, minutes=55),
        freq="5min",
    )
    print(f"\nFull index: {full_index[0]} → {full_index[-1]}  ({len(full_index):,} steps)\n")

    # --- Steps 1–4: load raw arrays ---
    traffic, sensor_order = load_sensor_data(raw_dir, full_index, gap_fill_limit, missing_threshold)
    weather  = load_weather(raw_dir, full_index)
    calendar = make_calendar_features(full_index)
    events   = load_event_features(raw_dir, full_index)

    # Combine context: weather (6) + calendar (7) + events (2) = 15 features
    context = np.concatenate([weather, calendar, events], axis=1)
    print(f"\nContext array: {context.shape}  "
          f"(weather={weather.shape[1]}, calendar={calendar.shape[1]}, events={events.shape[1]})")

    # --- Step 6: split indices (before normalizing so we know the train slice) ---
    print("\nStep 6: Computing split indices...")
    split_indices = get_split_indices(full_index, splits)
    train_s, train_e = split_indices["train"]
    train_slice = slice(train_s, train_e + 1)

    # --- Step 5: fit scalers on train, apply to all ---
    print("\nStep 5: Normalizing...")
    scaler_t, scaler_c = fit_scalers(traffic, context, train_slice)
    traffic_norm, context_norm = apply_scalers(traffic, context, scaler_t, scaler_c)
    print(f"  Traffic mean (train): {scaler_t.mean_.round(3)}")
    print(f"  Context mean (train, weather cols): {scaler_c.mean_[:6].round(3)}")

    # --- Load adjacency matrix from build_graph outputs, filter to kept sensors ---
    adj_full   = np.load(os.path.join(processed_dir, "adj_matrix.npy")).astype(np.float32)
    node_ids   = np.load(os.path.join(processed_dir, "node_ids.npy")).tolist()
    kept_mask  = [i for i, nid in enumerate(node_ids) if nid in sensor_order]
    if len(kept_mask) != len(sensor_order):
        raise RuntimeError(
            f"Sensor order mismatch: {len(sensor_order)} sensors kept but only "
            f"{len(kept_mask)} found in node_ids. Re-run build_graph.py."
        )
    adj = adj_full[np.ix_(kept_mask, kept_mask)]
    print(f"\nAdjacency matrix filtered: {adj_full.shape} → {adj.shape}")

    # --- Step 7: save to graph_dir (never overwrites processed_dir/build_graph outputs) ---
    print("\nStep 7: Saving outputs...")
    np.save(os.path.join(graph_dir, "traffic.npy"),    traffic_norm)
    np.save(os.path.join(graph_dir, "context.npy"),    context_norm)
    np.save(os.path.join(graph_dir, "adj_matrix.npy"), adj)
    np.save(os.path.join(graph_dir, "timestamps.npy"), full_index.values)

    with open(os.path.join(graph_dir, "sensor_order.json"), "w") as f:
        json.dump(sensor_order, f)
    with open(os.path.join(graph_dir, "split_indices.json"), "w") as f:
        json.dump(split_indices, f, indent=2)

    with open(os.path.join(graph_dir, "scaler_traffic.pkl"), "wb") as f:
        pickle.dump(scaler_t, f)
    with open(os.path.join(graph_dir, "scaler_context.pkl"), "wb") as f:
        pickle.dump(scaler_c, f)

    print(f"\nDone. Outputs in {graph_dir}/")
    print(f"  traffic.npy    {traffic_norm.shape}")
    print(f"  context.npy    {context_norm.shape}")
    print(f"  adj_matrix.npy {adj.shape}")
    print(f"  timestamps.npy {full_index.values.shape}")


if __name__ == "__main__":
    main()
