"""
preprocess.py
Transforms raw CSVs into model-ready tensors following the 7-step pipeline:

  1. Resample sensors to 5-min intervals; forward-fill gaps ≤15 min; NaN beyond
  2. Forward-fill hourly weather to 5-min
  3. Compute cyclic calendar features (hour, day-of-week, month, is_weekend, is_holiday)
  4. Create binary event flags from events CSV
  5. Normalize with StandardScaler (fit on train split only)
  6. Chronological split: train Jan-Oct 2022 / val Nov-Dec 2022 / test 2023
  7. Generate sliding windows of size T+H; save as .pt tensors

Usage:
    python src/data/preprocess.py
"""

# TODO: implement
