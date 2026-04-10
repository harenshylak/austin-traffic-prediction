"""
dataset.py
PyTorch Dataset + DataLoader for the preprocessed sliding-window tensors.

Each sample is a dict:
  {
    'traffic':   (T, N, F)  - speed, volume, occupancy
    'context':   (T, K)     - weather + calendar features
    'adjacency': (N, N)     - static adjacency matrix
    'target':    (H, N, 1)  - ground-truth speed
    'event_flag':(1,)       - binary event indicator
  }
"""

# TODO: implement
