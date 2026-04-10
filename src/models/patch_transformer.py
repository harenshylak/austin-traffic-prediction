"""
patch_transformer.py  —  Patch-Based Temporal Transformer (PTT)
Replaces the LSTM encoder.
  - Input (T=12, F=3) per sensor split into 6 non-overlapping patches of (2, 3)
  - Each patch flattened → linearly projected to d_model
  - Learnable positional embeddings added
  - 2-layer Transformer encoder (4 heads, d_model=64, ffn=256)
  - Mean pooling over 6 output tokens → (d_model,) per sensor
"""

# TODO: implement
