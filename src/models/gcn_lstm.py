"""
gcn_lstm.py  —  Layer 2
Adds spatial structure on top of the LSTM baseline.
At each input timestep: (N, F) → 2-layer GCN on static adjacency → (N, d_model)
The spatially-aware features are then fed as input to the LSTM encoder.
"""

# TODO: implement
