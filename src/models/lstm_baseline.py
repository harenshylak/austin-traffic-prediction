"""
lstm_baseline.py  —  Layer 0
Encoder-decoder LSTM. Processes each sensor independently.
  Encoder: (T, F) → 2-layer LSTM (hidden=64) → final hidden state
  Decoder: autoregressively generates H steps
           (teacher forcing during train, rollout during inference)
"""

# TODO: implement
