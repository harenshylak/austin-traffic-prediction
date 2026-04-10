"""
hmstgn.py  —  Full HM-STGN Assembly
forward() chains all modules:
  PTT → AGL + GATv2 → ECCE → Fusion → Prediction Heads

Prediction heads:
  (a) Speed regression     — Huber loss
  (b) Congestion classifier — binary cross-entropy
  (c) Anomaly detector     — reconstruction MSE

Combined loss: L = 0.5*L_speed + 0.25*L_congestion + 0.25*L_anomaly

Accepts the full sample dict from DataLoader; returns predictions + aux outputs.
"""

# TODO: implement
