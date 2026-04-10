"""
trainer.py
Training loop with:
  - AdamW optimizer + ReduceLROnPlateau scheduler
  - Gradient clipping (max_norm=5.0)
  - Early stopping on validation MAE (patience=10)
  - Checkpoint saving (best val MAE)
  - Optional wandb logging
"""

# TODO: implement
