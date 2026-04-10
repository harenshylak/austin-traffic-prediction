"""
context_encoder.py  —  Event-aware Contrastive Context Encoder (ECCE)
3-layer MLP: (T*K) → 128 → 64 → d_model
Auxiliary NT-Xent contrastive loss during training (dropped at inference):
  - Same condition (e.g. two SXSW Friday afternoons) → similar embeddings
  - Different conditions (SXSW vs normal Tuesday)    → distant embeddings
"""

# TODO: implement
