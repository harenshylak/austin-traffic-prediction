"""
patch_transformer.py  —  Patch-Based Temporal Transformer (PTT)
Replaces the LSTM encoder with a Transformer that operates on temporal patches.

Per sensor:
  (T, F) → split into P non-overlapping patches of size (patch_size, F)
          → flatten each patch → linear projection to d_model
          → add learnable positional embeddings
          → n_layers Transformer encoder (n_heads, ffn_dim)
          → mean pool over P tokens → (d_model,) embedding per sensor

For T=12, patch_size=2: P=6 patches.
"""

import math
import torch
import torch.nn as nn


class PatchTransformer(nn.Module):
    """
    Args:
        T          : input timesteps (default 12)
        F          : features per timestep (default 3)
        patch_size : timesteps per patch (default 2)
        d_model    : embedding dimension (default 64)
        n_heads    : attention heads (default 4)
        n_layers   : transformer encoder layers (default 2)
        ffn_dim    : feedforward hidden dim (default 256)
        dropout    : dropout rate (default 0.1)
    """

    def __init__(
        self,
        T: int = 12,
        F: int = 3,
        patch_size: int = 2,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ffn_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert T % patch_size == 0, f"T={T} must be divisible by patch_size={patch_size}"
        self.T          = T
        self.F          = F
        self.patch_size = patch_size
        self.d_model    = d_model
        self.n_patches  = T // patch_size

        patch_dim = patch_size * F  # flattened patch size

        # Linear projection: patch_dim → d_model
        self.patch_embed = nn.Linear(patch_dim, d_model)

        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T, N, F)  input traffic features

        Returns:
            h : (B, N, d_model)  per-sensor temporal embeddings
        """
        B, T, N, F = x.shape

        # Reshape to (B*N, T, F) — process each sensor independently
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, F)

        # Create patches: (B*N, n_patches, patch_size*F)
        x = x.reshape(B * N, self.n_patches, self.patch_size * F)

        # Patch embedding + positional encoding
        h = self.patch_embed(x) + self.pos_embed   # (B*N, n_patches, d_model)

        # Transformer encoder
        h = self.transformer(h)                     # (B*N, n_patches, d_model)
        h = self.norm(h)

        # Mean pool over patches → (B*N, d_model)
        h = h.mean(dim=1)

        # Reshape back to (B, N, d_model)
        h = h.reshape(B, N, self.d_model)
        return h
