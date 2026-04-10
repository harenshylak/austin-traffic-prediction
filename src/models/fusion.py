"""
fusion.py  —  Cross-Modal Attention Fusion
Each sensor node (spatial embedding) attends to the global context embedding.

Architecture:
  Q = h_spatial  (B, N, d_model)       — per-sensor temporal embeddings
  K = V = h_context expanded to (B, N, d_model)  — broadcast context
  MultiHeadAttention → residual + LayerNorm → FFN → residual + LayerNorm
  Output: h_fused (B, N, d_model)
"""

import torch
import torch.nn as nn


class CrossModalFusion(nn.Module):
    """
    Args:
        d_model  : embedding dimension (default 64)
        n_heads  : attention heads (default 4)
        ffn_dim  : feedforward hidden dim (default 256)
        dropout  : dropout rate (default 0.1)
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        h_spatial: torch.Tensor,
        h_context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h_spatial : (B, N, d_model)  spatial/temporal sensor embeddings
            h_context : (B, d_model)     global context embedding

        Returns:
            h_fused : (B, N, d_model)
        """
        B, N, D = h_spatial.shape

        # Expand context to match N sensor dimension for K and V
        # (B, d_model) → (B, 1, d_model) → (B, N, d_model)
        ctx = h_context.unsqueeze(1).expand(B, N, D)  # (B, N, d_model)

        # Cross-attention: Q=spatial, K=V=context
        attn_out, _ = self.attn(
            query=h_spatial,
            key=ctx,
            value=ctx,
        )                                              # (B, N, d_model)

        # Residual + LayerNorm
        h = self.norm1(h_spatial + attn_out)

        # FFN + residual + LayerNorm
        h = self.norm2(h + self.ffn(h))

        return h
