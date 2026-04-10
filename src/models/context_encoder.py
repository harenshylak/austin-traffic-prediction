"""
context_encoder.py  —  Event-aware Contrastive Context Encoder (ECCE)
Maps a window of context features (weather + calendar + events) to a single
d_model embedding that captures the "traffic condition" of the window.

Architecture:
  flatten (T, K) → MLP: T*K → hidden → 64 → d_model

Contrastive loss (NT-Xent) during training:
  - Positive pairs: same event type in same time-of-day bucket
  - Negative pairs: all other samples in the batch
  The loss is returned as an auxiliary output and added to the main loss
  with a small weight (0.1) during training. At inference it is ignored.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextEncoder(nn.Module):
    """
    Args:
        T        : input timesteps (default 12)
        K        : context features per timestep (default 15)
        hidden   : MLP hidden dim (default 128)
        d_model  : output embedding dim (default 64)
        dropout  : dropout rate (default 0.1)
    """

    def __init__(
        self,
        T: int = 12,
        K: int = 15,
        hidden: int = 128,
        d_model: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        in_dim = T * K
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context : (B, T, K)

        Returns:
            h : (B, d_model)
        """
        B = context.size(0)
        h = self.mlp(context.reshape(B, -1))  # (B, d_model)
        return self.norm(h)


def nt_xent_loss(
    z: torch.Tensor,
    event_flag: torch.Tensor,
    temperature: float = 0.5,
) -> torch.Tensor:
    """
    NT-Xent contrastive loss.

    Positive pair definition: two samples in the same batch where
    event_flag matches (both event or both non-event).
    Returns zero if no valid positive pairs exist in the batch.

    Args:
        z           : (B, d_model) embeddings (need not be pre-normalised)
        event_flag  : (B, 1) float, 1 = event, 0 = normal
        temperature : softmax temperature (default 0.5)

    Returns:
        scalar loss (0.0 if no positive pairs)
    """
    B = z.size(0)
    if B < 2:
        return z.new_zeros(1).squeeze()

    z = F.normalize(z, dim=1)                           # (B, d_model)

    # Positive mask: same event_flag, excluding self
    ef = event_flag.view(B)
    mask_self = torch.eye(B, dtype=torch.bool, device=z.device)
    pos_mask = (ef.unsqueeze(0) == ef.unsqueeze(1)) & ~mask_self  # (B, B)

    # If every sample is the only one of its class, no positives → skip
    has_pos = pos_mask.any(dim=1)
    if not has_pos.any():
        return z.new_zeros(1).squeeze()

    sim = torch.matmul(z, z.T) / temperature            # (B, B)

    # For numerical stability: subtract row max before softmax
    sim = sim - sim.detach().max(dim=1, keepdim=True).values

    # Exclude self from denominator
    sim_exp = torch.exp(sim) * (~mask_self).float()     # (B, B)
    denom = sim_exp.sum(dim=1, keepdim=True).clamp(min=1e-8)

    log_prob = sim - torch.log(denom)                   # (B, B)

    n_pos = pos_mask.float().sum(dim=1).clamp(min=1)
    loss_per_anchor = -(log_prob * pos_mask.float()).sum(dim=1) / n_pos

    return loss_per_anchor[has_pos].mean()
