"""
adaptive_graph.py  —  Adaptive Graph Learner (AGL)
Learns event-aware adjacency matrices that blend with the static road-network graph.

Two embedding pairs:
  normal embeddings : (E1_n, E2_n) — used on regular days
  event  embeddings : (E1_e, E2_e) — used when a special event is active

A_adaptive = softmax(ReLU(E1 @ E2.T))   shape (N, N)
A_final    = sigmoid(alpha) * A_static + (1 - sigmoid(alpha)) * A_adaptive

alpha is a learnable scalar (unconstrained; passed through sigmoid so blend ∈ (0,1)).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGraphLearner(nn.Module):
    """
    Args:
        N        : number of sensor nodes
        d_embed  : embedding dimension for node embeddings (default 16)
        alpha_init: initial value for the blend scalar (default 0.5)
    """

    def __init__(self, N: int, d_embed: int = 16, alpha_init: float = 0.5):
        super().__init__()
        self.N = N

        # Normal-day node embeddings
        self.E1_n = nn.Parameter(torch.empty(N, d_embed))
        self.E2_n = nn.Parameter(torch.empty(N, d_embed))

        # Event-day node embeddings
        self.E1_e = nn.Parameter(torch.empty(N, d_embed))
        self.E2_e = nn.Parameter(torch.empty(N, d_embed))

        # Blend scalar: alpha_init mapped through inverse-sigmoid so sigmoid(raw) ≈ alpha_init
        alpha_raw = torch.tensor(alpha_init).logit().item()
        self.alpha = nn.Parameter(torch.tensor(alpha_raw))

        # Initialise embeddings with Xavier uniform
        for p in [self.E1_n, self.E2_n, self.E1_e, self.E2_e]:
            tmp = torch.empty(1, N, d_embed)
            nn.init.xavier_uniform_(tmp)
            with torch.no_grad():
                p.copy_(tmp.squeeze(0))

    def forward(
        self,
        a_static: torch.Tensor,
        event_flag: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            a_static   : (N, N) or (B, N, N) static adjacency (from road network)
            event_flag : (B, 1) float tensor, 1 = event active, 0 = normal

        Returns:
            A_final : (B, N, N) blended adjacency matrix
        """
        B = event_flag.size(0)

        # Compute adaptive adjacency for both modes
        A_n = F.softmax(F.relu(self.E1_n @ self.E2_n.T), dim=-1)  # (N, N)
        A_e = F.softmax(F.relu(self.E1_e @ self.E2_e.T), dim=-1)  # (N, N)

        # Per-sample interpolation between normal and event graphs
        # event_flag: (B, 1) → (B, 1, 1) for broadcasting
        ef = event_flag.view(B, 1, 1)
        A_adaptive = (1 - ef) * A_n.unsqueeze(0) + ef * A_e.unsqueeze(0)  # (B, N, N)

        # Expand static adjacency to batch
        if a_static.dim() == 2:
            a_static = a_static.unsqueeze(0).expand(B, -1, -1)  # (B, N, N)

        alpha = torch.sigmoid(self.alpha)
        A_final = alpha * a_static + (1 - alpha) * A_adaptive    # (B, N, N)

        return A_final
