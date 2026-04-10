"""
gcn_lstm.py  —  Layer 2
Spatial + temporal fusion via GCN → LSTM.

Architecture:
  Per timestep:
    (N, F) → GraphConv x2 (ReLU + dropout) → (N, d_model)   [spatial mixing]
  Then per sensor:
    (T, d_model) → 2-layer LSTM → final (h, c)               [temporal encoding]
  Decoder:
    same auto-regressive decoder as LSTMBaseline              [H-step rollout]

The GCN uses the symmetric-normalised adjacency:
    A_hat = D^{-1/2} (A + I) D^{-1/2}
so each node aggregates its own features plus neighbour features.
Weights are shared across timesteps (parameter efficient).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Graph Convolution Layer
# ---------------------------------------------------------------------------

class GraphConv(nn.Module):
    """
    Single graph convolution:  H' = σ(A_hat H W)

    Args:
        in_dim  : input feature dimension
        out_dim : output feature dimension
        bias    : whether to include a bias term
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x     : (B, N, in_dim)
            a_hat : (N, N) or (B, N, N) normalised adjacency

        Returns:
            (B, N, out_dim)
        """
        # (B, N, out_dim) = A_hat @ x @ W
        if a_hat.dim() == 2:
            # broadcast over batch
            out = torch.einsum("nm,bmd->bnd", a_hat, x)
        else:
            out = torch.bmm(a_hat, x)
        return self.W(out)


# ---------------------------------------------------------------------------
# Normalised adjacency helper
# ---------------------------------------------------------------------------

def normalise_adj(adj: torch.Tensor) -> torch.Tensor:
    """
    Symmetric normalisation:  A_hat = D^{-1/2} (A + I) D^{-1/2}

    Args:
        adj : (N, N) raw adjacency (values in [0, 1])

    Returns:
        (N, N) normalised adjacency
    """
    N = adj.size(0)
    a_tilde = adj + torch.eye(N, device=adj.device, dtype=adj.dtype)
    deg = a_tilde.sum(dim=1)                          # (N,)
    d_inv_sqrt = torch.pow(deg, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    D = torch.diag(d_inv_sqrt)                        # (N, N)
    return D @ a_tilde @ D


# ---------------------------------------------------------------------------
# GCN + LSTM model
# ---------------------------------------------------------------------------

class GCNLSTMModel(nn.Module):
    """
    GCN spatial encoder + LSTM temporal encoder/decoder.

    Args:
        F          : raw input features per sensor (default 3)
        d_model    : hidden dimension for GCN output and LSTM (default 64)
        n_gcn      : number of GCN layers (default 2)
        n_lstm     : number of LSTM layers (default 2)
        H          : prediction horizon steps (default 12)
        dropout    : dropout applied between GCN layers and LSTM layers
    """

    def __init__(
        self,
        F: int = 3,
        d_model: int = 64,
        n_gcn: int = 2,
        n_lstm: int = 2,
        H: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.H       = H
        self.d_model = d_model
        self.n_lstm  = n_lstm
        self.dropout = dropout

        # GCN stack: F → d_model → d_model
        gcn_dims = [F] + [d_model] * n_gcn
        self.gcn_layers = nn.ModuleList([
            GraphConv(gcn_dims[i], gcn_dims[i + 1])
            for i in range(n_gcn)
        ])
        self.gcn_drop = nn.Dropout(dropout)

        # LSTM encoder: takes GCN output (d_model) per timestep
        self.encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_lstm,
            batch_first=True,
            dropout=dropout if n_lstm > 1 else 0.0,
        )

        # LSTM decoder: takes previous speed prediction (1-D)
        self.decoder_cell = nn.LSTM(
            input_size=1,
            hidden_size=d_model,
            num_layers=n_lstm,
            batch_first=True,
            dropout=dropout if n_lstm > 1 else 0.0,
        )

        self.head = nn.Linear(d_model, 1)

    def _apply_gcn(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        """
        Run GCN stack over a single timestep.

        Args:
            x     : (B, N, F)
            a_hat : (N, N)

        Returns:
            (B, N, d_model)
        """
        h = x
        for i, layer in enumerate(self.gcn_layers):
            h = layer(h, a_hat)
            if i < len(self.gcn_layers) - 1:
                h = F.relu(h)
                h = self.gcn_drop(h)
        return h  # no activation on final GCN layer (LSTM handles it)

    def forward(
        self,
        traffic: torch.Tensor,
        adjacency: torch.Tensor,
        target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        Args:
            traffic              : (B, T, N, F)
            adjacency            : (B, N, N) or (N, N) static adjacency
            target               : (B, H, N, 1)  required during training
            teacher_forcing_ratio: 1.0 = full teacher forcing, 0.0 = rollout

        Returns:
            preds : (B, H, N, 1)
        """
        B, T, N, F = traffic.shape

        # Use the first sample's adjacency (static, same for all in batch)
        adj = adjacency[0] if adjacency.dim() == 3 else adjacency  # (N, N)
        a_hat = normalise_adj(adj)                                  # (N, N)

        # --- Spatial encoding: GCN at each timestep ---
        # Stack timesteps: (B*T, N, F) → GCN → (B*T, N, d_model)
        x_flat = traffic.reshape(B * T, N, F)
        h_spatial = self._apply_gcn(x_flat, a_hat)               # (B*T, N, d_model)
        h_spatial = h_spatial.reshape(B, T, N, self.d_model)      # (B, T, N, d_model)

        # --- Temporal encoding: LSTM per sensor ---
        # Reshape to (B*N, T, d_model) — treat each sensor independently
        enc_in = h_spatial.permute(0, 2, 1, 3).reshape(B * N, T, self.d_model)
        _, (h, c) = self.encoder(enc_in)                           # (n_lstm, B*N, d_model)

        # --- Decoding: auto-regressive over H steps ---
        dec_input = traffic[:, -1, :, 0:1]                        # (B, N, 1) last speed
        dec_input = dec_input.permute(0, 2, 1).reshape(B * N, 1, 1)

        preds = []
        for step in range(self.H):
            out, (h, c) = self.decoder_cell(dec_input, (h, c))    # (B*N, 1, d_model)
            pred = self.head(out)                                   # (B*N, 1, 1)
            preds.append(pred)

            if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                gt = target[:, step, :, 0:1]                       # (B, N, 1)
                dec_input = gt.permute(0, 2, 1).reshape(B * N, 1, 1)
            else:
                dec_input = pred.detach()

        preds = torch.cat(preds, dim=1)                            # (B*N, H, 1)
        preds = preds.reshape(B, N, self.H, 1).permute(0, 2, 1, 3)  # (B, H, N, 1)
        return preds


def build_from_config(cfg: dict) -> GCNLSTMModel:
    return GCNLSTMModel(
        F=cfg["data"]["F"],
        d_model=cfg["model"]["d_model"],
        n_gcn=cfg["model"]["gatv2"]["n_layers"],   # reuse gatv2.n_layers for GCN depth
        n_lstm=2,
        H=cfg["data"]["H"],
        dropout=cfg["model"]["dropout"],
    )
