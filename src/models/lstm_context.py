"""
lstm_context.py  —  LSTM with Context Injection
Encoder-decoder LSTM that conditions the decoder on a global context embedding
(weather features, or weather + calendar + event features).

Architecture:
  Encoder     : same 2-layer LSTM as LSTMBaseline — (B*N, T, F) → hidden (h, c)
  Context enc : ContextEncoder MLP — (B, T, K) → (B, d_model)
  Injection   : h_ctx projected + broadcast over N → added to encoder hidden state
  Decoder     : same auto-regressive decoder as LSTMBaseline

This isolates the effect of adding contextual data modalities while keeping
the core sequence model identical to the baseline.
"""

import torch
import torch.nn as nn

from src.models.context_encoder import ContextEncoder


class LSTMWithContext(nn.Module):
    """
    Args:
        F          : traffic input features per sensor (default 3)
        K          : context features per timestep
        T          : input timesteps (needed by ContextEncoder)
        hidden_dim : LSTM hidden size (default 64)
        n_layers   : LSTM layers (default 2)
        H          : prediction horizon (default 12)
        dropout    : dropout (default 0.1)
        ctx_hidden : ContextEncoder MLP hidden dim (default 128)
    """

    def __init__(
        self,
        F: int = 3,
        K: int = 6,
        T: int = 12,
        hidden_dim: int = 64,
        n_layers: int = 2,
        H: int = 12,
        dropout: float = 0.1,
        ctx_hidden: int = 128,
    ):
        super().__init__()
        self.H          = H
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers

        # Traffic encoder (identical to LSTMBaseline)
        self.encoder = nn.LSTM(
            input_size=F,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # Context encoder: (B, T, K) → (B, hidden_dim)
        self.context_enc = ContextEncoder(
            T=T, K=K, hidden=ctx_hidden, d_model=hidden_dim, dropout=dropout,
        )

        # Project context embedding → one vector per LSTM layer, for both h and c
        # Output: (B, n_layers * hidden_dim * 2)  → split into h and c
        self.ctx_proj = nn.Sequential(
            nn.Linear(hidden_dim, n_layers * hidden_dim * 2),
            nn.Tanh(),
        )

        # Decoder (identical to LSTMBaseline)
        self.decoder_cell = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        traffic: torch.Tensor,
        context: torch.Tensor,
        target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        Args:
            traffic              : (B, T, N, F)
            context              : (B, T, K)
            target               : (B, H, N, 1)  required during training
            teacher_forcing_ratio: fraction of decoder steps using ground truth

        Returns:
            preds: (B, H, N, 1)
        """
        B, T, N, F = traffic.shape
        H = self.H

        # --- Encode traffic (per sensor) ---
        x = traffic.permute(0, 2, 1, 3).reshape(B * N, T, F)   # (B*N, T, F)
        _, (h_enc, c_enc) = self.encoder(x)                     # (n_layers, B*N, hidden)

        # --- Encode context (global) ---
        h_ctx = self.context_enc(context)                        # (B, hidden_dim)

        # Project to (n_layers, hidden) pairs for h and c
        proj = self.ctx_proj(h_ctx)                              # (B, n_layers*hidden*2)
        proj = proj.view(B, self.n_layers, self.hidden_dim, 2)  # (B, L, hidden, 2)
        h_delta = proj[..., 0]                                   # (B, L, hidden)
        c_delta = proj[..., 1]                                   # (B, L, hidden)

        # Expand context over N sensors and add to encoder hidden state
        # h_enc: (n_layers, B*N, hidden) → reshape to (n_layers, B, N, hidden)
        h_enc = h_enc.view(self.n_layers, B, N, self.hidden_dim)
        c_enc = c_enc.view(self.n_layers, B, N, self.hidden_dim)

        # h_delta: (B, L, hidden) → (L, B, 1, hidden) → broadcast over N
        h_delta = h_delta.permute(1, 0, 2).unsqueeze(2)        # (L, B, 1, hidden)
        c_delta = c_delta.permute(1, 0, 2).unsqueeze(2)        # (L, B, 1, hidden)

        h_dec = torch.tanh(h_enc + h_delta)                     # (L, B, N, hidden)
        c_dec = torch.tanh(c_enc + c_delta)                     # (L, B, N, hidden)

        # Flatten back to (n_layers, B*N, hidden) for LSTM cell
        h_dec = h_dec.reshape(self.n_layers, B * N, self.hidden_dim)
        c_dec = c_dec.reshape(self.n_layers, B * N, self.hidden_dim)

        # --- Auto-regressive decode ---
        dec_input = traffic[:, -1, :, 0:1]                      # (B, N, 1)
        dec_input = dec_input.permute(0, 2, 1).reshape(B * N, 1, 1)  # (B*N, 1, 1)

        preds = []
        for step in range(H):
            out, (h_dec, c_dec) = self.decoder_cell(dec_input, (h_dec, c_dec))
            pred = self.head(out)                                # (B*N, 1, 1)
            preds.append(pred)

            if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                gt = target[:, step, :, 0:1]
                dec_input = gt.permute(0, 2, 1).reshape(B * N, 1, 1)
            else:
                dec_input = pred.detach()

        preds = torch.cat(preds, dim=1)                         # (B*N, H, 1)
        preds = preds.reshape(B, N, H, 1).permute(0, 2, 1, 3)  # (B, H, N, 1)
        return preds


def build_from_config(cfg: dict) -> LSTMWithContext:
    m = cfg["model"]
    d = cfg["data"]
    return LSTMWithContext(
        F=d["F"],
        K=d["K"],
        T=d["T"],
        hidden_dim=m["d_model"],
        n_layers=2,
        H=d["H"],
        dropout=m["dropout"],
        ctx_hidden=m["context_encoder"]["hidden_dim"],
    )
