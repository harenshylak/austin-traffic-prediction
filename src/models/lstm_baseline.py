"""
lstm_baseline.py  —  Layer 0
Encoder-decoder LSTM. Processes each sensor independently (no graph).

Architecture:
  Encoder : 2-layer LSTM over (T, F) input per sensor → final hidden state (h, c)
  Decoder : autoregressively generates H speed predictions
            - Train: teacher forcing (feed ground-truth previous step)
            - Eval : rollout (feed own previous prediction)

Input / output contract (matches dataset.py batch dict):
  traffic    (B, T, N, F)  — normalized speed / volume / occupancy
  target     (B, H, N, 1)  — normalized ground-truth speed
  Returns    (B, H, N, 1)  — predicted speed
"""

import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    """
    Per-sensor encoder-decoder LSTM.

    Args:
        F          : number of input features per sensor (default 3)
        hidden_dim : LSTM hidden size (default 64, matches d_model in config)
        n_layers   : number of LSTM layers (default 2)
        H          : prediction horizon in steps (default 12)
        dropout    : dropout between LSTM layers (default 0.1)
    """

    def __init__(
        self,
        F: int = 3,
        hidden_dim: int = 64,
        n_layers: int = 2,
        H: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.H          = H
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers

        # Shared encoder across all sensors (weights are sensor-agnostic)
        self.encoder = nn.LSTM(
            input_size=F,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # Decoder: takes previous speed prediction (1-D) → next hidden → speed
        self.decoder_cell = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # Project hidden state → speed prediction
        self.head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        traffic: torch.Tensor,
        target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        Args:
            traffic              : (B, T, N, F)
            target               : (B, H, N, 1)  required during training
            teacher_forcing_ratio: fraction of decoder steps using ground truth input
                                   (1.0 = full teacher forcing, 0.0 = full rollout)

        Returns:
            preds: (B, H, N, 1)  normalized speed predictions
        """
        B, T, N, F = traffic.shape
        H = self.H

        # Reshape to (B*N, T, F) — treat each sensor as an independent sequence
        x = traffic.permute(0, 2, 1, 3).reshape(B * N, T, F)

        # Encode: run LSTM over the input window
        _, (h, c) = self.encoder(x)   # h, c: (n_layers, B*N, hidden)

        # Decoder: auto-regressive over H steps
        # Seed the first decoder input with the last observed speed (feature 0)
        dec_input = traffic[:, -1, :, 0:1]         # (B, N, 1)
        dec_input = dec_input.permute(0, 2, 1).reshape(B * N, 1, 1)  # (B*N, 1, 1)

        preds = []
        for step in range(H):
            out, (h, c) = self.decoder_cell(dec_input, (h, c))  # out: (B*N, 1, hidden)
            pred = self.head(out)                                 # (B*N, 1, 1)
            preds.append(pred)

            # Teacher forcing: during training (ratio=0.5), each step independently
            # uses ground-truth speed as the next input with probability=ratio.
            # During validation/test (ratio=0.0) the decoder is fully autoregressive —
            # each prediction feeds the next step. This train–test mismatch (exposure
            # bias) causes errors to compound across the H=12 horizon, which is why
            # MAE grows faster with horizon than a per-step error rate would predict.
            if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                gt = target[:, step, :, 0:1]                        # (B, N, 1)
                dec_input = gt.permute(0, 2, 1).reshape(B * N, 1, 1)
            else:
                dec_input = pred.detach()

        # Stack and reshape back to (B, H, N, 1)
        preds = torch.cat(preds, dim=1)          # (B*N, H, 1)
        preds = preds.reshape(B, N, H, 1).permute(0, 2, 1, 3)  # (B, H, N, 1)
        return preds


def build_from_config(cfg: dict) -> LSTMBaseline:
    """Instantiate LSTMBaseline from a loaded config dict."""
    return LSTMBaseline(
        F=cfg["data"]["F"],
        hidden_dim=cfg["model"]["d_model"],
        n_layers=2,
        H=cfg["data"]["H"],
        dropout=cfg["model"]["dropout"],
    )
