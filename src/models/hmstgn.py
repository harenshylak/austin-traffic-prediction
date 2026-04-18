"""
hmstgn.py  —  Full HM-STGN Assembly
Hierarchical Multi-modal Spatio-Temporal Graph Network.

Forward pass:
  1. PTT         : (B,T,N,F) → (B,N,d_model)   temporal patch embeddings per sensor
  2. AGL         : A_static + event_flag → (B,N,N) blended adjacency
  3. GCN         : (B,N,d_model) × A → (B,N,d_model)  spatial mixing
  4. ECCE        : (B,T,K) → (B,d_model)          global context embedding
  5. Fusion      : spatial + context → (B,N,d_model) fused embeddings
  6. Heads:
       speed      : (B,N,d_model) → (B,H,N,1)    speed regression
       congestion : (B,N,d_model) → (B,N,1)       binary congestion flag
       anomaly    : (B,N,d_model) → (B,N,d_model) reconstruction for anomaly

Combined loss:
  L = w_speed * L_huber + w_cong * L_bce + w_anom * L_mse + w_contrast * L_ntxent
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.patch_transformer import PatchTransformer
from src.models.adaptive_graph import AdaptiveGraphLearner
from src.models.context_encoder import ContextEncoder, nt_xent_loss
from src.models.fusion import CrossModalFusion
from src.models.gcn_lstm import GraphConv


# ---------------------------------------------------------------------------
# Prediction heads
# ---------------------------------------------------------------------------

class SpeedHead(nn.Module):
    """Projects fused sensor embeddings to H-step speed predictions."""

    def __init__(self, d_model: int, H: int):
        super().__init__()
        self.H = H
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, H),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h : (B, N, d_model)
        Returns:
            (B, H, N, 1)
        """
        out = self.proj(h)                        # (B, N, H)
        return out.permute(0, 2, 1).unsqueeze(-1) # (B, H, N, 1)


class CongestionHead(nn.Module):
    """Binary congestion classifier per sensor (speed < threshold → congested)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h : (B, N, d_model)
        Returns:
            logits (B, N, 1)
        """
        return self.proj(h)


class AnomalyHead(nn.Module):
    """Reconstruction-based anomaly detector: encoder → decoder → MSE."""

    def __init__(self, d_model: int):
        super().__init__()
        self.encoder = nn.Linear(d_model, d_model // 2)
        self.decoder = nn.Linear(d_model // 2, d_model)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h : (B, N, d_model)
        Returns:
            recon      : (B, N, d_model)  reconstructed embedding
            anomaly_score : (B, N, 1)    per-sensor reconstruction MSE
        """
        z     = F.relu(self.encoder(h))
        recon = self.decoder(z)
        score = ((h - recon) ** 2).mean(dim=-1, keepdim=True)  # (B, N, 1)
        return recon, score


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class HMSTGN(nn.Module):
    """
    Args:
        N        : number of sensor nodes
        T        : input timesteps (default 12)
        F_in     : input features per sensor (default 3)
        K        : context features per timestep (default 15)
        H        : prediction horizon (default 12)
        d_model  : shared embedding dim (default 64)
        patch_size: PTT patch size (default 2)
        n_ptt_heads: PTT attention heads (default 4)
        n_ptt_layers: PTT transformer layers (default 2)
        ptt_ffn  : PTT feedforward dim (default 256)
        n_gcn    : GCN layers (default 2)
        d_embed  : AGL node embedding dim (default 16)
        alpha_init: AGL blend init (default 0.5)
        ctx_hidden: ECCE MLP hidden dim (default 128)
        n_fuse_heads: fusion attention heads (default 4)
        fuse_ffn : fusion FFN dim (default 256)
        dropout  : dropout (default 0.1)
        w_speed  : speed loss weight (default 0.5)
        w_cong   : congestion loss weight (default 0.25)
        w_anom   : anomaly loss weight (default 0.25)
        w_contrast: contrastive loss weight (default 0.1)
        congestion_threshold: normalised speed below which sensor is congested (default -0.5)
    """

    def __init__(
        self,
        N: int,
        T: int = 12,
        F_in: int = 3,
        K: int = 15,
        H: int = 12,
        d_model: int = 64,
        patch_size: int = 2,
        n_ptt_heads: int = 4,
        n_ptt_layers: int = 2,
        ptt_ffn: int = 256,
        n_gcn: int = 2,
        d_embed: int = 16,
        alpha_init: float = 0.5,
        ctx_hidden: int = 128,
        n_fuse_heads: int = 4,
        fuse_ffn: int = 256,
        dropout: float = 0.1,
        w_speed: float = 0.5,
        w_cong: float = 0.25,
        w_anom: float = 0.25,
        w_contrast: float = 0.1,
        congestion_threshold: float = -0.5,
    ):
        super().__init__()
        self.H = H
        self.w_speed    = w_speed
        self.w_cong     = w_cong
        self.w_anom     = w_anom
        self.w_contrast = w_contrast
        self.congestion_threshold = congestion_threshold

        # 1. Patch Transformer (temporal encoder)
        self.ptt = PatchTransformer(
            T=T, F=F_in, patch_size=patch_size,
            d_model=d_model, n_heads=n_ptt_heads,
            n_layers=n_ptt_layers, ffn_dim=ptt_ffn, dropout=dropout,
        )

        # 2. Adaptive Graph Learner
        self.agl = AdaptiveGraphLearner(N=N, d_embed=d_embed, alpha_init=alpha_init)

        # 3. GCN spatial mixing (2 layers)
        gcn_dims = [d_model] + [d_model] * n_gcn
        self.gcn_layers = nn.ModuleList([
            GraphConv(gcn_dims[i], gcn_dims[i + 1])
            for i in range(n_gcn)
        ])
        self.gcn_drop = nn.Dropout(dropout)

        # 4. Context encoder
        self.ecce = ContextEncoder(T=T, K=K, hidden=ctx_hidden, d_model=d_model, dropout=dropout)

        # 5. Cross-modal fusion
        self.fusion = CrossModalFusion(d_model=d_model, n_heads=n_fuse_heads, ffn_dim=fuse_ffn, dropout=dropout)

        # 6. Prediction heads
        self.speed_head     = SpeedHead(d_model, H)
        self.congestion_head = CongestionHead(d_model)
        self.anomaly_head   = AnomalyHead(d_model)

    def _apply_gcn(self, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Batched symmetric normalisation: D^{-1/2}(A+I)D^{-1/2} vectorised over B.

        Args:
            h : (B, N, d_model)
            A : (B, N, N)
        Returns:
            (B, N, d_model)
        """
        B, N, _ = A.shape
        # Add self-loops
        A_tilde = A + torch.eye(N, device=A.device, dtype=A.dtype).unsqueeze(0)  # (B, N, N)
        # Degree vector
        deg = A_tilde.sum(dim=-1)                                                  # (B, N)
        d_inv_sqrt = deg.pow(-0.5)
        d_inv_sqrt = d_inv_sqrt.masked_fill(torch.isinf(d_inv_sqrt), 0.0)
        # D^{-1/2} as diagonal batched matrix
        D = torch.diag_embed(d_inv_sqrt)                                           # (B, N, N)
        A_hat = D @ A_tilde @ D                                                    # (B, N, N)

        for i, layer in enumerate(self.gcn_layers):
            h = layer(h, A_hat)
            if i < len(self.gcn_layers) - 1:
                h = F.relu(h)
                h = self.gcn_drop(h)
        return h

    def forward(
        self,
        traffic: torch.Tensor,
        context: torch.Tensor,
        adjacency: torch.Tensor,
        event_flag: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> dict:
        """
        Args:
            traffic    : (B, T, N, F)
            context    : (B, T, K)
            adjacency  : (B, N, N) or (N, N)
            event_flag : (B, 1)
            target     : (B, H, N, 1)  ground-truth speed (required for loss)

        Returns dict with keys:
            pred_speed      : (B, H, N, 1)
            pred_congestion : (B, N, 1) logits
            anomaly_score   : (B, N, 1)
            loss            : scalar (only if target is not None)
            loss_speed      : scalar
            loss_congestion : scalar
            loss_anomaly    : scalar
            loss_contrast   : scalar
        """
        B = traffic.size(0)

        # Static adjacency handling
        if adjacency.dim() == 2:
            a_static = adjacency.unsqueeze(0).expand(B, -1, -1)
        else:
            a_static = adjacency

        # 1. PTT: temporal embeddings
        h_temporal = self.ptt(traffic)                      # (B, N, d_model)

        # 2. AGL: blended adjacency
        A_blended = self.agl(a_static, event_flag)          # (B, N, N)

        # 3. GCN: spatial mixing
        h_spatial = self._apply_gcn(h_temporal, A_blended) # (B, N, d_model)

        # 4. ECCE: context embedding
        h_context = self.ecce(context)                      # (B, d_model)

        # 5. Fusion
        h_fused = self.fusion(h_spatial, h_context)        # (B, N, d_model)

        # 6. Prediction heads
        pred_speed      = self.speed_head(h_fused)         # (B, H, N, 1)
        pred_congestion = self.congestion_head(h_fused)    # (B, N, 1)
        recon, anomaly_score = self.anomaly_head(h_fused)  # (B, N, d_model), (B, N, 1)

        out = {
            "pred_speed":      pred_speed,
            "pred_congestion": pred_congestion,
            "anomaly_score":   anomaly_score,
        }

        if target is not None:
            # Speed loss (Huber)
            loss_speed = F.huber_loss(pred_speed, target, delta=1.0)

            # Congestion labels: normalised speed < threshold → congested (1)
            # Use last observed speed step as proxy target
            last_speed = traffic[:, -1, :, 0:1]             # (B, N, 1)
            cong_labels = (last_speed < self.congestion_threshold).float()
            loss_cong = F.binary_cross_entropy_with_logits(pred_congestion, cong_labels)

            # Anomaly loss (reconstruction MSE)
            loss_anom = F.mse_loss(recon, h_fused.detach())

            # Contrastive loss
            loss_contrast = nt_xent_loss(h_context, event_flag)

            loss = (
                self.w_speed    * loss_speed
                + self.w_cong   * loss_cong
                + self.w_anom   * loss_anom
                + self.w_contrast * loss_contrast
            )

            out.update({
                "loss":             loss,
                "loss_speed":       loss_speed,
                "loss_congestion":  loss_cong,
                "loss_anomaly":     loss_anom,
                "loss_contrast":    loss_contrast,
            })

        return out


def build_from_config(cfg: dict, N: int) -> HMSTGN:
    m   = cfg["model"]
    dat = cfg["data"]
    pw  = m["prediction_heads"]
    return HMSTGN(
        N=N,
        T=dat["T"],
        F_in=dat["F"],
        K=dat.get("K", 17),            # weather(6) + calendar(7) + events(4)
        H=dat["H"],
        d_model=m["d_model"],
        patch_size=m["patch_transformer"]["patch_size"],
        n_ptt_heads=m["patch_transformer"]["n_heads"],
        n_ptt_layers=m["patch_transformer"]["n_layers"],
        ptt_ffn=m["patch_transformer"]["feedforward_dim"],
        n_gcn=m["gatv2"]["n_layers"],
        d_embed=m["adaptive_graph"]["d_embed"],
        alpha_init=m["adaptive_graph"]["alpha_init"],
        ctx_hidden=m["context_encoder"]["hidden_dim"],
        n_fuse_heads=m["gatv2"]["n_heads"],
        fuse_ffn=m["patch_transformer"]["feedforward_dim"],
        dropout=m["dropout"],
        w_speed=pw["loss_weight_speed"],
        w_cong=pw["loss_weight_congestion"],
        w_anom=pw["loss_weight_anomaly"],
    )
