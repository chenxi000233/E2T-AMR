# src/e2tamr/models/emixer/fusion_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class EnergyWeightedFusion(nn.Module):
    """
    多模态融合（能量加权 Cross-Attention）：
      - 输入: 一组模态向量 [z_iq, z_spec, z_tf]，每个 (B, D)
      - 步骤:
         1) 将每个向量映射成 token (B, 1, D_model)
         2) 堆叠成序列 (B, M, D_model)，M=模态数
         3) 通过 MHA 融合，输出 (B, D_model)
         4) 能量门控 g (B,M) 对 value 做缩放
    """
    def __init__(self, in_dim: int = 128, d_model: int = 192, nhead: int = 3, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, in_dim)  # 回到 in_dim（可选）

    def forward(self, zs: List[torch.Tensor], g: torch.Tensor) -> torch.Tensor:
        """
        zs: [z_iq, z_spec, z_tf], 每个 (B, D_in)
        g : (B, M) 能量门控权重
        返回: z_fused (B, D_in)
        """
        B = zs[0].shape[0]
        tokens = [self.proj(z).unsqueeze(1) for z in zs]  # 列表[(B,1,d_model), ...]
        X = torch.cat(tokens, dim=1)  # (B, M, d_model)

        # 将 g 作为 value 缩放：乘在 token 上（等价于对 V 缩放）
        g_ = g.unsqueeze(-1)          # (B, M, 1)
        X_g = X * g_                  # (B, M, d_model)

        H = self.encoder(X_g)         # (B, M, d_model)
        # 池化：能量加权平均
        H_pool = (H * g_).sum(dim=1) / (g_.sum(dim=1) + 1e-8)   # (B, d_model)
        z = self.out(H_pool)                                    # (B, D_in)
        return z
