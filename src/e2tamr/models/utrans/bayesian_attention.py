# src/e2tamr/models/utrans/bayesian_attention.py
import torch
import torch.nn as nn
from typing import Tuple, Optional


class BayesianSelfAttention(nn.Module):
    """
    基于 TransformerEncoder 的贝叶斯注意力：
      - 使用 dropout 作为隐式变分近似
      - 支持 MC 采样求均值/方差输出
    """
    def __init__(self,
                 d_model: int = 192,
                 nhead: int = 3,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 return_var: bool = True,
                 mc_samples: int = 4):
        super().__init__()
        self.return_var = return_var
        self.mc_samples = mc_samples

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward_once(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (B, L, d_model)
        返回: (B, d_model)
        """
        H = self.encoder(X)       # (B, L, d_model)
        Ht = H.transpose(1, 2)    # (B, d_model, L)
        z = self.pool(Ht).squeeze(-1)   # (B, d_model)
        return z

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        若 return_var=True 且 mc_samples>1：返回 (mu, var)
        否则返回 (z, None)
        """
        if self.return_var and self.training and self.mc_samples > 1:
            zs = []
            for _ in range(self.mc_samples):
                zs.append(self.forward_once(X))
            Z = torch.stack(zs, dim=0)               # (S, B, d)
            mu = Z.mean(dim=0)                       # (B, d)
            var = Z.var(dim=0, unbiased=False)       # (B, d)
            return mu, var
        else:
            z = self.forward_once(X)
            return z, None
