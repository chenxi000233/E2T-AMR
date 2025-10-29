# src/e2tamr/models/emixer/energy_gate.py
import torch
import torch.nn as nn
from typing import Dict, List


class EnergyGate(nn.Module):
    """
    E -> gate weights g_m in [0,1]
    支持 softmax 或 sigmoid（默认 sigmoid 后再归一化，避免极端塌缩）
    """
    def __init__(self, in_dim: int = 3, hidden: int = 16, mode: str = "sigmoid"):
        super().__init__()
        assert mode in ["sigmoid", "softmax"]
        self.mode = mode
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_dim)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, energies: Dict[str, torch.Tensor], order: List[str]) -> torch.Tensor:
        """
        energies: {"E_iq":(B,1), "E_spec":(B,1), "E_tf":(B,1)}
        order:    ["E_iq","E_spec","E_tf"] 指定模态顺序
        返回: g (B, M)
        """
        B = None
        vecs = []
        for k in order:
            v = energies[k]  # (B,1)
            B = v.shape[0]
            vecs.append(v)
        E = torch.cat(vecs, dim=-1)  # (B,M)
        h = self.net(E)              # (B,M)
        if self.mode == "softmax":
            g = torch.softmax(h, dim=-1)  # (B,M)
        else:
            g = self.sigmoid(h)           # (B,M)
            g = g / (g.sum(dim=-1, keepdim=True) + 1e-8)  # 归一化
        return g  # (B,M)
