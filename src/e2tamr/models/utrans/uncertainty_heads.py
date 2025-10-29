# src/e2tamr/models/utrans/uncertainty_heads.py
import torch
import torch.nn as nn
from typing import Optional, Tuple


class TemperatureScaler(nn.Module):
    """可学习温度标定（全局一个标量 T>0）"""
    def __init__(self, init_temp: float = 1.0):
        super().__init__()
        self.logT = nn.Parameter(torch.log(torch.tensor(init_temp)))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        T = torch.exp(self.logT)
        return logits / T.clamp_min(1e-6)


class UncertaintyHead(nn.Module):
    """
    分类 + 不确定性：
      输入: 表征 z (B, d), 以及可选 var (B, d)
      输出: logits (B, C), aux = {"var": ..., "temperature": ...}
    """
    def __init__(self, in_dim: int, num_classes: int, use_temperature: bool = True):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, num_classes)
        )
        self.use_temperature = use_temperature
        self.temperature = TemperatureScaler(1.0) if use_temperature else None

    def forward(self, z: torch.Tensor, var: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        logits = self.classifier(z)  # (B,C)
        if self.use_temperature and self.training is False:
            logits = self.temperature(logits)
        aux = {}
        if var is not None:
            aux["var"] = var
        if self.temperature is not None:
            aux["temperature"] = torch.exp(self.temperature.logT).detach()
        return logits, aux
