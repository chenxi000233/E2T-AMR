# src/e2tamr/models/cget/steerable_posenc.py
import math
import torch
import torch.nn as nn

class SteerableFourierPosEnc(nn.Module):
    """
    P(t) = [sin(2π f t/T), cos(2π f t/T)]_f
    平移 τ -> 仅相位旋转：P(t+τ) = R(τ) P(t)
    产出 (B, Cpe, T)，可与特征拼接或相加
    """
    def __init__(self, T: int, freqs=(1,2,4,8), learnable_scale=False):
        super().__init__()
        self.T = T
        self.freqs = nn.Parameter(torch.tensor(freqs, dtype=torch.float32), requires_grad=False)
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=learnable_scale)

    def forward(self, B: int, T: int, device=None):
        t = torch.arange(T, dtype=torch.float32, device=device)
        pe = []
        for f in self.freqs:
            arg = 2*math.pi*f*t/self.T * self.alpha
            pe.append(torch.sin(arg))
            pe.append(torch.cos(arg))
        pe = torch.stack(pe, dim=0)                 # (2*F, T)
        pe = pe.unsqueeze(0).repeat(B, 1, 1)        # (B, 2*F, T)
        return pe
