# src/e2tamr/models/emixer/energy_estimator.py
import torch
import torch.nn as nn
from typing import Dict, Optional


class EnergyEstimator(nn.Module):
    """
    计算各模态能量置信度:
      - I/Q: 平均 |x|^2
      - Spectrum: 平均 PSD
      - TF: 平均 |STFT| (若输入为特征向量，则用其L2范数近似)
    支持 frame/token 级别的能量输出以做细粒度门控。
    """
    def __init__(self, normalize: bool = True, eps: float = 1e-12, mode: str = "global"):
        super().__init__()
        assert mode in ["global", "frame"], "mode must be 'global' or 'frame'"
        self.normalize = normalize
        self.eps = eps
        self.mode = mode

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        feats: {
          "iq_raw": (B,2,T)            # 可选：用于能量估计
          "spec_psd": (B,1,F)          # 可选：用于能量估计
          "tf_mag": (B,1,F,T')         # 可选：用于能量估计
          "z_iq": (B,D), "z_spec": (B,D), "z_tf": (B,D)  # 各模态编码后的向量
        }
        返回:
          {"E_iq": (B,1), "E_spec": (B,1), "E_tf": (B,1)}  # 或 (B,L) if mode=='frame'
        """
        out = {}

        # I/Q energy
        if "iq_raw" in feats:
            x = feats["iq_raw"]  # (B,2,T)
            E_iq = (x[:,0].pow(2) + x[:,1].pow(2)).mean(dim=-1)  # (B,)
        elif "z_iq" in feats:
            z = feats["z_iq"]
            E_iq = z.pow(2).sum(dim=-1) / (z.shape[-1] + self.eps)
        else:
            E_iq = None

        # Spectrum energy
        if "spec_psd" in feats:
            psd = feats["spec_psd"]  # (B,1,F)
            E_spec = psd.mean(dim=(-1,-2))  # (B,)
        elif "z_spec" in feats:
            z = feats["z_spec"]
            E_spec = z.pow(2).sum(dim=-1) / (z.shape[-1] + self.eps)
        else:
            E_spec = None

        # TF energy
        if "tf_mag" in feats:
            mag = feats["tf_mag"]  # (B,1,F,T')
            E_tf = mag.mean(dim=(-1,-2))  # (B,)
        elif "z_tf" in feats:
            z = feats["z_tf"]
            E_tf = z.pow(2).sum(dim=-1) / (z.shape[-1] + self.eps)
        else:
            E_tf = None

        energies = []
        if E_iq is not None:
            energies.append(("E_iq", E_iq))
        if E_spec is not None:
            energies.append(("E_spec", E_spec))
        if E_tf is not None:
            energies.append(("E_tf", E_tf))

        if self.normalize and energies:
            # softmax 归一化
            stacked = torch.stack([e for _, e in energies], dim=-1)  # (B, M)
            weights = torch.softmax(stacked, dim=-1)                 # (B, M)
            for i, (k, _) in enumerate(energies):
                out[k] = weights[:, i:i+1]                           # (B,1)
        else:
            for k, e in energies:
                out[k] = e.unsqueeze(-1) if e.ndim == 1 else e

        return out
