# src/e2tamr/models/e2t_amr.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class IQEncoder(nn.Module):
    """
    轻量 IQ 编码器：Conv1d -> GELU -> Conv1d -> GELU -> GAP -> Linear
    输入: (B, 2, T)
    输出: (B, hidden)
    """
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(2, hidden, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, stride=1, padding=1)
        self.proj  = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, T)
        h = F.gelu(self.conv1(x))
        h = F.gelu(self.conv2(h))          # (B, hidden, T)
        h = h.mean(dim=-1)                 # GAP over time -> (B, hidden)
        h = self.proj(h)                   # (B, hidden)
        return h


class SpecEncoder(nn.Module):
    """
    频谱编码器：MLP
    输入: (B, F)
    输出: (B, hidden)
    """
    def __init__(self, spec_dim: int = 129, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(spec_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F)
        return self.mlp(x)


class TFEncoder(nn.Module):
    """
    时频图编码器：展平 + MLP
    输入: (B, Tf, Ff)
    输出: (B, hidden)
    """
    def __init__(self, tf_shape: Tuple[int, int] = (33, 5), hidden: int = 64):
        super().__init__()
        self.tf_shape = tf_shape
        flat = tf_shape[0] * tf_shape[1]
        self.mlp = nn.Sequential(
            nn.Linear(flat, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, Tf, Ff)
        B = x.size(0)
        h = x.reshape(B, -1)  # (B, Tf*Ff)
        return self.mlp(h)


class E2TAMR(nn.Module):
    """
    简化版 E2T-AMR：三模态编码 + 级联融合 + 分类头
    - 支持参数:
        num_classes: 类别数
        hidden: 每个分支的隐空间维度
        spec_dim: 频谱维度（默认 129, 因为 rfft(n=256) -> 129 bins）
        tf_shape: (Tf, Ff) 时频图尺寸（默认 (33,5) 与你的 DataModule 对齐）
        use_iq/use_spec/use_tf: 模态开关
    - 前向输入:
        x: Dict[str, Tensor]，可能包含 'iq' (B,2,T), 'spec' (B,F), 'tf' (B,Tf,Ff)
    - 前向输出:
        logits: (B, num_classes)
    """
    def __init__(self,
                 num_classes: int,
                 hidden: int = 64,
                 spec_dim: int = 129,
                 tf_shape: Tuple[int, int] = (33, 5),
                 use_iq: bool = True,
                 use_spec: bool = True,
                 use_tf: bool = True):
        super().__init__()
        assert use_iq or use_spec or use_tf, "At least one modality must be enabled."

        self.use_iq = use_iq
        self.use_spec = use_spec
        self.use_tf = use_tf
        self.num_classes = num_classes
        self.hidden = hidden

        # 分支编码器
        if self.use_iq:
            self.iq_enc = IQEncoder(hidden=hidden)
        if self.use_spec:
            self.spec_enc = SpecEncoder(spec_dim=spec_dim, hidden=hidden)
        if self.use_tf:
            self.tf_enc = TFEncoder(tf_shape=tf_shape, hidden=hidden)

        # 融合后维度
        fused_dim = (hidden if use_iq else 0) + (hidden if use_spec else 0) + (hidden if use_tf else 0)

        # 分类头（可以加 Dropout 提升泛化）
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(0.1),
            nn.Linear(fused_dim, num_classes)
        )

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        feats = []
        if self.use_iq:
            # 期望 x['iq'] 形状为 (B, 2, T)
            iq = x.get("iq", None)
            if iq is None:
                raise RuntimeError("Model expects 'iq' in input dict, but got None while use_iq=True.")
            feats.append(self.iq_enc(iq))

        if self.use_spec:
            # 期望 x['spec'] 形状为 (B, F)
            sp = x.get("spec", None)
            if sp is None:
                raise RuntimeError("Model expects 'spec' in input dict, but got None while use_spec=True.")
            feats.append(self.spec_enc(sp))

        if self.use_tf:
            # 期望 x['tf'] 形状为 (B, Tf, Ff)
            tf = x.get("tf", None)
            if tf is None:
                raise RuntimeError("Model expects 'tf' in input dict, but got None while use_tf=True.")
            feats.append(self.tf_enc(tf))

        z = torch.cat(feats, dim=-1)  # (B, fused_dim)
        logits = self.head(z)         # (B, num_classes)
        return logits
