# src/e2tamr/losses/focal_ce.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CrossEntropyWithLS(nn.Module):
    def __init__(self, num_classes: int, label_smoothing: float = 0.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.num_classes = num_classes
        self.eps = float(label_smoothing)
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # label smoothing 的等价形式：KL + on-hot
        if self.eps > 0:
            log_probs = F.log_softmax(logits, dim=-1)
            with torch.no_grad():
                true_dist = torch.zeros_like(log_probs)
                true_dist.fill_(self.eps / (self.num_classes - 1))
                true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.eps)
            if self.weight is not None:
                # 对每个样本按其目标类加权
                sample_w = self.weight[target]  # (B,)
                loss = -(true_dist * log_probs).sum(dim=-1) * sample_w
                return loss.mean()
            else:
                return -(true_dist * log_probs).sum(dim=-1).mean()
        else:
            return F.cross_entropy(logits, target, weight=self.weight)


class FocalCrossEntropy(nn.Module):
    """
    Focal CE with label smoothing + class weights
    CE(p_t) = -w_y * log p_t
    Focal = (1-p_t)^gamma * CE
    """
    def __init__(self,
                 num_classes: int,
                 gamma: float = 2.0,
                 label_smoothing: float = 0.0,
                 weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        self.eps = float(label_smoothing)
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)               # (B,C)
        probs = torch.exp(log_probs)                            # (B,C)

        if self.eps > 0:
            with torch.no_grad():
                true_dist = torch.zeros_like(log_probs)
                true_dist.fill_(self.eps / (self.num_classes - 1))
                true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.eps)
        else:
            true_dist = torch.zeros_like(log_probs)
            true_dist.scatter_(1, target.unsqueeze(1), 1.0)

        # p_t = sum_c y_c * p_c
        pt = (true_dist * probs).sum(dim=-1).clamp_min(1e-8)    # (B,)
        focal_factor = (1.0 - pt).pow(self.gamma)               # (B,)

        # CE with smoothing: -sum y_smooth * log p
        ce = -(true_dist * log_probs).sum(dim=-1)               # (B,)

        if self.weight is not None:
            sample_w = self.weight[target]                      # (B,)
            loss = focal_factor * ce * sample_w
        else:
            loss = focal_factor * ce

        return loss.mean()
