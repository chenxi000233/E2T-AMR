# src/e2tamr/models/cget/invariance_losses.py
import torch

def shift_consistency_loss(f_xt, f_xt_tau):
    """
    平移一致性损失的占位函数：
    你可以设为 MSE(f(x), Shift(f(x), τ)) 或者特征对齐对比损失
    """
    return torch.tensor(0.0, device=f_xt.device)

def phase_norm(x, eps=1e-8):
    """
    相位归一化占位：z <- z / |z|
    x: (B, 2*C, T) 视作复特征
    """
    B, C2, T = x.shape
    C = C2 // 2
    real, imag = x[:, :C], x[:, C:]
    mag = torch.sqrt(real**2 + imag**2 + eps)
    real, imag = real / mag, imag / mag
    return torch.cat([real, imag], dim=1)
