# src/e2tamr/models/cget/complex_conv.py
import torch
import torch.nn as nn

class ComplexConv1d(nn.Module):
    """
    复数卷积：输入 (B, 2, T) -> 输出 (B, 2*out_ch, T)
    约定通道维度：x[:,0]=实部，x[:,1]=虚部
    wr/wi 均作用在单通道 (1) 上，然后按复卷积公式合成：
      y_r = conv(x_r, wr) - conv(x_i, wi)
      y_i = conv(x_i, wr) + conv(x_r, wi)
    """
    def __init__(self, in_ch=2, out_ch=32, kernel_size=5, padding=2, bias=False):
        super().__init__()
        # 注意：这里输入通道是 1（分别处理实/虚部）
        self.wr = nn.Conv1d(1, out_ch, kernel_size, padding=padding, bias=bias)
        self.wi = nn.Conv1d(1, out_ch, kernel_size, padding=padding, bias=bias)

    def forward(self, x):
        # x: (B, 2, T)
        xr, xi = x[:, 0:1], x[:, 1:2]      # (B,1,T), (B,1,T)
        yr = self.wr(xr) - self.wi(xi)     # (B,out_ch,T)
        yi = self.wr(xi) + self.wi(xr)     # (B,out_ch,T)
        return torch.cat([yr, yi], dim=1)  # (B, 2*out_ch, T)

class ComplexBlock(nn.Module):
    """复数卷积块：Conv -> BN -> GELU"""
    def __init__(self, in_ch=2, hidden=32, kernel=5):
        super().__init__()
        self.conv = ComplexConv1d(in_ch=in_ch, out_ch=hidden, kernel_size=kernel, padding=kernel//2)
        self.norm = nn.BatchNorm1d(2*hidden)
        self.act  = nn.GELU()

    def forward(self, x):
        y = self.conv(x)       # (B, 2*hidden, T)
        y = self.norm(y)
        y = self.act(y)
        return y
