# src/e2tamr/models/backbones/spectrogram.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def _safe_real_stft(x: torch.Tensor, n_fft: int, hop_length: int, win_length: int, window: torch.Tensor):
    """
    x: (B, T) 实数
    返回: 复 STFT 张量 (B, F, T_frames) ，复数形式（兼容新旧 torch）
    """
    # 新版 PyTorch (return_complex=True)
    try:
        S = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            return_complex=True
        )  # (B, F, T')
        return S
    except TypeError:
        # 旧版：return_complex 不存在，返回 real-imag 最后一维
        S_ri = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            return_complex=False
        )  # (B, F, T', 2)
        S = torch.view_as_complex(S_ri)  # (B, F, T')
        return S


class PowerSpectrumEncoder(nn.Module):
    """
    I/Q -> FFT -> 正频半边功率谱(PSD) -> 1D Conv Encoder -> vector
    输入: x (B, 2, T), x[:,0]=I, x[:,1]=Q
    输出: z (B, out_dim)
    """
    def __init__(self, input_len: int = 128, out_dim: int = 128, n_fft: int = 128):
        super().__init__()
        self.n_fft = n_fft
        self.input_len = input_len

        # 1D conv stack on PSD
        self.feat = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, T)
        B, _, T = x.shape
        # 合成复数基带
        xc = torch.complex(x[:, 0], x[:, 1])  # (B, T)

        # 对复数做 FFT（注意：复数不能用 rfft）
        Xf = torch.fft.fft(xc, n=self.n_fft, dim=-1)   # (B, F_full), complex

        # 功率谱 |X(f)|^2
        psd_full = Xf.abs().pow(2)                     # (B, F_full)

        # 只取正频半边（含 DC 和 Nyquist）：等价于 rfft 的频率范围
        F_pos = self.n_fft // 2 + 1
        psd = psd_full[..., :F_pos].unsqueeze(1)       # (B, 1, F_pos)

        # 可选归一化，避免随 n_fft 改变能量尺度
        psd = psd / (self.n_fft + 1e-12)

        # 1D 卷积特征提取
        h = self.feat(psd)         # (B, 128, F_pos)
        z = self.head(h)           # (B, out_dim)
        return z


class STFTEncoder(nn.Module):
    """
    I/Q -> STFT -> |S| -> 2D CNN -> vector
    为兼容旧版 torch：分别对 I、Q 做 STFT，再合成幅度 sqrt(|S_I|^2 + |S_Q|^2)。
    """
    def __init__(self,
                 n_fft: int = 64,
                 hop_length: int = 32,
                 win_length: int = 64,
                 out_dim: int = 128):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop_length
        self.win = win_length
        self.register_buffer("window", torch.hann_window(self.win, periodic=True))

        # small 2D CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, T)
        B, _, T = x.shape
        device = x.device
        window = self.window.to(device)

        # 分别对 I、Q 做 STFT
        I = x[:, 0]  # (B, T)
        Q = x[:, 1]  # (B, T)

        S_I = _safe_real_stft(I, n_fft=self.n_fft, hop_length=self.hop, win_length=self.win, window=window)  # (B,F,T')
        S_Q = _safe_real_stft(Q, n_fft=self.n_fft, hop_length=self.hop, win_length=self.win, window=window)  # (B,F,T')

        # 合成复幅值（等价于 I+jQ 的 STFT 幅度）
        mag = torch.sqrt(S_I.abs().pow(2) + S_Q.abs().pow(2))  # (B,F,T')
        mag = mag.unsqueeze(1)  # (B,1,F,T')

        h = self.cnn(mag)       # (B,128,F,T')
        z = self.head(h)        # (B,out_dim)
        return z
