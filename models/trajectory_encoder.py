"""
轨迹编码器: 将 (x, y) 坐标序列编码为 d_model 维特征

支持:
- 线性投影 + 位置编码
- 速度/加速度增强特征
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """正弦位置编码"""

    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """x: (B, T, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TrajectoryEncoder(nn.Module):
    """
    轨迹编码器

    输入: (B, T, 2)  — 行人坐标 (x, y)
    输出: (B, T, d_model) — 编码后的轨迹特征
    """

    def __init__(self, input_dim=2, d_model=128, dropout=0.1, use_velocity=True):
        super().__init__()
        self.use_velocity = use_velocity

        # 输入维度: pos(2) + vel(2) + acc(2) = 6 或 pos(2)
        feat_dim = input_dim * 3 if use_velocity else input_dim

        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)

    def _compute_motion_features(self, traj):
        """
        计算速度和加速度

        Args:
            traj: (B, T, 2)
        Returns:
            features: (B, T, 6) — [pos, vel, acc]
        """
        # 速度 (一阶差分)
        vel = torch.zeros_like(traj)
        vel[:, 1:, :] = traj[:, 1:, :] - traj[:, :-1, :]

        # 加速度 (二阶差分)
        acc = torch.zeros_like(traj)
        acc[:, 1:, :] = vel[:, 1:, :] - vel[:, :-1, :]

        return torch.cat([traj, vel, acc], dim=-1)  # (B, T, 6)

    def forward(self, traj):
        """
        Args:
            traj: (B, T, 2) — 观测轨迹坐标
        Returns:
            feat: (B, T, d_model)
        """
        if self.use_velocity:
            x = self._compute_motion_features(traj)  # (B, T, 6)
        else:
            x = traj

        x = self.input_proj(x)    # (B, T, d_model)
        x = self.pos_enc(x)
        x = self.norm(x)

        return x
