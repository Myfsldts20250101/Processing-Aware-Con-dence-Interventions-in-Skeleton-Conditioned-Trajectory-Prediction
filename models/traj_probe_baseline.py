"""历史轨迹 → 上肢关节偏移 baseline（无预训练主干，仅 obs_traj）。"""

from __future__ import annotations

import torch
import torch.nn as nn


def _obs_kinematics(obs_traj: torch.Tensor):
    """obs_traj (B,T,2) last_obs 相对系 → 位置 + 逐步速度 (B,T,4)。"""
    vel = torch.zeros_like(obs_traj)
    if obs_traj.shape[1] > 1:
        vel[:, 1:] = obs_traj[:, 1:] - obs_traj[:, :-1]
    return torch.cat([obs_traj, vel], dim=-1)


class TrajHistoryProbe(nn.Module):
    """
    obs_traj → 小头 → 未来质心轨迹 + 上肢关节 ΔJ。

    多模态 K：对隐向量加高斯扰动（训练/评测与 CVAE 采样类比，无 KL）。
    """

    def __init__(
        self,
        obs_len=8,
        pred_len=12,
        n_joints=6,
        hidden=128,
        arch='mlp',
        nhead=4,
        num_layers=2,
        dropout=0.1,
        noise_scale=0.15,
    ):
        super().__init__()
        self.obs_len = int(obs_len)
        self.pred_len = int(pred_len)
        self.n_joints = int(n_joints)
        self.hidden = int(hidden)
        self.arch = str(arch)
        self.noise_scale = float(noise_scale)
        feat_dim = 4

        if self.arch == 'transformer':
            self.step_embed = nn.Linear(feat_dim, hidden)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=nhead,
                dim_feedforward=hidden * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        else:
            in_dim = self.obs_len * feat_dim
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.centroid_head = nn.Linear(hidden, self.pred_len * 2)
        self.joint_head = nn.Linear(hidden, self.pred_len * self.n_joints * 2)

    def _encode(self, obs_traj: torch.Tensor) -> torch.Tensor:
        feats = _obs_kinematics(obs_traj)
        if self.arch == 'transformer':
            h = self.step_embed(feats)
            h = self.encoder(h)
            return h.mean(dim=1)
        return self.encoder(feats.reshape(feats.shape[0], -1))

    def forward(self, obs_traj: torch.Tensor, num_samples: int = 1):
        """
        Returns:
            pred_traj: (B, K, T_pred, 2)
            joint_delta: (B, K, T_pred, n_joints, 2)
        """
        h = self._encode(obs_traj)
        b = h.shape[0]
        k = max(1, int(num_samples))

        if k > 1:
            noise = torch.randn(b, k, self.hidden, device=h.device, dtype=h.dtype)
            h = h.unsqueeze(1) + self.noise_scale * noise
        else:
            h = h.unsqueeze(1)

        flat = h.reshape(b * k, self.hidden)
        pred = self.centroid_head(flat).view(b, k, self.pred_len, 2)
        jdelta = self.joint_head(flat).view(b, k, self.pred_len, self.n_joints, 2)
        return pred, jdelta

    def param_count(self):
        return sum(p.numel() for p in self.parameters())
