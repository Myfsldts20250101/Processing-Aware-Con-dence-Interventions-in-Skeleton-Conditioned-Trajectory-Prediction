"""关节 Probe 头：从 Transformer Decoder 隐层解出未来关节偏移。"""

import torch
import torch.nn as nn


class SkeletonProbeHead(nn.Module):
    """
    通用骨架 Probe 头（上肢 6 点或全 17 点）。

    输入:  H (B, K, T_pred, d_model)
    输出:  ΔJ (B, K, T_pred, n_joints, 2) — 相对末观测帧对应关节位置的偏移
    """

    def __init__(self, d_model=128, n_joints=17, hidden=64):
        super().__init__()
        self.n_joints = n_joints
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_joints * 2),
        )

    def forward(self, H):
        if H.dim() == 4:
            B, K, T, D = H.shape
            flat = H.reshape(B * K, T, D)
            out = self.mlp(flat).view(B * K, T, self.n_joints, 2)
            return out.view(B, K, T, self.n_joints, 2)
        out = self.mlp(H)
        return out.view(H.shape[0], H.shape[1], self.n_joints, 2)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


class UpperLimbProbeHead(SkeletonProbeHead):
    """上肢 6 点 Probe 头（兼容旧接口）。"""

    def __init__(self, d_model=128, n_joints=6, hidden=64):
        super().__init__(d_model=d_model, n_joints=n_joints, hidden=hidden)


FullSkeletonProbeHead = SkeletonProbeHead
