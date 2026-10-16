"""
训练期模态 Dropout：随机屏蔽轨迹或骨骼分支，增强单模态鲁棒性。

默认关闭（prob=0）；仅在 model.training=True 时生效，推理/评估不受影响。
"""

from __future__ import annotations

import torch


def apply_modality_dropout(
    traj_feat: torch.Tensor,
    skel_feat: torch.Tensor,
    prob: float,
    mode: str = "random",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    按样本随机丢弃一个模态的特征（置零后仍走融合，与缺失骨骼帧行为一致）。

    Args:
        traj_feat: (B, T, D)
        skel_feat: (B, T, D)
        prob: 每个样本触发 dropout 的概率（如 0.5）
        mode:
            - "random": 触发时 50/50 丢 traj 或 skel
            - "skel": 仅丢骨骼
            - "traj": 仅丢轨迹
    Returns:
        (traj_feat, skel_feat) 可能被置零
    """
    if prob <= 0.0:
        return traj_feat, skel_feat

    B = traj_feat.shape[0]
    device = traj_feat.device
    dropout = torch.rand(B, device=device) < prob
    if not dropout.any():
        return traj_feat, skel_feat

    traj_out = traj_feat.clone()
    skel_out = skel_feat.clone()

    if mode == "skel":
        drop_skel = dropout
        drop_traj = torch.zeros(B, dtype=torch.bool, device=device)
    elif mode == "traj":
        drop_traj = dropout
        drop_skel = torch.zeros(B, dtype=torch.bool, device=device)
    elif mode == "random":
        pick_skel = torch.rand(B, device=device) < 0.5
        drop_skel = dropout & pick_skel
        drop_traj = dropout & (~pick_skel)
    else:
        raise ValueError(f"Unknown modality_dropout_mode: {mode!r}")

    if drop_traj.any():
        traj_out[drop_traj] = 0.0
    if drop_skel.any():
        skel_out[drop_skel] = 0.0

    return traj_out, skel_out
