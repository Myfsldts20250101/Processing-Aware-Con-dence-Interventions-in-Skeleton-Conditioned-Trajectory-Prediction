"""
损失函数

- 轨迹重建损失 (L2 / Variety Loss)
- KL 散度 with annealing
- Best-of-K 训练策略
- 渐进式融合双分支损失权重调度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .progressive_fusion import ProgressiveFusionSchedule


class TrajectoryLoss(nn.Module):
    """
    轨迹预测损失函数

    支持:
    1. L2 重建损失
    2. Best-of-K (Variety Loss): 训练时采多条取最优
    3. KL annealing
    """

    def __init__(self, kl_weight=0.1, kl_weight_final=1.0,
                 kl_anneal_epochs=50, best_of_k=1):
        super().__init__()
        self.kl_weight = kl_weight
        self.kl_weight_final = kl_weight_final
        self.kl_anneal_epochs = kl_anneal_epochs
        self.best_of_k = best_of_k  # Best-of-K 采样数 (训练时)

    def get_kl_weight(self, epoch):
        """KL Annealing: 从 kl_weight 线性增长到 kl_weight_final"""
        if epoch >= self.kl_anneal_epochs:
            return self.kl_weight_final
        progress = epoch / self.kl_anneal_epochs
        return self.kl_weight + (self.kl_weight_final - self.kl_weight) * progress

    def l2_loss(self, pred, target):
        """
        L2 重建损失

        Args:
            pred: (B, T_pred, 2)
            target: (B, T_pred, 2)
        Returns:
            loss: scalar
        """
        return F.mse_loss(pred, target)

    def variety_loss(self, pred_trajs, target):
        """
        Best-of-K Variety Loss

        从 K 条预测中选最接近 GT 的一条计算损失。
        鼓励模型生成多样的合理轨迹。

        Args:
            pred_trajs: (B, K, T_pred, 2) — K 条预测轨迹
            target: (B, T_pred, 2) — GT
        Returns:
            loss: scalar
        """
        B, K, T, _ = pred_trajs.shape

        # 计算每条轨迹的 L2 距离
        target_exp = target.unsqueeze(1).expand_as(pred_trajs)  # (B, K, T, 2)
        dists = torch.norm(pred_trajs - target_exp, dim=-1)      # (B, K, T)
        ade_per_sample = dists.mean(dim=-1)                       # (B, K)

        # 选最优的一条
        best_idx = ade_per_sample.argmin(dim=-1)                  # (B,)
        best_pred = pred_trajs[torch.arange(B), best_idx]         # (B, T, 2)

        return F.mse_loss(best_pred, target)

    def forward(self, pred_traj, target_traj, kl_loss, epoch=0):
        """
        总损失

        Args:
            pred_traj: (B, T_pred, 2) 或 (B, K, T_pred, 2)
            target_traj: (B, T_pred, 2)
            kl_loss: scalar — CVAE KL 散度
            epoch: 当前 epoch (用于 KL annealing)
        Returns:
            total_loss: scalar
            loss_dict: dict — 各项损失分量
        """
        # 重建损失
        if pred_traj.dim() == 4:
            recon_loss = self.variety_loss(pred_traj, target_traj)
        else:
            recon_loss = self.l2_loss(pred_traj, target_traj)

        # KL 权重
        kl_w = self.get_kl_weight(epoch)
        weighted_kl = kl_w * kl_loss

        # 总损失
        total = recon_loss + weighted_kl

        loss_dict = {
            'total': total.item(),
            'recon': recon_loss.item(),
            'kl': kl_loss.item(),
            'kl_weight': kl_w,
        }

        return total, loss_dict


class ProgressiveFusionLoss(TrajectoryLoss):
    """
    渐进式融合损失：简单（轨迹-only）与复杂（Cross-Attn 融合）双分支重建，
    权重随 epoch 从简单主导过渡到融合主导。
    """

    def __init__(self, progressive_cfg, **kwargs):
        super().__init__(**kwargs)
        self.schedule = ProgressiveFusionSchedule({"progressive_fusion": progressive_cfg})

    def _recon_loss(self, pred_traj, target_traj):
        if pred_traj.dim() == 4:
            return self.variety_loss(pred_traj, target_traj)
        return self.l2_loss(pred_traj, target_traj)

    def forward(self, pred_simple, pred_fusion, target_traj, kl_loss, epoch=0):
        recon_simple = self._recon_loss(pred_simple, target_traj)
        recon_fusion = self._recon_loss(pred_fusion, target_traj)

        w_simple, w_fusion = self.schedule.get_loss_weights(epoch)
        kl_w = self.get_kl_weight(epoch)
        weighted_recon = w_simple * recon_simple + w_fusion * recon_fusion
        total = weighted_recon + kl_w * kl_loss

        loss_dict = {
            "total": total.item(),
            "recon": weighted_recon.item(),
            "recon_simple": recon_simple.item(),
            "recon_fusion": recon_fusion.item(),
            "kl": kl_loss.item(),
            "kl_weight": kl_w,
            "pf_w_simple": w_simple,
            "pf_w_fusion": w_fusion,
        }
        return total, loss_dict


class DualBranchLoss(TrajectoryLoss):
    """
    双分支固定权重损失：简单（轨迹-only）与融合分支等权或自定义权重。
    用于「共享 z vs 解耦 z」对照，不含渐进式 annealing。
    """

    def __init__(
        self,
        branch_weight_simple=0.5,
        branch_weight_fusion=0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.w_simple = float(branch_weight_simple)
        self.w_fusion = float(branch_weight_fusion)

    def _recon_loss(self, pred_traj, target_traj):
        if pred_traj.dim() == 4:
            return self.variety_loss(pred_traj, target_traj)
        return self.l2_loss(pred_traj, target_traj)

    def forward(
        self,
        pred_simple,
        pred_fusion,
        target_traj,
        kl_loss,
        epoch=0,
        kl_simple=None,
        kl_fusion=None,
    ):
        recon_simple = self._recon_loss(pred_simple, target_traj)
        recon_fusion = self._recon_loss(pred_fusion, target_traj)

        kl_w = self.get_kl_weight(epoch)
        weighted_recon = self.w_simple * recon_simple + self.w_fusion * recon_fusion
        total = weighted_recon + kl_w * kl_loss

        loss_dict = {
            "total": total.item(),
            "recon": weighted_recon.item(),
            "recon_simple": recon_simple.item(),
            "recon_fusion": recon_fusion.item(),
            "kl": kl_loss.item(),
            "kl_weight": kl_w,
            "db_w_simple": self.w_simple,
            "db_w_fusion": self.w_fusion,
        }
        if kl_simple is not None:
            loss_dict["kl_simple"] = kl_simple.item()
        if kl_fusion is not None:
            loss_dict["kl_fusion"] = kl_fusion.item()
        return total, loss_dict
