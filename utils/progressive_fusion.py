"""
渐进式融合调度器

训练早期强调简单特征（纯轨迹分支）的重建损失，后期逐步增大
复杂特征（Cross-Attn 融合分支）的损失权重，缓解骨骼噪声导致的初期不稳定。
"""

import math


class ProgressiveFusionSchedule:
    """按 epoch 线性 / 余弦插值损失权重。"""

    def __init__(self, cfg):
        pf = cfg.get("progressive_fusion") or {}
        self.anneal_epochs = max(1, int(pf.get("anneal_epochs", 50)))
        self.schedule = str(pf.get("schedule", "linear")).lower()
        self.simple_weight_init = float(pf.get("simple_weight_init", 1.0))
        self.simple_weight_final = float(pf.get("simple_weight_final", 0.0))
        self.fusion_weight_init = float(pf.get("fusion_weight_init", 0.0))
        self.fusion_weight_final = float(pf.get("fusion_weight_final", 1.0))
        self.skeleton_freeze_epochs = int(pf.get("skeleton_freeze_epochs", 0))

    def _progress(self, epoch):
        if epoch >= self.anneal_epochs:
            return 1.0
        t = epoch / float(self.anneal_epochs)
        if self.schedule == "cosine":
            return 0.5 * (1.0 - math.cos(math.pi * t))
        return t

    def _interp(self, epoch, init, final):
        p = self._progress(epoch)
        return init + (final - init) * p

    def get_loss_weights(self, epoch):
        w_simple = self._interp(epoch, self.simple_weight_init, self.simple_weight_final)
        w_fusion = self._interp(epoch, self.fusion_weight_init, self.fusion_weight_final)
        return w_simple, w_fusion

    def should_freeze_skeleton(self, epoch):
        return epoch < self.skeleton_freeze_epochs

    def summary(self):
        return (
            f"anneal={self.anneal_epochs} schedule={self.schedule} "
            f"simple {self.simple_weight_init}->{self.simple_weight_final} "
            f"fusion {self.fusion_weight_init}->{self.fusion_weight_final} "
            f"skel_freeze_epochs={self.skeleton_freeze_epochs}"
        )
