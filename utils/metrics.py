"""
评估指标: ADE / FDE / 上肢关节误差 / 安全预警

- ADE (Average Displacement Error): 所有预测帧的平均 L2 误差
- FDE (Final Displacement Error): 最后一帧的 L2 误差
- minADE / minFDE: K 条预测中最优的 ADE/FDE
- Joint ADE/FDE: 由质心预测 + 末帧骨骼刚性平移得到的上肢关节误差
- Safety warning: 以机械臂危险区为中心，统计肢体入侵预警成功率
"""

import torch
import numpy as np


# COCO-17 上肢关键点索引
UPPER_LIMB_JOINTS = (5, 6, 7, 8, 9, 10)
ELBOW_JOINTS = (7, 8)
WRIST_JOINTS = (9, 10)
SHOULDER_JOINTS = (5, 6)

JOINT_NAMES = {
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
}

DEFAULT_SAFETY_JOINTS = ELBOW_JOINTS + WRIST_JOINTS


def compute_ade(pred, target):
    """
    Average Displacement Error

    Args:
        pred: (B, T_pred, 2) 或 (T_pred, 2)
        target: (B, T_pred, 2) 或 (T_pred, 2)
    Returns:
        ade: scalar
    """
    error = torch.norm(pred - target, dim=-1)  # (B, T) 或 (T,)
    return error.mean()


def compute_fde(pred, target):
    """
    Final Displacement Error

    Args:
        pred: (B, T_pred, 2)
        target: (B, T_pred, 2)
    Returns:
        fde: scalar
    """
    error = torch.norm(pred[:, -1, :] - target[:, -1, :], dim=-1)  # (B,)
    return error.mean()


def compute_min_ade_fde(pred_trajs, target, return_best=False):
    """
    minADE / minFDE (Best-of-K)

    从 K 条预测轨迹中选最优的计算

    Args:
        pred_trajs: (B, K, T_pred, 2) — K 条预测轨迹
        target: (B, T_pred, 2) — GT
        return_best: 是否返回最优轨迹索引
    Returns:
        min_ade: scalar
        min_fde: scalar
        (optional) best_indices: (B,)
    """
    B, K, T, _ = pred_trajs.shape

    target_exp = target.unsqueeze(1).expand_as(pred_trajs)  # (B, K, T, 2)
    errors = torch.norm(pred_trajs - target_exp, dim=-1)     # (B, K, T)

    # ADE per sample per trajectory
    ade_per = errors.mean(dim=-1)  # (B, K)
    # FDE per sample per trajectory
    fde_per = errors[:, :, -1]     # (B, K)

    # Best of K
    min_ade_vals, best_ade_idx = ade_per.min(dim=-1)  # (B,)
    min_fde_vals, best_fde_idx = fde_per.min(dim=-1)  # (B,)

    min_ade = min_ade_vals.mean()
    min_fde = min_fde_vals.mean()

    if return_best:
        return min_ade, min_fde, best_ade_idx
    return min_ade, min_fde


def compute_multiframe_min_ade_fde(
    pred_trajs,
    target,
    horizons,
    hz: float = 2.5,
):
    """
    多预测步长 minADE / minFDE（Best-of-K）。

    Args:
        pred_trajs: (B, K, T_pred, 2)
        target: (B, T_pred, 2)
        horizons: 预测帧数列表，如 [5, 8, 12]（@2.5Hz → 2.0s / 3.2s / 4.8s）
        hz: 采样率，用于附带秒数说明
    Returns:
        dict: minADE@{h}f, minFDE@{h}f, meanADE@{h}f, meanFDE@{h}f, seconds@{h}f
    """
    if pred_trajs.dim() != 4:
        raise ValueError(f"pred_trajs must be (B,K,T,2), got {pred_trajs.shape}")
    B, K, T, _ = pred_trajs.shape
    target_exp = target.unsqueeze(1).expand_as(pred_trajs)
    errors = torch.norm(pred_trajs - target_exp, dim=-1)  # (B, K, T)

    out = {}
    for h in horizons:
        h = min(int(h), T)
        ade_per = errors[:, :, :h].mean(dim=-1)   # (B, K)
        fde_per = errors[:, :, h - 1]             # (B, K)
        mean_ade = ade_per.mean(dim=-1)           # (B,)
        mean_fde = fde_per.mean(dim=-1)

        out[f'minADE@{h}f'] = float(ade_per.min(dim=-1)[0].mean().item())
        out[f'minFDE@{h}f'] = float(fde_per.min(dim=-1)[0].mean().item())
        out[f'meanADE@{h}f'] = float(mean_ade.mean().item())
        out[f'meanFDE@{h}f'] = float(mean_fde.mean().item())
        out[f'seconds@{h}f'] = round(h / float(hz), 2)
    return out


DEFAULT_PRED_HORIZONS = (5, 8, 12)


def _nanmin(x, dim=-1):
    """兼容旧版 PyTorch 的 nanmin。"""
    xf = x.clone()
    xf[torch.isnan(xf)] = float('inf')
    vals, idx = xf.min(dim=dim)
    vals = vals.masked_fill(torch.isinf(vals), float('nan'))
    return vals, idx


def _nanmean(x, dim=-1):
    if hasattr(torch, 'nanmean'):
        return torch.nanmean(x, dim=dim)
    valid = ~torch.isnan(x)
    return x.sum(dim=dim) / valid.sum(dim=dim).clamp(min=1).float()


def predict_joints_from_trajectory(last_obs_joints, last_obs_pos, pred_traj):
    """由质心轨迹预测推断未来关节位置（刚性平移假设）。

    Args:
        last_obs_joints: (B, V, 2) — 末观测帧骨骼，window_origin 相对坐标
        last_obs_pos: (B, 2) — 末观测质心，window_origin 相对坐标
        pred_traj: (B, K, T, 2) — 预测质心，last_obs 相对坐标
    Returns:
        pred_joints: (B, K, T, V, 2) — 推断关节，last_obs 相对坐标
    """
    base = last_obs_joints - last_obs_pos.unsqueeze(1)
    return base.unsqueeze(1).unsqueeze(2) + pred_traj.unsqueeze(-2)


def _joint_valid_mask(conf, joint_indices, min_conf=0.1):
    """(B, T, V) bool — 指定关节在帧 t 是否有效。"""
    idx = list(joint_indices)
    return (conf[..., idx] > min_conf).all(dim=-1)


def compute_joint_ade_fde(
    pred_joints,
    gt_joints,
    gt_conf=None,
    joint_indices=None,
    min_conf=0.1,
    return_per_joint=False,
):
    """计算关节级 ADE/FDE（Best-of-K 口径与质心 minADE 一致）。

    Args:
        pred_joints: (B, K, T, V, 2)
        gt_joints: (B, T, V, 2)
        gt_conf: (B, T, V) 可选
        joint_indices: 参与统计的关节索引，默认 UPPER_LIMB_JOINTS
    Returns:
        dict: min_joint_ade, min_joint_fde, (optional) per_joint_ade/fde
    """
    if joint_indices is None:
        joint_indices = UPPER_LIMB_JOINTS
    idx = list(joint_indices)

    gt_sel = gt_joints[:, :, idx, :].unsqueeze(1)
    pred_sel = pred_joints[:, :, :, idx, :]
    errors = torch.norm(pred_sel - gt_sel, dim=-1)  # (B, K, T, J)

    if gt_conf is not None:
        valid = _joint_valid_mask(gt_conf, idx, min_conf=min_conf)
        valid = valid.unsqueeze(1)
        errors = errors.masked_fill(~valid.unsqueeze(-1), float('nan'))

    ade_per = _nanmean(errors, dim=2)  # (B, K, J)
    fde_per = errors[:, :, -1, :]

    min_ade_per_sample, _ = _nanmin(_nanmean(ade_per, dim=-1), dim=-1)
    min_fde_per_sample, _ = _nanmin(_nanmean(fde_per, dim=-1), dim=-1)

    out = {
        'min_joint_ade': float(min_ade_per_sample.nanmean().item()),
        'min_joint_fde': float(min_fde_per_sample.nanmean().item()),
    }

    if return_per_joint:
        best_idx = _nanmin(_nanmean(ade_per, dim=-1), dim=-1)[1]
        b = torch.arange(errors.shape[0], device=errors.device)
        best_ade = ade_per[b, best_idx]
        best_fde = fde_per[b, best_idx]
        for j_i, j in enumerate(idx):
            name = JOINT_NAMES.get(j, f'joint_{j}')
            out[f'min_joint_ade_{name}'] = float(best_ade[:, j_i].nanmean().item())
            out[f'min_joint_fde_{name}'] = float(best_fde[:, j_i].nanmean().item())

    return out


def _first_intrusion_frame(positions, danger_center, danger_radius, valid_mask=None):
    """positions: (N, T, J, 2) → 任一监测关节首次进入危险区的帧索引，无入侵返回 T。

    danger_center: (2,) 或 (N, 2)
    """
    if danger_center.ndim == 1:
        center = danger_center.view(1, 1, 1, 2)
    else:
        center = danger_center.view(-1, 1, 1, 2)
    dist = torch.norm(positions - center, dim=-1)
    if isinstance(danger_radius, torch.Tensor) and danger_radius.ndim > 0:
        radius = danger_radius.to(dist.device, dtype=dist.dtype).view(-1, 1, 1)
        inside = dist <= radius
    else:
        inside = dist <= float(danger_radius)
    if valid_mask is not None:
        if valid_mask.ndim == inside.ndim:
            inside = inside & valid_mask
        else:
            inside = inside & valid_mask.unsqueeze(-1)
    frame_inside = inside.any(dim=-1)
    has_any = frame_inside.any(dim=-1)
    first = frame_inside.float().argmax(dim=-1)
    T = positions.shape[1]
    first = torch.where(has_any, first, torch.full_like(first, T))
    return first


def compute_safety_warning_metrics(
    pred_joints,
    gt_joints,
    danger_center,
    danger_radius,
    gt_conf=None,
    monitor_joints=None,
    min_conf=0.1,
    lead_time_tolerance=0,
):
    """安全预警指标：统计模型是否成功预警肢体进入危险区。

    定义（逐样本、Best-of-K）:
      - GT 入侵: 监测关节（默认肘/腕）任一进入以 danger_center 为圆心、
        danger_radius 为半径的圆
      - 预测入侵: 同上，基于推断关节轨迹
      - 成功预警: GT 有入侵 且 pred 首次入侵帧 <= GT 首次入侵帧 + tolerance

    Args:
        pred_joints: (B, K, T, V, 2) last_obs 相对坐标
        gt_joints: (B, T, V, 2)
        danger_center: (2,) 与 gt_joints 同坐标系
        danger_radius: scalar 或 (B,) 逐样本半径
        gt_conf: (B, T, V)
        monitor_joints: 默认肘+腕
        lead_time_tolerance: 允许预测略晚于 GT 的帧数
    Returns:
        dict
    """
    if monitor_joints is None:
        monitor_joints = DEFAULT_SAFETY_JOINTS
    idx = list(monitor_joints)

    gt_sel = gt_joints[:, :, idx, :]
    pred_sel = pred_joints[:, :, :, idx, :]

    if gt_conf is not None:
        valid = gt_conf[:, :, idx] > min_conf
    else:
        valid = torch.ones(
            gt_sel.shape[0], gt_sel.shape[1], len(idx),
            dtype=torch.bool, device=gt_sel.device,
        )

    center = danger_center.to(gt_sel.device, dtype=gt_sel.dtype)
    if isinstance(danger_radius, torch.Tensor):
        radius = danger_radius.to(gt_sel.device, dtype=gt_sel.dtype)
    else:
        radius = float(danger_radius)

    gt_first = _first_intrusion_frame(
        gt_sel,
        center,
        radius,
        valid_mask=valid,
    )
    gt_has = gt_first < gt_sel.shape[1]

    B, K, T, J, _ = pred_sel.shape
    pred_flat = pred_sel.reshape(B * K, T, J, 2)
    valid_flat = valid.unsqueeze(1).expand(B, K, T, J).reshape(B * K, T, J)
    center_pred = center.unsqueeze(1).expand(B, K, 2).reshape(B * K, 2)
    if isinstance(radius, torch.Tensor) and radius.shape[0] == B:
        radius_pred = radius.unsqueeze(1).expand(B, K).reshape(B * K)
    else:
        radius_pred = radius
    pred_first_flat = _first_intrusion_frame(
        pred_flat, center_pred, radius_pred, valid_mask=valid_flat,
    )
    pred_first = pred_first_flat.view(B, K)
    pred_first_best = pred_first.min(dim=-1)[0]
    pred_has = pred_first_best < T

    tol = int(lead_time_tolerance)
    success = gt_has & (pred_first_best <= gt_first + tol)
    false_alarm = (~gt_has) & pred_has
    missed = gt_has & ~success

    lead = (gt_first - pred_first_best).float()
    lead_success = lead[success]

    return {
        'safety_gt_events': int(gt_has.sum().item()),
        'safety_success_count': int(success.sum().item()),
        'safety_success_rate': float(success.sum().item() / max(gt_has.sum().item(), 1)),
        'safety_missed_count': int(missed.sum().item()),
        'safety_false_alarm_count': int(false_alarm.sum().item()),
        'safety_false_alarm_rate': float(false_alarm.sum().item() / max(B, 1)),
        'safety_mean_lead_frames': float(lead_success.mean().item()) if success.any() else 0.0,
        'safety_num_samples': B,
    }


class MetricTracker:
    """训练过程中的指标跟踪器"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.ade_sum = 0.0
        self.fde_sum = 0.0
        self.min_ade_sum = 0.0
        self.min_fde_sum = 0.0
        self.count = 0
        self.loss_sums = {}

    def update(self, ade=None, fde=None, min_ade=None, min_fde=None,
               batch_size=1, **loss_dict):
        if ade is not None:
            self.ade_sum += ade * batch_size
        if fde is not None:
            self.fde_sum += fde * batch_size
        if min_ade is not None:
            self.min_ade_sum += min_ade * batch_size
        if min_fde is not None:
            self.min_fde_sum += min_fde * batch_size
        self.count += batch_size

        for k, v in loss_dict.items():
            if k not in self.loss_sums:
                self.loss_sums[k] = 0.0
            self.loss_sums[k] += v * batch_size

    def get_metrics(self):
        if self.count == 0:
            return {}
        metrics = {
            'ade': self.ade_sum / self.count,
            'fde': self.fde_sum / self.count,
            'min_ade': self.min_ade_sum / self.count,
            'min_fde': self.min_fde_sum / self.count,
        }
        for k, v in self.loss_sums.items():
            metrics[f'loss_{k}'] = v / self.count
        return metrics

    def summary_str(self):
        m = self.get_metrics()
        parts = []
        if m.get('min_ade', 0) > 0:
            parts.append(f"minADE={m['min_ade']:.4f}")
        if m.get('min_fde', 0) > 0:
            parts.append(f"minFDE={m['min_fde']:.4f}")
        if m.get('ade', 0) > 0:
            parts.append(f"ADE={m['ade']:.4f}")
        if m.get('fde', 0) > 0:
            parts.append(f"FDE={m['fde']:.4f}")
        for k, v in m.items():
            if k.startswith('loss_'):
                # KL 等量级可能远小于 1，保留足够小数避免误显示为 0
                parts.append(f"{k}={v:.6f}")
        return " | ".join(parts)
