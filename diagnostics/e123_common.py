#!/usr/bin/env python3
"""
E1/E2/E3 共享：双方有效规则、E2 五条件合成、K 采样指标。

口径（冻结）：
- 内容诊断总体 = T_B ∩ BV，不是全量 29158 moving 窗口。
- 双方有效在干预前一次筛选；E2 全输入置零触发占位须保留报告。
- E3 为共同基础噪声 ε，不是固定相同潜变量 z。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from e1_rev1_common import (
    CONF_JOINT_THR,
    MIN_VALID_JOINTS,
    CHANGE_TOL_DEFAULT,
    coord_rmse,
    conf_rmse,
    skeleton_quality_bundle,
    window_key,
)

# E2 条件名（顺序固定，写入协议）
E2_CONDITIONS = (
    'original',
    'coord_swap',
    'full_swap',
    'coord_zero',
    'all_zero',
)

BV_RULE_VERSION = 'BV1'


def joint_valid_from_conf(sk_conf: torch.Tensor, thr: float = CONF_JOINT_THR) -> torch.Tensor:
    return sk_conf > thr


def common_joint_valid(jv_t: torch.Tensor, jv_d: torch.Tensor) -> torch.Tensor:
    return jv_t & jv_d


def count_common_valid_joints(jv_t: torch.Tensor, jv_d: torch.Tensor) -> torch.Tensor:
    """(B,) 全时空共同有效关节总数。"""
    both = common_joint_valid(jv_t, jv_d)
    return both.reshape(both.shape[0], -1).sum(dim=-1)


def mean_common_valid_joints_per_frame(jv_t: torch.Tensor, jv_d: torch.Tensor) -> torch.Tensor:
    both = common_joint_valid(jv_t, jv_d).float()  # (B,T,V)
    return both.sum(dim=-1).mean(dim=-1)  # (B,)


def is_skel_input_valid(
    has_skel: bool,
    placeholder_t0: bool,
    n_valid_frames: int,
    *,
    require_has_skel: bool = True,
    forbid_placeholder: bool = True,
    min_valid_frames: int = 1,
) -> bool:
    if require_has_skel and not has_skel:
        return False
    if forbid_placeholder and placeholder_t0:
        return False
    if n_valid_frames < min_valid_frames:
        return False
    return True


def bv_pair_ok(
    *,
    target_has_skel: bool,
    target_placeholder: bool,
    target_n_valid_frames: int,
    donor_has_skel: bool,
    donor_placeholder: bool,
    donor_n_valid_frames: int,
    n_common_joints: int,
    content_changed: bool,
    min_common_joints: int = MIN_VALID_JOINTS,
    min_valid_frames: int = 1,
) -> tuple[bool, str]:
    """返回 (ok, fail_reason)。"""
    if not is_skel_input_valid(
        target_has_skel, target_placeholder, target_n_valid_frames,
        min_valid_frames=min_valid_frames,
    ):
        return False, 'target_invalid'
    if not is_skel_input_valid(
        donor_has_skel, donor_placeholder, donor_n_valid_frames,
        min_valid_frames=min_valid_frames,
    ):
        return False, 'donor_invalid'
    if not content_changed:
        return False, 'content_unchanged'
    if n_common_joints < min_common_joints:
        return False, 'insufficient_common_joints'
    return True, 'ok'


def build_e2_condition_inputs(
    sk_t: torch.Tensor,
    sc_t: torch.Tensor,
    sk_d: torch.Tensor,
    sc_d: torch.Tensor,
    *,
    conf_thr: float = CONF_JOINT_THR,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """
    五条件骨架输入。

    coord_swap 规则（冻结）：
      - confidence 始终保留 target
      - 仅在双方 conf>thr 的 (t,v) 写入 donor 坐标；其余保留 target 坐标
      - 禁止把 donor 缺失关节当成有效姿态
    all_zero：坐标与 confidence 全零（可触发占位保护，保留不剔除）
    """
    jv_t = joint_valid_from_conf(sc_t, conf_thr)
    jv_d = joint_valid_from_conf(sc_d, conf_thr)
    both = common_joint_valid(jv_t, jv_d).unsqueeze(-1)  # (B,T,V,1)

    sk_coord = torch.where(both, sk_d, sk_t)
    sc_coord = sc_t.clone()

    sk_zero = torch.zeros_like(sk_t)
    sc_zero = torch.zeros_like(sc_t)

    return {
        'original': (sk_t, sc_t),
        'coord_swap': (sk_coord, sc_coord),
        'full_swap': (sk_d, sc_d),
        'coord_zero': (sk_zero, sc_t.clone()),
        'all_zero': (sk_zero, sc_zero),
    }


def e2_condition_meta() -> dict[str, Any]:
    return {
        'conditions': list(E2_CONDITIONS),
        'coord_swap_rule': {
            'confidence': 'keep_target',
            'coords': 'donor_only_where_both_conf_gt_thr_else_keep_target',
            'conf_thr': CONF_JOINT_THR,
            'forbid_treating_donor_missing_as_valid': True,
        },
        'all_zero_note': (
            'May trigger sanitize placeholder; windows are NOT dropped after intervention; '
            'report placeholder as property of this condition.'
        ),
        'post_intervene_filter': False,
    }


@torch.no_grad()
def ade_fde_k_metrics(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    pred: (B,K,T,2) or (B,T,2); gt: (B,T,2)
    返回 min/mean ADE/FDE，以及 argmin 索引（ADE/FDE 可能不同）。
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    err = torch.norm(pred - gt.unsqueeze(1), dim=-1)  # (B,K,T)
    ade_k = err.mean(dim=-1)  # (B,K)
    fde_k = err[:, :, -1]  # (B,K)
    min_ade, argmin_ade = ade_k.min(dim=1)
    min_fde, argmin_fde = fde_k.min(dim=1)
    return {
        'min_ade': min_ade,
        'mean_ade': ade_k.mean(dim=1),
        'min_fde': min_fde,
        'mean_fde': fde_k.mean(dim=1),
        'argmin_ade': argmin_ade,
        'argmin_fde': argmin_fde,
        'ade_per_k': ade_k,
        'fde_per_k': fde_k,
    }


@torch.no_grad()
def paired_pred_ade_fde_per_k(pa: torch.Tensor, pb: torch.Tensor):
    """同编号候选 Pred-ADE/FDE: (B,K)"""
    if pa.dim() == 3:
        pa = pa.unsqueeze(1)
        pb = pb.unsqueeze(1)
    err = torch.norm(pa - pb, dim=-1)  # (B,K,T)
    return err.mean(dim=-1), err[:, :, -1]


def pack_float_list(xs) -> str:
    return ','.join(f'{float(x):.6g}' for x in xs)


def content_changed_flag(sk_a, sk_b, sc_a, sc_b, tol=CHANGE_TOL_DEFAULT) -> torch.Tensor:
    c = coord_rmse(sk_a, sk_b)
    f = conf_rmse(sc_a, sc_b)
    return (c > tol) | (f > tol)


@torch.no_grad()
def quality_snapshot(model, sk, sc):
    q = skeleton_quality_bundle(model, sk, sc)
    return q


def batch_window_keys(batch) -> list[str]:
    scene_ids = list(batch['scene_id'])
    ped_ids = batch['ped_id'].tolist() if torch.is_tensor(batch['ped_id']) else list(batch['ped_id'])
    starts = (
        batch['window_start'].tolist()
        if torch.is_tensor(batch['window_start']) else list(batch['window_start'])
    )
    return [window_key(scene_ids[i], ped_ids[i], starts[i]) for i in range(len(scene_ids))]
