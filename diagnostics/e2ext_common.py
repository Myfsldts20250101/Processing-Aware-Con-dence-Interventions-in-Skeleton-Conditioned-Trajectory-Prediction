#!/usr/bin/env python3
"""
E2-ext-v1 共享：confidence-only 条件合成 + 固定 mask 硬核验。

版本：E2-ext-v1
设计：项目描述/JRDB_E2ext_confidence-only_20261003.md

口径：
- 不修改 E123 / E2 五条件语义；本模块独立版本号。
- C1 conf_swap：坐标保留 target；双方 conf>thr 处写 donor conf。
- C2 conf_zero：坐标保留 target；confidence 全 0。
- C3：输入同 C1，主表仅 mask_identical==1（san_pad 全帧一致）。
- 「固定 mask」必须 preprocessor 后核验，不能只靠配置声明。
"""

from __future__ import annotations

from typing import Any

import torch

from e1_rev1_common import CONF_JOINT_THR, CHANGE_TOL_DEFAULT, coord_rmse, conf_rmse
from e123_common import joint_valid_from_conf, common_joint_valid

SCHEME_VERSION = 'E2-ext-v1'

E2EXT_CONDITIONS = (
    'conf_swap',       # C1
    'conf_zero',       # C2
    'conf_swap_fixed_mask',  # C3（输入同 C1；过滤在评测层）
)

# C3 主表判据：sanitize 后的 key-padding 与 clean 全帧一致
MASK_IDENTICAL_USES = 'san_pad'


def build_e2ext_condition_inputs(
    sk_t: torch.Tensor,
    sc_t: torch.Tensor,
    sk_d: torch.Tensor,
    sc_d: torch.Tensor,
    *,
    conf_thr: float = CONF_JOINT_THR,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """
    返回 C1/C2（及 C3 共用输入）的 (sk, sc)。

    conf_swap 规则（冻结，对称于 E2 coord_swap）：
      - 坐标始终保留 target
      - 仅在双方 conf>thr 的 (t,v) 写入 donor confidence；其余保留 target conf
      - 禁止把 donor 缺失关节写成“有效高 conf”
    conf_zero：坐标保留 target，confidence 全 0。
    conf_swap_fixed_mask：张量输入与 conf_swap 相同；mask 过滤在评测/汇总层。
    """
    jv_t = joint_valid_from_conf(sc_t, conf_thr)
    jv_d = joint_valid_from_conf(sc_d, conf_thr)
    both = common_joint_valid(jv_t, jv_d)  # (B,T,V)

    sc_swap = torch.where(both, sc_d, sc_t)
    sk_keep = sk_t
    sc_zero = torch.zeros_like(sc_t)

    return {
        'conf_swap': (sk_keep, sc_swap),
        'conf_zero': (sk_keep, sc_zero),
        'conf_swap_fixed_mask': (sk_keep.clone(), sc_swap.clone()),
    }


def mask_compare(
    q_clean: dict[str, torch.Tensor],
    q_int: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    比较 preprocessor 后的 mask / 有效 key（batch 维保留）。

    需要 q_* 含：
      san_pad: (B,T) True=ignore
      raw_pad: (B,T) True=ignore
      key_valid_san / key_valid_raw: (B,T) int
      placeholder_t0: (B,) bool/int
      n_valid_frames: (B,)
    """
    san_c = q_clean['san_pad'].bool()
    san_i = q_int['san_pad'].bool()
    raw_c = q_clean['raw_pad'].bool()
    raw_i = q_int['raw_pad'].bool()

    san_equal = san_c == san_i  # (B,T)
    raw_equal = raw_c == raw_i

    mask_identical = san_equal.all(dim=-1)  # (B,) 主判据
    frac_frames_mask_equal = san_equal.float().mean(dim=-1)
    mask_hamming_san = (~san_equal).sum(dim=-1).to(torch.int64)
    mask_hamming_raw = (~raw_equal).sum(dim=-1).to(torch.int64)

    n_valid_san_c = q_clean['key_valid_san'].sum(dim=-1).to(torch.int64)
    n_valid_san_i = q_int['key_valid_san'].sum(dim=-1).to(torch.int64)
    n_valid_raw_c = q_clean['key_valid_raw'].sum(dim=-1).to(torch.int64)
    n_valid_raw_i = q_int['key_valid_raw'].sum(dim=-1).to(torch.int64)

    return {
        'mask_identical': mask_identical,
        'frac_frames_mask_equal': frac_frames_mask_equal,
        'mask_hamming_san': mask_hamming_san,
        'mask_hamming_raw': mask_hamming_raw,
        'n_valid_keys_san_clean': n_valid_san_c,
        'n_valid_keys_san_int': n_valid_san_i,
        'n_valid_keys_raw_clean': n_valid_raw_c,
        'n_valid_keys_raw_int': n_valid_raw_i,
        'placeholder_t0_clean': q_clean['placeholder_t0'].bool(),
        'placeholder_t0_int': q_int['placeholder_t0'].bool(),
        'n_valid_frames_clean': q_clean['n_valid_frames'],
        'n_valid_frames_int': q_int['n_valid_frames'],
    }


def mask_fields_for_row(cmp: dict[str, torch.Tensor], i: int) -> dict[str, Any]:
    """逐窗落盘字段。"""
    return {
        'mask_identical': int(cmp['mask_identical'][i].item()),
        'frac_frames_mask_equal': float(cmp['frac_frames_mask_equal'][i].item()),
        'mask_hamming_san': int(cmp['mask_hamming_san'][i].item()),
        'mask_hamming_raw': int(cmp['mask_hamming_raw'][i].item()),
        'n_valid_keys_san_clean': int(cmp['n_valid_keys_san_clean'][i].item()),
        'n_valid_keys_san_int': int(cmp['n_valid_keys_san_int'][i].item()),
        'n_valid_keys_raw_clean': int(cmp['n_valid_keys_raw_clean'][i].item()),
        'n_valid_keys_raw_int': int(cmp['n_valid_keys_raw_int'][i].item()),
        'placeholder_t0_clean': int(cmp['placeholder_t0_clean'][i].item()),
        'placeholder_t0_int': int(cmp['placeholder_t0_int'][i].item()),
        'n_valid_frames_clean': int(cmp['n_valid_frames_clean'][i].item()),
        'n_valid_frames_int': int(cmp['n_valid_frames_int'][i].item()),
        'mask_identical_uses': MASK_IDENTICAL_USES,
    }


def e2ext_condition_meta() -> dict[str, Any]:
    return {
        'scheme_version': SCHEME_VERSION,
        'conditions': list(E2EXT_CONDITIONS),
        'conf_swap_rule': {
            'coords': 'keep_target',
            'confidence': 'donor_only_where_both_conf_gt_thr_else_keep_target',
            'conf_thr': CONF_JOINT_THR,
            'forbid_treating_donor_missing_as_valid': True,
            'symmetric_to': 'E2.coord_swap',
        },
        'conf_zero_rule': {
            'coords': 'keep_target',
            'confidence': 'all_zero',
            'fixed_mask_required': False,
            'note': 'May change mask / trigger placeholder; report, do not claim pure-conf.',
        },
        'conf_swap_fixed_mask_rule': {
            'input_same_as': 'conf_swap',
            'main_table_requires': 'mask_identical==1',
            'mask_identical_uses': MASK_IDENTICAL_USES,
            'exclude_reason_if_fail': 'mask_changed',
        },
        'coord_rmse_expect_near_zero': True,
        'coord_rmse_tol': CHANGE_TOL_DEFAULT,
        'post_intervene_filter': False,
        'does_not_modify_e123_e2_conditions': True,
    }


def assert_coords_unchanged(
    sk_t: torch.Tensor,
    sk_int: torch.Tensor,
    *,
    tol: float = 0.0,
) -> torch.Tensor:
    """(B,) bool：干预后坐标与 target 一致（容差 tol 作用于 coord_rmse）。"""
    return coord_rmse(sk_t, sk_int) <= tol


def content_conf_changed(
    sc_t: torch.Tensor,
    sc_int: torch.Tensor,
    *,
    tol: float = CHANGE_TOL_DEFAULT,
) -> torch.Tensor:
    """(B,) bool：confidence 内容是否改变。"""
    return conf_rmse(sc_t, sc_int) > tol
