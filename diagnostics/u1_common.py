#!/usr/bin/env python3
"""
U1 conf_swap_full：整幅 donor confidence，坐标保留 target。

正式赋值在 preprocessor 之前完成并核验（含单侧无效位置）。
不要调用 e2ext_common.build_e2ext_condition_inputs 的 both-mask conf_swap。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch

from e1_rev1_common import CONF_JOINT_THR, coord_rmse, conf_rmse
from e2ext_common import mask_compare, mask_fields_for_row
from e123_common import joint_valid_from_conf

SCHEME_VERSION = 'U1_conf_swap_v2.1'
CONDITION = 'conf_swap_full'
ASSIGN_TOL = 0.0
PRED_DET_ATOL = 1e-6
ORIGINAL_ADE_ATOL = 1e-4


def build_conf_swap_full(
    sk_t: torch.Tensor,
    sc_t: torch.Tensor,
    sk_d: torch.Tensor,
    sc_d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del sc_t  # 正式条件不使用 target conf
    return sk_t.clone(), sc_d.clone()


def visibility_counts(
    sc_t: torch.Tensor,
    sc_d: torch.Tensor,
    *,
    conf_thr: float = CONF_JOINT_THR,
) -> dict[str, torch.Tensor]:
    jv_t = joint_valid_from_conf(sc_t, conf_thr)
    jv_d = joint_valid_from_conf(sc_d, conf_thr)
    uni_t = jv_t & ~jv_d
    uni_d = jv_d & ~jv_t
    both = jv_t & jv_d
    neither = ~jv_t & ~jv_d
    return {
        'jv_t': jv_t,
        'jv_d': jv_d,
        'uni_t': uni_t,
        'uni_d': uni_d,
        'both': both,
        'neither': neither,
        'n_jv_t': jv_t.reshape(jv_t.shape[0], -1).sum(dim=-1),
        'n_jv_d': jv_d.reshape(jv_d.shape[0], -1).sum(dim=-1),
        'n_uni_t_only': uni_t.reshape(uni_t.shape[0], -1).sum(dim=-1),
        'n_uni_d_only': uni_d.reshape(uni_d.shape[0], -1).sum(dim=-1),
        'n_both': both.reshape(both.shape[0], -1).sum(dim=-1),
        'n_neither': neither.reshape(neither.shape[0], -1).sum(dim=-1),
        'n_preproc_zeroed_t': (~jv_t).reshape(jv_t.shape[0], -1).sum(dim=-1),
        'n_preproc_zeroed_d': (~jv_d).reshape(jv_d.shape[0], -1).sum(dim=-1),
    }


def assign_gate_batch(
    sk_t: torch.Tensor,
    sc_t: torch.Tensor,
    sk_d: torch.Tensor,
    sc_d: torch.Tensor,
    sk_int: torch.Tensor,
    sc_int: torch.Tensor,
    in_mask: list[bool],
    *,
    tol: float = ASSIGN_TOL,
) -> dict[str, Any]:
    """Preprocessor 之前。in_mask 为 donor ok 的 batch 下标。"""
    vis = visibility_counts(sc_t, sc_d)
    ok_idx = [i for i, m in enumerate(in_mask) if m]
    max_sk = 0.0
    max_sc = 0.0
    max_sk_uni = 0.0
    max_sc_uni = 0.0
    n_uni = 0
    for i in ok_idx:
        dsk = (sk_int[i] - sk_t[i]).abs().max().item()
        dsc = (sc_int[i] - sc_d[i]).abs().max().item()
        max_sk = max(max_sk, dsk)
        max_sc = max(max_sc, dsc)
        uni = vis['uni_t'][i] | vis['uni_d'][i]
        n_u = int(uni.sum().item())
        n_uni += n_u
        if n_u:
            dsk_u = (sk_int[i] - sk_t[i]).abs()[uni].max().item()
            dsc_u = (sc_int[i] - sc_d[i]).abs()[uni].max().item()
            max_sk_uni = max(max_sk_uni, dsk_u)
            max_sc_uni = max(max_sc_uni, dsc_u)
    pass_sk = max_sk <= tol
    pass_sc = max_sc <= tol
    pass_uni = (n_uni == 0) or (max_sk_uni <= tol and max_sc_uni <= tol)
    return {
        'n_ok': len(ok_idx),
        'n_unilateral_positions': n_uni,
        'max_abs_sk_int_minus_sk_t': max_sk,
        'max_abs_sc_int_minus_sc_d': max_sc,
        'max_abs_sk_unilateral': max_sk_uni,
        'max_abs_sc_unilateral': max_sc_uni,
        'pass_sk': pass_sk,
        'pass_sc': pass_sc,
        'pass_unilateral': pass_uni,
        'pass': pass_sk and pass_sc and pass_uni,
        'vis': vis,
        'coord_rmse': coord_rmse(sk_t, sk_int),
        'conf_rmse_vs_donor': conf_rmse(sc_d, sc_int),
        'conf_rmse_vs_target': conf_rmse(sc_t, sc_int),
    }


def preproc_export_row(q_clean: dict, q_int: dict, vis: dict, cmp: dict, i: int) -> dict[str, Any]:
    row = {
        'n_jv_t': int(vis['n_jv_t'][i].item()),
        'n_jv_d': int(vis['n_jv_d'][i].item()),
        'n_both': int(vis['n_both'][i].item()),
        'n_uni_t_only': int(vis['n_uni_t_only'][i].item()),
        'n_uni_d_only': int(vis['n_uni_d_only'][i].item()),
        'n_neither': int(vis['n_neither'][i].item()),
        'n_preproc_zeroed_clean': int(vis['n_preproc_zeroed_t'][i].item()),
        'n_preproc_zeroed_int': int(vis['n_preproc_zeroed_d'][i].item()),
        'n_jv_post_clean': int(q_clean['joint_valid'][i].sum().item()),
        'n_jv_post_int': int(q_int['joint_valid'][i].sum().item()),
        'n_jv_post_xor': int(
            (q_clean['joint_valid'][i] != q_int['joint_valid'][i]).sum().item()
        ),
    }
    row.update(mask_fields_for_row(cmp, i))
    return row


def join_u1_to_e2(
    u1_rows: list[dict],
    e2_csv: Path,
    *,
    original_ade_atol: float = ORIGINAL_ADE_ATOL,
) -> dict[str, Any]:
    e2_by_key: dict[str, dict] = {}
    if not e2_csv.is_file():
        return {
            'pass': False,
            'reason': 'e2_csv_missing',
            'e2_csv': str(e2_csv),
            'n_u1_ok': 0,
            'n_matched': 0,
            'n_missing_e2': 0,
            'n_donor_key_mismatch': 0,
            'original_ade_max_abs': None,
            'original_ade_n': 0,
        }
    with open(e2_csv, newline='', encoding='utf-8') as f:
        for rec in csv.DictReader(f):
            e2_by_key[rec['key']] = rec

    ok = [r for r in u1_rows if r.get('include_main') == 1]
    n_missing = 0
    n_mismatch = 0
    n_matched = 0
    ade_abs = []
    for r in ok:
        rec = e2_by_key.get(r['key'])
        if rec is None:
            n_missing += 1
            continue
        n_matched += 1
        if str(rec.get('donor_key', '')) != str(r.get('donor_key', '')):
            n_mismatch += 1
        try:
            a_u = float(r['ade_clean'])
            a_e = float(rec['original_ade'])
            ade_abs.append(abs(a_u - a_e))
        except (TypeError, ValueError, KeyError):
            pass
    max_ade = max(ade_abs) if ade_abs else None
    orig_pass = max_ade is not None and max_ade <= original_ade_atol
    join_pass = n_missing == 0 and n_mismatch == 0 and len(ok) > 0
    return {
        'pass': bool(join_pass and orig_pass),
        'join_pass': join_pass,
        'original_align_pass': orig_pass,
        'e2_csv': str(e2_csv),
        'n_u1_ok': len(ok),
        'n_matched': n_matched,
        'n_missing_e2': n_missing,
        'n_donor_key_mismatch': n_mismatch,
        'original_ade_max_abs': max_ade,
        'original_ade_n': len(ade_abs),
        'original_ade_atol': original_ade_atol,
    }


def u1_condition_meta() -> dict[str, Any]:
    return {
        'scheme_version': SCHEME_VERSION,
        'condition': CONDITION,
        'formula': 'sk_int=sk_t; sc_int=sc_d',
        'check_before_preprocessor': True,
        'includes_unilateral_invalid': True,
        'not_e2ext_C1': True,
        'mask_identical_not_required_for_main': True,
    }
