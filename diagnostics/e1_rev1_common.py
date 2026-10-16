#!/usr/bin/env python3
"""E1-rev1 共享工具：指标、donor 分配、mask/占位、四格汇总。"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.fusion import _sanitize_key_padding_mask

CHANGE_TOL_DEFAULT = 1e-6
CONF_JOINT_THR = 0.1
MIN_VALID_JOINTS = 3


def now_local() -> str:
    return datetime.now().astimezone().strftime('%Y-%m-%dT%H:%M:%S%z')


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sha256_json_obj(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(blob).hexdigest()


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    keys = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def coord_rmse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    flat = (a - b).reshape(a.shape[0], -1)
    return flat.norm(dim=-1) / max(flat.shape[1] ** 0.5, 1.0)


def conf_rmse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    flat = (a - b).reshape(a.shape[0], -1)
    return flat.norm(dim=-1) / max(flat.shape[1] ** 0.5, 1.0)


def ade_fde_vs_gt(pred: torch.Tensor, gt: torch.Tensor):
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    err = torch.norm(pred - gt.unsqueeze(1), dim=-1)
    ade = err.mean(dim=-1).squeeze(1)
    fde = err[:, :, -1].squeeze(1)
    return ade, fde


def pred_pair_ade_fde(pa: torch.Tensor, pb: torch.Tensor):
    if pa.dim() == 3:
        pa = pa.unsqueeze(1)
        pb = pb.unsqueeze(1)
    err = torch.norm(pa - pb, dim=-1)
    return err.mean(dim=-1).mean(dim=-1), err[:, :, -1].mean(dim=-1)


def traj_disp_l2(obs: torch.Tensor) -> torch.Tensor:
    return (obs[:, -1] - obs[:, 0]).norm(dim=-1)


@torch.no_grad()
def skeleton_quality_bundle(model, sk_kp: torch.Tensor, sk_conf: torch.Tensor):
    """
    返回逐样本质量与 mask 诊断（P0 链路字段）。
    sk_* : (B,T,V,*) dataloader 归一化后张量。
    """
    _, skel_mask = model.skeleton_preprocessor(sk_kp, sk_conf)  # (B,T) 1=valid frame
    raw_pad = (skel_mask == 0)  # True = ignore key
    san_pad = _sanitize_key_padding_mask(raw_pad.clone())
    joint_valid = (sk_conf > CONF_JOINT_THR)  # (B,T,V) bool
    valid_joints = joint_valid.float().sum(dim=-1)  # (B,T)
    n_valid_frames = skel_mask.sum(dim=-1)
    all_masked_raw = raw_pad.all(dim=-1)
    placeholder = all_masked_raw & (~san_pad[:, 0])
    kp_abs_mean = sk_kp.abs().reshape(sk_kp.shape[0], -1).mean(dim=-1)
    conf_mean = sk_conf.reshape(sk_conf.shape[0], -1).mean(dim=-1)
    kp_all_zero = (sk_kp.abs().reshape(sk_kp.shape[0], -1).max(dim=-1).values <= 0)
    key_valid_raw = (~raw_pad).to(torch.int64)
    key_valid_san = (~san_pad).to(torch.int64)
    frame_mask = skel_mask.to(torch.int64)
    return {
        'skel_mask': skel_mask,
        'raw_pad': raw_pad,
        'san_pad': san_pad,
        'joint_valid': joint_valid,
        'valid_joints': valid_joints,
        'n_valid_frames': n_valid_frames,
        'placeholder_t0': placeholder,
        'kp_abs_mean': kp_abs_mean,
        'conf_mean': conf_mean,
        'kp_all_zero': kp_all_zero,
        'key_valid_raw': key_valid_raw,
        'key_valid_san': key_valid_san,
        'frame_mask': frame_mask,
    }


def pack_1d_int(t: torch.Tensor) -> str:
    return ','.join(str(int(x)) for x in t.detach().cpu().flatten().tolist())


def pack_2d_int(t: torch.Tensor) -> str:
    """(T,V) -> 't0v0,t0v1,...;t1v0,...' """
    arr = t.detach().cpu().numpy().astype(np.int64)
    rows = [','.join(str(int(x)) for x in row.tolist()) for row in arr]
    return ';'.join(rows)


def p0_fields_for_row(q: dict, i: int, prefix: str = 'target') -> dict:
    return {
        f'{prefix}_kp_all_zero': int(q['kp_all_zero'][i].item()),
        f'{prefix}_n_valid_frames': int(q['n_valid_frames'][i].item()),
        f'{prefix}_conf_mean': float(q['conf_mean'][i].item()),
        f'{prefix}_kp_abs_mean': float(q['kp_abs_mean'][i].item()),
        f'{prefix}_n_valid_keys_raw': int(q['key_valid_raw'][i].sum().item()),
        f'{prefix}_n_valid_keys_sanitized': int(q['key_valid_san'][i].sum().item()),
        f'{prefix}_placeholder_t0': int(q['placeholder_t0'][i].item()),
        f'{prefix}_frame_mask': pack_1d_int(q['frame_mask'][i]),
        f'{prefix}_valid_joints_per_frame': pack_1d_int(q['valid_joints'][i]),
        f'{prefix}_key_mask_raw': pack_1d_int(q['key_valid_raw'][i]),
        f'{prefix}_key_mask_san': pack_1d_int(q['key_valid_san'][i]),
        # joint_valid 全量较大：用紧凑 0/1 串 (T;V flattened per frame)
        f'{prefix}_joint_valid_TV': pack_2d_int(q['joint_valid'][i].to(torch.int64)),
    }


def load_donor_map(path: Path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    pairs = data.get('pairs') or {}
    return data, pairs


def lookup_donor_in_batch(pairs: dict, target_key: str, batch_keys: list[str]):
    """从冻结表取 donor_key，并在当前 batch 中定位下标；失败返回 status。"""
    rec = pairs.get(target_key)
    if rec is None:
        return -1, 'not_in_map', None
    st = rec.get('donor_status', 'ok')
    if st != 'ok':
        return -1, st, rec
    dkey = rec.get('donor_key')
    if not dkey:
        return -1, 'map_missing_donor_key', rec
    try:
        j = batch_keys.index(dkey)
    except ValueError:
        return -1, 'donor_not_in_batch', rec
    return j, 'ok', rec


@torch.no_grad()
def encode_prior_z(model, obs, sk_kp, sk_conf):
    memory, _ = model._encode(obs, sk_kp, sk_conf)
    mem_agg = model.cvae._aggregate_memory(memory)
    prior_mu, prior_logvar = model.cvae.prior_net(mem_agg)
    prior_logvar = prior_logvar.clamp(model.cvae.logvar_min, model.cvae.logvar_max)
    z = prior_mu
    return memory, prior_mu, prior_logvar, z


def window_key(scene_id: str, ped_id: int, window_start: int) -> str:
    return f'{scene_id}|{int(ped_id)}|{int(window_start)}'


def assign_donors_within_batch(
    scene_ids,
    ped_ids,
    window_starts,
    seed: int,
    *,
    has_skel=None,
    placeholder=None,
    require_donor_valid: bool = False,
    sk_kp: torch.Tensor | None = None,
    sk_conf: torch.Tensor | None = None,
    change_tol: float = CHANGE_TOL_DEFAULT,
):
    """
    within-batch 分配。
    require_donor_valid=True 时：donor 须 has_skel=1、非 placeholder，且相对 target 内容改变。
    """
    b = len(scene_ids)
    rng = random.Random(int(seed))
    donor_idx = [-1] * b
    status = ['no_candidate'] * b
    n_cands = [0] * b

    for i in range(b):
        cands = []
        for j in range(b):
            if j == i:
                continue
            if scene_ids[j] == scene_ids[i] and int(ped_ids[j]) == int(ped_ids[i]):
                continue
            if require_donor_valid:
                if has_skel is None or placeholder is None:
                    raise ValueError('require_donor_valid needs has_skel and placeholder')
                if not bool(has_skel[j]):
                    continue
                if bool(placeholder[j]):
                    continue
                if sk_kp is not None and sk_conf is not None:
                    c = float(coord_rmse(sk_kp[i:i+1], sk_kp[j:j+1])[0].item())
                    f = float(conf_rmse(sk_conf[i:i+1], sk_conf[j:j+1])[0].item())
                    if c <= change_tol and f <= change_tol:
                        continue
            cands.append(j)
        n_cands[i] = len(cands)
        if not cands:
            status[i] = 'no_candidate'
            continue
        j = rng.choice(cands)
        donor_idx[i] = j
        status[i] = 'ok'
    return donor_idx, status, n_cands


def summarize_numeric(rows: list[dict], keys: list[str], delta_eps=0.01, pred_hi=0.05):
    if not rows:
        return {'n': 0}
    out = {'n': len(rows)}

    def arr(k):
        return np.asarray([float(r[k]) for r in rows], dtype=np.float64)

    for k in keys:
        a = arr(k)
        out[k] = {
            'mean': float(a.mean()),
            'median': float(np.median(a)),
            'p10': float(np.percentile(a, 10)),
            'p90': float(np.percentile(a, 90)),
            'p95': float(np.percentile(a, 95)),
        }
    if 'delta_ade' in keys:
        d = arr('delta_ade')
        p = arr('pred_ade') if 'pred_ade' in keys else None
        out['frac_improve'] = float((d < -delta_eps).mean())
        out['frac_worsen'] = float((d > delta_eps).mean())
        out['frac_stable'] = float((np.abs(d) <= delta_eps).mean())
        if p is not None:
            out['frac_changed_but_err_stable'] = float(
                ((p > pred_hi) & (np.abs(d) < delta_eps)).mean()
            )
    out['thresholds'] = {'delta_eps': delta_eps, 'pred_ade_hi': pred_hi}
    return out


def quadrant_name(clean_ph: int, donor_ph: int) -> str:
    return f'clean_ph{clean_ph}_donor_ph{donor_ph}'


def availability_bucket(clean_valid: bool, donor_valid: bool) -> str:
    if clean_valid and donor_valid:
        return 'valid_to_valid'
    if clean_valid and not donor_valid:
        return 'valid_to_missing'
    if (not clean_valid) and donor_valid:
        return 'missing_to_valid'
    return 'missing_to_missing'


def load_target_set(path: Path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    keys = set(data['target_keys'])
    return data, keys
