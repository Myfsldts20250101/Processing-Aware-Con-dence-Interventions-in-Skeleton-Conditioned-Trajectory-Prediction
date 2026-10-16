#!/usr/bin/env python3
"""U1F 三项补做 · 只读统计。禁止前向/训练/重跑40格。

可从任意工作目录执行：
  python diagnostics/runners/41_u1f_supplement_readonly.py
产物：
  local_runs/u1f_lock_Jt_v1/READONLY_SUPPLEMENT_20261005.json
"""
from __future__ import annotations
import csv, json
from pathlib import Path
from collections import defaultdict
import numpy as np

import os
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("DIAGNOSTIC_INPUT_ROOT", str(PACKAGE_ROOT / "statistics/u1f_lock_Jt_v1"))).resolve()
OUTPUT_ROOT = Path(os.environ.get("DIAGNOSTIC_OUTPUT_ROOT", str(PACKAGE_ROOT / "local_runs/u1f_lock_Jt_v1"))).resolve()
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
OUT = OUTPUT_ROOT / "READONLY_SUPPLEMENT_20261005.json"
SEEDS = list(range(42, 50))
DONORS = list(range(42, 47))
QS = (0.75, 0.90, 0.95)
TAUS = (0.0, 0.0005, 0.001, 0.002)
EPS = 1e-12
ALIGN_FALSE_SEEDS = {42, 43, 44, 45}  # e2_original_align_pass=false


def mean_sd(xs):
    a = np.asarray([x for x in xs if x is not None], float)
    if a.size == 0:
        return {"mean": None, "sample_sd": None, "n": 0, "min": None, "max": None}
    sd = 0.0 if a.size == 1 else float(a.std(ddof=1))
    return {
        "mean": float(a.mean()),
        "sample_sd": sd,
        "n": int(a.size),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def hierarchy(cell_vals):
    per_seed = {}
    for s in SEEDS:
        ds = [
            cell_vals[(s, d)]
            for d in DONORS
            if (s, d) in cell_vals and cell_vals[(s, d)] is not None
        ]
        per_seed[str(s)] = float(np.mean(ds)) if ds else None
    xs = [per_seed[str(s)] for s in SEEDS if per_seed[str(s)] is not None]
    out = mean_sd(xs)
    out["per_seed"] = per_seed
    return out


def fget(r, k):
    v = r.get(k, "")
    return np.nan if v is None or v == "" else float(v)


def load_cell(s, d):
    path = ROOT / "by_ckpt" / f"s{s}" / f"d{d}" / "per_window_F.csv"
    rows = []
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            if int(float(r["include_main"])) != 1:
                continue
            rows.append(r)
    return rows


def high_resp(pred, delta, q):
    pred = np.asarray(pred, float)
    delta = np.asarray(delta, float)
    m = np.isfinite(pred) & np.isfinite(delta)
    pred, delta = pred[m], delta[m]
    thr = float(np.quantile(pred, q))
    hi = pred >= thr
    d_hi = delta[hi]
    return {
        "thr": thr,
        "n_hi": int(hi.sum()),
        "mean_delta_hi": float(d_hi.mean()) if d_hi.size else None,
        "contrib_harm_to_all_pos": float(
            np.clip(d_hi, 0, None).sum() / (np.clip(delta, 0, None).sum() + EPS)
        ),
        "contrib_improve_to_all_neg": float(
            (-np.clip(d_hi, None, 0)).sum() / ((-np.clip(delta, None, 0)).sum() + EPS)
        ),
    }


# ---- §1 cross-baseline ----
c1 = {
    k: {}
    for k in [
        "A_NF_ade",
        "D_NF_sameO",
        "D_NF_cross",
        "resid_A_minus_Dcross",
        "A_NF_pred",
        "abs_resid_mean",
        "frac_abs_resid_gt_1e4",
        "n_windows",
        "n_resid_gt_1e4",
    ]
}
drift_windows = []

for s in SEEDS:
    for d in DONORS:
        rows = load_cell(s, d)
        N = np.array([fget(r, "u1_ade_intervene") for r in rows])
        F = np.array([fget(r, "ade_F") for r in rows])
        O = np.array([fget(r, "ade_O") for r in rows])
        e2o = np.array([fget(r, "e2_original_ade") for r in rows])
        dF = np.array([fget(r, "delta_ade") for r in rows])
        dN = np.array([fget(r, "u1_delta_ade") for r in rows])
        pN = np.array([fget(r, "u1_pred_ade") for r in rows])
        pF = np.array([fget(r, "pred_ade") for r in rows])
        A = N - F
        D_same = dN - dF  # ≡ A if same O
        D_cross = (N - e2o) - (F - O)  # N vs e2_original, F vs O
        resid = A - D_cross  # ≡ O - e2o
        c1["A_NF_ade"][(s, d)] = float(np.nanmean(A))
        c1["D_NF_sameO"][(s, d)] = float(np.nanmean(D_same))
        c1["D_NF_cross"][(s, d)] = float(np.nanmean(D_cross))
        c1["resid_A_minus_Dcross"][(s, d)] = float(np.nanmean(resid))
        c1["A_NF_pred"][(s, d)] = float(np.nanmean(pN - pF))
        c1["abs_resid_mean"][(s, d)] = float(np.nanmean(np.abs(resid)))
        hit = np.abs(resid) > 1e-4
        c1["frac_abs_resid_gt_1e4"][(s, d)] = float(np.mean(hit))
        c1["n_windows"][(s, d)] = float(len(rows))
        c1["n_resid_gt_1e4"][(s, d)] = float(hit.sum())
        if s in ALIGN_FALSE_SEEDS:
            for i in np.where(hit)[0][:3]:
                drift_windows.append(
                    {
                        "cell": f"s{s}d{d}",
                        "key": rows[i]["key"],
                        "ade_O": float(O[i]),
                        "e2_original_ade": float(e2o[i]),
                        "resid": float(resid[i]),
                        "A_NF": float(A[i]),
                        "D_NF_cross": float(D_cross[i]),
                    }
                )

sec1 = {k: hierarchy(c1[k]) for k in c1}
max_same_gap = max(
    abs(c1["A_NF_ade"][sd] - c1["D_NF_sameO"][sd]) for sd in c1["A_NF_ade"]
)

# ---- §2 thresholds + N-key fixed group ----
c2_tau = {f"tau_{t}": {x: {} for x in ("improve", "worsen", "stable")} for t in TAUS}
c2_q = {}
c2_nkey = {}

for s in SEEDS:
    for d in DONORS:
        rows = load_cell(s, d)
        dF = np.array([fget(r, "delta_ade") for r in rows])
        pF = np.array([fget(r, "pred_ade") for r in rows])
        pN = np.array([fget(r, "u1_pred_ade") for r in rows])
        for t in TAUS:
            c2_tau[f"tau_{t}"]["improve"][(s, d)] = float(np.mean(dF < -t))
            c2_tau[f"tau_{t}"]["worsen"][(s, d)] = float(np.mean(dF > t))
            c2_tau[f"tau_{t}"]["stable"][(s, d)] = float(np.mean(np.abs(dF) <= t))
        for q in QS:
            tag = f"F_self_p{int(q * 100)}"
            hs = high_resp(pF, dF, q)
            for kk in (
                "mean_delta_hi",
                "contrib_harm_to_all_pos",
                "contrib_improve_to_all_neg",
                "n_hi",
            ):
                c2_q.setdefault(f"{tag}.{kk}", {})[(s, d)] = hs[kk]
            thr_n = float(np.quantile(pN[np.isfinite(pN)], q))
            hi_n = pN >= thr_n
            d_hi = dF[hi_n]
            c2_nkey.setdefault(f"Nkey_p{int(q * 100)}.n_hi", {})[(s, d)] = float(
                hi_n.sum()
            )
            c2_nkey.setdefault(f"Nkey_p{int(q * 100)}.mean_delta_F", {})[(s, d)] = (
                float(d_hi.mean()) if d_hi.size else None
            )
            c2_nkey.setdefault(f"Nkey_p{int(q * 100)}.contrib_harm_to_all_pos", {})[
                (s, d)
            ] = float(
                np.clip(d_hi, 0, None).sum() / (np.clip(dF, 0, None).sum() + EPS)
            )
            c2_nkey.setdefault(f"Nkey_p{int(q * 100)}.contrib_improve_to_all_neg", {})[
                (s, d)
            ] = float(
                (-np.clip(d_hi, None, 0)).sum() / ((-np.clip(dF, None, 0)).sum() + EPS)
            )

sec2 = {
    "tau_grid": {
        tk: {x: hierarchy(c2_tau[tk][x]) for x in c2_tau[tk]} for tk in c2_tau
    },
    "F_self_quantiles": {k: hierarchy(v) for k, v in c2_q.items()},
    "N_key_fixed_report_F": {k: hierarchy(v) for k, v in c2_nkey.items()},
    "note_eps_closeout": "closeout used eps=0.01; compare with tau_0.0..0.002 here",
}

# ---- §3 algorithm note only (cite closeout, no recompute) ----
closeout = json.loads(((OUTPUT_ROOT / "READONLY_CLOSEOUT_20261005.json" if (OUTPUT_ROOT / "READONLY_CLOSEOUT_20261005.json").exists() else ROOT / "READONLY_CLOSEOUT_20261005.json")).read_text())
sec3 = {
    "adopt": {
        "B": 1000,
        "seed": 20261005,
        "source": "READONLY_CLOSEOUT_20261005.json",
    },
    "ci_numbers": closeout["scene"]["bootstrap_ci95_on_seedmean_scenes"],
    "algorithm": {
        "window_weight": (
            "scene_n[sc]=AGGREGATE.n_windows_over_pass_cells//40; "
            "seed统计= sum(scene_mean*w)/sum(w)；equal则w=1"
        ),
        "scene_mean_per_seed": "格内该scene窗均值 → 5 donor 均值 → 得 seed×scene",
        "bootstrap_point_map": (
            "先对8个seed的scene均值再取平均，得到27维 scene 点向量，"
            "再对scene有放回重采样"
        ),
        "shared_index": (
            "每次 bootstrap 只抽一套 scene 下标 idx，同时用于 vals 与 weights"
            "（非按seed各抽一套）"
        ),
        "order": (
            "CSV窗 → cell×scene均值 → donor均 → seed×scene → "
            "(可选)跨seed scene点估计 → bootstrap加权均值"
        ),
        "percentile": "np.quantile(boots, 0.025/0.975)，B=1000，default_rng(20261005)",
        "no_rerun_reason": "仅次数/历史B不同不重算；以本 closeout 为准",
    },
}

out = {
    "meta": {
        "root": str(ROOT),
        "readonly": True,
        "no_forward_no_train_no_40cell_rerun": True,
        "paper_NF_recommendation": (
            "主文 Pred/绝对 ADE 的 N−F 用同窗 A=N−F（相对本轮O的D_same≡A）；"
            "D_cross 与残差仅作跨次基线兼容附录，勿与主 N−F 混用"
        ),
        "max_abs_A_minus_D_sameO_across_cells": max_same_gap,
    },
    "section1_cross_baseline": {
        "hierarchy": sec1,
        "drift_window_examples": drift_windows[:60],
        "identity": "A-D_cross = ade_O - e2_original_ade；A-D_sameO ≡ 0",
    },
    "section2_sensitivity": sec2,
    "section3_scene_ci": sec3,
}
OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
print("WROTE", OUT)
print(
    "§1 A_NF_ade",
    sec1["A_NF_ade"]["mean"],
    "D_same",
    sec1["D_NF_sameO"]["mean"],
    "D_cross",
    sec1["D_NF_cross"]["mean"],
    "resid",
    sec1["resid_A_minus_Dcross"]["mean"],
)
print("§1 max|A-D_same|", max_same_gap)
print("§2 tau_0.001 stable", sec2["tau_grid"]["tau_0.001"]["stable"]["mean"])
print(
    "§2 Nkey_p90 contrib_harm",
    sec2["N_key_fixed_report_F"]["Nkey_p90.contrib_harm_to_all_pos"]["mean"],
)
print("§3 adopt CI pred_win", sec3["ci_numbers"]["pred_ade_window_weighted"])
