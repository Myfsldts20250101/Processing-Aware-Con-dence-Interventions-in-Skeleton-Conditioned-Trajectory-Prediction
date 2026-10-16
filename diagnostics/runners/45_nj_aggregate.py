#!/usr/bin/env python3
"""N_J 只读汇总：主表、Δ_policy / R_policy / Δ_support、分层、H/I。

输入默认读取包内 statistics/，输出写入 local_runs/。禁止新前向 / 训练 / 重跑 40 格。
层级：格内窗均值 → 5 donor 均值 → 8 seed mean±样本 SD（ddof=1）。
40 格不是 40 次独立训练。
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import os
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("DIAGNOSTIC_INPUT_ROOT", str(PACKAGE_ROOT / "statistics/nj_same_assign_v1"))).resolve()
OUTPUT_ROOT = Path(os.environ.get("DIAGNOSTIC_OUTPUT_ROOT", str(PACKAGE_ROOT / "local_runs/nj_same_assign_v1"))).resolve()
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
TRAIN = list(range(42, 50))
DONOR = list(range(42, 47))
N_SBV = 15160
OUT = OUTPUT_ROOT / "READONLY_AGGREGATE_20261005.json"

CELL_KEYS = [
    "ade_O", "ade_F", "ade_NJ", "ade_N",
    "fde_O", "fde_F", "fde_NJ", "fde_N",
    "delta_ade_F", "delta_ade_NJ", "delta_ade_N",
    "pred_ade_F_vs_O", "pred_ade_NJ_vs_O", "pred_ade_N_vs_O",
    "pred_fde_F_vs_O", "pred_fde_NJ_vs_O", "pred_fde_N_vs_O",
    "delta_policy", "R_policy", "delta_support", "R_support",
    "R_policy_fde", "R_support_fde",
    "identity_resid",
    "H_policy", "I_policy",
    "frac_pos_policy", "frac_neg_policy", "frac_zero_policy",
]


def fmean(xs):
    return float(np.mean(xs)) if len(xs) else float("nan")


def cell_stats(path: Path) -> dict:
    cols = defaultdict(list)
    stratum_n = {1: 0, 2: 0, 3: 0, 4: 0, 0: 0}
    strat_dp = defaultdict(list)
    strat_rp = defaultdict(list)
    n = 0
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("include_main", "")) != "1":
                continue
            n += 1
            sid = int(float(row["stratum"]))
            stratum_n[sid] = stratum_n.get(sid, 0) + 1
            dp = float(row["delta_policy"])
            rp = float(row["R_policy"])
            strat_dp[sid].append(dp)
            strat_rp[sid].append(rp)
            for k in (
                "ade_O", "ade_F", "ade_NJ", "ade_N",
                "fde_O", "fde_F", "fde_NJ", "fde_N",
                "delta_ade_F", "delta_ade_NJ", "delta_ade_N",
                "pred_ade_F_vs_O", "pred_ade_NJ_vs_O", "pred_ade_N_vs_O",
                "pred_fde_F_vs_O", "pred_fde_NJ_vs_O", "pred_fde_N_vs_O",
                "delta_policy", "R_policy", "delta_support", "R_support",
                "R_policy_fde", "R_support_fde", "identity_resid",
            ):
                cols[k].append(float(row[k]))
            cols["H_policy"].append(max(dp, 0.0))
            cols["I_policy"].append(max(-dp, 0.0))
            cols["frac_pos_policy"].append(1.0 if dp > 0.0 else 0.0)
            cols["frac_neg_policy"].append(1.0 if dp < 0.0 else 0.0)
            cols["frac_zero_policy"].append(1.0 if dp == 0.0 else 0.0)

    if n != N_SBV:
        raise SystemExit(f"{path} n={n} != {N_SBV}")
    out = {"n": n}
    for k in CELL_KEYS:
        out[k] = fmean(cols[k])
    hi = out["H_policy"] - out["I_policy"]
    out["HI_minus_mean_policy"] = float(hi - out["delta_policy"])
    out["stratum_n"] = {str(k): int(v) for k, v in sorted(stratum_n.items())}
    out["stratum_frac"] = {str(k): (v / n) for k, v in sorted(stratum_n.items())}
    out["stratum"] = {}
    for sid in (1, 2, 3, 4, 0):
        ns = stratum_n.get(sid, 0)
        out["stratum"][str(sid)] = {
            "n": ns,
            "frac": ns / n,
            "delta_policy": fmean(strat_dp[sid]) if ns else None,
            "R_policy": fmean(strat_rp[sid]) if ns else None,
        }
    return out


def seed_spread(vals):
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "range": float(arr.max() - arr.min()),
        "mean": float(arr.mean()),
        "sample_sd_donors": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "per_donor": {str(d): float(v) for d, v in zip(DONOR, arr)},
        "sign": int(np.sign(arr.mean())) if arr.mean() != 0 else 0,
    }


def mean_list(vals):
    if not vals:
        return None
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "range": float(arr.max() - arr.min()),
        "sample_sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
    }


def hier8(seed_means):
    arr = np.asarray(seed_means, dtype=np.float64)
    npos = int((arr > 0).sum())
    nneg = int((arr < 0).sum())
    nzer = int((arr == 0).sum())
    return {
        "mean": float(arr.mean()),
        "sample_sd": float(arr.std(ddof=1)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n_seed_pos": npos,
        "n_seed_neg": nneg,
        "n_seed_zero": nzer,
        "per_seed": (
            {str(s): float(v) for s, v in zip(TRAIN, arr)}
            if len(arr) == len(TRAIN)
            else {str(i): float(v) for i, v in enumerate(arr)}
        ),
    }


def main():
    if not ROOT.is_dir():
        raise SystemExit(f"找不到数据根 {ROOT} ——请检查 DIAGNOSTIC_INPUT_ROOT")

    gates_ok = 0
    for s in TRAIN:
        for d in DONOR:
            g = ROOT / "by_ckpt" / f"s{s}" / f"d{d}" / "NJ_GATE.json"
            if not g.is_file():
                raise SystemExit(f"缺 GATE {g}")
            gj = json.loads(g.read_text(encoding="utf-8"))
            if not gj.get("pass"):
                raise SystemExit(f"GATE fail {g}")
            if int(gj.get("n_ok", 0)) != N_SBV:
                raise SystemExit(f"n_ok {g}")
            gates_ok += 1
    if gates_ok != 40:
        raise SystemExit(f"gates {gates_ok}/40")

    cells = {}
    by_seed = defaultdict(list)
    for s in TRAIN:
        for d in DONOR:
            p = ROOT / "by_ckpt" / f"s{s}" / f"d{d}" / "per_window_NJ.csv"
            st = cell_stats(p)
            cells[f"s{s}d{d}"] = st
            by_seed[s].append(st)
            print(
                f"  s{s} d{d} n={st['n']} "
                f"d_policy={st['delta_policy']:.6g} R_policy={st['R_policy']:.6g} "
                f"d_support={st['delta_support']:.6g}"
            )

    per_seed = {}
    for s in TRAIN:
        rec = {"train_seed": s}
        for k in CELL_KEYS:
            rec[k] = seed_spread([c[k] for c in by_seed[s]])
        rec["HI_check"] = rec["H_policy"]["mean"] - rec["I_policy"]["mean"] - rec["delta_policy"]["mean"]
        rec["stratum_empty"] = {}
        rec["stratum_frac"] = {}
        rec["stratum_delta_policy"] = {}
        rec["stratum_R_policy"] = {}
        for sid in ("1", "2", "3", "4"):
            rec["stratum_frac"][sid] = seed_spread(
                [c["stratum_frac"].get(sid, 0.0) for c in by_seed[s]]
            )
            dps = [c["stratum"][sid]["delta_policy"] for c in by_seed[s]]
            rps = [c["stratum"][sid]["R_policy"] for c in by_seed[s]]
            rec["stratum_empty"][sid] = sum(1 for x in dps if x is None)
            rec["stratum_delta_policy"][sid] = mean_list([x for x in dps if x is not None])
            rec["stratum_R_policy"][sid] = mean_list([x for x in rps if x is not None])
        per_seed[str(s)] = rec

    hierarchy = {k: hier8([per_seed[str(s)][k]["mean"] for s in TRAIN]) for k in CELL_KEYS}

    strat_h = {}
    for sid in ("1", "2", "3", "4"):
        fr = hier8([per_seed[str(s)]["stratum_frac"][sid]["mean"] for s in TRAIN])
        # 若该种子五 donor 全空，均值被写成 0，仅作占比；delta 用非空种子
        dp_vals, rp_vals = [], []
        for s in TRAIN:
            dpm = per_seed[str(s)]["stratum_delta_policy"][sid]
            rpm = per_seed[str(s)]["stratum_R_policy"][sid]
            if dpm is None:
                continue
            dp_vals.append(dpm["mean"])
            rp_vals.append(rpm["mean"])
        strat_h[sid] = {
            "frac": fr,
            "n_seeds_with_windows": len(dp_vals),
            "delta_policy": hier8(dp_vals) if dp_vals else None,
            "R_policy": hier8(rp_vals) if rp_vals else None,
        }

    payload = {
        "meta": {
            "readonly": True,
            "no_forward_no_train_no_40cell_rerun": True,
            "scheme": "NJ_same_assign_v1",
            "n_cells": 40,
            "n_S_BV": N_SBV,
            "hierarchy": "cell window-mean → 5 donor mean → 8 seed mean±sample SD (ddof=1)",
            "not_40_independent_trainings": True,
            "H_policy": "mean(max(delta_policy,0)) over all include_main windows",
            "I_policy": "mean(max(-delta_policy,0)) over all include_main windows",
            "R_policy": "ADE(pred_NJ, pred_F); not Pred-ADE_NJ minus Pred-ADE_F",
            "stratum": "1 new placeholder; 2 mask change; 3 jv/xy change; 4 none; 0 other",
            "donor_range_is_not_CI": True,
        },
        "gate": {"n_pass": 40, "n_ok": N_SBV},
        "hierarchy": hierarchy,
        "stratum_hierarchy": strat_h,
        "per_seed_five_donor_spread": per_seed,
        "cells": {k: {kk: cells[k][kk] for kk in (
            "n", "delta_policy", "R_policy", "delta_support", "H_policy", "I_policy",
            "stratum_n",
        )} for k in cells},
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def row(name, h):
        print(
            f"  {name:22s} {h['mean']:.6g} ± {h['sample_sd']:.6g}  "
            f"[{h['min']:.6g}, {h['max']:.6g}]  "
            f"sign +{h['n_seed_pos']}/-{h['n_seed_neg']}/0{h['n_seed_zero']}"
        )

    print("\n# 八种子层级（donor 均值后再对 seed）")
    print("绝对 ADE")
    for k in ("ade_O", "ade_F", "ade_NJ", "ade_N"):
        row(k, hierarchy[k])
    print("相对 O 的 ΔADE / Pred-ADE")
    for k in ("delta_ade_F", "delta_ade_NJ", "delta_ade_N",
              "pred_ade_F_vs_O", "pred_ade_NJ_vs_O", "pred_ade_N_vs_O"):
        row(k, hierarchy[k])
    print("主对照 处理策略 / 次要 替换范围")
    for k in ("delta_policy", "R_policy", "delta_support", "R_support",
              "H_policy", "I_policy", "frac_pos_policy", "frac_neg_policy", "frac_zero_policy"):
        row(k, hierarchy[k])
    print(
        "H-I vs mean Δ_policy (seed-level check):",
        hierarchy["H_policy"]["mean"] - hierarchy["I_policy"]["mean"] - hierarchy["delta_policy"]["mean"],
    )
    print("\n# 分层占比与组内 Δ_policy / R_policy")
    for sid, lab in (("1", "new_placeholder"), ("2", "mask_chg"), ("3", "jv_or_xy"), ("4", "unchanged")):
        sh = strat_h[sid]
        print(f"  stratum {sid} {lab}: frac={sh['frac']['mean']:.4g} seeds_with={sh['n_seeds_with_windows']}")
        if sh["delta_policy"] is None:
            print("    empty")
        else:
            row("    d_policy", sh["delta_policy"])
            row("    R_policy", sh["R_policy"])
    print("\nWrote", OUT)
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
