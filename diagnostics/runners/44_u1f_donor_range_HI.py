#!/usr/bin/env python3
"""U1F 只读：每种子五 donor 变化范围 + 正/负误差绝对量 H/I。

输入默认读取包内 statistics/，输出写入 local_runs/。禁止新前向 / 训练 / 重跑 40 格。
H = mean(max(delta, 0))；I = mean(max(-delta, 0))；净均值 = H - I。
层级：格内窗均值 → 5 donor → 8 seed mean±样本 SD（ddof=1）。
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import os
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("DIAGNOSTIC_INPUT_ROOT", str(PACKAGE_ROOT / "statistics/u1f_lock_Jt_v1"))).resolve()
OUTPUT_ROOT = Path(os.environ.get("DIAGNOSTIC_OUTPUT_ROOT", str(PACKAGE_ROOT / "local_runs/u1f_lock_Jt_v1"))).resolve()
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
TRAIN = list(range(42, 50))
DONOR = list(range(42, 47))
OUT = OUTPUT_ROOT / "READONLY_DONOR_RANGE_HI_20261005.json"


def cell_stats(path: Path) -> dict:
    h_ade, i_ade, h_fde, i_fde = [], [], [], []
    pred, dade, dfde, ao, af = [], [], [], [], []
    n = 0
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("include_main", "1") not in ("1", "1.0", "true", "True"):
                continue
            n += 1
            da = float(row["delta_ade"])
            df = float(row["delta_fde"])
            h_ade.append(max(da, 0.0))
            i_ade.append(max(-da, 0.0))
            h_fde.append(max(df, 0.0))
            i_fde.append(max(-df, 0.0))
            pred.append(float(row["pred_ade"]))
            dade.append(da)
            dfde.append(df)
            ao.append(float(row["ade_O"]))
            af.append(float(row["ade_F"]))
    a = lambda x: float(np.mean(x)) if x else float("nan")
    return {
        "n": n,
        "pred_ade": a(pred),
        "delta_ade": a(dade),
        "delta_fde": a(dfde),
        "ade_O": a(ao),
        "ade_F": a(af),
        "H_ade": a(h_ade),
        "I_ade": a(i_ade),
        "H_fde": a(h_fde),
        "I_fde": a(i_fde),
    }


def spread(vals):
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "range": float(arr.max() - arr.min()),
        "mean": float(arr.mean()),
        "sample_sd_donors": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "per_donor": {str(d): float(v) for d, v in zip(DONOR, arr)},
    }


def main():
    if not ROOT.is_dir():
        raise SystemExit(f"找不到数据根 {ROOT} ——请检查 DIAGNOSTIC_INPUT_ROOT")

    cells = {}
    by_seed = defaultdict(list)
    for s in TRAIN:
        for d in DONOR:
            p = ROOT / "by_ckpt" / f"s{s}" / f"d{d}" / "per_window_F.csv"
            if not p.is_file():
                raise SystemExit(f"缺 CSV {p}")
            st = cell_stats(p)
            cells[f"s{s}d{d}"] = st
            by_seed[s].append(st)
            print(f"  s{s} d{d} n={st['n']} pred={st['pred_ade']:.6f} H={st['H_ade']:.6f} I={st['I_ade']:.6f}")

    keys = ["pred_ade", "delta_ade", "delta_fde", "H_ade", "I_ade", "H_fde", "I_fde"]
    per_seed = {}
    for s in TRAIN:
        rec = {"train_seed": s}
        for k in keys:
            rec[k] = spread([c[k] for c in by_seed[s]])
        rec["net_check_ade"] = rec["H_ade"]["mean"] - rec["I_ade"]["mean"]
        per_seed[str(s)] = rec

    def seed_hier(k):
        means = [per_seed[str(s)][k]["mean"] for s in TRAIN]
        arr = np.asarray(means, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "sample_sd": float(arr.std(ddof=1)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "per_seed_donor_mean": {str(s): float(v) for s, v in zip(TRAIN, arr)},
        }

    payload = {
        "meta": {
            "readonly": True,
            "no_forward_no_train_no_40cell_rerun": True,
            "definition": {
                "H": "mean(max(delta,0)) over include_main windows",
                "I": "mean(max(-delta,0)) over include_main windows",
                "identity": "mean(delta) = H - I",
                "spread": "min/max/range of the 5 donor cell-means within a train seed",
                "hierarchy": "cell window-mean → 5 donor mean → 8 seed mean±sample SD (ddof=1)",
                "subset": "include_main=1, n=15160 per cell",
            },
        },
        "hierarchy": {k: seed_hier(k) for k in keys},
        "per_seed_five_donor_spread": per_seed,
        "cells": cells,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    h = payload["hierarchy"]
    print("== hierarchy (8 seed, 5-donor means) ==")
    for k in keys:
        x = h[k]
        print(f"  {k}: {x['mean']:.6f} ± {x['sample_sd']:.6f}  seed-range [{x['min']:.6f}, {x['max']:.6f}]")
    print("== per-seed Pred-ADE 5-donor range ==")
    for s in TRAIN:
        sp = per_seed[str(s)]["pred_ade"]
        print(
            f"  s{s}: min={sp['min']:.6f} max={sp['max']:.6f} range={sp['range']:.6f} "
            f"H_ade={per_seed[str(s)]['H_ade']['mean']:.6f} I_ade={per_seed[str(s)]['I_ade']['mean']:.6f}"
        )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
