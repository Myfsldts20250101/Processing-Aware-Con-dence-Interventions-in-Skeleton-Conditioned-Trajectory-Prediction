#!/usr/bin/env python3
"""Aggregate U1F 40 cells: 5-donor mean then 8-seed mean±sample SD; join U1/E2; scene table."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

TRAIN = list(range(42, 50))
DONOR = list(range(42, 47))
KEYS = [
    "pred_ade",
    "pred_fde",
    "ade_O",
    "ade_F",
    "delta_ade",
    "delta_fde",
    "fde_O",
    "fde_F",
    "e2_full_swap_pred_ade",
    "e2_full_swap_delta_ade",
    "u1_pred_ade",
    "u1_delta_ade",
]


def mean_sd(xs):
    xs = [float(x) for x in xs if x is not None and x == x]
    n = len(xs)
    if n == 0:
        return None, None, n
    m = sum(xs) / n
    if n == 1:
        return m, 0.0, n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var), n


def cell_means(path: Path, changed_only: bool):
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("include_main") != "1":
                continue
            if changed_only and r.get("valid_numeric_intervention") != "1":
                continue
            rows.append(r)
    out = {"n": len(rows)}
    for k in KEYS:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[k]))
            except (TypeError, ValueError, KeyError):
                pass
        if vals:
            a = np.asarray(vals, dtype=np.float64)
            out[k] = {
                "mean": float(a.mean()),
                "median": float(np.median(a)),
                "p90": float(np.quantile(a, 0.9)),
            }
        else:
            out[k] = None
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--allow_incomplete", type=int, default=0)
    args = p.parse_args()
    root = Path(args.root)
    cells = {}
    missing = []
    for s in TRAIN:
        for d in DONOR:
            g = root / f"by_ckpt/s{s}/d{d}/U1F_GATE.json"
            csvp = root / f"by_ckpt/s{s}/d{d}/per_window_F.csv"
            if not g.is_file() or not csvp.is_file():
                missing.append(f"s{s}d{d}")
                continue
            gate = json.loads(g.read_text())
            if not gate.get("pass"):
                missing.append(f"s{s}d{d}:gate_fail")
                continue
            cells[(s, d)] = {
                "all": cell_means(csvp, False),
                "changed": cell_means(csvp, True),
                "gate": gate,
            }
    complete = len(cells) == 40 and not missing
    report = {
        "n_cells_pass": len(cells),
        "n_expected": 40,
        "complete_8x5": complete,
        "missing_or_fail": missing,
        "note": "incomplete must not be cited as full 8-seed" if not complete else "full 8x5",
    }
    if not complete and not args.allow_incomplete:
        (root / "AGGREGATE.json").write_text(json.dumps(report, indent=2) + "\n")
        print("INCOMPLETE", report)
        return

    def seed_from_donors(subset, metric):
        seed_vals = []
        per_seed = {}
        for s in TRAIN:
            ds = []
            for d in DONOR:
                rec = cells.get((s, d))
                if rec is None or rec[subset][metric] is None:
                    ds = None
                    break
                ds.append(rec[subset][metric]["mean"])
            if ds is None or len(ds) != 5:
                continue
            sm = sum(ds) / 5
            seed_vals.append(sm)
            per_seed[s] = sm
        m, sd, n = mean_sd(seed_vals)
        return {"mean": m, "sample_sd": sd, "n_seeds": n, "per_seed": per_seed, "seeds_used": sorted(per_seed)}

    tables = {}
    for subset in ("all", "changed"):
        tables[subset] = {}
        for metric in ("pred_ade", "pred_fde", "delta_ade", "delta_fde", "ade_O", "ade_F", "u1_pred_ade", "e2_full_swap_pred_ade"):
            tables[subset][metric] = seed_from_donors(subset, metric)

    scene_acc = defaultdict(list)
    for (s, d), rec in cells.items():
        csvp = root / f"by_ckpt/s{s}/d{d}/per_window_F.csv"
        with csvp.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("include_main") != "1":
                    continue
                try:
                    scene_acc[r["scene_id"]].append(float(r["pred_ade"]))
                except (TypeError, ValueError, KeyError):
                    pass
    scene_tbl = []
    for sc, vs in sorted(scene_acc.items()):
        a = np.asarray(vs, dtype=np.float64)
        scene_tbl.append({"scene": sc, "n_windows_over_pass_cells": int(a.size), "pred_ade_mean": float(a.mean())})

    out = {
        **report,
        "tables": tables,
        "n_scenes": len(scene_tbl),
        "scene_pred_ade": scene_tbl,
    }
    (root / "AGGREGATE.json").write_text(json.dumps(out, indent=2) + "\n")
    md = ["# U1F 40 格汇总", ""]
    if not complete:
        md.append(f"**不完整 {len(cells)}/40。下列数字不得当作完整八种子结论。** 缺：{missing}")
        md.append("")
    md.append("五 donor 格 mean → 八种子 mean±样本 SD（ddof=1）。")
    md.append("")
    md.append("| 子集 | Pred-ADE | Pred-FDE | ΔADE | ΔFDE | U1 Pred-ADE | full_swap Pred-ADE | n_seeds |")
    md.append("|---|---|---|---|---|---|---|---|")
    for subset, label in (("all", "全窗含 conf 未变"), ("changed", "仅 conf 有变化")):
        t = tables[subset]
        def fmt(k):
            x = t[k]
            if x["mean"] is None:
                return "NA"
            return f"{x['mean']:.4f}±{x['sample_sd']:.4f}"
        md.append(
            f"| {label} | {fmt('pred_ade')} | {fmt('pred_fde')} | {fmt('delta_ade')} | {fmt('delta_fde')} | {fmt('u1_pred_ade')} | {fmt('e2_full_swap_pred_ade')} | {t['pred_ade']['n_seeds']} |"
        )
    md.append("")
    (root / "AGGREGATE.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("wrote", root / "AGGREGATE.json", "complete", complete, "n", len(cells))


if __name__ == "__main__":
    main()
