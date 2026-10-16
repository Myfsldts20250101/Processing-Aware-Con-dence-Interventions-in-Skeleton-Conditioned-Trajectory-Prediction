#!/usr/bin/env python3
"""N_J 逐种子图：donor 范围为 min–max，不是置信区间。

输入默认读取包内 statistics/，输出写入 local_runs/。只读 READONLY_AGGREGATE JSON。禁止前向/重跑 40 格。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit("需要 matplotlib。" + str(e))

import os
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("DIAGNOSTIC_INPUT_ROOT", str(PACKAGE_ROOT / "statistics/nj_same_assign_v1"))).resolve()
OUTPUT_ROOT = Path(os.environ.get("DIAGNOSTIC_OUTPUT_ROOT", str(PACKAGE_ROOT / "local_runs/nj_same_assign_v1"))).resolve()
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
AGG = (OUTPUT_ROOT / "READONLY_AGGREGATE_20261005.json" if (OUTPUT_ROOT / "READONLY_AGGREGATE_20261005.json").exists() else ROOT / "READONLY_AGGREGATE_20261005.json")
OUT = OUTPUT_ROOT / "figs_seed_donor_range"
TRAIN = [str(s) for s in range(42, 50)]
X = np.arange(len(TRAIN))


def seed_mm(spread_by_seed, key):
    means, lo, hi = [], [], []
    for s in TRAIN:
        b = spread_by_seed[s][key]
        means.append(b["mean"])
        lo.append(b["min"])
        hi.append(b["max"])
    return np.asarray(means), np.asarray(lo), np.asarray(hi)


def draw_range(ax, means, lo, hi, ylabel, title, color="#1f4e79"):
    yerr = np.vstack([means - lo, hi - means])
    ax.errorbar(
        X, means, yerr=yerr, fmt="o", color=color, ecolor="#7a8a99",
        capsize=4, markersize=7, linewidth=1.2,
        label="seed mean (5-donor avg); whiskers = donor min–max, not CI",
    )
    ax.axhline(0.0, color="#888888", linewidth=0.8, linestyle="--")
    ax.set_xticks(X)
    ax.set_xticklabels(TRAIN)
    ax.set_xlabel("train seed")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, frameon=False)
    ax.grid(True, axis="y", alpha=0.3)


def main():
    if not AGG.is_file():
        raise SystemExit(f"缺 {AGG} ——先跑 45_nj_aggregate.py，或检查 DIAGNOSTIC_INPUT_ROOT")
    data = json.loads(AGG.read_text(encoding="utf-8"))
    sp = data["per_seed_five_donor_spread"]
    OUT.mkdir(parents=True, exist_ok=True)

    specs = [
        ("delta_policy", r"$\Delta_{\mathrm{policy}}=L_{NJ}-L_F$", "fig1_delta_policy.png"),
        ("R_policy", r"$R_{\mathrm{policy}}=\mathrm{ADE}(\hat{y}_{NJ},\hat{y}_F)$", "fig2_R_policy.png"),
        ("delta_support", r"$\Delta_{\mathrm{support}}=L_N-L_{NJ}$", "fig3_delta_support.png"),
        ("R_support", r"$R_{\mathrm{support}}=\mathrm{ADE}(\hat{y}_N,\hat{y}_{NJ})$", "fig4_R_support.png"),
    ]
    pngs = []
    for key, ylab, name in specs:
        m, lo, hi = seed_mm(sp, key)
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        draw_range(ax, m, lo, hi, ylab, ylab + " by train seed", "#1f4e79")
        fig.tight_layout()
        p = OUT / name
        fig.savefig(p, dpi=200)
        plt.close(fig)
        pngs.append(str(p))
        print("wrote", p)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    mp, lp, hp = seed_mm(sp, "delta_policy")
    ms, ls, hs = seed_mm(sp, "delta_support")
    ax.errorbar(
        X - 0.08, mp, yerr=np.vstack([mp - lp, hp - mp]),
        fmt="o", color="#1f4e79", ecolor="#9bb0c4", capsize=3, label=r"$\Delta_{\mathrm{policy}}$",
    )
    ax.errorbar(
        X + 0.08, ms, yerr=np.vstack([ms - ls, hs - ms]),
        fmt="s", color="#b85c38", ecolor="#d4a090", capsize=3, label=r"$\Delta_{\mathrm{support}}$",
    )
    ax.axhline(0.0, color="#888888", linewidth=0.8, linestyle="--")
    ax.set_xticks(X)
    ax.set_xticklabels(TRAIN)
    ax.set_xlabel("train seed")
    ax.set_ylabel("GT ADE difference")
    ax.set_title(r"Policy vs support (whiskers = donor min–max, not CI)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p5 = OUT / "fig5_policy_vs_support.png"
    fig.savefig(p5, dpi=200)
    plt.close(fig)
    pngs.append(str(p5))
    print("wrote", p5)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    mh, lh, hh = seed_mm(sp, "H_policy")
    mi, li, hi = seed_mm(sp, "I_policy")
    ax.errorbar(
        X - 0.08, mh, yerr=np.vstack([mh - lh, hh - mh]),
        fmt="o", color="#8c1d18", ecolor="#c47a76", capsize=3, label=r"$H_{\mathrm{policy}}$",
    )
    ax.errorbar(
        X + 0.08, mi, yerr=np.vstack([mi - li, hi - mi]),
        fmt="o", color="#2f6f4e", ecolor="#88b39a", capsize=3, label=r"$I_{\mathrm{policy}}$",
    )
    ax.set_xticks(X)
    ax.set_xticklabels(TRAIN)
    ax.set_xlabel("train seed")
    ax.set_ylabel("mean |signed| GT ADE")
    ax.set_title(r"$H=\mathrm{mean}(\max(\Delta,0))$, $I=\mathrm{mean}(\max(-\Delta,0))$; whiskers not CI")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p6 = OUT / "fig6_HI_policy.png"
    fig.savefig(p6, dpi=200)
    plt.close(fig)
    pngs.append(str(p6))
    print("wrote", p6)

    summary = {
        "readonly": True,
        "whiskers": "donor min-max of cell window-means; NOT confidence intervals",
        "n_png": len(pngs),
        "png": pngs,
        "source_json": str(AGG),
        "note_s43": "delta_policy seed 43 is the sole negative 5-donor mean",
    }
    (OUT / "FIGS_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("RESULT: PASS", OUT)


if __name__ == "__main__":
    main()
