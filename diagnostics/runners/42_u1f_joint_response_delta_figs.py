#!/usr/bin/env python3
"""U1F 逐窗联合图：响应大小(pred_ade) × 改善/恶化/描述带内(delta_ade)。

只读已有 per_window_F.csv。禁止前向/训练/重跑40格。

可从任意工作目录执行：
  python diagnostics/runners/42_u1f_joint_response_delta_figs.py

产物目录：
  local_runs/u1f_lock_Jt_v1/figs_joint_response_delta/
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit(
        "需要 matplotlib。请先安装后再跑本脚本。原始错误: " + str(e)
    )

import os
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("DIAGNOSTIC_INPUT_ROOT", str(PACKAGE_ROOT / "statistics/u1f_lock_Jt_v1"))).resolve()
OUTPUT_ROOT = Path(os.environ.get("DIAGNOSTIC_OUTPUT_ROOT", str(PACKAGE_ROOT / "local_runs/u1f_lock_Jt_v1"))).resolve()
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
OUT_DIR = OUTPUT_ROOT / "figs_joint_response_delta"
SEEDS = list(range(42, 50))
DONORS = list(range(42, 47))
TAU_MAIN = 0.01
TAU_APPENDIX = (0.0, 0.001, 0.002)
# 全量 40×15160 过大：每格均匀降采样，保证可复现
PER_CELL_CAP = 2000
SAMPLE_SEED = 20261005
Q_HI = 0.90


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


def band_masks(delta, tau):
    worsen = delta > tau
    improve = delta < -tau
    stable = np.abs(delta) <= tau
    return improve, worsen, stable


def subsample_indices(n, cap, rng):
    if n <= cap:
        return np.arange(n)
    return np.sort(rng.choice(n, size=cap, replace=False))


def collect_pool():
    """返回降采样后的 pred_ade, delta_ade, u1_pred_ade, train_seed 数组。"""
    rng = np.random.default_rng(SAMPLE_SEED)
    preds, deltas, u1preds, seeds = [], [], [], []
    n_raw = 0
    for s in SEEDS:
        for d in DONORS:
            rows = load_cell(s, d)
            n = len(rows)
            n_raw += n
            idx = subsample_indices(n, PER_CELL_CAP, rng)
            for i in idx:
                r = rows[i]
                preds.append(fget(r, "pred_ade"))
                deltas.append(fget(r, "delta_ade"))
                u1preds.append(fget(r, "u1_pred_ade"))
                seeds.append(s)
    pred = np.asarray(preds, float)
    delta = np.asarray(deltas, float)
    u1p = np.asarray(u1preds, float)
    seed = np.asarray(seeds, int)
    m = np.isfinite(pred) & np.isfinite(delta)
    meta = {
        "n_raw_include_main": n_raw,
        "n_plot": int(m.sum()),
        "per_cell_cap": PER_CELL_CAP,
        "sample_seed": SAMPLE_SEED,
        "tau_main": TAU_MAIN,
        "note": "图为关系展示；主表数字仍以 closeout/supplement 层级汇总为准",
    }
    return pred[m], delta[m], u1p[m], seed[m], meta


def frac_by_response_bins(pred, delta, tau, n_bins=10):
    """按 pred 分位箱统计三类带占比。"""
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(pred, qs)
    # 去重边缘
    edges = np.unique(edges)
    if edges.size < 3:
        return None
    centers, fi, fw, fs, counts = [], [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo == edges[-2]:
            sel = (pred >= lo) & (pred <= hi)
        else:
            sel = (pred >= lo) & (pred < hi)
        if sel.sum() == 0:
            continue
        d = delta[sel]
        imp, wor, stb = band_masks(d, tau)
        centers.append(0.5 * (lo + hi))
        fi.append(float(imp.mean()))
        fw.append(float(wor.mean()))
        fs.append(float(stb.mean()))
        counts.append(int(sel.sum()))
    return {
        "centers": np.asarray(centers),
        "improve": np.asarray(fi),
        "worsen": np.asarray(fw),
        "stable": np.asarray(fs),
        "counts": np.asarray(counts),
        "edges": edges,
    }


def fig_main_scatter_hex(pred, delta, tau, out_path):
    """主图：hexbin(log1p pred vs delta) + 带边界线 + 右上占比。"""
    imp, wor, stb = band_masks(delta, tau)
    x = np.log1p(np.clip(pred, 0, None))
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    hb = ax.hexbin(x, delta, gridsize=60, mincnt=1, cmap="viridis", linewidths=0.0)
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label("window count (subsampled pool)")
    ax.axhline(tau, color="C3", ls="--", lw=1.0, label=f"worsen > +{tau}")
    ax.axhline(-tau, color="C2", ls="--", lw=1.0, label=f"improve < -{tau}")
    ax.axhline(0.0, color="0.5", ls=":", lw=0.8)
    ax.set_xlabel(r"response size: $\log(1+\mathrm{Pred\text{-}ADE}_F)$")
    ax.set_ylabel(r"$\Delta\mathrm{ADE}=ADE_F-ADE_O$")
    ax.set_title(
        f"Joint: response vs delta bands (tau={tau})\n"
        f"improve={imp.mean():.3f}  worsen={wor.mean():.3f}  stable={stb.mean():.3f}"
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def fig_band_vs_response_bins(pred, delta, tau, out_path, n_bins=10):
    """响应分位箱上的三类带堆叠占比。"""
    pack = frac_by_response_bins(pred, delta, tau, n_bins=n_bins)
    if pack is None:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(pack["centers"]))
    ax.bar(x, pack["improve"], color="#2ca02c", label="improve", width=0.9)
    ax.bar(
        x,
        pack["stable"],
        bottom=pack["improve"],
        color="#7f7f7f",
        label="stable (in band)",
        width=0.9,
    )
    ax.bar(
        x,
        pack["worsen"],
        bottom=pack["improve"] + pack["stable"],
        color="#d62728",
        label="worsen",
        width=0.9,
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction within response bin")
    ax.set_xlabel("Pred-ADE quantile bins (low → high response)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{i+1}" for i in x], fontsize=8)
    ax.set_title(f"Band mix vs response size (tau={tau}, {len(x)} bins)")
    ax.legend(loc="upper right", fontsize=8)
    # 次轴：箱中心 pred
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{c:.3f}" for c in pack["centers"]], fontsize=7, rotation=45)
    ax2.set_xlabel("bin-center Pred-ADE")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return pack


def fig_high_response_bars(pred, delta, tau, out_path, q=Q_HI):
    """全体 vs 高响应(p90) 三类带占比对比。"""
    thr = float(np.quantile(pred, q))
    hi = pred >= thr
    rows = []
    for name, sel in [("all", np.ones_like(pred, dtype=bool)), (f"pred≥p{int(q*100)}", hi)]:
        d = delta[sel]
        imp, wor, stb = band_masks(d, tau)
        rows.append((name, float(imp.mean()), float(stb.mean()), float(wor.mean()), int(sel.sum())))
    labels = [r[0] for r in rows]
    improve = np.array([r[1] for r in rows])
    stable = np.array([r[2] for r in rows])
    worsen = np.array([r[3] for r in rows])
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    x = np.arange(len(labels))
    ax.bar(x, improve, color="#2ca02c", label="improve", width=0.65)
    ax.bar(x, stable, bottom=improve, color="#7f7f7f", label="stable", width=0.65)
    ax.bar(
        x,
        worsen,
        bottom=improve + stable,
        color="#d62728",
        label="worsen",
        width=0.65,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lb}\n(n={rows[i][4]})" for i, lb in enumerate(labels)])
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction")
    ax.set_title(f"Band mix: all vs high-response (tau={tau}, thr={thr:.4f})")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return {"thr": thr, "rows": [
        {"name": a, "improve": b, "stable": c, "worsen": d, "n": e}
        for a, b, c, d, e in rows
    ]}


def fig_tau_appendix(pred, delta, out_path):
    """附录：多个 tau 下全体三类带占比。"""
    taus = list(TAU_APPENDIX) + [TAU_MAIN]
    improve, worsen, stable = [], [], []
    for t in taus:
        imp, wor, stb = band_masks(delta, t)
        improve.append(float(imp.mean()))
        worsen.append(float(wor.mean()))
        stable.append(float(stb.mean()))
    x = np.arange(len(taus))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(x, improve, "o-", color="#2ca02c", label="improve")
    ax.plot(x, worsen, "o-", color="#d62728", label="worsen")
    ax.plot(x, stable, "o-", color="#7f7f7f", label="stable")
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in taus])
    ax.set_xlabel("tau")
    ax.set_ylabel("fraction")
    ax.set_ylim(0, 1)
    ax.set_title("Band fractions vs tau (subsampled pool)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def fig_nkey_fixed_optional(pred, delta, u1p, tau, out_path, q=Q_HI):
    """用 N 的高响应 key 固定集合，对比该集合上 F 的带占比 vs F 自划 p90。"""
    thr_n = float(np.quantile(u1p[np.isfinite(u1p)], q))
    thr_f = float(np.quantile(pred, q))
    hi_n = u1p >= thr_n
    hi_f = pred >= thr_f
    packs = []
    for name, sel in [
        ("all", np.ones_like(pred, dtype=bool)),
        (f"F-self p{int(q*100)}", hi_f),
        (f"N-key p{int(q*100)}", hi_n),
    ]:
        d = delta[sel]
        imp, wor, stb = band_masks(d, tau)
        packs.append((name, float(imp.mean()), float(stb.mean()), float(wor.mean()), int(sel.sum())))
    labels = [p[0] for p in packs]
    improve = np.array([p[1] for p in packs])
    stable = np.array([p[2] for p in packs])
    worsen = np.array([p[3] for p in packs])
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    x = np.arange(len(labels))
    ax.bar(x, improve, color="#2ca02c", label="improve", width=0.65)
    ax.bar(x, stable, bottom=improve, color="#7f7f7f", label="stable", width=0.65)
    ax.bar(
        x,
        worsen,
        bottom=improve + stable,
        color="#d62728",
        label="worsen",
        width=0.65,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lb}\n(n={packs[i][4]})" for i, lb in enumerate(labels)], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel(r"fraction of $\Delta$ADE bands (F)")
    ax.set_title(f"F delta bands under fixed groups (tau={tau})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return {
        "thr_N": thr_n,
        "thr_F": thr_f,
        "rows": [
            {"name": a, "improve": b, "stable": c, "worsen": d, "n": e}
            for a, b, c, d, e in packs
        ],
    }


def main():
    if not ROOT.is_dir():
        raise SystemExit(
            f"找不到 {ROOT}。请检查 DIAGNOSTIC_INPUT_ROOT。"
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred, delta, u1p, seed, meta = collect_pool()

    p_hex = OUT_DIR / "fig1_hex_response_vs_delta_tau0p01.png"
    p_bin = OUT_DIR / "fig2_band_frac_vs_response_bins_tau0p01.png"
    p_hi = OUT_DIR / "fig3_band_frac_all_vs_highresp_tau0p01.png"
    p_tau = OUT_DIR / "fig4_band_frac_vs_tau_appendix.png"
    p_nkey = OUT_DIR / "fig5_F_bands_Nkey_vs_Fself_p90.png"

    fig_main_scatter_hex(pred, delta, TAU_MAIN, p_hex)
    bin_pack = fig_band_vs_response_bins(pred, delta, TAU_MAIN, p_bin, n_bins=10)
    hi_pack = fig_high_response_bars(pred, delta, TAU_MAIN, p_hi, q=Q_HI)
    fig_tau_appendix(pred, delta, p_tau)
    nkey_pack = fig_nkey_fixed_optional(pred, delta, u1p, TAU_MAIN, p_nkey, q=Q_HI)

    summary = {
        "meta": meta,
        "outputs": [str(p) for p in (p_hex, p_bin, p_hi, p_tau, p_nkey)],
        "pool_band_frac_tau_main": {
            "tau": TAU_MAIN,
            "improve": float((delta < -TAU_MAIN).mean()),
            "worsen": float((delta > TAU_MAIN).mean()),
            "stable": float((np.abs(delta) <= TAU_MAIN).mean()),
        },
        "response_bins": None
        if bin_pack is None
        else {
            "centers": bin_pack["centers"].tolist(),
            "improve": bin_pack["improve"].tolist(),
            "worsen": bin_pack["worsen"].tolist(),
            "stable": bin_pack["stable"].tolist(),
            "counts": bin_pack["counts"].tolist(),
        },
        "high_response": hi_pack,
        "nkey_vs_fself": nkey_pack,
        "caption_zh": (
            "逐窗联合：横轴为 F 的 Pred-ADE（响应大小），纵轴/分色为 "
            "ΔADE=ADE_F−ADE_O 相对 tau 的改善/恶化/描述带内。"
            f"主图 tau={TAU_MAIN}；池化为每格最多 {PER_CELL_CAP} 窗、"
            f"sample_seed={SAMPLE_SEED}。图为关系展示，论文主表仍用层级汇总。"
        ),
    }
    sum_path = OUT_DIR / "FIGS_SUMMARY.json"
    sum_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("WROTE", OUT_DIR)
    for p in summary["outputs"]:
        print(" ", p)
    print("SUMMARY", sum_path)
    print(
        "pool_frac",
        summary["pool_band_frac_tau_main"],
        "n_plot",
        meta["n_plot"],
        "/",
        meta["n_raw_include_main"],
    )


if __name__ == "__main__":
    main()
