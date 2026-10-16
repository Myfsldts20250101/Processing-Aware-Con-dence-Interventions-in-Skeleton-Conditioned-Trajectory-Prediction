#!/usr/bin/env python3
"""U1F 可复现核对：指纹 + 关键字段（只读，不改写产物）。

可从任意工作目录执行：
  python diagnostics/runners/43_u1f_repro_check.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import os
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("DIAGNOSTIC_INPUT_ROOT", str(PACKAGE_ROOT / "statistics/u1f_lock_Jt_v1"))).resolve()
OUTPUT_ROOT = Path(os.environ.get("DIAGNOSTIC_OUTPUT_ROOT", str(PACKAGE_ROOT / "local_runs/u1f_lock_Jt_v1"))).resolve()
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# 2026-10-05 本机冻结指纹（前 16）；重跑 A/B 后若内容等价但序列化不同，sha 可能变
FINGERPRINTS = {
    "READONLY_CLOSEOUT_20261005.json": ("ee54be3dbf51137a", 77403),
    "READONLY_SUPPLEMENT_20261005.json": ("11d9643b5d216c74", 35110),
    "RUN_MANIFEST.json": ("f8edf8da26f64760", 472),
    "AGGREGATE.json": ("dff74df6712c2cd8", 13726),
    "RUN_STATUS.txt": ("34a4a631c639711d", 47),
}

EXPECTED = {
    "n_gate_pass": 40,
    "n_hard_pass": 40,
    "hard_lock_all_zero": True,
    "pred_ade": 0.07988604770016644,
    "ade_O": 0.5898770507989043,
    "delta_ade": 0.003968508456533638,
    "nf_pred_ade": 0.03168248327159647,
    "ci95_lo": 0.07014198044306534,
    "ci95_hi": 0.08965761648137496,
    "B": 1000,
    "boot_seed": 20261005,
    "figs_n_plot": 80000,
    "figs_sample_seed": 20261005,
}


def sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def near(a, b, tol=1e-12):
    return abs(float(a) - float(b)) <= tol


def main():
    if not ROOT.is_dir():
        print("FAIL: 找不到数据根", ROOT, "——请检查 DIAGNOSTIC_INPUT_ROOT")
        return 2

    ok = True
    print("== fingerprints ==")
    for name, (exp_h, exp_sz) in FINGERPRINTS.items():
        p = ROOT / name
        if not p.is_file():
            print("FAIL missing", name)
            ok = False
            continue
        h, sz = sha16(p), p.stat().st_size
        match = h == exp_h and sz == exp_sz
        print(
            ("OK  " if match else "DIFF")
            + f" {name}: sha16={h} (exp {exp_h}) size={sz} (exp {exp_sz})"
        )
        # fingerprint DIFF 不直接判整页失败（允许重跑后序列化差异），仅提示

    print("== closeout key fields ==")
    cpath = ROOT / "READONLY_CLOSEOUT_20261005.json"
    if not cpath.is_file():
        print("FAIL missing closeout")
        return 1
    c = json.loads(cpath.read_text())
    acc = c["acceptance"]
    x = c["crosscheck_vs_AGGREGATE"]
    nf = c["nf_paired_same_window"]["metrics"]["nf_pred_ade"]["mean"]
    ci = c["scene"]["bootstrap_ci95_on_seedmean_scenes"]["pred_ade_window_weighted"]

    checks = [
        ("n_gate_pass", acc["n_gate_pass"], EXPECTED["n_gate_pass"], True),
        ("n_hard_pass", acc["n_hard_pass"], EXPECTED["n_hard_pass"], True),
        ("hard_lock_all_zero", acc["hard_lock_all_zero"], EXPECTED["hard_lock_all_zero"], True),
        ("cross pred_ade", x["pred_ade_all_mean"], EXPECTED["pred_ade"], False),
        ("cross aggregate_pred", x["aggregate_pred_ade_mean"], EXPECTED["pred_ade"], False),
        ("cross ade_O", x["ade_O_mean"], EXPECTED["ade_O"], False),
        ("cross delta_ade", x["delta_ade_mean"], EXPECTED["delta_ade"], False),
        ("nf_pred_ade", nf, EXPECTED["nf_pred_ade"], False),
        ("ci point", ci["point"], EXPECTED["pred_ade"], False),
        ("ci lo", ci["ci95"][0], EXPECTED["ci95_lo"], False),
        ("ci hi", ci["ci95"][1], EXPECTED["ci95_hi"], False),
        ("B", ci["B"], EXPECTED["B"], True),
        ("boot_seed", ci["seed"], EXPECTED["boot_seed"], True),
    ]
    for name, got, exp, exact in checks:
        good = (got == exp) if exact else near(got, exp)
        if not good:
            ok = False
        print(("OK  " if good else "FAIL") + f" {name}: got={got} exp={exp}")

    print("== figs summary ==")
    fsum = ROOT / "figs_joint_response_delta" / "FIGS_SUMMARY.json"
    figs = list((ROOT / "figs_joint_response_delta").glob("fig*.png")) if (
        ROOT / "figs_joint_response_delta"
    ).is_dir() else []
    if not fsum.is_file():
        print("FAIL missing FIGS_SUMMARY.json")
        ok = False
    else:
        s = json.loads(fsum.read_text())
        good_n = s["meta"]["n_plot"] == EXPECTED["figs_n_plot"]
        good_seed = s["meta"]["sample_seed"] == EXPECTED["figs_sample_seed"]
        if not (good_n and good_seed):
            ok = False
        print(
            ("OK  " if good_n else "FAIL")
            + f" n_plot={s['meta']['n_plot']} exp={EXPECTED['figs_n_plot']}"
        )
        print(
            ("OK  " if good_seed else "FAIL")
            + f" sample_seed={s['meta']['sample_seed']} exp={EXPECTED['figs_sample_seed']}"
        )
    print(f"png_count={len(figs)} (exp 5)")
    if len(figs) != 5:
        ok = False
        print("FAIL expected 5 fig*.png")

    print("== RUN_STATUS ==")
    st = (ROOT / "RUN_STATUS.txt").read_text().strip()
    print(st)
    if "pass_cells=40" not in st:
        ok = False
        print("FAIL pass_cells!=40")

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
