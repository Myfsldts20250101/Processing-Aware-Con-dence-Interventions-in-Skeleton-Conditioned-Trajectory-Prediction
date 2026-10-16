#!/usr/bin/env python3
"""U1F F-input gate: assign donor conf on clean J_t only. No preprocessor replay, no ckpt, no forward."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

ROSTER_PAIRS_SHA = "d93ba2e1e2325f6f57a5f4ab803431e2426f07374d787d22a66d5cd14b385e58"
C0_NPZ_SHA = "6959d26577afc0023a5e1f0bfb3155ae5cf9c142529ad4b1d4902dbe0bae702e"
C01_NPZ_SHA = "f44c3f48181d71a7be51e5455bf23e2cdbe98b1137956036fe9bdd00f8353c4f"
G4_NPZ_SHA = "2de521013ab9992cf09d5e5bf10f6818355d2bb3fb3682040cff1c63f98270a0"
TZ = timezone(timedelta(hours=8))
SCHEME = "U1F_F_input_gate_s42d42_n256"
CONF_THR = 0.1
N = 256

LOCKED_C0 = [
    ("kp_loader_target", "kp_loader"),
    ("x_preproc_target", "x_preproc_clean"),
    ("frame_mask_target", "frame_mask"),
    ("joint_valid_target", "joint_valid"),
    ("key_mask_raw_target", "key_mask_raw"),
    ("key_mask_san_target", "key_mask_san"),
    ("placeholder_t0_target", "placeholder_t0"),
]
LOCKED_C01 = ["window_origin", "last_obs_pos", "obs_traj", "pred_traj"]
LOCKED_G4 = [
    ("center_hip_target", "center_hip"),
    ("xy_centered_target", "xy_centered"),
    ("xy_zeroed_target", "xy_zeroed"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def arr_sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def now_local() -> str:
    return datetime.now(TZ).strftime("%Y-%m-%dT%H:%M:%S%z")


def float_diff(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        return {"pass_exact": False, "max_abs": None, "n_mismatch": None, "gap": "shape"}
    if np.issubdtype(a.dtype, np.floating) or np.issubdtype(b.dtype, np.floating):
        aa = np.asarray(a, dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)
        nan_xor = np.isnan(aa) ^ np.isnan(bb)
        both = (~np.isnan(aa)) & (~np.isnan(bb))
        n_nan_mis = int(nan_xor.sum())
        if both.any():
            d = np.abs(aa[both] - bb[both])
            max_abs = float(d.max()) if d.size else 0.0
            n_val = int(np.sum(d != 0))
        else:
            max_abs = 0.0
            n_val = 0
        n_mis = n_nan_mis + n_val
        inf = bool((~np.isfinite(aa) & ~np.isnan(aa)).any() or (~np.isfinite(bb) & ~np.isnan(bb)).any())
        return {
            "pass_exact": n_mis == 0 and max_abs == 0.0 and not inf,
            "max_abs": max_abs,
            "n_mismatch": n_mis,
            "n_nan_xor": n_nan_mis,
            "nonfinite_nonnan": inf,
        }
    n_mis = int(np.sum(np.asarray(a) != np.asarray(b)))
    return {"pass_exact": n_mis == 0, "max_abs": 0 if n_mis == 0 else 1, "n_mismatch": n_mis}


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--roster",
        default="eval_logs/u1f_c0_loader_state_s42d42_n256_DESIGN/ROSTER.json",
    )
    p.add_argument(
        "--c0_npz",
        default="eval_logs/u1f_c0_loader_state_s42d42_n256/state_s42d42_n256.npz",
    )
    p.add_argument(
        "--c01_npz",
        default="eval_logs/u1f_c0_1_dataset_traj_s42d42_n256/dataset_traj_s42d42_n256.npz",
    )
    p.add_argument(
        "--g4_npz",
        default="eval_logs/u1f_g4g5_replay_s42d42_n256/replay_s42d42_n256.npz",
    )
    p.add_argument(
        "--out_dir",
        default="eval_logs/u1f_f_input_gate_s42d42_n256",
    )
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gaps = []

    def stop(code: str, detail: str):
        gaps.append({"gap_code": code, "detail": detail})
        (out_dir / "GAPS.json").write_text(
            json.dumps({"n": len(gaps), "gaps": gaps, "stopped": True}, indent=2) + "\n"
        )
        raise SystemExit(f"STOP {code}: {detail}")

    roster = json.loads(Path(args.roster).read_text(encoding="utf-8"))
    if roster.get("pairs_sha256") != ROSTER_PAIRS_SHA:
        stop("roster_sha", str(roster.get("pairs_sha256")))
    if len(roster["windows"]) != N:
        stop("roster_n", str(len(roster["windows"])))

    c0_sha = sha256_file(Path(args.c0_npz))
    c01_sha = sha256_file(Path(args.c01_npz))
    g4_sha = sha256_file(Path(args.g4_npz))
    if c0_sha != C0_NPZ_SHA:
        stop("c0_sha", c0_sha)
    if c01_sha != C01_NPZ_SHA:
        stop("c01_sha", c01_sha)
    if g4_sha != G4_NPZ_SHA:
        stop("g4_sha", g4_sha)

    c0 = np.load(args.c0_npz, allow_pickle=False)
    c01 = np.load(args.c01_npz, allow_pickle=False)
    g4 = np.load(args.g4_npz, allow_pickle=False)

    keys = [str(x) for x in c0["key"]]
    roster_keys = [w["key"] for w in sorted(roster["windows"], key=lambda x: int(x["roster_idx"]))]
    if keys != roster_keys:
        stop("key_order_c0_roster", "mismatch")
    if [str(x) for x in c01["key"]] != keys or [str(x) for x in g4["key"]] != keys:
        stop("key_join", "C0.1 or G4 key order")
    if [str(x) for x in c0["donor_key"]] != [str(x) for x in c01["donor_key"]]:
        stop("donor_key_join", "C0 vs C0.1")

    conf_t = np.array(c0["conf_loader_target"], dtype=np.float32, copy=True)
    conf_d = np.array(c0["conf_loader_donor"], dtype=np.float32, copy=True)
    Jt = np.array(c0["joint_valid_target"], dtype=np.int8, copy=True) == 1
    if conf_t.shape != (N, 8, 17) or conf_d.shape != (N, 8, 17) or Jt.shape != (N, 8, 17):
        stop("conf_shape", f"{conf_t.shape}/{conf_d.shape}/{Jt.shape}")

    # J_t from locked clean valid only — never from donor conf.
    Jt_from_conf_t = conf_t > CONF_THR
    if not np.array_equal(Jt, Jt_from_conf_t):
        stop("Jt_not_from_clean_conf", "joint_valid_target != (conf_loader_target>0.1)")
    Jt_from_donor = conf_d > CONF_THR
    n_valid_disagree_if_recomputed = int(np.sum(Jt != Jt_from_donor))

    conf_F = np.where(Jt, conf_d, conf_t).astype(np.float32)
    x_clean = np.array(c0["x_preproc_target"], dtype=np.float32, copy=True)
    x_F = x_clean.copy()
    x_F[:, 2, :, :] = conf_F

    assign_on = float_diff(conf_F[Jt], conf_d[Jt])
    assign_off = float_diff(conf_F[~Jt], conf_t[~Jt])
    if (not assign_on.get("pass_exact")) or (not assign_off.get("pass_exact")):
        stop("assignment_identity", json.dumps({"on": assign_on, "off": assign_off}))

    arrays = {
        "key": np.array(c0["key"]),
        "donor_key": np.array(c0["donor_key"]),
        "roster_idx": np.array(c0["roster_idx"]),
        "joint_valid_Jt": Jt.astype(np.int8),
        "conf_loader_target": conf_t,
        "conf_loader_donor": conf_d,
        "conf_F": conf_F,
        "x_preproc_F": x_F,
        "x_preproc_clean": x_clean,
    }
    for src, dst in LOCKED_C0:
        arrays[dst] = np.array(c0[src])
    for name in LOCKED_C01:
        arrays[name] = np.array(c01[name])
    for src, dst in LOCKED_G4:
        arrays[dst] = np.array(g4[src])

    diffs = {
        "assign_Jt_equals_donor_conf": assign_on,
        "assign_outside_Jt_equals_clean_conf": assign_off,
        "x_F_xy_equals_clean": float_diff(x_F[:, :2], x_clean[:, :2]),
        "x_F_conf_equals_conf_F": float_diff(x_F[:, 2], conf_F),
        "kp_copied_clean": float_diff(arrays["kp_loader"], c0["kp_loader_target"]),
        "center_copied_g4": float_diff(arrays["center_hip"], g4["center_hip_target"]),
        "xy_centered_copied_g4": float_diff(arrays["xy_centered"], g4["xy_centered_target"]),
        "xy_zeroed_copied_g4": float_diff(arrays["xy_zeroed"], g4["xy_zeroed_target"]),
        "frame_mask": float_diff(arrays["frame_mask"], c0["frame_mask_target"]),
        "joint_valid": float_diff(arrays["joint_valid"], c0["joint_valid_target"]),
        "key_mask_raw": float_diff(arrays["key_mask_raw"], c0["key_mask_raw_target"]),
        "key_mask_san": float_diff(arrays["key_mask_san"], c0["key_mask_san_target"]),
        "placeholder_t0": float_diff(arrays["placeholder_t0"], c0["placeholder_t0_target"]),
        "window_origin": float_diff(arrays["window_origin"], c01["window_origin"]),
        "last_obs_pos": float_diff(arrays["last_obs_pos"], c01["last_obs_pos"]),
        "obs_traj": float_diff(arrays["obs_traj"], c01["obs_traj"]),
        "pred_traj": float_diff(arrays["pred_traj"], c01["pred_traj"]),
        "xy_zeroed_equals_x_F_xy": float_diff(
            np.transpose(arrays["xy_zeroed"], (0, 3, 1, 2)), x_F[:, :2]
        ),
    }

    # Must not equal a naive preprocessor-style recompute of validity from conf_F
    J_from_conf_F = conf_F > CONF_THR
    diffs["Jt_not_recomputed_from_conf_F"] = {
        "n_cells_disagree": int(np.sum(Jt != J_from_conf_F)),
        "pass_exact": True,
        "note": "disagreement is expected if donor conf on J_t is <=0.1; J_t stays clean",
    }

    changed = Jt & (conf_F != conf_t)
    n_support = Jt.reshape(N, -1).sum(axis=1).astype(np.int32)
    n_changed = changed.reshape(N, -1).sum(axis=1).astype(np.int32)
    abs_on_J = np.abs(conf_F.astype(np.float64) - conf_t.astype(np.float64))
    abs_on_J = np.where(Jt, abs_on_J, np.nan)
    with np.errstate(all="ignore"):
        max_abs_w = np.nanmax(abs_on_J, axis=(1, 2))
        mean_abs_w = np.nanmean(abs_on_J, axis=(1, 2))
    max_abs_w = np.where(n_support == 0, 0.0, max_abs_w).astype(np.float32)
    mean_abs_w = np.where(n_support == 0, 0.0, mean_abs_w).astype(np.float32)
    empty_support = n_support == 0
    invalid_numeric = (n_support > 0) & (n_changed == 0)
    valid_numeric = n_changed > 0

    arrays["n_support_joints"] = n_support
    arrays["n_changed_joints"] = n_changed
    arrays["max_abs_conf_on_Jt"] = max_abs_w
    arrays["mean_abs_conf_on_Jt"] = mean_abs_w
    arrays["flag_empty_support"] = empty_support.astype(np.int8)
    arrays["flag_invalid_numeric_intervention"] = invalid_numeric.astype(np.int8)
    arrays["flag_valid_numeric_intervention"] = valid_numeric.astype(np.int8)

    finite_ok = True
    for name, a in arrays.items():
        if np.issubdtype(a.dtype, np.floating):
            bad = ~np.isfinite(a)
            if bad.any():
                finite_ok = False
                gaps.append({"gap_code": "nonfinite", "detail": f"{name}:{int(bad.sum())}"})

    structure_keys = [
        k
        for k in diffs
        if k
        not in (
            "assign_Jt_equals_donor_conf",
            "assign_outside_Jt_equals_clean_conf",
            "x_F_conf_equals_conf_F",
            "Jt_not_recomputed_from_conf_F",
        )
    ]
    structure_pass = all(diffs[k].get("pass_exact") for k in structure_keys)
    assignment_pass = diffs["assign_Jt_equals_donor_conf"]["pass_exact"] and diffs[
        "assign_outside_Jt_equals_clean_conf"
    ]["pass_exact"] and diffs["x_F_conf_equals_conf_F"]["pass_exact"]

    n_win_changed = int(valid_numeric.sum())
    n_win_invalid = int(invalid_numeric.sum())
    n_win_empty = int(empty_support.sum())
    n_joint_changed = int(n_changed.sum())
    n_joint_support = int(n_support.sum())
    global_max = float(np.max(max_abs_w)) if n_win_changed or n_joint_changed else 0.0
    if n_joint_changed:
        global_mean = float(np.nanmean(abs_on_J[changed]))
    else:
        global_mean = 0.0

    numeric_ok = n_win_changed > 0
    complete = all(k in arrays for k, _ in []) or True
    required = [
        "conf_F",
        "x_preproc_F",
        "joint_valid_Jt",
        "center_hip",
        "xy_zeroed",
        "frame_mask",
        "key_mask_san",
        "placeholder_t0",
        "obs_traj",
        "pred_traj",
        "window_origin",
        "last_obs_pos",
        "kp_loader",
    ]
    missing_fields = [k for k in required if k not in arrays]
    if missing_fields:
        complete = False
        gaps.append({"gap_code": "missing_field", "detail": ",".join(missing_fields)})

    npz_path = out_dir / "f_input_s42d42_n256.npz"
    np.savez_compressed(npz_path, **arrays)

    index_rows = []
    for w in roster["windows"]:
        i = int(w["roster_idx"])
        index_rows.append(
            {
                "roster_idx": i,
                "key": w["key"],
                "donor_key": w["donor_key"],
                "n_support_joints": int(n_support[i]),
                "n_changed_joints": int(n_changed[i]),
                "max_abs_conf_on_Jt": float(max_abs_w[i]),
                "mean_abs_conf_on_Jt": float(mean_abs_w[i]),
                "empty_support": int(empty_support[i]),
                "invalid_numeric_intervention": int(invalid_numeric[i]),
                "valid_numeric_intervention": int(valid_numeric[i]),
            }
        )
    with (out_dir / "INDEX.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
        wr.writeheader()
        wr.writerows(index_rows)

    (out_dir / "DIFF.json").write_text(json.dumps(diffs, indent=2) + "\n")
    with (out_dir / "DIFF.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["check", "pass_exact", "max_abs", "n_mismatch"])
        wr.writeheader()
        for name, d in diffs.items():
            wr.writerow(
                {
                    "check": name,
                    "pass_exact": int(bool(d.get("pass_exact"))),
                    "max_abs": d.get("max_abs", ""),
                    "n_mismatch": d.get("n_mismatch", d.get("n_cells_disagree", "")),
                }
            )

    change = {
        "n_windows": N,
        "n_windows_valid_numeric_intervention": n_win_changed,
        "n_windows_invalid_numeric_intervention": n_win_invalid,
        "n_windows_empty_support": n_win_empty,
        "n_support_joints_total": n_joint_support,
        "n_changed_joints_total": n_joint_changed,
        "max_abs_conf_on_Jt_global": global_max,
        "mean_abs_conf_on_changed_joints": global_mean,
        "n_cells_Jt_vs_donor_valid_disagree": n_valid_disagree_if_recomputed,
        "note_invalid": "invalid_numeric_intervention means J_t nonempty but conf_F==conf_clean on all J_t; not model insensitivity",
        "note_empty": "empty_support means no clean-valid joint; assignment is a no-op by spec",
    }
    (out_dir / "CHANGE_REPORT.json").write_text(json.dumps(change, indent=2) + "\n")

    identity = {
        "scheme_id": SCHEME,
        "created_at": now_local(),
        "n": N,
        "roster_pairs_sha256": ROSTER_PAIRS_SHA,
        "c0_npz_sha256": c0_sha,
        "c01_npz_sha256": c01_sha,
        "g4_npz_sha256": g4_sha,
        "assignment": "conf_F = donor_conf on J_t=(joint_valid_target==1), else clean conf; includes donor 0 and low values",
        "no_preprocessor_call": True,
        "no_ckpt": True,
        "no_forward": True,
        "no_40cell": True,
        "npz_sha256": sha256_file(npz_path),
        "arrays": {k: arr_sha(v) for k, v in arrays.items()},
    }
    (out_dir / "IDENTITY.json").write_text(json.dumps(identity, indent=2) + "\n")
    (out_dir / "PRECISION.json").write_text(
        json.dumps(
            {
                "dtype": "float32 conf/x; int8 masks; change uses exact != on float32 bits",
                "no_silent_zero": True,
                "locked_diff_rule": "max_abs==0 and n_mismatch==0 vs clean C0/C0.1/G4",
            },
            indent=2,
        )
        + "\n"
    )

    gate_pass = bool(
        structure_pass and assignment_pass and finite_ok and complete and numeric_ok and not missing_fields
    )
    gate = {
        "assignment_pass": assignment_pass,
        "structure_lock_pass": structure_pass,
        "finite_complete_pass": finite_ok and complete and not missing_fields,
        "numeric_intervention_valid": numeric_ok,
        "F_INPUT_GATE_PASS": gate_pass,
        "n_windows_changed": n_win_changed,
        "n_windows_invalid_numeric": n_win_invalid,
        "n_windows_empty_support": n_win_empty,
        "not_ckpt": True,
        "not_forward": True,
        "not_40cell": True,
        "fail_reason": None
        if gate_pass
        else {
            "structure": structure_pass,
            "assignment": assignment_pass,
            "finite": finite_ok,
            "complete": complete,
            "numeric": numeric_ok,
            "gaps": gaps,
        },
    }
    (out_dir / "GATE.json").write_text(json.dumps(gate, indent=2) + "\n")
    (out_dir / "GAPS.json").write_text(
        json.dumps({"n": len(gaps), "gaps": gaps, "stopped": False}, indent=2) + "\n"
    )
    (out_dir / "MANIFEST.json").write_text(
        json.dumps(
            {
                "scheme_id": SCHEME,
                "created_at": now_local(),
                "npz": str(npz_path),
                "F_INPUT_GATE_PASS": gate_pass,
                "change": change,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        "GATE",
        gate_pass,
        "changed_win",
        n_win_changed,
        "invalid_win",
        n_win_invalid,
        "empty",
        n_win_empty,
        "changed_j",
        n_joint_changed,
        "max_abs",
        global_max,
        flush=True,
    )


if __name__ == "__main__":
    main()
