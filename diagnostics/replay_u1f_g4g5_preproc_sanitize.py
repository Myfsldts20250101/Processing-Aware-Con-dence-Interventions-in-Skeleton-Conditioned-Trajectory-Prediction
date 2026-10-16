#!/usr/bin/env python3
"""U1F G4/G5: replay SkeletonPreprocessor + sanitize on frozen 256 clean windows.

Reads C0 npz only. No Dataset reload, no donor assignment, no F, no ckpt, no model.forward.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import torch
import yaml

from models.ctr_gcn import SkeletonPreprocessor
from models.fusion import _sanitize_key_padding_mask

ROSTER_PAIRS_SHA = "d93ba2e1e2325f6f57a5f4ab803431e2426f07374d787d22a66d5cd14b385e58"
C0_NPZ_SHA = "6959d26577afc0023a5e1f0bfb3155ae5cf9c142529ad4b1d4902dbe0bae702e"
CONFIG_SHA = "95dc8e80801ed3d794da6b120cfae68fa9da1296455dc71d03ce04f33f513335"
CONF_THR = 0.1
HIP_L, HIP_R = 11, 12
TZ = timezone(timedelta(hours=8))
SCHEME = "U1F_G4G5_replay_s42d42_n256"
SIDES = ("target", "donor")

C0_REQUIRED = [
    "key",
    "donor_key",
    "roster_idx",
    "kp_loader_target",
    "conf_loader_target",
    "kp_loader_donor",
    "conf_loader_donor",
    "x_preproc_target",
    "x_preproc_donor",
    "frame_mask_target",
    "frame_mask_donor",
    "joint_valid_target",
    "joint_valid_donor",
    "key_mask_raw_target",
    "key_mask_san_target",
    "key_mask_raw_donor",
    "key_mask_san_donor",
    "placeholder_t0_target",
    "placeholder_t0_donor",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now_local() -> str:
    return datetime.now(TZ).strftime("%Y-%m-%dT%H:%M:%S%z")


def arr_sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def float_diff(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        return {
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
            "max_abs": None,
            "n_mismatch": None,
            "n_nan_a": int(np.isnan(a).sum()) if np.issubdtype(a.dtype, np.floating) else None,
            "pass_exact": False,
            "gap": "shape_mismatch",
        }
    if np.issubdtype(a.dtype, np.floating) or np.issubdtype(b.dtype, np.floating):
        aa = a.astype(np.float64, copy=False)
        bb = b.astype(np.float64, copy=False)
        nan_a = np.isnan(aa)
        nan_b = np.isnan(bb)
        nan_xor = nan_a ^ nan_b
        both = (~nan_a) & (~nan_b)
        if both.any():
            max_abs = float(np.max(np.abs(aa[both] - bb[both])))
        else:
            max_abs = 0.0
        n_mis = int(nan_xor.sum()) + int(both.sum() and np.sum(np.abs(aa[both] - bb[both]) != 0))
        if both.any():
            n_mis = int(nan_xor.sum()) + int(np.sum(np.abs(aa[both] - bb[both]) != 0))
        else:
            n_mis = int(nan_xor.sum())
        return {
            "shape": list(a.shape),
            "dtype_a": str(a.dtype),
            "dtype_b": str(b.dtype),
            "max_abs": max_abs,
            "n_mismatch": n_mis,
            "n_nan_a": int(nan_a.sum()),
            "n_nan_b": int(nan_b.sum()),
            "pass_exact": bool(n_mis == 0 and max_abs == 0.0),
        }
    n_mis = int(np.sum(a != b))
    return {
        "shape": list(a.shape),
        "dtype_a": str(a.dtype),
        "dtype_b": str(b.dtype),
        "max_abs": 0 if n_mis == 0 else 1,
        "n_mismatch": n_mis,
        "pass_exact": n_mis == 0,
    }


def sanitize_from_frame_mask(frame_mask: np.ndarray):
    """frame_mask 1=valid frame → pad True=masked. Returns key_raw, key_san, placeholder, pad_raw, pad_san."""
    fm = torch.from_numpy(frame_mask.astype(np.float32))
    raw_pad = fm == 0
    san_pad = _sanitize_key_padding_mask(raw_pad.clone())
    all_masked = raw_pad.all(dim=-1)
    placeholder = all_masked & (~san_pad[:, 0])
    key_raw = (~raw_pad).to(torch.int8).numpy()
    key_san = (~san_pad).to(torch.int8).numpy()
    ph = placeholder.to(torch.int8).numpy()
    return key_raw, key_san, ph, raw_pad.numpy(), san_pad.numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/jrdb_v2_decoupled_z.yaml")
    p.add_argument(
        "--roster",
        default="eval_logs/u1f_c0_loader_state_s42d42_n256_DESIGN/ROSTER.json",
    )
    p.add_argument(
        "--c0_npz",
        default="eval_logs/u1f_c0_loader_state_s42d42_n256/state_s42d42_n256.npz",
    )
    p.add_argument(
        "--out_dir",
        default="eval_logs/u1f_g4g5_replay_s42d42_n256",
    )
    args = p.parse_args()

    root = Path(".").resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gaps = []

    def stop_gap(code: str, detail: str):
        gaps.append({"gap_code": code, "detail": detail})
        (out_dir / "GAPS.json").write_text(
            json.dumps({"n": len(gaps), "gaps": gaps, "stopped": True}, indent=2) + "\n"
        )
        raise SystemExit(f"STOP {code}: {detail}")

    cfg_path = Path(args.config)
    cfg_sha = sha256_file(cfg_path)
    if cfg_sha != CONFIG_SHA:
        stop_gap("config_sha_mismatch", f"{cfg_sha} != {CONFIG_SHA}")

    roster = json.loads(Path(args.roster).read_text(encoding="utf-8"))
    if roster.get("pairs_sha256") != ROSTER_PAIRS_SHA:
        stop_gap("roster_sha_mismatch", str(roster.get("pairs_sha256")))
    if len(roster["windows"]) != 256:
        stop_gap("roster_n", str(len(roster["windows"])))

    c0_path = Path(args.c0_npz)
    c0_sha = sha256_file(c0_path)
    if c0_sha != C0_NPZ_SHA:
        stop_gap("c0_npz_sha_mismatch", f"{c0_sha} != {C0_NPZ_SHA}")

    c0 = np.load(c0_path, allow_pickle=False)
    missing = [k for k in C0_REQUIRED if k not in c0.files]
    if missing:
        stop_gap("c0_missing_arrays", ",".join(missing))

    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    skel_cfg = cfg["skeleton"]
    num_joints = int(skel_cfg["num_joints"])
    use_confidence = bool(skel_cfg.get("use_confidence", True))
    center_joint = skel_cfg.get("center_joint", None)
    if center_joint is not None:
        stop_gap("unexpected_center_joint", str(center_joint))
    if num_joints != 17 or not use_confidence:
        stop_gap("unexpected_skel_cfg", json.dumps({"num_joints": num_joints, "use_confidence": use_confidence}))
    if skel_cfg.get("pretrained_weights"):
        stop_gap("skel_pretrained_weights_set", str(skel_cfg.get("pretrained_weights")))

    pre = SkeletonPreprocessor(
        num_joints=num_joints,
        use_confidence=use_confidence,
        center_joint=None,
    )
    pre.eval()
    n_param = sum(p.numel() for p in pre.parameters())
    if n_param != 0:
        stop_gap("preprocessor_has_learned_params", str(n_param))

    n = 256
    keys_c0 = c0["key"]
    roster_keys = [w["key"] for w in sorted(roster["windows"], key=lambda x: int(x["roster_idx"]))]
    if [str(k) for k in keys_c0] != roster_keys:
        stop_gap("key_order_mismatch_c0_roster", "C0 key vs ROSTER")

    replay = {
        "key": keys_c0.copy(),
        "donor_key": c0["donor_key"].copy(),
        "roster_idx": c0["roster_idx"].copy(),
    }

    diffs = {}
    n_pass = 0
    n_fail = 0

    for side in SIDES:
        kp = np.array(c0[f"kp_loader_{side}"], dtype=np.float32, copy=True)
        conf = np.array(c0[f"conf_loader_{side}"], dtype=np.float32, copy=True)
        if kp.shape != (n, 8, 17, 2) or conf.shape != (n, 8, 17):
            stop_gap("loader_shape", f"{side} kp={kp.shape} conf={conf.shape}")
        if np.isnan(kp).any() or np.isnan(conf).any():
            stop_gap("loader_nan", side)

        kp_t = torch.from_numpy(kp)
        conf_t = torch.from_numpy(conf)
        with torch.no_grad():
            x_t, mask_t = pre(kp_t, conf_t)
        x = x_t.cpu().numpy().astype(np.float32)
        frame = mask_t.cpu().numpy()
        frame_i8 = np.rint(frame).astype(np.int8)
        if not np.allclose(frame, frame_i8.astype(np.float32), atol=0, rtol=0):
            stop_gap("frame_mask_not_01", side)

        center = (kp[:, :, HIP_L : HIP_L + 1, :] + kp[:, :, HIP_R : HIP_R + 1, :]) / 2.0
        xy_centered = kp - center
        valid = (conf > CONF_THR).astype(np.float32)
        xy_zeroed = xy_centered * valid[..., None]
        x_xy_ref = np.transpose(xy_zeroed, (0, 3, 1, 2))
        x_conf_ref = conf[:, None, :, :]
        joint_valid = (conf > CONF_THR).astype(np.int8)
        frame_from_valid = (valid.sum(axis=-1) >= 3).astype(np.int8)

        key_raw_from_replay, key_san_from_replay, ph_from_replay, _, _ = sanitize_from_frame_mask(frame_i8)
        key_raw_from_c0fm, key_san_from_c0fm, ph_from_c0fm, _, _ = sanitize_from_frame_mask(
            np.array(c0[f"frame_mask_{side}"])
        )

        replay[f"center_hip_{side}"] = center.astype(np.float32)
        replay[f"xy_centered_{side}"] = xy_centered.astype(np.float32)
        replay[f"xy_zeroed_{side}"] = xy_zeroed.astype(np.float32)
        replay[f"x_preproc_{side}"] = x
        replay[f"conf_channel_{side}"] = x[:, 2].astype(np.float32)
        replay[f"frame_mask_{side}"] = frame_i8
        replay[f"joint_valid_{side}"] = joint_valid
        replay[f"key_mask_raw_{side}"] = key_raw_from_replay
        replay[f"key_mask_san_{side}"] = key_san_from_replay
        replay[f"placeholder_t0_{side}"] = ph_from_replay
        replay[f"key_mask_san_from_c0_frame_{side}"] = key_san_from_c0fm
        replay[f"placeholder_from_c0_frame_{side}"] = ph_from_c0fm

        checks = {
            f"G4_x_preproc_{side}": float_diff(x, c0[f"x_preproc_{side}"]),
            f"G4_x_xy_vs_center_zero_{side}": float_diff(x[:, :2], x_xy_ref.astype(np.float32)),
            f"G4_x_conf_channel_vs_loader_{side}": float_diff(x[:, 2], conf),
            f"G4_frame_mask_{side}": float_diff(frame_i8, c0[f"frame_mask_{side}"]),
            f"G4_frame_vs_joint_count_{side}": float_diff(frame_i8, frame_from_valid),
            f"G4_joint_valid_{side}": float_diff(joint_valid, c0[f"joint_valid_{side}"]),
            f"G5_key_raw_from_replay_fm_{side}": float_diff(key_raw_from_replay, c0[f"key_mask_raw_{side}"]),
            f"G5_key_san_from_replay_fm_{side}": float_diff(key_san_from_replay, c0[f"key_mask_san_{side}"]),
            f"G5_placeholder_from_replay_fm_{side}": float_diff(ph_from_replay, c0[f"placeholder_t0_{side}"]),
            f"G5_key_raw_from_c0_fm_{side}": float_diff(key_raw_from_c0fm, c0[f"key_mask_raw_{side}"]),
            f"G5_key_san_from_c0_fm_{side}": float_diff(key_san_from_c0fm, c0[f"key_mask_san_{side}"]),
            f"G5_placeholder_from_c0_fm_{side}": float_diff(ph_from_c0fm, c0[f"placeholder_t0_{side}"]),
        }
        for name, d in checks.items():
            diffs[name] = d
            if d.get("pass_exact"):
                n_pass += 1
            else:
                n_fail += 1
                gaps.append(
                    {
                        "gap_code": name,
                        "detail": json.dumps({k: v for k, v in d.items() if k != "shape"}),
                    }
                )

    npz_path = out_dir / "replay_s42d42_n256.npz"
    np.savez_compressed(npz_path, **replay)

    identity = {
        "scheme_id": SCHEME,
        "created_at": now_local(),
        "n": 256,
        "roster_pairs_sha256": ROSTER_PAIRS_SHA,
        "c0_npz_sha256": c0_sha,
        "config_sha256": cfg_sha,
        "config": str(cfg_path),
        "preprocessor": {
            "class": "SkeletonPreprocessor",
            "num_joints": num_joints,
            "use_confidence": use_confidence,
            "center_joint": None,
            "center_rule": "mean(COCO 11 left_hip, 12 right_hip)",
            "conf_thr_zero_and_frame": CONF_THR,
            "min_valid_joints_per_frame": 3,
            "n_parameters": n_param,
            "ckpt_loaded": False,
            "dataset_reloaded": False,
            "donor_assigned": False,
            "model_forward": False,
        },
        "sanitize": {
            "fn": "models.fusion._sanitize_key_padding_mask",
            "rule": "if a row is all True (all keys padded), set t0 pad False",
        },
        "source_file_sha256": {
            "models/ctr_gcn.py": sha256_file(root / "models" / "ctr_gcn.py"),
            "models/fusion.py": sha256_file(root / "models" / "fusion.py"),
        },
        "npz_replay_sha256": sha256_file(npz_path),
        "arrays": {k: arr_sha(v) for k, v in replay.items()},
    }
    (out_dir / "IDENTITY.json").write_text(json.dumps(identity, indent=2) + "\n")

    precision = {
        "compute": "float32 tensors into SkeletonPreprocessor; diffs also reported in float64 abs",
        "masks": "int8 after rint of preprocessor mask; exact equality vs C0 int8",
        "no_float64_as_official": True,
        "no_silent_zero_fill": True,
        "pass_rule": "max_abs==0 and n_mismatch==0 (bit-identical after float32 module out)",
        "new_diagnostic_not_in_C0": [
            "center_hip_* (T,1,2)",
            "xy_centered_* before conf zero",
            "xy_zeroed_* after conf zero",
            "conf_channel_* == x[:,2]",
        ],
    }
    (out_dir / "PRECISION.json").write_text(json.dumps(precision, indent=2) + "\n")

    (out_dir / "DIFF.json").write_text(json.dumps(diffs, indent=2) + "\n")
    with (out_dir / "DIFF.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["check", "pass_exact", "max_abs", "n_mismatch", "n_nan_a", "n_nan_b", "gap"],
        )
        w.writeheader()
        for name, d in diffs.items():
            w.writerow(
                {
                    "check": name,
                    "pass_exact": int(bool(d.get("pass_exact"))),
                    "max_abs": d.get("max_abs"),
                    "n_mismatch": d.get("n_mismatch"),
                    "n_nan_a": d.get("n_nan_a", ""),
                    "n_nan_b": d.get("n_nan_b", ""),
                    "gap": d.get("gap", ""),
                }
            )

    g4_keys = [k for k in diffs if k.startswith("G4_")]
    g5_keys = [k for k in diffs if k.startswith("G5_")]
    g4_ok = all(diffs[k].get("pass_exact") for k in g4_keys)
    g5_ok = all(diffs[k].get("pass_exact") for k in g5_keys)
    n_ph_t = int(replay["placeholder_t0_target"].sum())
    n_ph_d = int(replay["placeholder_t0_donor"].sum())

    gate = {
        "G4_pass": g4_ok,
        "G5_pass": g5_ok,
        "n_checks_pass": n_pass,
        "n_checks_fail": n_fail,
        "n_placeholder_target": n_ph_t,
        "n_placeholder_donor": n_ph_d,
        "not_F": True,
        "not_ckpt": True,
        "not_forward": True,
    }
    (out_dir / "GATE.json").write_text(json.dumps(gate, indent=2) + "\n")
    (out_dir / "GAPS.json").write_text(
        json.dumps({"n": len(gaps), "gaps": gaps, "stopped": False}, indent=2) + "\n"
    )
    man = {
        "scheme_id": SCHEME,
        "created_at": now_local(),
        "cwd": str(root),
        "n": 256,
        "G4_pass": g4_ok,
        "G5_pass": g5_ok,
        "n_gap_rows": len(gaps),
        "npz": str(npz_path),
        "forbid_done": [
            "donor_assignment",
            "U1-F",
            "load_best.pt",
            "model.forward",
            "roster_expand",
        ],
        "c0_preserved": True,
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(man, indent=2) + "\n")
    print("G4", g4_ok, "G5", g5_ok, "fail", n_fail, "placeholder_t/d", n_ph_t, n_ph_d, flush=True)


if __name__ == "__main__":
    main()
