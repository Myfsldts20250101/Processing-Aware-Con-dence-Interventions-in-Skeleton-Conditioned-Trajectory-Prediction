#!/usr/bin/env python3
"""同会话 N/F 诊断：一次进程内算 O / N / F。不写入冻结 40 格，不改 20 格历史。"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import torch
import yaml

from data.jrdb_dataset import build_dataloader
from e1_rev1_common import (
    ade_fde_vs_gt,
    load_donor_map,
    lookup_donor_in_batch,
    now_local,
    pred_pair_ade_fde,
    sha256_file,
    write_csv,
    write_json,
)
from e123_common import batch_window_keys
from eval_u1f_cell import assign_conf_F, predict_locked
from models.full_model import build_model
from u1_common import ORIGINAL_ADE_ATOL, build_conf_swap_full

N_SBV = 15160
SCHEME = "NF_same_session_diag_v1"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--train_seed", type=int, required=True)
    p.add_argument("--donor_assign_seed", type=int, required=True)
    p.add_argument("--diagnostic_set_json", required=True)
    p.add_argument("--donor_map_json", required=True)
    p.add_argument("--e2_csv", required=True)
    p.add_argument("--u1_csv", required=True)
    p.add_argument("--u1f_csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--expected_ckpt_sha256", default=None)
    p.add_argument("--skip_if_done", type=int, default=1)
    p.add_argument(
        "--cudnn_mode",
        choices=["u1f", "legacy"],
        default="u1f",
        help="u1f=cudnn.deterministic (与 U1F 评测一致); legacy=不改 cudnn (与 E2/U1 评测一致)",
    )
    return p.parse_args()


def load_json_set(path: Path):
    import json
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data, set(data["target_keys"])


def load_csv_by_key(path: Path) -> dict:
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            out[rec["key"]] = rec
    return out


def fnum(x):
    if x is None or x == "":
        return None
    return float(x)


def max_abs_pairs(a, b):
    mx = 0.0
    n_gt = 0
    n_miss = 0
    for x, y in zip(a, b):
        if x is None or y is None:
            n_miss += 1
            continue
        d = abs(x - y)
        mx = max(mx, d)
        if d > ORIGINAL_ADE_ATOL:
            n_gt += 1
    return mx, n_gt, n_miss


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    done = out / "DIAG_DONE.json"
    if args.skip_if_done and done.exists():
        print("[skip done]", out)
        return

    ckpt_sha = sha256_file(args.checkpoint)
    if args.expected_ckpt_sha256 and ckpt_sha != args.expected_ckpt_sha256:
        write_json(out / "DIAG_GATE.json", {"pass": False, "reason": "ckpt_sha", "got": ckpt_sha})
        raise SystemExit(2)

    sbv_meta, sbv_keys = load_json_set(Path(args.diagnostic_set_json))
    if len(sbv_keys) != N_SBV:
        write_json(out / "DIAG_GATE.json", {"pass": False, "reason": "n_S_BV", "n": len(sbv_keys)})
        raise SystemExit(2)
    map_meta, pairs = load_donor_map(Path(args.donor_map_json))
    if int(map_meta.get("donor_assign_seed", -1)) != int(args.donor_assign_seed):
        raise SystemExit("donor_map seed mismatch")

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["data"].setdefault("motion_filter", {})["enabled"] = True

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if args.cudnn_mode == "u1f":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    blob = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "config" in blob:
        saved = blob["config"]
        saved["data"] = cfg["data"]
        cfg = saved
    model = build_model(cfg, verbose=False)
    miss, unexp = model.load_state_dict(blob["model_state_dict"], strict=False)
    if miss or unexp:
        write_json(out / "DIAG_GATE.json", {
            "pass": False, "reason": "state_dict",
            "missing": list(miss), "unexpected": list(unexp),
        })
        raise SystemExit(2)
    model.to(device).eval()

    write_json(out / "CELL_MANIFEST.json", {
        "scheme": SCHEME,
        "not_frozen_40cell": True,
        "does_not_rewrite_u1f": True,
        "train_seed": args.train_seed,
        "donor_assign_seed": args.donor_assign_seed,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": ckpt_sha,
        "cudnn_mode": args.cudnn_mode,
        "n_S_BV": N_SBV,
        "diagnostic_set_sha256": sbv_meta.get("diagnostic_set_sha256"),
        "donor_map_sha256": map_meta.get("donor_map_sha256"),
        "decode": "most_likely",
        "K": 1,
        "started_at": now_local(),
    })

    e2_by = load_csv_by_key(Path(args.e2_csv))
    u1_by = load_csv_by_key(Path(args.u1_csv))
    u1f_by = load_csv_by_key(Path(args.u1f_csv))

    loader = build_dataloader(cfg, split=args.split)
    status = Counter()
    rows = []
    replay_o_max = 0.0
    replay_n_max = 0.0
    det_o = None
    n_ok = 0

    for batch_idx, batch in enumerate(loader):
        obs = batch["obs_traj"].to(device)
        gt = batch["pred_traj"].to(device)
        sk = batch["skeleton_kp"].to(device)
        sc = batch["skeleton_conf"].to(device)
        b = obs.shape[0]
        keys = batch_window_keys(batch)
        donor_idx = [-1] * b
        donor_st = ["not_in_S_BV"] * b
        for i, key in enumerate(keys):
            if key not in sbv_keys:
                continue
            j, st, _ = lookup_donor_in_batch(pairs, key, keys)
            donor_idx[i] = j
            donor_st[i] = st if st != "ok" else "ok"

        in_mask = [keys[i] in sbv_keys and donor_st[i] == "ok" for i in range(b)]

        x_c, m_c = model.skeleton_preprocessor(sk, sc)
        pred_o = model.predict(obs, sk, sc, num_samples=1, predict_mode="most_likely")
        pred_o_lock = predict_locked(model, obs, x_c, m_c)
        replay_o_max = max(replay_o_max, float((pred_o - pred_o_lock).abs().max().item()))
        if det_o is None:
            pred_o2 = model.predict(obs, sk, sc, num_samples=1, predict_mode="most_likely")
            det_o = float((pred_o - pred_o2).abs().max().item())

        sk_d = sk.clone()
        sc_d = sc.clone()
        for i, j in enumerate(donor_idx):
            if in_mask[i] and j >= 0:
                sk_d[i] = sk[j]
                sc_d[i] = sc[j]

        sk_n, sc_n = build_conf_swap_full(sk, sc, sk_d, sc_d)
        pred_n = model.predict(obs, sk_n, sc_n, num_samples=1, predict_mode="most_likely")
        x_n, m_n = model.skeleton_preprocessor(sk_n, sc_n)
        pred_n_lock = predict_locked(model, obs, x_n, m_n)
        replay_n_max = max(replay_n_max, float((pred_n - pred_n_lock).abs().max().item()))

        sc_f, _jt = assign_conf_F(sc, sc_d)
        x_f = x_c.clone()
        x_f[:, 2] = sc_f
        pred_f = predict_locked(model, obs, x_f, m_c)

        ade_o, fde_o = ade_fde_vs_gt(pred_o, gt)
        ade_n, fde_n = ade_fde_vs_gt(pred_n, gt)
        ade_f, fde_f = ade_fde_vs_gt(pred_f, gt)
        p_nf, pf_nf = pred_pair_ade_fde(pred_n, pred_f)
        p_of, pf_of = pred_pair_ade_fde(pred_o, pred_f)
        p_on, pf_on = pred_pair_ade_fde(pred_o, pred_n)

        for i in range(b):
            if keys[i] not in sbv_keys:
                status["not_in_S_BV"] += 1
                continue
            st = donor_st[i]
            status[st] += 1
            include = int(st == "ok")
            if include:
                n_ok += 1
            j = donor_idx[i]
            e2 = e2_by.get(keys[i], {})
            u1 = u1_by.get(keys[i], {})
            u1f = u1f_by.get(keys[i], {})
            row = {
                "train_seed": args.train_seed,
                "donor_assign_seed": args.donor_assign_seed,
                "cudnn_mode": args.cudnn_mode,
                "key": keys[i],
                "donor_key": keys[j] if include else "",
                "donor_status": st,
                "include_main": include,
                "ade_O": float(ade_o[i].item()) if include else "",
                "fde_O": float(fde_o[i].item()) if include else "",
                "ade_N": float(ade_n[i].item()) if include else "",
                "fde_N": float(fde_n[i].item()) if include else "",
                "ade_F": float(ade_f[i].item()) if include else "",
                "fde_F": float(fde_f[i].item()) if include else "",
                "delta_N_minus_O": float((ade_n[i] - ade_o[i]).item()) if include else "",
                "delta_F_minus_O": float((ade_f[i] - ade_o[i]).item()) if include else "",
                "A_N_minus_F": float((ade_n[i] - ade_f[i]).item()) if include else "",
                "pred_ade_NF": float(p_nf[i].item()) if include else "",
                "pred_fde_NF": float(pf_nf[i].item()) if include else "",
                "pred_ade_OF": float(p_of[i].item()) if include else "",
                "pred_ade_ON": float(p_on[i].item()) if include else "",
                "e2_original_ade": e2.get("original_ade", "") if include else "",
                "u1_ade_clean": u1.get("ade_clean", "") if include else "",
                "u1_ade_intervene": u1.get("ade_intervene", "") if include else "",
                "u1_pred_ade": u1.get("pred_ade", "") if include else "",
                "u1_delta_ade": u1.get("delta_ade", "") if include else "",
                "u1f_ade_O": u1f.get("ade_O", "") if include else "",
                "u1f_ade_F": u1f.get("ade_F", "") if include else "",
                "u1f_pred_ade": u1f.get("pred_ade", "") if include else "",
                "u1f_delta_ade": u1f.get("delta_ade", "") if include else "",
            }
            rows.append(row)

    if n_ok != N_SBV:
        write_json(out / "DIAG_GATE.json", {
            "pass": False, "reason": "n_ok", "n_ok": n_ok, "n_S_BV": N_SBV,
            "donor_status": dict(status),
        })
        raise SystemExit(2)

    write_csv(out / "per_window_NF.csv", rows)
    ok = [r for r in rows if r["include_main"] == 1]

    this_o = [fnum(r["ade_O"]) for r in ok]
    this_n = [fnum(r["ade_N"]) for r in ok]
    this_f = [fnum(r["ade_F"]) for r in ok]
    this_a = [fnum(r["A_N_minus_F"]) for r in ok]
    this_pnf = [fnum(r["pred_ade_NF"]) for r in ok]
    arch_o_e2 = [fnum(r["e2_original_ade"]) for r in ok]
    arch_o_u1 = [fnum(r["u1_ade_clean"]) for r in ok]
    arch_n = [fnum(r["u1_ade_intervene"]) for r in ok]
    arch_f = [fnum(r["u1f_ade_F"]) for r in ok]
    arch_o_u1f = [fnum(r["u1f_ade_O"]) for r in ok]
    # 归档 Pred N−F 无直接列（U1 pred_ade 是 N vs clean）。对照用 A = ADE_N − ADE_F。
    arch_a = []
    for r in ok:
        n = fnum(r["u1_ade_intervene"])
        f = fnum(r["u1f_ade_F"])
        arch_a.append(None if n is None or f is None else n - f)

    mx_o_e2, n_o_e2, miss_o_e2 = max_abs_pairs(this_o, arch_o_e2)
    mx_o_u1, n_o_u1, miss_o_u1 = max_abs_pairs(this_o, arch_o_u1)
    mx_o_u1f, n_o_u1f, miss_o_u1f = max_abs_pairs(this_o, arch_o_u1f)
    mx_n, n_n, miss_n = max_abs_pairs(this_n, arch_n)
    mx_f, n_f, miss_f = max_abs_pairs(this_f, arch_f)
    mx_a, n_a, miss_a = max_abs_pairs(this_a, arch_a)

    # 归档 Pred(N,F) 没有直接列；用当次 Pred NF vs |U1 pred_ade - U1F pred_ade| 无定义。
    # 报告当次 Pred NF 均值，以及归档 A 均值。
    summary = {
        "scheme": SCHEME,
        "not_a_replacement_for_frozen_40cell": True,
        "historical_20cell_atol_unchanged": True,
        "train_seed": args.train_seed,
        "donor_assign_seed": args.donor_assign_seed,
        "cudnn_mode": args.cudnn_mode,
        "n_ok": n_ok,
        "replay_O_predict_vs_locked_max_abs": replay_o_max,
        "replay_N_predict_vs_locked_max_abs": replay_n_max,
        "det_O_repeat_max_abs": det_o,
        "this_mean_A_N_minus_F": mean(this_a),
        "this_mean_pred_ade_NF": mean(this_pnf),
        "arch_mean_A_u1N_minus_u1fF": mean([x for x in arch_a if x is not None]),
        "max_abs_thisA_minus_archA": mx_a,
        "n_windows_A_gt_atol": n_a,
        "n_missing_arch_A": miss_a,
        "vs_archive": {
            "ade_O_vs_e2_original": {"max_abs": mx_o_e2, "n_gt_atol": n_o_e2, "n_missing": miss_o_e2},
            "ade_O_vs_u1_clean": {"max_abs": mx_o_u1, "n_gt_atol": n_o_u1, "n_missing": miss_o_u1},
            "ade_O_vs_u1f_O": {"max_abs": mx_o_u1f, "n_gt_atol": n_o_u1f, "n_missing": miss_o_u1f},
            "ade_N_vs_u1_intervene": {"max_abs": mx_n, "n_gt_atol": n_n, "n_missing": miss_n},
            "ade_F_vs_u1f_F": {"max_abs": mx_f, "n_gt_atol": n_f, "n_missing": miss_f},
        },
        "readout": {
            "if_N_and_F_both_near_archive": "跨次执行差小，归档 N−F 可视为同会话干预差的近似",
            "if_F_near_archive_N_far": "归档 N 含跨次差；以本会话 A / Pred NF 为准判断干预量级",
            "if_O_still_far_from_e2": "20 格 resid 仍在，符合预期，不追改",
            "replay_N_nonzero": "N 上 predict vs predict_locked 仍有差，实现路径会混入 N−F",
        },
        "original_ade_atol": ORIGINAL_ADE_ATOL,
        "donor_status": dict(status),
        "written_at": now_local(),
    }
    write_json(out / "DIAG_SUMMARY.json", summary)
    write_json(out / "DIAG_GATE.json", {
        "pass": True,
        "meaning": "identity/n_ok only; not original_ade_align; not frozen-table rewrite",
        "n_ok": n_ok,
        "replay_O_max_abs": replay_o_max,
        "replay_N_max_abs": replay_n_max,
    })
    write_json(done, {"done": True, "written_at": now_local()})
    print(
        "DIAG", args.train_seed, args.donor_assign_seed, args.cudnn_mode,
        "n_ok", n_ok,
        "meanA", summary["this_mean_A_N_minus_F"],
        "archA", summary["arch_mean_A_u1N_minus_u1fF"],
        "max|A-archA|", mx_a,
        "max|N-u1N|", mx_n,
        "max|F-u1fF|", mx_f,
        "replayO", replay_o_max,
        "replayN", replay_n_max,
        "->", out,
    )


if __name__ == "__main__":
    main()
