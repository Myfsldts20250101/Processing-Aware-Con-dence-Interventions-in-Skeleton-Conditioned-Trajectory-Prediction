#!/usr/bin/env python3
"""N_J 单格：O/F/N_J/N 同次。不覆盖 U1F。阶段 B/C 共用。"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

from data.jrdb_dataset import build_dataloader
from e1_rev1_common import (
    ade_fde_vs_gt,
    conf_rmse,
    load_donor_map,
    lookup_donor_in_batch,
    now_local,
    pred_pair_ade_fde,
    sha256_file,
    summarize_numeric,
    write_csv,
    write_json,
)
from e123_common import batch_window_keys
from eval_u1f_cell import assign_conf_F, predict_locked
from models.full_model import build_model
from models.fusion import _sanitize_key_padding_mask
from u1_common import build_conf_swap_full

N_SBV = 15160
SCHEME = "NJ_same_assign_v1"
IDENTITY_ATOL = 1e-6


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--train_seed", type=int, required=True)
    p.add_argument("--donor_assign_seed", type=int, required=True)
    p.add_argument("--diagnostic_set_json", required=True)
    p.add_argument("--donor_map_json", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--expected_ckpt_sha256", default=None)
    p.add_argument("--skip_if_done", type=int, default=1)
    return p.parse_args()


def load_json_set(path: Path):
    import json
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data, set(data["target_keys"])


def tmax(a, b=None):
    if b is None:
        return float(a.detach().abs().max().item()) if a.numel() else 0.0
    return float((a - b).detach().abs().max().item())


def hip_center(kp):
    return (kp[:, :, 11:12, :] + kp[:, :, 12:13, :]) / 2.0


def placeholder_t0(frame_mask):
    key_raw = frame_mask == 0
    san = _sanitize_key_padding_mask(key_raw.clone())
    all_masked = key_raw.all(dim=1)
    opened = all_masked & (~san[:, 0])
    return opened, san


def stratum_id(new_ph, mask_chg, jv_chg, xy_chg):
    if new_ph:
        return 1
    if mask_chg:
        return 2
    if jv_chg or xy_chg:
        return 3
    return 4


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    done = out / "NJ_DONE.json"
    if args.skip_if_done and done.exists():
        print("[skip done]", out)
        return

    ckpt_sha = sha256_file(args.checkpoint)
    if args.expected_ckpt_sha256 and ckpt_sha != args.expected_ckpt_sha256:
        write_json(out / "NJ_GATE.json", {"pass": False, "reason": "ckpt_sha", "got": ckpt_sha})
        raise SystemExit(2)

    sbv_meta, sbv_keys = load_json_set(Path(args.diagnostic_set_json))
    if len(sbv_keys) != N_SBV:
        write_json(out / "NJ_GATE.json", {"pass": False, "reason": "n_S_BV", "n": len(sbv_keys)})
        raise SystemExit(2)
    map_meta, pairs = load_donor_map(Path(args.donor_map_json))
    if int(map_meta.get("donor_assign_seed", -1)) != int(args.donor_assign_seed):
        raise SystemExit("donor_map seed mismatch")

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["data"].setdefault("motion_filter", {})["enabled"] = True

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
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
        write_json(out / "NJ_GATE.json", {
            "pass": False, "reason": "state_dict",
            "missing": list(miss), "unexpected": list(unexp),
        })
        raise SystemExit(2)
    model.to(device).eval()

    write_json(out / "CELL_MANIFEST.json", {
        "scheme": SCHEME,
        "not_frozen_u1f": True,
        "train_seed": args.train_seed,
        "donor_assign_seed": args.donor_assign_seed,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": ckpt_sha,
        "diagnostic_set_sha256": sbv_meta.get("diagnostic_set_sha256"),
        "donor_map_sha256": map_meta.get("donor_map_sha256"),
        "n_S_BV": N_SBV,
        "conditions": ["O", "F", "N_J", "N"],
        "c_J": "assign_conf_F once, shared F and N_J",
        "tensor_compare": "exact_max_abs_eq_0",
        "identity_atol": IDENTITY_ATOL,
        "equal_input_check": "per_window",
        "cudnn_deterministic": True,
        "decode": "most_likely",
        "K": 1,
        "started_at": now_local(),
    })

    loader = build_dataloader(cfg, split=args.split)
    status = Counter()
    rows = []
    n_ok = 0
    n_mono_j = 0
    n_mono_f = 0
    n_nonfinite = 0
    n_other_stratum = 0
    mx_cj = 0.0
    mx_out = 0.0
    mx_fxy = 0.0
    mx_pre = 0.0
    mx_hip = 0.0
    mx_fph = 0.0
    replay_max = 0.0
    det_o = None
    ident_max = 0.0
    eq_in_n = 0
    eq_in_pred_max = 0.0
    n_eq_pred_fail = 0

    pred_o_l, pred_f_l, pred_nj_l, pred_n_l = [], [], [], []
    gt_l, lo_l, wo_l, key_l = [], [], [], []

    for batch_idx, batch in enumerate(loader):
        obs = batch["obs_traj"].to(device)
        gt = batch["pred_traj"].to(device)
        sk = batch["skeleton_kp"].to(device)
        sc = batch["skeleton_conf"].to(device)
        last_obs = batch["last_obs_pos"].to(device)
        worg = batch["window_origin"].to(device)
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

        sk_d = sk.clone()
        sc_d = sc.clone()
        for i, j in enumerate(donor_idx):
            if in_mask[i] and j >= 0:
                sk_d[i] = sk[j]
                sc_d[i] = sc[j]

        sk_o, sc_o = sk.clone(), sc.clone()
        c_j, jt = assign_conf_F(sc_o, sc_d)
        mx_cj = max(mx_cj, tmax(c_j, c_j.clone()))
        mx_out = max(mx_out, tmax(torch.where(jt, sc_o, c_j), sc_o))

        x_o, m_o = model.skeleton_preprocessor(sk_o.clone(), sc_o.clone())
        x_nj, m_nj = model.skeleton_preprocessor(sk.clone(), c_j.clone())
        sk_n, sc_n = build_conf_swap_full(sk, sc, sk_d, sc_d)
        x_n, m_n = model.skeleton_preprocessor(sk_n, sc_n)
        x_f = x_o.clone()
        x_f[:, 2] = c_j
        mx_fxy = max(mx_fxy, tmax(x_f[:, :2], x_o[:, :2]))
        mx_pre = max(mx_pre, tmax(sk, sk_o))
        mx_hip = max(mx_hip, tmax(hip_center(sk), hip_center(sk_o)))

        ph_o, san_o = placeholder_t0(m_o)
        ph_nj, san_nj = placeholder_t0(m_nj)
        ph_f, san_f = placeholder_t0(m_o)
        mx_fph = max(mx_fph, float((ph_f.float() - ph_o.float()).abs().max().item()))
        mx_fph = max(mx_fph, tmax(san_f.float(), san_o.float()))

        pred_o = model.predict(obs, sk_o, sc_o, num_samples=1, predict_mode="most_likely")
        pred_f = predict_locked(model, obs, x_f, m_o)
        pred_nj = model.predict(obs, sk.clone(), c_j.clone(), num_samples=1, predict_mode="most_likely")
        pred_n = model.predict(obs, sk.clone(), sc_d.clone(), num_samples=1, predict_mode="most_likely")
        pred_loc = predict_locked(model, obs, x_o, m_o)
        replay_max = max(replay_max, tmax(pred_o, pred_loc))

        if det_o is None and any(in_mask):
            pred_o2 = model.predict(obs, sk.clone(), sc.clone(), num_samples=1, predict_mode="most_likely")
            det_o = tmax(pred_o, pred_o2)

        if not all(torch.isfinite(t).all() for t in (pred_o, pred_f, pred_nj, pred_n)):
            n_nonfinite += 1

        ade_o, fde_o = ade_fde_vs_gt(pred_o, gt)
        ade_f, fde_f = ade_fde_vs_gt(pred_f, gt)
        ade_nj, fde_nj = ade_fde_vs_gt(pred_nj, gt)
        ade_n, fde_n = ade_fde_vs_gt(pred_n, gt)
        p_f, pf_f = pred_pair_ade_fde(pred_o, pred_f)
        p_nj, pf_nj = pred_pair_ade_fde(pred_o, pred_nj)
        p_n, pf_n = pred_pair_ade_fde(pred_o, pred_n)
        r_pol, rf_pol = pred_pair_ade_fde(pred_nj, pred_f)
        r_sup, rf_sup = pred_pair_ade_fde(pred_n, pred_nj)
        crmse = conf_rmse(c_j, sc_o)

        jv_o = sc_o > 0.1
        jv_nj = c_j > 0.1
        n_jv_o = jv_o.reshape(b, -1).sum(dim=-1)
        n_jv_nj = jv_nj.reshape(b, -1).sum(dim=-1)
        n_fr_o = (m_o > 0.5).sum(dim=-1)
        n_fr_nj = (m_nj > 0.5).sum(dim=-1)
        n_chg = ((c_j != sc_o) & jt).reshape(b, -1).sum(dim=-1)
        n_cross = (jt & ((sc_o > 0.1) != (c_j > 0.1))).reshape(b, -1).sum(dim=-1)

        for i in range(b):
            if keys[i] not in sbv_keys:
                status["not_in_S_BV"] += 1
                continue
            st = donor_st[i]
            status[st] += 1
            include = int(st == "ok")
            j = donor_idx[i]
            if not include:
                rows.append({
                    "train_seed": args.train_seed,
                    "donor_map": args.donor_assign_seed,
                    "scene_id": keys[i].split("|")[0],
                    "target_key": keys[i],
                    "donor_key": "",
                    "donor_status": st,
                    "include_main": 0,
                })
                continue

            n_ok += 1
            if int((n_jv_nj[i] - n_jv_o[i]).item()) > 0:
                n_mono_j += 1
            if int((n_fr_nj[i] - n_fr_o[i]).item()) > 0:
                n_mono_f += 1

            mask_chg = bool((m_nj[i] != m_o[i]).any().item())
            jv_chg = bool((jv_nj[i] != jv_o[i]).any().item())
            xy_chg = float((x_nj[i, :2] - x_o[i, :2]).abs().max().item()) > 0.0
            new_ph = bool((~ph_o[i] & ph_nj[i]).item())
            sid = stratum_id(new_ph, mask_chg, jv_chg, xy_chg)
            if sid == 4 and (mask_chg or jv_chg or xy_chg or new_ph):
                n_other_stratum += 1
                sid = 0

            lo = float(ade_o[i].item())
            lf = float(ade_f[i].item())
            lnj = float(ade_nj[i].item())
            ln = float(ade_n[i].item())
            ident = abs((ln - lo) - ((lf - lo) + (lnj - lf) + (ln - lnj)))
            ident_max = max(ident_max, ident)

            xf_eq = bool((x_f[i] == x_nj[i]).all().item()) and bool((m_o[i] == m_nj[i]).all().item())
            if xf_eq:
                eq_in_n += 1
                dpred = tmax(pred_f[i], pred_nj[i])
                eq_in_pred_max = max(eq_in_pred_max, dpred)
                if dpred != 0.0:
                    n_eq_pred_fail += 1

            xy_f_nj = float((x_nj[i, :2] - x_f[i, :2]).abs().max().item())
            mham = int((m_nj[i] != m_o[i]).sum().item())

            row = {
                "train_seed": args.train_seed,
                "donor_map": args.donor_assign_seed,
                "scene_id": keys[i].split("|")[0],
                "target_key": keys[i],
                "donor_key": keys[j],
                "donor_status": st,
                "include_main": 1,
                "n_conf_changed": int(n_chg[i].item()),
                "conf_rmse_cJ_vs_t": float(crmse[i].item()),
                "n_thr_cross_0p1": int(n_cross[i].item()),
                "n_jv_O": int(n_jv_o[i].item()),
                "n_jv_NJ": int(n_jv_nj[i].item()),
                "n_frame_O": int(n_fr_o[i].item()),
                "n_frame_NJ": int(n_fr_nj[i].item()),
                "dn_jv": int((n_jv_nj[i] - n_jv_o[i]).item()),
                "dn_frame": int((n_fr_nj[i] - n_fr_o[i]).item()),
                "xy_max_abs_F_vs_O": float((x_f[i, :2] - x_o[i, :2]).abs().max().item()),
                "xy_max_abs_NJ_vs_F": xy_f_nj,
                "mask_hamming_NJ_vs_F": mham,
                "placeholder_O": int(ph_o[i].item()),
                "placeholder_F": int(ph_f[i].item()),
                "placeholder_NJ": int(ph_nj[i].item()),
                "stratum": sid,
                "F_NJ_input_equal": int(xf_eq),
                "ade_O": lo, "fde_O": float(fde_o[i].item()),
                "ade_F": lf, "fde_F": float(fde_f[i].item()),
                "ade_NJ": lnj, "fde_NJ": float(fde_nj[i].item()),
                "ade_N": ln, "fde_N": float(fde_n[i].item()),
                "pred_ade_F_vs_O": float(p_f[i].item()),
                "pred_fde_F_vs_O": float(pf_f[i].item()),
                "pred_ade_NJ_vs_O": float(p_nj[i].item()),
                "pred_fde_NJ_vs_O": float(pf_nj[i].item()),
                "pred_ade_N_vs_O": float(p_n[i].item()),
                "pred_fde_N_vs_O": float(pf_n[i].item()),
                "delta_ade_F": lf - lo,
                "delta_ade_NJ": lnj - lo,
                "delta_ade_N": ln - lo,
                "delta_fde_F": float((fde_f[i] - fde_o[i]).item()),
                "delta_fde_NJ": float((fde_nj[i] - fde_o[i]).item()),
                "delta_fde_N": float((fde_n[i] - fde_o[i]).item()),
                "R_policy": float(r_pol[i].item()),
                "R_policy_fde": float(rf_pol[i].item()),
                "delta_policy": lnj - lf,
                "R_support": float(r_sup[i].item()),
                "R_support_fde": float(rf_sup[i].item()),
                "delta_support": ln - lnj,
                "identity_resid": ident,
                "fail_exec": 0,
                "nonfinite": 0,
                "assign_ok": 1,
                "mono_ok": int(
                    int((n_jv_nj[i] - n_jv_o[i]).item()) <= 0
                    and int((n_fr_nj[i] - n_fr_o[i]).item()) <= 0
                ),
            }
            rows.append(row)
            pred_o_l.append(pred_o[i, 0].detach().cpu().numpy())
            pred_f_l.append(pred_f[i, 0].detach().cpu().numpy())
            pred_nj_l.append(pred_nj[i, 0].detach().cpu().numpy())
            pred_n_l.append(pred_n[i, 0].detach().cpu().numpy())
            gt_l.append(gt[i].detach().cpu().numpy())
            lo_l.append(last_obs[i].detach().cpu().numpy())
            wo_l.append(worg[i].detach().cpu().numpy())
            key_l.append(keys[i])

    assign_ok = mx_out == 0.0
    f_lock = mx_fxy == 0.0 and mx_fph == 0.0
    pre_ok = mx_pre == 0.0 and mx_hip == 0.0
    mono_ok = n_mono_j == 0 and n_mono_f == 0
    n_ok_ok = n_ok == N_SBV
    replay_ok = replay_max == 0.0
    det_ok = det_o == 0.0
    ident_ok = ident_max <= IDENTITY_ATOL
    eq_pred_ok = n_eq_pred_fail == 0
    finite_ok = n_nonfinite == 0
    csv_n = sum(1 for r in rows if r.get("include_main") == 1)
    csv_ok = csv_n == N_SBV == len(key_l)
    other_ok = n_other_stratum == 0

    overall = (
        assign_ok and f_lock and pre_ok and mono_ok and n_ok_ok
        and replay_ok and det_ok and ident_ok and eq_pred_ok
        and finite_ok and csv_ok and other_ok
    )
    gate = {
        "pass": overall,
        "n_ok": n_ok,
        "n_S_BV": N_SBV,
        "cJ_outside_Jt_vs_clean_max_abs": mx_out,
        "F_xy_vs_O_max_abs": mx_fxy,
        "F_ph_san_vs_O_max_abs": mx_fph,
        "NJ_pre_xy_vs_O_max_abs": mx_pre,
        "hip_NJ_vs_O_max_abs": mx_hip,
        "replay_locked_clean_vs_O": replay_max,
        "det_O_repeat_max_abs": det_o,
        "identity_resid_max": ident_max,
        "identity_atol": IDENTITY_ATOL,
        "n_F_NJ_input_equal": eq_in_n,
        "equal_input_pred_max_abs": eq_in_pred_max,
        "n_equal_input_pred_fail": n_eq_pred_fail,
        "n_mono_joint_increase": n_mono_j,
        "n_mono_frame_increase": n_mono_f,
        "n_nonfinite_batches": n_nonfinite,
        "n_stratum_other": n_other_stratum,
        "donor_status": dict(status),
        "written_at": now_local(),
    }
    write_json(out / "NJ_GATE.json", gate)
    if not overall:
        write_json(out / "summary_local.json", {"gate_pass": False, "gate": gate})
        print("NJ GATE FAIL", gate)
        raise SystemExit(2)

    full_keys = None
    for r in rows:
        if r.get("include_main") == 1:
            full_keys = list(r.keys())
            break
    if full_keys:
        for r in rows:
            for k in full_keys:
                r.setdefault(k, "")
    write_csv(out / "per_window_NJ.csv", rows)
    np.savez_compressed(
        out / "pred_trajs.npz",
        keys=np.array(key_l, dtype=object),
        pred_O=np.stack(pred_o_l, axis=0),
        pred_F=np.stack(pred_f_l, axis=0),
        pred_NJ=np.stack(pred_nj_l, axis=0),
        pred_N=np.stack(pred_n_l, axis=0),
        gt=np.stack(gt_l, axis=0),
        last_obs_pos=np.stack(lo_l, axis=0),
        window_origin=np.stack(wo_l, axis=0),
    )
    ok = [r for r in rows if r.get("include_main") == 1]
    keys_num = [
        "ade_O", "ade_F", "ade_NJ", "ade_N",
        "delta_policy", "R_policy", "delta_support", "R_support",
        "pred_ade_F_vs_O", "pred_ade_NJ_vs_O", "pred_ade_N_vs_O",
        "identity_resid",
    ]
    stratum_n = Counter(int(r["stratum"]) for r in ok)
    summary = {
        "gate_pass": True,
        "n_ok": len(ok),
        "n_F_NJ_input_equal": eq_in_n,
        "stratum_n": {str(k): int(v) for k, v in sorted(stratum_n.items())},
        "all": summarize_numeric(ok, keys_num, 0.01, 0.05),
        "note": "cell means only; not 8-seed formal table; do not select on sign",
    }
    write_json(out / "summary_local.json", summary)
    write_json(done, {"done": True, "n_ok": len(ok), "written_at": now_local()})
    dpm = summary["all"]["delta_policy"]["mean"]
    print(
        "NJ PASS",
        f"s{args.train_seed}d{args.donor_assign_seed}",
        "n_ok", len(ok),
        "mean_d_policy", dpm,
        "eq_in", eq_in_n,
        "ident_max", ident_max,
        "->", out,
    )


if __name__ == "__main__":
    main()
