#!/usr/bin/env python3
"""N_J 阶段 A：s42d42 按输入状态选窗 + 方案 §6 检查。不覆盖 U1F，不跑 40 格。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import torch
import yaml

from data.jrdb_dataset import build_dataloader
from e1_rev1_common import (
    ade_fde_vs_gt,
    load_donor_map,
    lookup_donor_in_batch,
    now_local,
    sha256_file,
    write_json,
)
from e123_common import batch_window_keys
from eval_u1f_cell import assign_conf_F, predict_locked
from models.full_model import build_model
from models.fusion import _sanitize_key_padding_mask
from u1_common import build_conf_swap_full

N_SBV = 15160
SCHEME = "NJ_same_assign_stageA_v1"
TAGS = (
    "conf_unchanged",
    "valid_unchanged",
    "n_valid_joints_down",
    "frame_mask_changed",
    "new_placeholder",
    "xy_changed_mask_same",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--train_seed", type=int, default=42)
    p.add_argument("--donor_assign_seed", type=int, default=42)
    p.add_argument("--diagnostic_set_json", required=True)
    p.add_argument("--donor_map_json", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--expected_ckpt_sha256", default=None)
    p.add_argument("--skip_if_done", type=int, default=1)
    p.add_argument("--max_per_tag", type=int, default=1)
    return p.parse_args()


def load_json_set(path: Path):
    import json
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data, set(data["target_keys"])


def tmax(a, b=None):
    if b is None:
        return float(a.abs().max().item()) if a.numel() else 0.0
    return float((a - b).abs().max().item())


def hip_center(kp):
    return (kp[:, :, 11:12, :] + kp[:, :, 12:13, :]) / 2.0


def placeholder_t0(frame_mask):
    key_raw = frame_mask == 0
    san = _sanitize_key_padding_mask(key_raw.clone())
    all_masked = key_raw.all(dim=1)
    opened = all_masked & (~san[:, 0])
    return opened, san


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    done = out / "STAGEA_DONE.json"
    if args.skip_if_done and done.exists():
        print("[skip done]", out)
        return

    ckpt_sha = sha256_file(args.checkpoint)
    if args.expected_ckpt_sha256 and ckpt_sha != args.expected_ckpt_sha256:
        write_json(out / "STAGEA_GATE.json", {"pass": False, "reason": "ckpt_sha", "got": ckpt_sha})
        raise SystemExit(2)

    sbv_meta, sbv_keys = load_json_set(Path(args.diagnostic_set_json))
    if len(sbv_keys) != N_SBV:
        write_json(out / "STAGEA_GATE.json", {"pass": False, "reason": "n_S_BV", "n": len(sbv_keys)})
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
        write_json(out / "STAGEA_GATE.json", {
            "pass": False, "reason": "state_dict",
            "missing": list(miss), "unexpected": list(unexp),
        })
        raise SystemExit(2)
    model.to(device).eval()

    write_json(out / "MANIFEST.json", {
        "scheme": SCHEME,
        "stage": "A",
        "not_frozen_u1f": True,
        "train_seed": args.train_seed,
        "donor_assign_seed": args.donor_assign_seed,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": ckpt_sha,
        "diagnostic_set_sha256": sbv_meta.get("diagnostic_set_sha256"),
        "donor_map_sha256": map_meta.get("donor_map_sha256"),
        "n_S_BV": N_SBV,
        "tensor_compare": "exact_max_abs_eq_0",
        "pred_repeat_gate": "max_abs_eq_0",
        "historical_original_ade_atol_not_used_for_tensors": 1e-4,
        "cudnn_deterministic": True,
        "decode": "most_likely",
        "K": 1,
        "started_at": now_local(),
    })

    loader = build_dataloader(cfg, split=args.split)
    status = Counter()
    n_ok = 0
    mx = defaultdict(float)
    n_mono_joint_up = 0
    n_mono_frame_up = 0
    n_nonfinite = 0
    selected = {t: [] for t in TAGS}
    tag_counts = Counter()
    sample_rows = []
    det_o = None
    order_max = None
    replay_clean = 0.0
    equal_input_pred_max = 0.0
    n_equal_input = 0

    def bump(name, val):
        mx[name] = max(mx[name], float(val))

    @torch.no_grad()
    def four_preds(obs, sk, sc, sc_d, c_j, x_o, m_o):
        sk_o, sc_o = sk.clone(), sc.clone()
        pred_o = model.predict(obs, sk_o, sc_o, num_samples=1, predict_mode="most_likely")
        x_f = x_o.clone()
        x_f[:, 2] = c_j
        pred_f = predict_locked(model, obs, x_f, m_o.clone())
        pred_nj = model.predict(obs, sk.clone(), c_j.clone(), num_samples=1, predict_mode="most_likely")
        pred_n = model.predict(obs, sk.clone(), sc_d.clone(), num_samples=1, predict_mode="most_likely")
        return pred_o, pred_f, pred_nj, pred_n, x_f

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
        if not any(in_mask) and not any(k in sbv_keys for k in keys):
            for i in range(b):
                status["not_in_S_BV"] += 1
            continue

        sk_d = sk.clone()
        sc_d = sc.clone()
        for i, j in enumerate(donor_idx):
            if in_mask[i] and j >= 0:
                sk_d[i] = sk[j]
                sc_d[i] = sc[j]

        # 独立 clean 副本；c_J 只算一次
        sk_o, sc_o = sk.clone(), sc.clone()
        c_j, jt = assign_conf_F(sc_o, sc_d)
        c_j_f = c_j.clone()
        c_j_nj = c_j.clone()
        bump("cJ_F_vs_NJ", tmax(c_j_f, c_j_nj))
        bump("cJ_outside_Jt_vs_clean", tmax(torch.where(jt, sc_o, c_j), sc_o))

        x_o, m_o = model.skeleton_preprocessor(sk_o.clone(), sc_o.clone())
        x_nj, m_nj = model.skeleton_preprocessor(sk.clone(), c_j_nj)
        sk_n, sc_n = build_conf_swap_full(sk, sc, sk_d, sc_d)
        x_n, m_n = model.skeleton_preprocessor(sk_n, sc_n)

        x_f = x_o.clone()
        x_f[:, 2] = c_j_f
        bump("F_xy_vs_O", tmax(x_f[:, :2], x_o[:, :2]))
        bump("F_mask_vs_O", tmax(m_o, m_o.clone()))  # m_F := m_O
        bump("NJ_pre_xy_vs_O", tmax(sk.clone(), sk_o))
        bump("hip_NJ_vs_O", tmax(hip_center(sk), hip_center(sk_o)))

        jv_o = sc_o > 0.1
        jv_nj = c_j > 0.1
        n_jv_o = jv_o.reshape(b, -1).sum(dim=-1)
        n_jv_nj = jv_nj.reshape(b, -1).sum(dim=-1)
        n_fr_o = (m_o > 0.5).sum(dim=-1)
        n_fr_nj = (m_nj > 0.5).sum(dim=-1)
        ph_o, san_o = placeholder_t0(m_o)
        ph_nj, san_nj = placeholder_t0(m_nj)
        ph_f, san_f = placeholder_t0(m_o)

        bump("F_ph_vs_O", float((ph_f.float() - ph_o.float()).abs().max().item()))
        bump("F_san_vs_O", tmax(san_f.float(), san_o.float()))

        need_fwd = False
        pick_i = []
        for i in range(b):
            if keys[i] not in sbv_keys:
                status["not_in_S_BV"] += 1
                continue
            st = donor_st[i]
            status[st] += 1
            if st != "ok":
                continue
            n_ok += 1
            if int((n_jv_nj[i] - n_jv_o[i]).item()) > 0:
                n_mono_joint_up += 1
            if int((n_fr_nj[i] - n_fr_o[i]).item()) > 0:
                n_mono_frame_up += 1

            conf_un = bool((c_j[i] == sc_o[i]).all().item())
            val_un = bool((jv_nj[i] == jv_o[i]).all().item())
            jdown = bool((n_jv_nj[i] < n_jv_o[i]).item())
            mchg = bool((m_nj[i] != m_o[i]).any().item())
            new_ph = bool((~ph_o[i] & ph_nj[i]).item())
            xy_chg = bool(
                (not mchg)
                and float((x_nj[i, :2] - x_o[i, :2]).abs().max().item()) > 0.0
            )
            flags = {
                "conf_unchanged": conf_un,
                "valid_unchanged": val_un,
                "n_valid_joints_down": jdown,
                "frame_mask_changed": mchg,
                "new_placeholder": new_ph,
                "xy_changed_mask_same": xy_chg,
            }
            for t, on in flags.items():
                if on:
                    tag_counts[t] += 1
                    if len(selected[t]) < args.max_per_tag:
                        selected[t].append({
                            "key": keys[i],
                            "donor_key": keys[donor_idx[i]],
                            "batch_idx": batch_idx,
                            "i_in_batch": i,
                        })
                        need_fwd = True
                        pick_i.append(i)

        if not need_fwd:
            continue

        pred_o, pred_f, pred_nj, pred_n, x_f2 = four_preds(obs, sk, sc, sc_d, c_j, x_o, m_o)
        pred_o2 = model.predict(obs, sk.clone(), sc.clone(), num_samples=1, predict_mode="most_likely")
        if det_o is None:
            det_o = tmax(pred_o, pred_o2)
        else:
            det_o = max(det_o, tmax(pred_o, pred_o2))
        pred_loc = predict_locked(model, obs, x_o, m_o)
        replay_clean = max(replay_clean, tmax(pred_o, pred_loc))

        # 条件顺序反向：N → N_J → F → O
        pred_n_b = model.predict(obs, sk.clone(), sc_d.clone(), num_samples=1, predict_mode="most_likely")
        pred_nj_b = model.predict(obs, sk.clone(), c_j.clone(), num_samples=1, predict_mode="most_likely")
        x_fb = x_o.clone()
        x_fb[:, 2] = c_j
        pred_f_b = predict_locked(model, obs, x_fb, m_o.clone())
        pred_o_b = model.predict(obs, sk.clone(), sc.clone(), num_samples=1, predict_mode="most_likely")
        om = max(tmax(pred_o, pred_o_b), tmax(pred_f, pred_f_b), tmax(pred_nj, pred_nj_b), tmax(pred_n, pred_n_b))
        order_max = om if order_max is None else max(order_max, om)

        same_in = (tmax(x_f2, x_nj) == 0.0) and (tmax(m_o, m_nj) == 0.0)
        if same_in:
            n_equal_input += int(sum(1 for i in pick_i if in_mask[i]))
            equal_input_pred_max = max(equal_input_pred_max, tmax(pred_f, pred_nj))

        if not torch.isfinite(pred_o).all() or not torch.isfinite(pred_f).all() \
                or not torch.isfinite(pred_nj).all() or not torch.isfinite(pred_n).all():
            n_nonfinite += 1

        ade_o, _ = ade_fde_vs_gt(pred_o, gt)
        ade_f, _ = ade_fde_vs_gt(pred_f, gt)
        ade_nj, _ = ade_fde_vs_gt(pred_nj, gt)
        ade_n, _ = ade_fde_vs_gt(pred_n, gt)
        for i in sorted(set(pick_i)):
            sample_rows.append({
                "key": keys[i],
                "donor_key": keys[donor_idx[i]],
                "batch_idx": batch_idx,
                "ade_O": float(ade_o[i].item()),
                "ade_F": float(ade_f[i].item()),
                "ade_NJ": float(ade_nj[i].item()),
                "ade_N": float(ade_n[i].item()),
                "delta_policy": float((ade_nj[i] - ade_f[i]).item()),
                "note": "sample_only_not_for_case_finding",
            })

    uncovered = [t for t in TAGS if not selected[t]]
    assign_ok = mx["cJ_F_vs_NJ"] == 0.0 and mx["cJ_outside_Jt_vs_clean"] == 0.0
    pre_xy_ok = mx["NJ_pre_xy_vs_O"] == 0.0
    hip_ok = mx["hip_NJ_vs_O"] == 0.0
    f_lock_ok = mx["F_xy_vs_O"] == 0.0 and mx["F_ph_vs_O"] == 0.0
    mono_ok = n_mono_joint_up == 0 and n_mono_frame_up == 0
    n_ok_ok = n_ok == N_SBV
    det_ok = det_o == 0.0
    replay_ok = replay_clean == 0.0
    order_ok = order_max == 0.0
    equal_pred_ok = (n_equal_input == 0) or (equal_input_pred_max == 0.0)
    finite_ok = n_nonfinite == 0

    overall = (
        assign_ok and pre_xy_ok and hip_ok and f_lock_ok and mono_ok
        and n_ok_ok and det_ok and replay_ok and order_ok and equal_pred_ok
        and finite_ok
    )

    checks = {
        "max_abs": dict(mx),
        "n_ok": n_ok,
        "n_S_BV": N_SBV,
        "n_mono_joint_increase": n_mono_joint_up,
        "n_mono_frame_increase": n_mono_frame_up,
        "n_nonfinite_pred_batches": n_nonfinite,
        "det_O_repeat_max_abs": det_o,
        "replay_locked_clean_vs_predict_O": replay_clean,
        "order_swap_pred_max_abs": order_max,
        "n_selected_windows_equal_F_NJ_input": n_equal_input,
        "equal_input_pred_max_abs": equal_input_pred_max,
        "assign_ok": assign_ok,
        "NJ_pre_xy_eq_O": pre_xy_ok,
        "hip_NJ_eq_O": hip_ok,
        "F_lock_ok": f_lock_ok,
        "monotonicity_ok": mono_ok,
        "det_ok": det_ok,
        "replay_ok": replay_ok,
        "order_ok": order_ok,
        "equal_input_pred_ok": equal_pred_ok,
        "finite_ok": finite_ok,
        "n_ok_ok": n_ok_ok,
        "donor_status": dict(status),
        "tag_window_counts_all_S_BV": dict(tag_counts),
        "uncovered_tags": uncovered,
        "selection_rule": "first_max_per_tag_by_input_state_not_ADE",
        "written_at": now_local(),
    }
    write_json(out / "STAGEA_CHECKS.json", checks)
    write_json(out / "stageA_keys.json", {
        "selected": selected,
        "uncovered": uncovered,
        "tag_counts": dict(tag_counts),
        "sample_ade_not_used_for_selection": sample_rows,
    })
    write_json(out / "STAGEA_GATE.json", {
        "pass": overall,
        "uncovered_ok": True,
        "note_uncovered": "missing tags recorded, not a fail",
        "hard_fail_reasons": [k for k, v in {
            "assign_ok": assign_ok,
            "NJ_pre_xy_eq_O": pre_xy_ok,
            "hip_NJ_eq_O": hip_ok,
            "F_lock_ok": f_lock_ok,
            "monotonicity_ok": mono_ok,
            "n_ok_ok": n_ok_ok,
            "det_ok": det_ok,
            "replay_ok": replay_ok,
            "order_ok": order_ok,
            "equal_input_pred_ok": equal_pred_ok,
            "finite_ok": finite_ok,
        }.items() if not v],
        "written_at": now_local(),
    })
    if overall:
        write_json(done, {"done": True, "written_at": now_local()})
    print(
        "STAGEA", "PASS" if overall else "FAIL",
        "n_ok", n_ok,
        "uncovered", uncovered,
        "mono_up", n_mono_joint_up, n_mono_frame_up,
        "replay", replay_clean,
        "det", det_o,
        "order", order_max,
        "->", out,
    )
    if not overall:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
