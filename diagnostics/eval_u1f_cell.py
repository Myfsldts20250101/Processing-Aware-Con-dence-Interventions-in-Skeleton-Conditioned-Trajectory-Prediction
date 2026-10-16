#!/usr/bin/env python3
"""U1F 40-cell: locked conf-on-J_t F, most_likely K=1, S_BV=15160. Reuse E2/U1; no K20/train."""
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
from models.full_model import build_model
from models.fusion import _sanitize_key_padding_mask

CONF_THR = 0.1
SCHEME = "U1F_lock_Jt_conf_v1"
N_SBV = 15160
ORIGINAL_ADE_ATOL = 1e-4


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
    p.add_argument("--out_dir", required=True)
    p.add_argument("--weight_gate_json", default="")
    p.add_argument("--phase", choices=["weight", "cell"], default="cell")
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


@torch.no_grad()
def predict_locked(model, obs, x, frame_mask):
    traj_feat = model.traj_encoder(obs)
    skel_feat = model._project_skel_feat(model.skeleton_encoder(x))
    fused = model.fusion(traj_feat, skel_feat, frame_mask)
    memory = model.transformer_encoder(fused)
    z, mem = model.cvae.sample_prior(memory, num_samples=1, mode="most_likely")
    pred = model.transformer_decoder(mem, z, None)
    b = obs.shape[0]
    return pred.reshape(b, 1, model.pred_len, 2)


def assign_conf_F(sc_t, sc_d):
    jt = sc_t > CONF_THR
    return torch.where(jt, sc_d, sc_t), jt


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prog = out / "CKPT_PROGRESS.json"
    if args.skip_if_done and prog.exists():
        import json
        cur = json.loads(prog.read_text())
        if cur.get("U1F_done"):
            print("[skip done]", out)
            return

    ckpt_sha = sha256_file(args.checkpoint)
    if args.expected_ckpt_sha256 and ckpt_sha != args.expected_ckpt_sha256:
        write_json(out / "U1F_GATE.json", {"pass": False, "reason": "ckpt_sha", "got": ckpt_sha})
        raise SystemExit(2)

    sbv_meta, sbv_keys = load_json_set(Path(args.diagnostic_set_json))
    if len(sbv_keys) != N_SBV:
        write_json(out / "U1F_GATE.json", {"pass": False, "reason": "n_S_BV", "n": len(sbv_keys)})
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
        write_json(out / "U1F_GATE.json", {"pass": False, "reason": "state_dict", "missing": list(miss), "unexpected": list(unexp)})
        raise SystemExit(2)
    model.to(device).eval()

    write_json(out / "CELL_MANIFEST.json", {
        "scheme": SCHEME,
        "phase": args.phase,
        "train_seed": args.train_seed,
        "donor_assign_seed": args.donor_assign_seed,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": ckpt_sha,
        "n_S_BV": N_SBV,
        "diagnostic_set_sha256": sbv_meta.get("diagnostic_set_sha256"),
        "donor_map_sha256": map_meta.get("donor_map_sha256"),
        "decode": "most_likely",
        "K": 1,
        "not_K20": True,
        "not_train": True,
        "started_at": now_local(),
    })

    loader = build_dataloader(cfg, split=args.split)
    status = Counter()
    rows = []
    replay_max = 0.0
    det_o = None
    det_f = None
    lock_max_xy = 0.0
    lock_mask_fail = 0
    n_ok = 0
    n_conf_changed = 0
    n_would_frame = 0
    orig_abs = []

    e2_by = {}
    with open(args.e2_csv, newline="", encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            e2_by[rec["key"]] = rec
    u1_by = {}
    u1_path = Path(args.u1_csv)
    if u1_path.is_file():
        with open(u1_path, newline="", encoding="utf-8") as f:
            for rec in csv.DictReader(f):
                u1_by[rec["key"]] = rec
    elif args.phase == "cell":
        write_json(out / "U1F_GATE.json", {"pass": False, "reason": "u1_csv_missing", "path": str(u1_path)})
        raise SystemExit(2)

    n_scan = 0
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
        pred_off = model.predict(obs, sk, sc, num_samples=1, predict_mode="most_likely")
        pred_loc = predict_locked(model, obs, x_c, m_c)
        replay_max = max(replay_max, float((pred_off - pred_loc).abs().max().item()))
        if det_o is None:
            pred_loc2 = predict_locked(model, obs, x_c, m_c)
            det_o = float((pred_loc - pred_loc2).abs().max().item())

        sk_d = sk.clone()
        sc_d = sc.clone()
        for i, j in enumerate(donor_idx):
            if in_mask[i] and j >= 0:
                sk_d[i] = sk[j]
                sc_d[i] = sc[j]
        sc_F, jt = assign_conf_F(sc, sc_d)
        x_F = x_c.clone()
        x_F[:, 2] = sc_F
        lock_max_xy = max(lock_max_xy, float((x_F[:, :2] - x_c[:, :2]).abs().max().item()))
        would = (sc_F > CONF_THR).float().sum(dim=-1) >= 3
        in_t = torch.tensor(in_mask, device=device)
        n_would_frame += int(((would != (m_c > 0.5)).any(dim=-1) & in_t).sum().item())
        if float((x_F[:, 2] - sc_F).abs().max().item()) != 0.0:
            lock_mask_fail += 1

        pred_f = None
        if args.phase == "cell":
            pred_f = predict_locked(model, obs, x_F, m_c)
            if det_f is None:
                pred_f2 = predict_locked(model, obs, x_F, m_c)
                det_f = float((pred_f - pred_f2).abs().max().item())
            san = _sanitize_key_padding_mask((m_c == 0).clone())
            would_san = _sanitize_key_padding_mask((would == 0).clone())
            # fusion uses m_c; fail if we had passed would-mask equal check incorrectly
            if float((san.float() - would_san.float()).abs().max().item()) > 0:
                pass  # expected often; not a fail
            ade_o, fde_o = ade_fde_vs_gt(pred_off, gt)
            ade_f, fde_f = ade_fde_vs_gt(pred_f, gt)
            pade, pfde = pred_pair_ade_fde(pred_off, pred_f)

        ade_c, fde_c = ade_fde_vs_gt(pred_off, gt)
        chg = (sc_F != sc) & jt

        for i in range(b):
            if keys[i] not in sbv_keys:
                status["not_in_S_BV"] += 1
                continue
            st = donor_st[i]
            status[st] += 1
            include = int(st == "ok")
            if include:
                n_ok += 1
                e2 = e2_by.get(keys[i])
                if e2 is not None:
                    try:
                        orig_abs.append(abs(float(ade_c[i].item()) - float(e2["original_ade"])))
                    except (TypeError, ValueError, KeyError):
                        orig_abs.append(1e9)
                n_chg_i = int(chg[i].sum().item())
                if n_chg_i > 0:
                    n_conf_changed += 1
            if args.phase == "weight":
                continue
            j = donor_idx[i]
            e2 = e2_by.get(keys[i], {})
            u1 = u1_by.get(keys[i], {})
            row = {
                "train_seed": args.train_seed,
                "donor_assign_seed": args.donor_assign_seed,
                "key": keys[i],
                "donor_key": keys[j] if include else "",
                "donor_status": st,
                "include_main": include,
                "n_changed_joints": int(chg[i].sum().item()) if include else "",
                "valid_numeric_intervention": int(chg[i].any().item()) if include else "",
                "ade_O": float(ade_o[i].item()) if include else "",
                "fde_O": float(fde_o[i].item()) if include else "",
                "ade_F": float(ade_f[i].item()) if include else "",
                "fde_F": float(fde_f[i].item()) if include else "",
                "delta_ade": float((ade_f[i] - ade_o[i]).item()) if include else "",
                "delta_fde": float((fde_f[i] - fde_o[i]).item()) if include else "",
                "pred_ade": float(pade[i].item()) if include else "",
                "pred_fde": float(pfde[i].item()) if include else "",
                "conf_rmse_F_vs_t": float(conf_rmse(sc_F[i : i + 1], sc[i : i + 1]).item()) if include else "",
                "e2_original_ade": e2.get("original_ade", "") if include else "",
                "e2_full_swap_ade": e2.get("full_swap_ade", "") if include else "",
                "e2_full_swap_pred_ade": e2.get("full_swap_pred_ade", "") if include else "",
                "e2_full_swap_delta_ade": e2.get("full_swap_delta_ade", "") if include else "",
                "e2_full_swap_fde": e2.get("full_swap_fde", "") if include else "",
                "e2_full_swap_pred_fde": e2.get("full_swap_pred_fde", "") if include else "",
                "e2_full_swap_delta_fde": e2.get("full_swap_delta_fde", "") if include else "",
                "u1_pred_ade": u1.get("pred_ade", "") if include else "",
                "u1_ade_intervene": u1.get("ade_intervene", "") if include else "",
                "u1_delta_ade": u1.get("delta_ade", "") if include else "",
                "u1_fde_intervene": u1.get("fde_intervene", "") if include else "",
                "u1_pred_fde": u1.get("pred_fde", "") if include else "",
                "scene_id": keys[i].split("|")[0],
            }
            if include and e2 and str(e2.get("donor_key", "")) != keys[j]:
                row["donor_key_vs_e2"] = "mismatch"
            rows.append(row)
        n_scan += b

    orig_max = max(orig_abs) if orig_abs else None
    n_e2_gt_atol = int(sum(1 for x in orig_abs if x > ORIGINAL_ADE_ATOL))
    replay_ok = replay_max == 0.0
    det_ok = (det_o == 0.0) and (args.phase == "weight" or det_f == 0.0)
    lock_ok = lock_max_xy == 0.0 and lock_mask_fail == 0
    n_ok_ok = n_ok == N_SBV
    orig_ok = orig_max is not None and orig_max <= ORIGINAL_ADE_ATOL
    hard_ok = replay_ok and det_ok and lock_ok and n_ok_ok
    u1_join = True
    n_u1_mis = 0
    n_e2_mis = 0
    if args.phase == "cell":
        okrows = [r for r in rows if r["include_main"] == 1]
        for r in okrows:
            if r["e2_original_ade"] == "" or r.get("donor_key_vs_e2") == "mismatch":
                n_e2_mis += 1
            if r["u1_pred_ade"] == "":
                n_u1_mis += 1
        u1_join = n_u1_mis == 0 and n_e2_mis == 0

    if args.phase == "weight":
        overall = hard_ok
        gate = {
            "pass": overall,
            "phase": "weight",
            "hard_pass": hard_ok,
            "replay_max_abs": replay_max,
            "det_O_max_abs": det_o,
            "lock_xy_max_abs": lock_max_xy,
            "original_ade_max_abs": orig_max,
            "original_ade_atol": ORIGINAL_ADE_ATOL,
            "e2_original_align_pass": orig_ok,
            "n_windows_e2_ade_gt_atol": n_e2_gt_atol,
            "n_ok": n_ok,
            "n_S_BV": N_SBV,
            "note_e2": "this-run official O is the F baseline after replay==0; E2 original_ade join is reported, not a reason to skip F if only ~1e-4 kernel drift",
            "written_at": now_local(),
        }
        write_json(out / "WEIGHT_GATE.json", gate)
        write_json(prog, {"U1F_done": overall, "phase": "weight", "gate_pass": overall})
        print("WEIGHT_GATE", overall, "replay", replay_max, "orig", orig_max, "n_ok", n_ok)
        if not overall:
            raise SystemExit(2)
        return

    overall = hard_ok and u1_join
    gate = {
        "pass": overall,
        "phase": "cell",
        "hard_pass": hard_ok,
        "replay_max_abs": replay_max,
        "det_O_max_abs": det_o,
        "det_F_max_abs": det_f,
        "lock_xy_max_abs": lock_max_xy,
        "original_ade_max_abs": orig_max,
        "e2_original_align_pass": orig_ok,
        "n_windows_e2_ade_gt_atol": n_e2_gt_atol,
        "n_ok": n_ok,
        "n_conf_changed": n_conf_changed,
        "n_conf_unchanged_kept": n_ok - n_conf_changed,
        "n_would_recompute_frame": n_would_frame,
        "n_u1_missing": n_u1_mis,
        "n_e2_missing_or_donor_mismatch": n_e2_mis,
        "donor_status": dict(status),
        "written_at": now_local(),
    }
    write_json(out / "U1F_GATE.json", gate)
    if not overall:
        write_json(out / "summary_local.json", {"gate_pass": False, "gate": gate})
        write_json(prog, {"U1F_done": False, "gate_pass": False})
        print("GATE FAIL", gate)
        raise SystemExit(2)

    write_csv(out / "per_window_F.csv", rows)
    ok = [r for r in rows if r["include_main"] == 1]
    chg = [r for r in ok if int(r["valid_numeric_intervention"]) == 1]
    keys_num = ["pred_ade", "pred_fde", "ade_O", "ade_F", "delta_ade", "delta_fde", "fde_O", "fde_F"]
    summary = {
        "gate_pass": True,
        "n_ok": len(ok),
        "n_changed": len(chg),
        "all": summarize_numeric(ok, keys_num, 0.01, 0.05),
        "changed_only": summarize_numeric(chg, keys_num, 0.01, 0.05) if chg else {"n": 0},
    }
    write_json(out / "summary_local.json", summary)
    write_json(prog, {"U1F_done": True, "gate_pass": True, "n_ok": len(ok)})
    print("GATE PASS n_ok", len(ok), "changed", len(chg), "->", out)


if __name__ == "__main__":
    main()
