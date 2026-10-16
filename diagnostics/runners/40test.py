from __future__ import annotations
import json, math, csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import os
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("DIAGNOSTIC_INPUT_ROOT", str(PACKAGE_ROOT / "statistics/u1f_lock_Jt_v1"))).resolve()
OUTPUT_ROOT = Path(os.environ.get("DIAGNOSTIC_OUTPUT_ROOT", str(PACKAGE_ROOT / "local_runs/u1f_lock_Jt_v1"))).resolve()
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
OUT = OUTPUT_ROOT / "READONLY_CLOSEOUT_20261005.json"
SEEDS = list(range(42, 50))
DONORS = list(range(42, 47))
B_BOOT, BOOT_SEED = 1000, 20261005
EPS = 1e-12
def mean_sd(xs):
  a = np.asarray(xs, dtype=float)
  if a.size == 0:
      return {"mean": None, "sample_sd": None, "n": 0, "min": None, "max": None, "values": []}
  if a.size == 1:
      sd = 0.0
  else:
      sd = float(a.std(ddof=1))
  return {
      "mean": float(a.mean()),
      "sample_sd": sd,
      "n": int(a.size),
      "min": float(a.min()),
      "max": float(a.max()),
      "values": [float(x) for x in a],
  }
def hierarchy_cell_to_seed(cell_vals):
  """cell_vals[(s,d)] -> per-seed mean over donors -> seed mean±sd"""
  per_seed = {}
  for s in SEEDS:
      ds = [cell_vals[(s, d)] for d in DONORS if (s, d) in cell_vals and cell_vals[(s, d)] is not None]
      per_seed[str(s)] = float(np.mean(ds)) if ds else None
  xs = [per_seed[str(s)] for s in SEEDS if per_seed[str(s)] is not None]
  out = mean_sd(xs)
  out["per_seed"] = per_seed
  return out
# ---------- identity / gates ----------
manifest = json.loads((ROOT / "RUN_MANIFEST.json").read_text())
status = (ROOT / "RUN_STATUS.txt").read_text().strip()
agg = json.loads((ROOT / "AGGREGATE.json").read_text())
gates, manifests, summaries = {}, {}, {}
for s in SEEDS:
  for d in DONORS:
      cell = ROOT / "by_ckpt" / f"s{s}" / f"d{d}"
      g = json.loads((cell / "U1F_GATE.json").read_text())
      m = json.loads((cell / "CELL_MANIFEST.json").read_text())
      sm = json.loads((cell / "summary_local.json").read_text())
      gates[(s, d)] = g
      manifests[(s, d)] = m
      summaries[(s, d)] = sm
ckpt_by_seed = {}
for s in SEEDS:
  m0 = manifests[(s, DONORS[0])]
  ckpt_by_seed[str(s)] = {
      "checkpoint": m0["checkpoint"],
      "checkpoint_sha256": m0["checkpoint_sha256"],
      "diagnostic_set_sha256": m0["diagnostic_set_sha256"],
      "donor_map_sha256": m0["donor_map_sha256"],
      "n_S_BV": m0["n_S_BV"],
      "decode": m0.get("decode"),
      "K": m0.get("K"),
  }
gate_rows = []
hard_keys = ["replay_max_abs", "det_O_max_abs", "det_F_max_abs", "lock_xy_max_abs"]
for s in SEEDS:
  for d in DONORS:
      g = gates[(s, d)]
      gate_rows.append({
          "train_seed": s, "donor": d,
          "pass": g.get("pass"), "hard_pass": g.get("hard_pass"),
          **{k: g.get(k) for k in hard_keys},
          "original_ade_max_abs": g.get("original_ade_max_abs"),
          "e2_original_align_pass": g.get("e2_original_align_pass"),
          "n_windows_e2_ade_gt_atol": g.get("n_windows_e2_ade_gt_atol"),
          "n_ok": g.get("n_ok"),
          "n_conf_changed": g.get("n_conf_changed"),
          "n_conf_unchanged_kept": g.get("n_conf_unchanged_kept"),
          "n_would_recompute_frame": g.get("n_would_recompute_frame"),
          "n_u1_missing": g.get("n_u1_missing"),
          "n_e2_missing_or_donor_mismatch": g.get("n_e2_missing_or_donor_mismatch"),
      })
n_pass = sum(1 for r in gate_rows if r["pass"] is True)
n_hard = sum(1 for r in gate_rows if r["hard_pass"] is True)
align_false = [f"s{r['train_seed']}d{r['donor']}" for r in gate_rows if r["e2_original_align_pass"] is False]
hard_ok = all((r[k] == 0.0 or r[k] == 0) for r in gate_rows for k in hard_keys)
# coverage: donor-mean per seed then seed stats on rate
cov_cell, unch_cell, would_cell = {}, {}, {}
for s in SEEDS:
  for d in DONORS:
      g = gates[(s, d)]
      n_ok = float(g["n_ok"])
      cov_cell[(s, d)] = g["n_conf_changed"] / n_ok
      unch_cell[(s, d)] = g["n_conf_unchanged_kept"] / n_ok
      would_cell[(s, d)] = g["n_would_recompute_frame"] / n_ok
coverage = {
  "conf_changed_rate": hierarchy_cell_to_seed(cov_cell),
  "conf_unchanged_kept_rate": hierarchy_cell_to_seed(unch_cell),
  "would_recompute_frame_rate_if_unlocked": hierarchy_cell_to_seed(would_cell),
  "note": "主口径保留 conf 未变窗；would_recompute 为反事实（若按 donor conf 重算帧 mask）",
}
# ---------- helpers on CSV ----------
def load_cell_csv(s, d):
  path = ROOT / "by_ckpt" / f"s{s}" / f"d{d}" / "per_window_F.csv"
  rows = []
  with path.open(newline="") as f:
      for r in csv.DictReader(f):
          if int(float(r["include_main"])) != 1:
              continue
          if r.get("donor_status") not in (None, "", "ok") and r["donor_status"] != "ok":
              # still keep if include_main=1; status recorded
              pass
          rows.append(r)
  return rows
def fget(r, k):
  v = r.get(k, "")
  if v is None or v == "":
      return np.nan
  return float(v)
# accumulate cell-level means for absolute O/F and paired N-F
metrics_abs = ["ade_O", "fde_O", "ade_F", "fde_F", "pred_ade", "pred_fde", "delta_ade", "delta_fde",
             "u1_pred_ade", "u1_pred_fde", "u1_delta_ade", "e2_full_swap_pred_ade"]
cell_means = {k: {} for k in metrics_abs}
cell_means.update({
  "nf_pred_ade": {}, "nf_pred_fde": {}, "nf_delta_ade": {},
  "frac_delta_pos": {}, "frac_delta_neg": {}, "frac_delta_zero": {},
  "frac_improve_eps": {}, "frac_worsen_eps": {}, "frac_stable_eps": {},
})
# for high-response / scene: pool all include_main rows across 40 cells with weights later via hierarchy
# We'll compute per-cell first, then seed hierarchy for scalar metrics.
# For high-response shares: compute per cell on that cell's windows, then hierarchy.
def high_resp_shares(pred, delta, q=0.90):
  pred = np.asarray(pred, float); delta = np.asarray(delta, float)
  m = np.isfinite(pred) & np.isfinite(delta)
  pred, delta = pred[m], delta[m]
  if pred.size == 0:
      return None
  thr = float(np.quantile(pred, q))
  hi = pred >= thr
  d_hi = delta[hi]
  harm = d_hi > 0; imp = d_hi < 0
  abs_all = np.abs(d_hi)
  abs_sum = float(abs_all.sum()) + EPS
  return {
      "thr_p90": thr,
      "n_hi": int(hi.sum()),
      "n_all": int(pred.size),
      "count_harm_share": float(harm.mean()) if d_hi.size else None,
      "count_improve_share": float(imp.mean()) if d_hi.size else None,
      "mass_harm_share": float(np.abs(d_hi[harm]).sum() / abs_sum) if abs_sum > 0 else None,
      "mass_improve_share": float(np.abs(d_hi[imp]).sum() / abs_sum) if abs_sum > 0 else None,
      "mean_delta_hi": float(d_hi.mean()) if d_hi.size else None,
      # contribution of hi layer to overall signed delta mass
      "contrib_harm_to_all_pos": float(np.clip(d_hi, 0, None).sum() / (np.clip(delta, 0, None).sum() + EPS)),
      "contrib_improve_to_all_neg": float((-np.clip(d_hi, None, 0)).sum() / ((-np.clip(delta, None, 0)).sum() + EPS)),
  }
hi_cell = defaultdict(dict)  # metric_name -> (s,d) -> value
scene_cell_pred = defaultdict(lambda: defaultdict(list))  # scene -> (s,d) -> list pred
scene_cell_delta = defaultdict(lambda: defaultdict(list))
delta_eps = 0.01  # same as summary_local thresholds
for s in SEEDS:
  for d in DONORS:
      rows = load_cell_csv(s, d)
      arr = {k: np.array([fget(r, k) for r in rows], float) for k in
             ["ade_O","fde_O","ade_F","fde_F","pred_ade","pred_fde","delta_ade","delta_fde",
              "u1_pred_ade","u1_pred_fde","u1_delta_ade","e2_full_swap_pred_ade","n_changed_joints"]}
      scenes = [r["scene_id"] for r in rows]
      # absolute / pred means
      for k in metrics_abs:
          cell_means[k][(s, d)] = float(np.nanmean(arr[k]))
      # N-F paired (same window)
      nf_pa = arr["u1_pred_ade"] - arr["pred_ade"]
      nf_pf = arr["u1_pred_fde"] - arr["pred_fde"]
      nf_da = arr["u1_delta_ade"] - arr["delta_ade"]
      cell_means["nf_pred_ade"][(s, d)] = float(np.nanmean(nf_pa))
      cell_means["nf_pred_fde"][(s, d)] = float(np.nanmean(nf_pf))
      cell_means["nf_delta_ade"][(s, d)] = float(np.nanmean(nf_da))
      da = arr["delta_ade"]
      cell_means["frac_delta_pos"][(s, d)] = float(np.mean(da > 0))
      cell_means["frac_delta_neg"][(s, d)] = float(np.mean(da < 0))
      cell_means["frac_delta_zero"][(s, d)] = float(np.mean(da == 0))
      cell_means["frac_worsen_eps"][(s, d)] = float(np.mean(da > delta_eps))
      cell_means["frac_improve_eps"][(s, d)] = float(np.mean(da < -delta_eps))
      cell_means["frac_stable_eps"][(s, d)] = float(np.mean(np.abs(da) <= delta_eps))
      # coverage from joints
      changed = arr["n_changed_joints"] > 0
      # high response on F pred_ade vs F delta; also on N and N-F pred
      for tag, pred, delta in [
          ("F_pred_vs_F_delta", arr["pred_ade"], arr["delta_ade"]),
          ("N_pred_vs_N_delta", arr["u1_pred_ade"], arr["u1_delta_ade"]),
          ("NF_pred_vs_F_delta", np.abs(arr["u1_pred_ade"] - arr["pred_ade"]), arr["delta_ade"]),
      ]:
          hs = high_resp_shares(pred, delta, 0.90)
          if hs:
              for kk, vv in hs.items():
                  if kk in ("thr_p90", "n_hi", "n_all"):
                      continue
                  hi_cell[f"{tag}.{kk}"][(s, d)] = vv
              hi_cell[f"{tag}.thr_p90"][(s, d)] = hs["thr_p90"]
              hi_cell[f"{tag}.n_hi"][(s, d)] = float(hs["n_hi"])
      # scenes
      for i, sc in enumerate(scenes):
          scene_cell_pred[sc][(s, d)].append(arr["pred_ade"][i])
          scene_cell_delta[sc][(s, d)].append(arr["delta_ade"][i])
main_table = {k: hierarchy_cell_to_seed(cell_means[k]) for k in [
  "ade_O","fde_O","ade_F","fde_F","pred_ade","pred_fde","delta_ade","delta_fde",
  "u1_pred_ade","u1_pred_fde","u1_delta_ade","e2_full_swap_pred_ade",
]}
nf_table = {k: hierarchy_cell_to_seed(cell_means[k]) for k in [
  "nf_pred_ade","nf_pred_fde","nf_delta_ade",
]}
# sign summary for nf_pred_ade per seed
nf_pa = nf_table["nf_pred_ade"]
nf_sign = {
  "n_seeds_pos": sum(1 for v in nf_pa["per_seed"].values() if v is not None and v > 0),
  "n_seeds_neg": sum(1 for v in nf_pa["per_seed"].values() if v is not None and v < 0),
  "n_seeds_zero": sum(1 for v in nf_pa["per_seed"].values() if v is not None and v == 0),
  "rule": "paired same-window mean(N-F) per cell → donor mean → seed; SD is sample SD of 8 seed means; NOT sd(N)-sd(F)",
}
sign_table = {k: hierarchy_cell_to_seed(cell_means[k]) for k in [
  "frac_delta_pos","frac_delta_neg","frac_delta_zero",
  "frac_improve_eps","frac_worsen_eps","frac_stable_eps",
]}
hi_table = {k: hierarchy_cell_to_seed(v) for k, v in hi_cell.items()}
# ---------- scene: per-seed scene means (donor-avg of cell scene means), then window-weighted / equal-weight across scenes
# Build for each seed: scene -> mean pred / mean delta (avg over donors of cell-level scene means)
scene_names = sorted(scene_cell_pred.keys())
def scene_seed_maps(metric_dict):
  """metric_dict[scene][(s,d)] = list -> returns per_seed[str(s)][scene] = donor-mean of cell means"""
  out = {str(s): {} for s in SEEDS}
  for sc in scene_names:
      for s in SEEDS:
          donor_means = []
          for d in DONORS:
              xs = metric_dict[sc].get((s, d), [])
              if xs:
                  donor_means.append(float(np.mean(xs)))
          if donor_means:
              out[str(s)][sc] = float(np.mean(donor_means))
  return out
pred_by_seed_scene = scene_seed_maps(scene_cell_pred)
delta_by_seed_scene = scene_seed_maps(scene_cell_delta)
# window counts per scene from roster: use first cell's counts / 40 already in AGGREGATE as n_windows_over_pass_cells / 40
scene_n = {r["scene"]: int(r["n_windows_over_pass_cells"] // 40) for r in agg["scene_pred_ade"]}
def weighted_mean(sc_map, equal=False):
  nums, dens = [], []
  for sc, v in sc_map.items():
      w = 1.0 if equal else float(scene_n.get(sc, 0))
      if w <= 0 and not equal:
          continue
      nums.append(v * w); dens.append(w)
  if not dens or sum(dens) == 0:
      return None
  return float(sum(nums) / sum(dens))
def seed_hierarchy_from_scene_maps(maps, equal=False):
  per_seed = {}
  for s in SEEDS:
      per_seed[str(s)] = weighted_mean(maps[str(s)], equal=equal)
  xs = [per_seed[str(s)] for s in SEEDS if per_seed[str(s)] is not None]
  out = mean_sd(xs); out["per_seed"] = per_seed
  return out
scene_pred_win = seed_hierarchy_from_scene_maps(pred_by_seed_scene, equal=False)
scene_pred_eq = seed_hierarchy_from_scene_maps(pred_by_seed_scene, equal=True)
scene_delta_win = seed_hierarchy_from_scene_maps(delta_by_seed_scene, equal=False)
scene_delta_eq = seed_hierarchy_from_scene_maps(delta_by_seed_scene, equal=True)
# bootstrap CI on scenes using mean-over-seeds scene values (point), cluster by scene
def scene_point_map(maps):
  """average seed means per scene"""
  out = {}
  for sc in scene_names:
      vals = [maps[str(s)][sc] for s in SEEDS if sc in maps[str(s)]]
      out[sc] = float(np.mean(vals)) if vals else None
  return out
pred_sc = scene_point_map(pred_by_seed_scene)
delta_sc = scene_point_map(delta_by_seed_scene)
def bootstrap_scene(sc_map, equal=False, B=B_BOOT, seed=BOOT_SEED):
  scenes = [sc for sc in scene_names if sc_map.get(sc) is not None]
  vals = np.array([sc_map[sc] for sc in scenes], float)
  w = np.array([1.0 if equal else scene_n[sc] for sc in scenes], float)
  rng = np.random.default_rng(seed)
  boots = []
  n = len(scenes)
  for _ in range(B):
      idx = rng.integers(0, n, size=n)
      vv, ww = vals[idx], w[idx]
      boots.append(float(np.sum(vv * ww) / np.sum(ww)))
  boots = np.sort(np.asarray(boots))
  point = float(np.sum(vals * w) / np.sum(w))
  return {
      "point": point,
      "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
      "B": B, "seed": seed, "n_scenes": n, "equal_weight": equal,
  }
scene_boot = {
  "pred_ade_window_weighted": bootstrap_scene(pred_sc, equal=False),
  "pred_ade_scene_equal": bootstrap_scene(pred_sc, equal=True),
  "delta_ade_window_weighted": bootstrap_scene(delta_sc, equal=False),
  "delta_ade_scene_equal": bootstrap_scene(delta_sc, equal=True),
}
# per-scene table (seed-averaged)
scene_rows = []
for sc in scene_names:
  scene_rows.append({
      "scene": sc,
      "n_windows_per_cell": scene_n.get(sc),
      "pred_ade_seedmean": pred_sc[sc],
      "delta_ade_seedmean": delta_sc[sc],
  })
scene_rows.sort(key=lambda r: (-(r["pred_ade_seedmean"] or -1)))
# nf range/sign already in nf_table
closeout = {
  "meta": {
      "created_for": "U1F 40-cell read-only closeout",
      "root": str(ROOT),
      "scheme": manifest.get("scheme"),
      "run_manifest": manifest,
      "run_status_txt": status,
      "aggregate_complete_8x5": agg.get("complete_8x5"),
      "n_cells_pass_aggregate": agg.get("n_cells_pass"),
      "hierarchy": "cell window-mean → arithmetic mean over 5 donors → mean±sample SD over 8 train seeds (ddof=1)",
      "decode": "most_likely K=1",
      "not_K20": True,
      "high_response": "Pred-ADE >= p90 within cell before hierarchy on share metrics",
      "bootstrap": {"B": B_BOOT, "seed": BOOT_SEED, "unit": "scene"},
      "delta_eps": delta_eps,
  },
  "acceptance": {
      "n_gate_pass": n_pass,
      "n_hard_pass": n_hard,
      "n_expected": 40,
      "hard_lock_all_zero": hard_ok,
      "identity_by_train_seed": ckpt_by_seed,
      "e2_original_align_pass_false_cells": align_false,
      "gate_rows": gate_rows,
      "u1_e2_reuse": {
          "n_u1_missing_max": max(r["n_u1_missing"] for r in gate_rows),
          "n_e2_missing_or_donor_mismatch_max": max(r["n_e2_missing_or_donor_mismatch"] for r in gate_rows),
          "note": "N/E2 columns embedded in per_window_F.csv; missing counts from U1F_GATE",
      },
  },
  "coverage": coverage,
  "main_table_OF_abs_and_pred": main_table,
  "nf_paired_same_window": {"metrics": nf_table, "sign_summary_nf_pred_ade": nf_sign},
  "delta_sign_fractions": sign_table,
  "high_response_p90": hi_table,
  "scene": {
      "n_scenes": len(scene_names),
      "pred_ade_window_weighted_seed_hierarchy": scene_pred_win,
      "pred_ade_scene_equal_seed_hierarchy": scene_pred_eq,
      "delta_ade_window_weighted_seed_hierarchy": scene_delta_win,
      "delta_ade_scene_equal_seed_hierarchy": scene_delta_eq,
      "bootstrap_ci95_on_seedmean_scenes": scene_boot,
      "per_scene_seedmean": scene_rows,
  },
  "crosscheck_vs_AGGREGATE": {
      "pred_ade_all_mean": main_table["pred_ade"]["mean"],
      "aggregate_pred_ade_mean": agg["tables"]["all"]["pred_ade"]["mean"],
      "ade_O_mean": main_table["ade_O"]["mean"],
      "aggregate_ade_O_mean": agg["tables"]["all"]["ade_O"]["mean"],
      "delta_ade_mean": main_table["delta_ade"]["mean"],
      "aggregate_delta_ade_mean": agg["tables"]["all"]["delta_ade"]["mean"],
  },
}
OUT.write_text(json.dumps(closeout, ensure_ascii=False, indent=2))
print("WROTE", OUT)
print("gate_pass", n_pass, "hard_pass", n_hard, "hard_lock_all_zero", hard_ok)
print("coverage_changed_rate", coverage["conf_changed_rate"]["mean"], "±", coverage["conf_changed_rate"]["sample_sd"])
print("O_ADE", main_table["ade_O"]["mean"], "F_ADE", main_table["ade_F"]["mean"])
print("O_FDE", main_table["fde_O"]["mean"], "F_FDE", main_table["fde_F"]["mean"])
print("F_pred_ADE", main_table["pred_ade"]["mean"], "N_pred_ADE", main_table["u1_pred_ade"]["mean"])
print("N-F_pred_ADE", nf_table["nf_pred_ade"]["mean"], "±", nf_table["nf_pred_ade"]["sample_sd"],
    "range", nf_table["nf_pred_ade"]["min"], nf_table["nf_pred_ade"]["max"], nf_sign)
print("frac_pos/neg", sign_table["frac_delta_pos"]["mean"], sign_table["frac_delta_neg"]["mean"])
print("scene_pred_win", scene_boot["pred_ade_window_weighted"])
print("scene_delta_win", scene_boot["delta_ade_window_weighted"])
print("crosscheck pred_ade", closeout["crosscheck_vs_AGGREGATE"])
