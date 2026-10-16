#!/usr/bin/env bash
# U1F 8×5. Per-weight gate then cells. Failed cell recorded; do not treat incomplete 8-seed as full.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${ROOT}/diagnostics${PYTHONPATH:+:${PYTHONPATH}}"
export CTR_GCN_TRAJ_MOTION_CACHE_ROOT="${CTR_GCN_TRAJ_MOTION_CACHE_ROOT:-${ROOT}/local_runs/motion_filter_cache}"
PYTHON="${PYTHON:-python3}"
GPU="${GPU:-0}"
CFG="${CFG:-configs/jrdb_v2_decoupled_z.yaml}"
E123_ROOT="${E123_ROOT:-protocols}"
U1_ROOT="${U1_ROOT:-statistics/u1_conf_swap_v2}"
OUT_ROOT="${OUT_ROOT:-local_runs/u1f_lock_Jt_v1}"
TRAIN_SEEDS="${TRAIN_SEEDS:-42,43,44,45,46,47,48,49}"
DONOR_SEEDS="${DONOR_SEEDS:-42,43,44,45,46}"
FORCE="${FORCE:-0}"

declare -A CKPT_DIR=(
  [42]="jrdb_v2_decoupled_z_s42_20260619_1432"
  [43]="jrdb_v2_decoupled_z_s43_paired3"
  [44]="jrdb_v2_decoupled_z_s44_paired3"
  [45]="jrdb_v2_decoupled_z_s45_paired5"
  [46]="jrdb_v2_decoupled_z_s46_paired5"
  [47]="jrdb_v2_decoupled_z_s47_paired8"
  [48]="jrdb_v2_decoupled_z_s48_paired8"
  [49]="jrdb_v2_decoupled_z_s49_paired8"
)
declare -A CKPT_SHA=(
  [42]="94cb5326200fa2544340c87d59200b9e15f397c66fe99e39a6727f885c6ce07f"
  [43]="7767867582a9dff71ceb787da149023c419e4faec4d984b542e777a79c6339fe"
  [44]="15d12a1fa535de7b713a726dc342ac01a449b73eab09b7033196c8cf77393374"
  [45]="eb2f367fe9aa6e197180562978ebc7053fc2c8f0bfed3ddad4feece5e8142d27"
  [46]="83d3a816765a1b55a8b9b911840d817ae2592aa610bad1735a59992f52cbdde0"
  [47]="5257aad963453a7f971137a9504b95d970b716beb3cb0e8df99e5f6771966d60"
  [48]="bf59925031a14672858518fe0268eea37408b59f38eff4a47e73e3755673eb38"
  [49]="39525c1aa7f612bb96787bad7bc727fd6e8b8ed48eb7018e4cfdde2547b59465"
)

mkdir -p "${OUT_ROOT}"
FAILED="${OUT_ROOT}/FAILED_CELLS.md"
echo -e "# U1F failed / skipped cells\n" > "${FAILED}"
BV_JSON="${E123_ROOT}/diagnostic_BV/diagnostic_set_BV.json"
MAP_DIR="${E123_ROOT}/donor_maps"

# Historical U1F replay needs E2/U1 reference CSVs absent from the core package.
for reference in "${E123_ROOT}/by_ckpt/s42/d42/E2/per_window.csv" "${U1_ROOT}/by_ckpt/s42/d42/per_window_conf_swap_full.csv"; do
  [[ -f "${reference}" ]] || { echo "Missing historical reference: ${reference}. Use run_nj_40.sh for the self-contained four-condition evaluation." >&2; exit 2; }
done

IFS=',' read -r -a TSEEDS <<< "${TRAIN_SEEDS}"
IFS=',' read -r -a DSEEDS <<< "${DONOR_SEEDS}"

n_fail=0
n_pass=0
blocked_seeds=""

"${PYTHON}" - "${OUT_ROOT}" "${TRAIN_SEEDS}" "${DONOR_SEEDS}" <<'PY'
import json, sys, time
from pathlib import Path
out = Path(sys.argv[1])
man = {
  "scheme": "U1F_lock_Jt_conf_v1",
  "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
  "train_seeds": [int(x) for x in sys.argv[2].split(",") if x.strip()],
  "donor_seeds": [int(x) for x in sys.argv[3].split(",") if x.strip()],
  "n_S_BV": 15160,
  "decode": "most_likely K=1",
  "reuse": ["E2 original/full_swap", "U1 conf_swap_full"],
  "not_K20": True,
  "not_train": True,
  "registration_status": "pre_frozen_followup_plan",
  "preregistered": False,
}
(out / "RUN_MANIFEST.json").write_text(json.dumps(man, indent=2) + "\n")
print("[manifest]", out / "RUN_MANIFEST.json")
PY

for ts in "${TSEEDS[@]}"; do
  ts="${ts// /}"
  ckpt="checkpoints/${CKPT_DIR[$ts]}/best.pt"
  sha="${CKPT_SHA[$ts]}"
  if [[ ! -f "${ckpt}" ]]; then
    echo "- missing ckpt s${ts} ${ckpt}" | tee -a "${FAILED}"
    blocked_seeds="${blocked_seeds} ${ts}"
    n_fail=$((n_fail+1))
    continue
  fi
  wdir="${OUT_ROOT}/weight_gate/s${ts}"
  mkdir -p "${wdir}"
  e2w="${E123_ROOT}/by_ckpt/s${ts}/d42/E2/per_window.csv"
  u1w="${U1_ROOT}/by_ckpt/s${ts}/d42/per_window_conf_swap_full.csv"
  mapw="${MAP_DIR}/donor_pairs_B_d42.json"
  echo "[weight] s${ts}"
  set +e
  "${PYTHON}" diagnostics/eval_u1f_cell.py \
    --phase weight --config "${CFG}" --checkpoint "${ckpt}" \
    --train_seed "${ts}" --donor_assign_seed 42 \
    --diagnostic_set_json "${BV_JSON}" --donor_map_json "${mapw}" \
    --e2_csv "${e2w}" --u1_csv "${u1w}" --out_dir "${wdir}" \
    --split test --gpu "${GPU}" --expected_ckpt_sha256 "${sha}"
  wrc=$?
  set -e
  if [[ ${wrc} -ne 0 ]]; then
    echo "- WEIGHT FAIL s${ts} (all 5 donor cells blocked)" | tee -a "${FAILED}"
    blocked_seeds="${blocked_seeds} ${ts}"
    n_fail=$((n_fail+5))
    continue
  fi
  for ds in "${DSEEDS[@]}"; do
    ds="${ds// /}"
    cell="${OUT_ROOT}/by_ckpt/s${ts}/d${ds}"
    mkdir -p "${cell}"
    echo "[cell] s${ts} d${ds}"
    set +e
    "${PYTHON}" diagnostics/eval_u1f_cell.py \
      --phase cell --config "${CFG}" --checkpoint "${ckpt}" \
      --train_seed "${ts}" --donor_assign_seed "${ds}" \
      --diagnostic_set_json "${BV_JSON}" \
      --donor_map_json "${MAP_DIR}/donor_pairs_B_d${ds}.json" \
      --e2_csv "${E123_ROOT}/by_ckpt/s${ts}/d${ds}/E2/per_window.csv" \
      --u1_csv "${U1_ROOT}/by_ckpt/s${ts}/d${ds}/per_window_conf_swap_full.csv" \
      --out_dir "${cell}" --split test --gpu "${GPU}" \
      --expected_ckpt_sha256 "${sha}"
    rc=$?
    set -e
    if [[ ${rc} -ne 0 ]]; then
      echo "- CELL FAIL s${ts} d${ds} exit=${rc}" | tee -a "${FAILED}"
      n_fail=$((n_fail+1))
    else
      n_pass=$((n_pass+1))
    fi
  done
done

echo "pass_cells=${n_pass} fail_or_blocked=${n_fail}" | tee "${OUT_ROOT}/RUN_STATUS.txt"
echo "blocked_seeds:${blocked_seeds}" >> "${OUT_ROOT}/RUN_STATUS.txt"
if [[ ${n_pass} -eq 40 ]]; then
  "${PYTHON}" diagnostics/analyze_u1f_40.py --root "${OUT_ROOT}" || true
else
  echo "INCOMPLETE: not aggregating as full 8-seed (pass ${n_pass}/40)" | tee -a "${FAILED}"
  "${PYTHON}" diagnostics/analyze_u1f_40.py --root "${OUT_ROOT}" --allow_incomplete 1 || true
fi
echo "Done -> ${OUT_ROOT}"
