#!/usr/bin/env bash
# N_J 阶段 C：8×5=40 格。s42d42 / s46d42 若已有 NJ_DONE 则跳过（阶段 B 计入）。
# 不写 u1f_lock_Jt_v1。缺格必须披露，禁止成功子集冒充全量。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${ROOT}/diagnostics${PYTHONPATH:+:${PYTHONPATH}}"
export CTR_GCN_TRAJ_MOTION_CACHE_ROOT="${CTR_GCN_TRAJ_MOTION_CACHE_ROOT:-${ROOT}/local_runs/motion_filter_cache}"
PYTHON="${PYTHON:-python3}"
GPU="${GPU:-0}"
CFG="${CFG:-configs/jrdb_v2_decoupled_z.yaml}"
E123_ROOT="${E123_ROOT:-protocols}"
OUT_ROOT="${OUT_ROOT:-local_runs/nj_same_assign_v1}"
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
echo -e "# N_J failed / skipped cells\n" > "${FAILED}"
BV_JSON="${E123_ROOT}/diagnostic_BV/diagnostic_set_BV.json"
MAP_DIR="${E123_ROOT}/donor_maps"

"${PYTHON}" - "${OUT_ROOT}" "${TRAIN_SEEDS}" "${DONOR_SEEDS}" <<'PY'
import json, sys, time
from pathlib import Path
out = Path(sys.argv[1])
man = {
  "scheme": "NJ_same_assign_v1",
  "stage": "C",
  "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
  "train_seeds": [int(x) for x in sys.argv[2].split(",") if x.strip()],
  "donor_seeds": [int(x) for x in sys.argv[3].split(",") if x.strip()],
  "n_S_BV": 15160,
  "decode": "most_likely K=1",
  "conditions": ["O", "F", "N_J", "N"],
  "not_overwrite_u1f": True,
  "stageB_cells_count_if_NJ_DONE": ["s42d42", "s46d42"],
  "registration_status": "pre_frozen_followup_plan",
  "preregistered": False,
}
(out / "RUN_MANIFEST.json").write_text(json.dumps(man, indent=2) + "\n")
print("[manifest]", out / "RUN_MANIFEST.json")
PY

IFS=',' read -r -a TSEEDS <<< "${TRAIN_SEEDS}"
IFS=',' read -r -a DSEEDS <<< "${DONOR_SEEDS}"

n_fail=0
n_pass=0
n_skip=0

for ts in "${TSEEDS[@]}"; do
  ts="${ts// /}"
  ckpt="checkpoints/${CKPT_DIR[$ts]}/best.pt"
  sha="${CKPT_SHA[$ts]}"
  if [[ ! -f "${ckpt}" ]]; then
    echo "- missing ckpt s${ts} ${ckpt}" | tee -a "${FAILED}"
    n_fail=$((n_fail+5))
    continue
  fi
  for ds in "${DSEEDS[@]}"; do
    ds="${ds// /}"
    cell="${OUT_ROOT}/by_ckpt/s${ts}/d${ds}"
    mkdir -p "${cell}"
    if [[ "${FORCE}" == "1" ]]; then
      rm -f "${cell}/NJ_DONE.json"
    fi
    if [[ -f "${cell}/NJ_DONE.json" ]]; then
      echo "[skip done] s${ts} d${ds}"
      n_skip=$((n_skip+1))
      n_pass=$((n_pass+1))
      continue
    fi
    echo "[nj C] s${ts} d${ds}"
    set +e
    "${PYTHON}" diagnostics/eval_nj_cell.py \
      --config "${CFG}" --checkpoint "${ckpt}" \
      --train_seed "${ts}" --donor_assign_seed "${ds}" \
      --diagnostic_set_json "${BV_JSON}" \
      --donor_map_json "${MAP_DIR}/donor_pairs_B_d${ds}.json" \
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

{
  echo "pass_cells=${n_pass} fail_or_blocked=${n_fail} skipped_done=${n_skip}"
  echo "note: skipped_done counted in pass_cells if NJ_DONE present (stage B)"
} | tee "${OUT_ROOT}/RUN_STATUS.txt"
if [[ ${n_pass} -ne 40 ]]; then
  echo "INCOMPLETE: ${n_pass}/40 — do not treat as full matrix" | tee -a "${FAILED}"
fi
echo "Done -> ${OUT_ROOT}"
