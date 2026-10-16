#!/usr/bin/env bash
# 同会话 N/F 诊断：仅 s42d42（过阈）与 s46d42（未过阈）。
# 不写 eval_logs/u1f_lock_Jt_v1，不改 20 格历史。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INNER="${ROOT}/ctr_gcn_traj/ctr_gcn_traj"
cd "${INNER}"
PYTHON="${PYTHON:-python3}"
GPU="${GPU:-0}"
CFG="${CFG:-configs/jrdb_v2_decoupled_z.yaml}"
E123_ROOT="${E123_ROOT:-eval_logs/e123}"
U1_ROOT="${U1_ROOT:-eval_logs/u1_conf_swap_v2}"
U1F_ROOT="${U1F_ROOT:-eval_logs/u1f_lock_Jt_v1}"
OUT_ROOT="${OUT_ROOT:-eval_logs/nf_same_session_diag}"
CUDNN_MODE="${CUDNN_MODE:-u1f}"
FORCE="${FORCE:-0}"

declare -A CKPT_DIR=(
  [42]="jrdb_v2_decoupled_z_s42_20260619_1432"
  [46]="jrdb_v2_decoupled_z_s46_paired5"
)
declare -A CKPT_SHA=(
  [42]="94cb5326200fa2544340c87d59200b9e15f397c66fe99e39a6727f885c6ce07f"
  [46]="83d3a816765a1b55a8b9b911840d817ae2592aa610bad1735a59992f52cbdde0"
)

BV_JSON="${E123_ROOT}/diagnostic_BV/diagnostic_set_BV.json"
MAP_DIR="${E123_ROOT}/donor_maps"
mkdir -p "${OUT_ROOT}"

run_one() {
  local ts="$1" ds="$2"
  local ckpt="checkpoints/${CKPT_DIR[$ts]}/best.pt"
  local sha="${CKPT_SHA[$ts]}"
  local cell="${OUT_ROOT}/s${ts}d${ds}_${CUDNN_MODE}"
  mkdir -p "${cell}"
  if [[ "${FORCE}" == "1" ]]; then
    rm -f "${cell}/DIAG_DONE.json"
  fi
  echo "[diag] s${ts} d${ds} cudnn=${CUDNN_MODE}"
  "${PYTHON}" eval_nf_same_session_diag.py \
    --config "${CFG}" --checkpoint "${ckpt}" \
    --train_seed "${ts}" --donor_assign_seed "${ds}" \
    --diagnostic_set_json "${BV_JSON}" \
    --donor_map_json "${MAP_DIR}/donor_pairs_B_d${ds}.json" \
    --e2_csv "${E123_ROOT}/by_ckpt/s${ts}/d${ds}/E2/per_window.csv" \
    --u1_csv "${U1_ROOT}/by_ckpt/s${ts}/d${ds}/per_window_conf_swap_full.csv" \
    --u1f_csv "${U1F_ROOT}/by_ckpt/s${ts}/d${ds}/per_window_F.csv" \
    --out_dir "${cell}" --split test --gpu "${GPU}" \
    --expected_ckpt_sha256 "${sha}" \
    --cudnn_mode "${CUDNN_MODE}"
}

run_one 42 42
run_one 46 42
echo "[done] ${OUT_ROOT}"
