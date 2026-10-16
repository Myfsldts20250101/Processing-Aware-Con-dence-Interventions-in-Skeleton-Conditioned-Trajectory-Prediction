#!/usr/bin/env bash
# N_J 阶段 A：仅 s42 d42。不写 u1f_lock_Jt_v1。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INNER="${ROOT}/ctr_gcn_traj/ctr_gcn_traj"
cd "${INNER}"
PYTHON="${PYTHON:-python3}"
GPU="${GPU:-0}"
CFG="${CFG:-configs/jrdb_v2_decoupled_z.yaml}"
E123_ROOT="${E123_ROOT:-eval_logs/e123}"
OUT="${OUT:-eval_logs/nj_same_assign_v1/stageA_s42d42}"
FORCE="${FORCE:-0}"
CKPT="checkpoints/jrdb_v2_decoupled_z_s42_20260619_1432/best.pt"
SHA="94cb5326200fa2544340c87d59200b9e15f397c66fe99e39a6727f885c6ce07f"
BV_JSON="${E123_ROOT}/diagnostic_BV/diagnostic_set_BV.json"
MAP_JSON="${E123_ROOT}/donor_maps/donor_pairs_B_d42.json"

mkdir -p "${OUT}"
if [[ "${FORCE}" == "1" ]]; then
  rm -f "${OUT}/STAGEA_DONE.json"
fi

echo "[nj stageA] s42 d42 -> ${OUT}"
"${PYTHON}" eval_nj_stageA.py \
  --config "${CFG}" --checkpoint "${CKPT}" \
  --train_seed 42 --donor_assign_seed 42 \
  --diagnostic_set_json "${BV_JSON}" \
  --donor_map_json "${MAP_JSON}" \
  --out_dir "${OUT}" --split test --gpu "${GPU}" \
  --expected_ckpt_sha256 "${SHA}"
echo "[done] ${OUT}"
