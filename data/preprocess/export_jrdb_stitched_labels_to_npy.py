#!/usr/bin/env python3
"""
从 JRDB 的 `labels_2d_stitched/*.json` 导出 `trajectories/*.npy`（与 data/jrdb_dataset.py 约定一致），并写出同目录 `{scene}_ped_ids.npy`
（形状 `(P,)`，`ped_ids[j]` 为该场景轨迹矩阵列 `j` 对应的 JRDB 行人整数 id）。

常见布局（`--labels_subdir auto` 时按顺序自动探测）：
  - `<JRDB根>/train_dataset_with_activity/labels/labels_2d_stitched`（官方 zip 常见）
  - `<某根>/labels/labels_2d_stitched`
  - `<某根>/labels_2d_stitched`

依赖 `configs/jrdb_scene_splits.json` 中 `official.{train,val,test}` 划分。

注意：仅 train 包时，test 场景 json 会缺失并跳过。仅当存在 `{labels_dir}/{scene}.json` 才写出该场景。

Box: [u,v,w,h] 中心为像素坐标 (u+w/2, v+h/2)。
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any

import numpy as np


def _natural_frame_key(name: str) -> tuple[int, str]:
    m = re.match(r"^(\d+)\.jpg$", name, re.I)
    if m:
        return (int(m.group(1)), name)
    return (10**9, name)


def _parse_ped_id(label_id: str) -> int | None:
    if not label_id:
        return None
    parts = str(label_id).split(":")
    try:
        return int(parts[-1])
    except ValueError:
        return None


def _box_center_xy(box: list[Any]) -> tuple[float, float]:
    """JRDB 2D box: 左上角 (u,v) 与宽高 (w,h)，见官网检测格式说明。"""
    if len(box) < 4:
        raise ValueError(f"box too short: {box}")
    u, v, w, h = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    return (u + w / 2.0, v + h / 2.0)


def json_labels_to_traj_npy(
    path: str, skip_no_eval: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """返回 (轨迹数组 (F,P,2), ped_ids (P,) int64)，列 j 对应行人 id ped_ids[j]。"""
    with open(path, "r", encoding="utf-8") as f:
        root = json.load(f)
    labels = root.get("labels")
    if not isinstance(labels, dict):
        raise ValueError(f"无 labels 字典: {path}")

    frame_keys = sorted(labels.keys(), key=lambda k: _natural_frame_key(k))
    rows: list[tuple[int, int, float, float]] = []
    peds_set: set[int] = set()

    for fi, fk in enumerate(frame_keys):
        objs = labels.get(fk, [])
        if not isinstance(objs, list):
            continue
        for obj in objs:
            if not isinstance(obj, dict):
                continue
            attr = obj.get("attributes") or {}
            if skip_no_eval and attr.get("no_eval") is True:
                continue
            pid = _parse_ped_id(obj.get("label_id", ""))
            if pid is None:
                continue
            box = obj.get("box")
            if not isinstance(box, (list, tuple)) or len(box) < 4:
                continue
            cx, cy = _box_center_xy([float(box[i]) for i in range(4)])
            rows.append((fi, pid, cx, cy))
            peds_set.add(pid)

    if not rows:
        raise ValueError(f"无有效轨迹点: {path}")

    peds = sorted(peds_set)
    p2i = {p: i for i, p in enumerate(peds)}
    F, P = len(frame_keys), len(peds)
    arr = np.full((F, P, 2), -1.0, dtype=np.float32)
    for fi, pid, cx, cy in rows:
        j = p2i[pid]
        arr[fi, j, 0] = cx
        arr[fi, j, 1] = cy
    ped_arr = np.array(peds, dtype=np.int64)
    return arr, ped_arr


def resolve_labels_2d_stitched_dir(jrdb_root: str, labels_subdir: str) -> str:
    """
    labels_subdir:
      'auto' — 在 jrdb_root 下依次尝试常见相对路径（覆盖 train_dataset 下嵌套结构）。
      其它值 — 视为相对 jrdb_root 的显式子路径。
    """
    root = os.path.abspath(jrdb_root)
    if labels_subdir and labels_subdir != "auto":
        p = os.path.join(root, labels_subdir)
        if not os.path.isdir(p):
            raise SystemExit(f"[错误] 不是目录: {p}")
        return p

    candidates = [
        os.path.join(root, "train_dataset_with_activity", "labels", "labels_2d_stitched"),
        os.path.join(root, "labels", "labels_2d_stitched"),
        os.path.join(root, "labels_2d_stitched"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise SystemExit(
        "[错误] 未找到 labels_2d_stitched。请用 --labels_subdir 指定相对 --jrdb_root 的路径，例如:\n"
        f"  train_dataset_with_activity/labels/labels_2d_stitched\n  已尝试: {candidates!r}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="JRDB labels_2d_stitched -> trajectories/*.npy")
    ap.add_argument(
        "--jrdb_root",
        type=str,
        default="/root/autodl-tmp/JRDB/train_dataset",
        help="JRDB 数据根，一般为 .../train_dataset；其下常含 train_dataset_with_activity/labels/labels_2d_stitched",
    )
    ap.add_argument(
        "--labels_subdir",
        type=str,
        default="auto",
        help="相对 jrdb_root 的 stitched 标注目录；默认 auto 自动探测",
    )
    ap.add_argument(
        "--splits_json",
        type=str,
        default="/root/work/ctr_gcn_traj/ctr_gcn_traj/ctr_gcn_traj/configs/jrdb_scene_splits.json",
        help="含 official.train/val/test 的划分文件",
    )
    ap.add_argument(
        "--output_root",
        type=str,
        default="/root/work/ctr_gcn_traj/ctr_gcn_traj/ctr_gcn_traj/data/jrdb_processed",
        help="写入 {train,val,test}/trajectories/*.npy",
    )
    ap.add_argument(
        "--include_no_eval",
        action="store_true",
        help="包含 attributes.no_eval=true 的框（默认跳过，与检测 benchmark 口径更接近）",
    )
    args = ap.parse_args()

    labels_dir = resolve_labels_2d_stitched_dir(args.jrdb_root, args.labels_subdir)
    print(f"[labels] 使用标注目录: {labels_dir}")
    with open(args.splits_json, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    official = cfg["official"]

    skip_no_eval = not args.include_no_eval
    for split in ("train", "val", "test"):
        scenes = official.get(split) or []
        out_t = os.path.join(args.output_root, split, "trajectories")
        os.makedirs(out_t, exist_ok=True)
        n_ok, n_miss = 0, 0
        for scene in scenes:
            jp = os.path.join(labels_dir, f"{scene}.json")
            if not os.path.isfile(jp):
                print(f"[missing] {split} 无标注文件，跳过: {jp}")
                n_miss += 1
                continue
            try:
                arr, ped_arr = json_labels_to_traj_npy(jp, skip_no_eval=skip_no_eval)
            except Exception as e:
                print(f"[fail] {jp}: {e}")
                n_miss += 1
                continue
            out_p = os.path.join(out_t, f"{scene}.npy")
            np.save(out_p, arr)
            ped_path = os.path.join(out_t, f"{scene}_ped_ids.npy")
            np.save(ped_path, ped_arr)
            print(f"[ok] {split} {scene} -> {out_p} shape={arr.shape}  ped_ids -> {ped_path}")
            n_ok += 1
        print(f"--- {split}: 写出 {n_ok}，缺/失败 {n_miss}")

    out_abs = os.path.abspath(args.output_root)
    print(
        f"\n完成。请将 configs/jrdb_default.yaml 中 data.data_dir 设为:\n  \"{out_abs}\""
    )


if __name__ == "__main__":
    main()
