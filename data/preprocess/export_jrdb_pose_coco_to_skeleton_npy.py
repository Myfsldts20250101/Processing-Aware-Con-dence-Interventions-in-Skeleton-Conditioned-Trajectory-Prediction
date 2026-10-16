#!/usr/bin/env python3
"""
从 JRDB `labels_2d_pose_stitched_coco/*.json`（COCO 格式，17 点）导出
`skeletons/*.npy`，与 `data/jrdb_dataset.py` 及已导出的 `trajectories/*.npy` 对齐。

输出形状: (num_frames, max_peds, 17, 3)，最后一维为 (x, y, conf)。
行人列顺序、帧顺序与 `export_jrdb_stitched_labels_to_npy.py` 使用同一套
`labels_2d_stitched` 规则（含 `skip_no_eval`），以保证与 `trajectories/{scene}.npy`
的 (F, P) 一致。

COCO keypoints 可见性 v∈{0,1,2} → conf = v / 2.0（0 / 0.5 / 1.0）。

若某场景无 pose 文件或某帧无图像，对应位置保持 0（与 Dataset 中「无骨骼」一致）。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
from pathlib import Path
import numpy as np


def _load_traj_export_module():
    """动态加载同目录下的轨迹导出模块（文件名含下划线，避免重复实现）。"""
    here = Path(__file__).resolve().parent
    path = here / "export_jrdb_stitched_labels_to_npy.py"
    spec = importlib.util.spec_from_file_location("jrdb_traj_export", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_traj_mod = None


def _traj():
    global _traj_mod
    if _traj_mod is None:
        _traj_mod = _load_traj_export_module()
    return _traj_mod


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


def stitched_frame_keys_and_ped_map(
    stitched_json_path: str, *, skip_no_eval: bool
) -> tuple[list[str], dict[int, int], int, int]:
    """
    与轨迹导出相同的帧序与行人列索引。
    Returns:
        frame_keys, pid -> column index, F, P
    """
    with open(stitched_json_path, "r", encoding="utf-8") as f:
        root = json.load(f)
    labels = root.get("labels")
    if not isinstance(labels, dict):
        raise ValueError(f"无 labels 字典: {stitched_json_path}")

    frame_keys = sorted(labels.keys(), key=lambda k: _natural_frame_key(k))
    peds_set: set[int] = set()

    for fk in frame_keys:
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
            peds_set.add(pid)

    if not peds_set:
        raise ValueError(f"无有效行人（与轨迹口径一致）: {stitched_json_path}")

    peds = sorted(peds_set)
    p2i = {p: i for i, p in enumerate(peds)}
    return frame_keys, p2i, len(frame_keys), len(peds)


def resolve_pose_coco_dir(jrdb_root: str, labels_subdir: str) -> str:
    root = os.path.abspath(jrdb_root)
    if labels_subdir and labels_subdir != "auto":
        p = os.path.join(root, labels_subdir)
        if not os.path.isdir(p):
            raise SystemExit(f"[错误] 不是目录: {p}")
        return p

    candidates = [
        os.path.join(
            root,
            "train_dataset_with_activity",
            "labels",
            "labels_2d_pose_stitched_coco",
        ),
        os.path.join(root, "labels", "labels_2d_pose_stitched_coco"),
        os.path.join(root, "labels_2d_pose_stitched_coco"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise SystemExit(
        "[错误] 未找到 labels_2d_pose_stitched_coco。请用 --pose_subdir 指定相对 "
        "--jrdb_root 的路径，例如:\n"
        "  train_dataset_with_activity/labels/labels_2d_pose_stitched_coco"
    )


def fill_skeleton_from_coco(
    skel: np.ndarray,
    frame_keys: list[str],
    p2i: dict[int, int],
    coco_path: str,
) -> None:
    """原地写入 skel[F,P,17,3]。"""
    fi_by_basename = {fk: i for i, fk in enumerate(frame_keys)}

    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    image_id_to_fi: dict[int, int] = {}
    for im in coco.get("images", []):
        if not isinstance(im, dict):
            continue
        fn = im.get("file_name", "")
        base = os.path.basename(str(fn))
        fi = fi_by_basename.get(base)
        if fi is not None:
            image_id_to_fi[int(im["id"])] = fi

    for ann in coco.get("annotations", []):
        if not isinstance(ann, dict):
            continue
        fi = image_id_to_fi.get(int(ann.get("image_id", -1)))
        if fi is None:
            continue
        tid = int(ann["track_id"])
        j = p2i.get(tid)
        if j is None:
            continue
        kp = ann.get("keypoints")
        if not isinstance(kp, list) or len(kp) < 17 * 3:
            continue
        a = np.asarray(kp[: 17 * 3], dtype=np.float32).reshape(17, 3)
        for ji in range(17):
            x, y, v = float(a[ji, 0]), float(a[ji, 1]), float(a[ji, 2])
            vv = int(round(v))
            if vv not in (0, 1, 2):
                vv = 0
            conf = vv / 2.0
            skel[fi, j, ji, 0] = x
            skel[fi, j, ji, 1] = y
            skel[fi, j, ji, 2] = conf


def main() -> None:
    ap = argparse.ArgumentParser(
        description="JRDB labels_2d_pose_stitched_coco -> skeletons/*.npy"
    )
    ap.add_argument(
        "--jrdb_root",
        type=str,
        default="/root/autodl-tmp/JRDB/train_dataset",
        help="JRDB 根目录（与轨迹导出一致）",
    )
    ap.add_argument(
        "--labels_subdir",
        type=str,
        default="auto",
        help="相对 jrdb_root 的 labels_2d_stitched 路径；默认 auto",
    )
    ap.add_argument(
        "--pose_subdir",
        type=str,
        default="auto",
        help="相对 jrdb_root 的 COCO pose 目录；默认 auto",
    )
    ap.add_argument(
        "--splits_json",
        type=str,
        default=str(
            Path(__file__).resolve().parent.parent
            / "ctr_gcn_traj"
            / "ctr_gcn_traj"
            / "configs"
            / "jrdb_scene_splits.json"
        ),
        help="含 official.train/val/test",
    )
    ap.add_argument(
        "--output_root",
        type=str,
        default=str(
            Path(__file__).resolve().parent.parent
            / "ctr_gcn_traj"
            / "ctr_gcn_traj"
            / "data"
            / "jrdb_processed"
        ),
        help="写入 {train,val,test}/skeletons/*.npy（与 trajectories 并列）",
    )
    ap.add_argument(
        "--include_no_eval",
        action="store_true",
        help="与轨迹导出一致：包含 no_eval 框对应的行人列（默认跳过）",
    )
    ap.add_argument(
        "--verify_trajectories",
        action="store_true",
        help="若已有 trajectories/*.npy，校验 (F,P) 与之一致",
    )
    args = ap.parse_args()

    traj_mod = _traj()
    labels_stitched_dir = traj_mod.resolve_labels_2d_stitched_dir(
        args.jrdb_root, args.labels_subdir
    )
    labels_pose_dir = resolve_pose_coco_dir(args.jrdb_root, args.pose_subdir)
    print(f"[stitched] {labels_stitched_dir}")
    print(f"[pose]     {labels_pose_dir}")

    skip_no_eval = not args.include_no_eval
    with open(args.splits_json, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    official = cfg["official"]

    for split in ("train", "val", "test"):
        scenes = official.get(split) or []
        out_s = os.path.join(args.output_root, split, "skeletons")
        os.makedirs(out_s, exist_ok=True)
        n_ok, n_miss_pose, n_skip = 0, 0, 0

        for scene in scenes:
            st_path = os.path.join(labels_stitched_dir, f"{scene}.json")
            if not os.path.isfile(st_path):
                print(f"[skip] {split} 无 stitched，跳过: {st_path}")
                n_skip += 1
                continue

            try:
                frame_keys, p2i, F, P = stitched_frame_keys_and_ped_map(
                    st_path, skip_no_eval=skip_no_eval
                )
            except ValueError as e:
                print(f"[fail] {st_path}: {e}")
                n_skip += 1
                continue

            skel = np.zeros((F, P, 17, 3), dtype=np.float32)
            pose_path = os.path.join(labels_pose_dir, f"{scene}.json")
            if os.path.isfile(pose_path):
                try:
                    fill_skeleton_from_coco(skel, frame_keys, p2i, pose_path)
                except Exception as e:
                    print(f"[warn] pose 解析失败，保持全 0: {pose_path}: {e}")
                    n_miss_pose += 1
            else:
                print(f"[warn] 无 pose 文件，写全 0: {pose_path}")
                n_miss_pose += 1

            if args.verify_trajectories:
                t_path = os.path.join(
                    args.output_root, split, "trajectories", f"{scene}.npy"
                )
                if os.path.isfile(t_path):
                    tr = np.load(t_path)
                    if tr.shape[:2] != (F, P):
                        print(
                            f"[错误] 与轨迹形状不一致 {t_path} {tr.shape} vs skel ({F},{P})"
                        )
                        n_skip += 1
                        continue

            out_p = os.path.join(out_s, f"{scene}.npy")
            np.save(out_p, skel)
            nz = float(np.count_nonzero(skel)) / max(skel.size, 1)
            print(
                f"[ok] {split} {scene} -> {out_p} shape={skel.shape} "
                f"nonzero_frac={nz:.4f}"
            )
            n_ok += 1

        print(
            f"--- {split}: skeletons 写出 {n_ok}, stitched 跳过 {n_skip}, "
            f"缺 pose 或解析警告累计 {n_miss_pose}"
        )

    out_abs = os.path.abspath(args.output_root)
    print(
        f"\n完成。skeletons 与 trajectories 在同一 output_root 下。\n"
        f"data.data_dir 仍为:\n  \"{out_abs}\""
    )


if __name__ == "__main__":
    main()
