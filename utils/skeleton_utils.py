"""
骨骼数据预处理工具

从 JRDB 原始数据中提取和对齐骨骼关键点。
"""

import numpy as np
import os
import json


def interpolate_missing_keypoints(keypoints, confidence, min_conf=0.1):
    """
    插值缺失的关键点

    对置信度低于阈值的关键点，用前后帧线性插值填充

    Args:
        keypoints: (T, V, 2)
        confidence: (T, V)
        min_conf: 最低置信度阈值
    Returns:
        keypoints_filled: (T, V, 2)
        confidence_filled: (T, V)
    """
    T, V, _ = keypoints.shape
    kp = keypoints.copy()
    conf = confidence.copy()

    for v in range(V):
        valid = conf[:, v] >= min_conf
        if valid.sum() < 2:
            continue

        valid_idx = np.where(valid)[0]
        invalid_idx = np.where(~valid)[0]

        if len(invalid_idx) == 0:
            continue

        # 线性插值
        for dim in range(2):
            kp[invalid_idx, v, dim] = np.interp(
                invalid_idx,
                valid_idx,
                kp[valid_idx, v, dim],
            )
        # 更新置信度 (插值的标记为较低置信度)
        conf[invalid_idx, v] = 0.3

    return kp, conf


def normalize_skeleton(keypoints, image_width=1920, image_height=1080):
    """
    归一化骨骼坐标到 [-1, 1]

    Args:
        keypoints: (..., 2) — 像素坐标
        image_width, image_height: 原始图像尺寸
    Returns:
        normalized: (..., 2)
    """
    normalized = keypoints.copy().astype(np.float32)
    normalized[..., 0] = (normalized[..., 0] / image_width) * 2 - 1
    normalized[..., 1] = (normalized[..., 1] / image_height) * 2 - 1
    return normalized


def compute_bone_features(keypoints):
    """
    计算骨骼向量特征 (骨骼连接的方向和长度)

    Args:
        keypoints: (T, V, 2) — 关键点坐标
    Returns:
        bone_feat: (T, num_bones, 2) — 骨骼向量
    """
    # COCO-17 骨骼连接
    bones = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (0, 5), (0, 6), (5, 7), (7, 9),
        (6, 8), (8, 10), (5, 11), (6, 12),
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    ]

    T = keypoints.shape[0]
    bone_feat = np.zeros((T, len(bones), 2), dtype=np.float32)

    for i, (src, dst) in enumerate(bones):
        bone_feat[:, i, :] = keypoints[:, dst, :] - keypoints[:, src, :]

    return bone_feat


def align_skeleton_to_trajectory(traj_timestamps, skel_timestamps,
                                  skeleton_data, tolerance=0.05):
    """
    将骨骼数据与轨迹时间戳对齐

    Args:
        traj_timestamps: (T_traj,) — 轨迹时间戳
        skel_timestamps: (T_skel,) — 骨骼时间戳
        skeleton_data: (T_skel, V, 3) — 骨骼数据 (x, y, conf)
        tolerance: 时间匹配容差 (秒)
    Returns:
        aligned: (T_traj, V, 3) — 对齐后的骨骼
    """
    V = skeleton_data.shape[1]
    aligned = np.zeros((len(traj_timestamps), V, 3), dtype=np.float32)

    for i, t in enumerate(traj_timestamps):
        diffs = np.abs(skel_timestamps - t)
        best_idx = np.argmin(diffs)

        if diffs[best_idx] < tolerance:
            aligned[i] = skeleton_data[best_idx]
        # else: 保持零值 (表示缺失)

    return aligned


def create_skeleton_availability_mask(skeleton_conf, min_joints=3, min_conf=0.1):
    """
    创建骨骼可用性 mask

    Args:
        skeleton_conf: (T, V) — 关键点置信度
        min_joints: 最少有效关键点数
        min_conf: 置信度阈值
    Returns:
        mask: (T,) — 1=有效, 0=缺失
    """
    valid_joints = (skeleton_conf > min_conf).sum(axis=-1)  # (T,)
    mask = (valid_joints >= min_joints).astype(np.float32)
    return mask
