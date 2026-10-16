"""
JRDB 数据集加载器

处理:
1. 轨迹数据: (obs_len + pred_len) 帧的 (x, y) 坐标
2. 骨骼数据: 对应时间窗口的关键点序列
3. 数据增强: 随机旋转、翻转

目录结构见文件头原版说明。

运动阈值过滤、缓存：见仓库根目录《版本_II.md》§3.3；与训练空间一致（agent-centric 后除以 traj_scale 的完整 obs+pred 窗）。
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, DataLoader

# 像素域轨迹在 agent-centric 后仍可能有 O(1e3) 量级的位移，与 MLP/Transformer 稳定域对齐
_DEFAULT_TRAJ_SCALE = 1000.0

# COCO-17: 0 nose … 16 right_ankle（与 models/ctr_gcn.py 一致）
_COCO17_FULL = tuple(range(17))
_COCO17_ROOT = (11, 12)  # 左右髋，作根节点去全局平移
_COCO17_LOWER_LEG = (13, 14, 15, 16)  # 膝、踝 — 用于判定下肢遮挡
# 上肢 + 躯干 8 点：肩、肘、腕、髋（无头、无小腿）
_COCO17_UPPER_TORSO_8 = (5, 6, 7, 8, 9, 10, 11, 12)
# 纯上肢 6 点：左右肩、肘、腕（不含髋）
_COCO17_UPPER_LIMB_6 = (5, 6, 7, 8, 9, 10)
_DEFAULT_POSE_THRESHOLD_M = 0.08
_DEFAULT_TORSO_REF_M = 0.45
_DEFAULT_POSE_MIN_CONF = 0.10


def _pose_change_config(motion_filter):
    """合并 YAML data.motion_filter.pose_change 与顶层 threshold（米）。"""
    mf = motion_filter or {}
    pc = dict(mf.get('pose_change') or {})
    top_thr = mf.get('threshold', _DEFAULT_POSE_THRESHOLD_M)
    pc.setdefault('threshold_m', float(top_thr))
    if 'threshold' in mf:
        pc['threshold_m'] = float(mf['threshold'])
    pc.setdefault('min_conf', _DEFAULT_POSE_MIN_CONF)
    pc.setdefault('lower_body_joints', list(_COCO17_LOWER_LEG))
    pc.setdefault('upper_torso_joints', list(_COCO17_UPPER_TORSO_8))
    pc.setdefault('upper_limb_joints', list(_COCO17_UPPER_LIMB_6))
    pc.setdefault('lower_body_occlusion_conf', 0.3)
    pc.setdefault('lower_body_min_valid_ratio', 0.5)
    pc.setdefault('reference_torso_m', _DEFAULT_TORSO_REF_M)
    # path_mean：整窗髋相对路径均值（旧）
    # fut_upper6_max：pred 段首尾，纯上肢 6 点 max 位移（工业/实拍推荐）
    pc.setdefault('criterion', 'path_mean')
    pc.setdefault('joint_set', 'auto')
    if 'meters_per_pixel' not in pc:
        pc['meters_per_pixel'] = None
    return pc


def _pose_change_meta_signature(pose_cfg):
    """写入 motion_filter 缓存 meta，参数变更时自动失效。"""
    keys = (
        'threshold_m', 'min_conf', 'criterion', 'joint_set', 'upper_limb_joints',
        'lower_body_joints', 'upper_torso_joints', 'lower_body_occlusion_conf',
        'lower_body_min_valid_ratio', 'reference_torso_m', 'meters_per_pixel',
    )
    return {k: pose_cfg.get(k) for k in keys}


def _is_lower_body_severely_occluded(conf_seq, pose_cfg):
    """
    下肢（膝、踝）置信度整体偏低 → 视为遮挡严重（仅 joint_set=auto 时使用）。

    注意：工业/实拍骨骼在遮挡时常经**软件补全**，补全点 conf 可能不低，
    不能用此 heuristics 识别「真遮挡」；此类数据应设 joint_set=upper8。
    """
    min_conf = float(pose_cfg['min_conf'])
    lower_joints = tuple(int(j) for j in pose_cfg['lower_body_joints'])
    occl_conf_thr = float(pose_cfg['lower_body_occlusion_conf'])
    occl_ratio_thr = float(pose_cfg['lower_body_min_valid_ratio'])

    c = np.asarray(conf_seq, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] <= max(lower_joints):
        return True
    lb = c[:, lower_joints]
    if lb.size == 0:
        return True
    mean_lb = float(np.mean(lb))
    valid_ratio = float(np.mean(lb >= min_conf))
    return (mean_lb < occl_conf_thr) or (valid_ratio < occl_ratio_thr)


def _select_pose_joint_indices(conf_seq, pose_cfg):
    """
    选择参与 pose_change 的关节集。
    joint_set=upper8：固定上肢+躯干（工业坐姿、下肢软件补全场景）；
    full17：始终 17 点；auto：低 conf 判遮挡时退 8 点。
    """
    joint_set = str(pose_cfg.get('joint_set', 'auto')).lower()
    upper = tuple(int(j) for j in pose_cfg['upper_torso_joints'])
    if joint_set == 'upper8':
        return upper, 'upper8'
    if joint_set == 'full17':
        return _COCO17_FULL, 'full17'
    if _is_lower_body_severely_occluded(conf_seq, pose_cfg):
        return upper, 'upper8'
    return _COCO17_FULL, 'full17'


def _estimate_meters_per_pixel(skel_xy, conf_seq, pose_cfg):
    """
    用肩–髋像素距离估计 m/px（reference_torso_m 默认 0.45 m）。
    skeleton 坐标为像素系。
    """
    fixed = pose_cfg.get('meters_per_pixel')
    if fixed is not None:
        v = float(fixed)
        return v if v > 0 else None

    min_conf = float(pose_cfg['min_conf'])
    ref_m = float(pose_cfg['reference_torso_m'])
    xy = np.asarray(skel_xy, dtype=np.float64)
    conf = np.asarray(conf_seq, dtype=np.float64)
    sh = xy[:, [5, 6], :].mean(axis=1)
    hp = xy[:, [11, 12], :].mean(axis=1)
    conf_ok = (conf[:, [5, 6]].mean(axis=1) >= min_conf) & (
        conf[:, [11, 12]].mean(axis=1) >= min_conf
    )
    if conf_ok.sum() == 0:
        return None
    dists = np.linalg.norm(sh - hp, axis=-1)
    med_px = float(np.median(dists[conf_ok]))
    if med_px <= 1e-6:
        return None
    return ref_m / med_px


def _motion_metric_pose_change_path_mean(xy, conf, pose_cfg):
    """旧口径：整 obs+pred 窗、髋相对、关节路径长度均值（米）。"""
    if xy.shape[0] < 2:
        return 0.0

    joint_idx, _ = _select_pose_joint_indices(conf, pose_cfg)
    min_conf = float(pose_cfg['min_conf'])
    mpp = _estimate_meters_per_pixel(xy, conf, pose_cfg)
    if mpp is None:
        return 0.0

    root = xy[:, list(_COCO17_ROOT), :].mean(axis=1, keepdims=True)
    rel = xy - root

    per_joint_path_px = []
    for j in joint_idx:
        if j >= rel.shape[1]:
            continue
        valid = (conf[:-1, j] >= min_conf) & (conf[1:, j] >= min_conf)
        if valid.sum() == 0:
            continue
        step = np.linalg.norm(np.diff(rel[:, j, :], axis=0), axis=-1)
        per_joint_path_px.append(float(step[valid].sum()))

    if not per_joint_path_px:
        return 0.0
    motion_px = float(np.mean(per_joint_path_px))
    return motion_px * mpp


def _motion_metric_pose_change_fut_upper6_max(xy, conf, pose_cfg, obs_len, pred_len):
    """
    工业/实拍 moving：pred 段首尾，纯上肢 6 点 max 位移（米）。

    在 fut 首尾两帧都有效的上肢点 j 中，
    max_j ‖pos_fut[-1,j] - pos_fut[0,j]‖（像素）× m/px ≥ θ。
    """
    if obs_len < 0 or pred_len < 1 or xy.shape[0] < obs_len + pred_len:
        return 0.0

    fut_xy = xy[obs_len:obs_len + pred_len]
    fut_cf = conf[obs_len:obs_len + pred_len]
    if fut_xy.shape[0] < 2:
        return 0.0

    min_conf = float(pose_cfg['min_conf'])
    mpp = _estimate_meters_per_pixel(xy, conf, pose_cfg)
    if mpp is None:
        return 0.0

    joints = tuple(int(j) for j in pose_cfg.get('upper_limb_joints', _COCO17_UPPER_LIMB_6))
    max_disp_m = 0.0
    any_valid = False
    for j in joints:
        if j >= fut_xy.shape[1]:
            continue
        if fut_cf[0, j] >= min_conf and fut_cf[-1, j] >= min_conf:
            d_px = float(np.linalg.norm(fut_xy[-1, j] - fut_xy[0, j]))
            max_disp_m = max(max_disp_m, d_px * mpp)
            any_valid = True
    return max_disp_m if any_valid else 0.0


def _motion_metric_pose_change(skel_xy, conf_seq, pose_cfg, obs_len=8, pred_len=12):
    """
    坐姿/工业 pose_change 运动量（米）。
    skel_xy: (T, 17, 2) 像素坐标；conf_seq: (T, 17)。
    criterion=fut_upper6_max（默认工业配置）或 path_mean（旧）。
    """
    xy = np.asarray(skel_xy, dtype=np.float64)
    conf = np.asarray(conf_seq, dtype=np.float64)
    if xy.ndim != 3 or xy.shape[1] < 17:
        return 0.0

    criterion = str(pose_cfg.get('criterion', 'path_mean')).lower()
    if criterion in ('fut_upper6_max', 'fut_endpoint_max', 'fut_upper_limb_max'):
        return _motion_metric_pose_change_fut_upper6_max(
            xy, conf, pose_cfg, int(obs_len), int(pred_len),
        )
    return _motion_metric_pose_change_path_mean(xy, conf, pose_cfg)


def _track_quality_config(motion_filter, data_track_quality=None):
    """
    滑窗质量：目标行人全程在画面内、轨迹/ID 不中断。
    可写在 data.track_quality 或 data.motion_filter.track_quality（前者优先）。
    """
    mf = motion_filter or {}
    tq = dict(mf.get('track_quality') or {})
    if data_track_quality:
        tq = {**tq, **data_track_quality}
    if not tq.get('enabled', False):
        return None
    tq = dict(tq)
    tq['enabled'] = True
    tq.setdefault('min_conf', _DEFAULT_POSE_MIN_CONF)
    tq.setdefault('min_frame_skel_conf', float(tq['min_conf']))
    tq.setdefault('min_joint_valid_ratio', 0.30)
    tq.setdefault('require_skel', True)
    tq.setdefault('reject_zero_traj_frame', True)
    tq.setdefault('reject_negative_traj', True)
    in_frame = dict(tq.get('in_frame') or {})
    in_frame.setdefault('enabled', True)
    in_frame.setdefault('width', 1280)
    in_frame.setdefault('height', 720)
    in_frame.setdefault('margin_px', 5.0)
    tq['in_frame'] = in_frame
    return tq


def _track_quality_meta_signature(tq_cfg):
    if tq_cfg is None:
        return None
    keys = (
        'min_conf', 'min_frame_skel_conf', 'min_joint_valid_ratio',
        'require_skel', 'reject_zero_traj_frame', 'reject_negative_traj', 'in_frame',
    )
    return {k: tq_cfg.get(k) for k in keys}


def _window_passes_track_quality(traj_window, skel_full_window, tq_cfg, parse_skel_fn):
    """
    traj_window: (T, 2) 像素系绝对坐标（未 agent-centric）
    skel_full_window: (T, 17, 3) 或 None
    """
    if tq_cfg is None:
        return True

    tw = np.asarray(traj_window, dtype=np.float64)
    if tw.ndim != 2 or tw.shape[0] < 1 or tw.shape[1] < 2:
        return False

    if tq_cfg.get('reject_negative_traj', True):
        if np.any(tw < 0):
            return False

    if tq_cfg.get('reject_zero_traj_frame', True):
        if np.any(np.all(tw == 0, axis=-1)):
            return False

    if np.any(np.isnan(tw)):
        return False

    if np.all(tw == 0) or np.all(tw == -1):
        return False

    iframe = tq_cfg.get('in_frame') or {}
    if iframe.get('enabled', True):
        m = float(iframe.get('margin_px', 0))
        w = float(iframe['width']) - m
        h = float(iframe['height']) - m
        if w <= m or h <= m:
            return False
        if (
            np.any(tw[:, 0] < m)
            or np.any(tw[:, 0] > w)
            or np.any(tw[:, 1] < m)
            or np.any(tw[:, 1] > h)
        ):
            return False

    if not tq_cfg.get('require_skel', True):
        return True

    if skel_full_window is None or np.all(skel_full_window == 0):
        return False

    sk_xy, sk_cf = parse_skel_fn(skel_full_window, squeeze_time=True)
    if sk_xy is None or sk_cf is None:
        return False

    min_conf = float(tq_cfg['min_conf'])
    min_frame_conf = float(tq_cfg['min_frame_skel_conf'])
    min_jratio = float(tq_cfg['min_joint_valid_ratio'])
    T = sk_cf.shape[0]
    for t in range(T):
        cf = sk_cf[t]
        if float(np.mean(cf)) < min_frame_conf:
            return False
        if float(np.mean(cf >= min_conf)) < min_jratio:
            return False
        if iframe.get('enabled', True):
            m = float(iframe.get('margin_px', 0))
            w = float(iframe['width']) - m
            h = float(iframe['height']) - m
            valid = cf >= min_conf
            if valid.sum() == 0:
                return False
            xs = sk_xy[t, valid, 0]
            ys = sk_xy[t, valid, 1]
            if (
                np.any(xs < m)
                or np.any(xs > w)
                or np.any(ys < m)
                or np.any(ys > h)
            ):
                return False

    return True


def _endpoint_min_scaled(motion_filter, split=None):
    """pose_change 等 metric 下的可选质心 endpoint 下界（scaled 空间）；None 表示不启用。"""
    mf = motion_filter or {}
    v = mf.get('endpoint_min_scaled')
    if v is None:
        return None
    v = float(v)
    if v <= 0.0:
        return None
    if split is not None:
        ep_splits = mf.get(
            'endpoint_min_scaled_splits',
            mf.get('apply_splits', ['train', 'val', 'test']),
        )
        if split not in ep_splits:
            return None
    return v


def _motion_metric_full_window(traj_seq_scaled, metric):
    """
    traj_seq_scaled: (seq_len, 2)，首帧已为窗内原点（0），已与 traj_scale 对齐。
    metric: endpoint（末帧相对首帧 L2）| path_length
    """
    t = np.asarray(traj_seq_scaled, dtype=np.float64)
    if metric == 'path_length':
        if t.shape[0] < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(t, axis=0), axis=-1)))
    return float(np.linalg.norm(t[-1] - t[0]))


def _scene_list_signature(scene_files):
    """与轨迹 npy 文件名列表绑定，合并/增删场景时缓存自动失效"""
    blob = '\n'.join(scene_files).encode('utf-8')
    return hashlib.sha256(blob).hexdigest()


def _default_motion_cache_root():
    """默认使用包内 local_runs 缓存；可用环境变量或 YAML motion_filter.cache_dir 覆盖"""
    return os.environ.get(
        'CTR_GCN_TRAJ_MOTION_CACHE_ROOT',
        str(Path(__file__).resolve().parents[1] / 'local_runs' / 'motion_filter_cache'),
    )


class JRDBTrajectoryDataset(Dataset):
    """
    JRDB 轨迹 + 骨骼数据集

    返回滑动窗口样本:
    - obs_traj: (T_obs, 2)
    - pred_traj: (T_pred, 2)
    - skeleton_kp: (T_obs, 17, 2)
    - skeleton_conf: (T_obs, 17)
    - has_skeleton: bool
    """

    def __init__(
        self,
        data_dir,
        split='train',
        obs_len=8,
        pred_len=12,
        skip=1,
        min_ped=1,
        augment=True,
        traj_scale=_DEFAULT_TRAJ_SCALE,
        motion_filter=None,
        track_quality=None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.domain_name = Path(str(data_dir)).name
        self.split = split
        self.traj_scale = float(traj_scale) if traj_scale else 1.0
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.skip = skip
        self.augment = augment and (split == 'train')
        self.motion_filter = motion_filter if motion_filter is not None else {}
        self.track_quality_cfg = _track_quality_config(
            self.motion_filter, data_track_quality=track_quality,
        )

        self.samples = []
        self._load_data()

    def _motion_filter_active(self):
        mf = self.motion_filter
        if not mf.get('enabled'):
            return False
        splits = mf.get('apply_splits', ['train', 'val', 'test'])
        return self.split in splits

    def _motion_cache_dir_signature(self):
        """data_dir 绝对路径 SHA256（全_hex），写入 meta 做跨机校验"""
        ap = os.path.abspath(self.data_dir)
        return hashlib.sha256(ap.encode('utf-8')).hexdigest()

    def _motion_cache_bundle_dir(self):
        mf = self.motion_filter
        root = mf.get('cache_dir') or _default_motion_cache_root()
        root = os.path.expanduser(os.path.expandvars(str(root)))
        data_sig = self._motion_cache_dir_signature()
        bundle = os.path.join(root, data_sig, self.split)
        return bundle

    def _try_load_triple_cache(self, scene_sig):
        mf = self.motion_filter
        if not mf.get('use_cache', True):
            return None
        bundle_dir = self._motion_cache_bundle_dir()
        meta_path = os.path.join(bundle_dir, '_motion_filter_meta.json')
        trip_path = os.path.join(bundle_dir, '_motion_filter_triples.npy')
        if not (os.path.isfile(meta_path) and os.path.isfile(trip_path)):
            return None
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
        metric = mf.get('metric', 'endpoint')
        thr = float(mf['threshold'])
        eps = 1e-9
        data_sig = self._motion_cache_dir_signature()
        checks = [
            meta.get('scene_sig') == scene_sig,
            meta.get('data_dir_signature_sha256') == data_sig,
            meta.get('metric') == metric,
            abs(float(meta.get('threshold', -1)) - thr) < eps,
            int(meta.get('obs_len', -1)) == self.obs_len,
            int(meta.get('pred_len', -1)) == self.pred_len,
            int(meta.get('skip', -1)) == self.skip,
            abs(float(meta.get('traj_scale', -1.0)) - self.traj_scale) < eps,
            meta.get('split') == self.split,
        ]
        if metric == 'pose_change':
            checks.append(
                meta.get('pose_change') == _pose_change_meta_signature(_pose_change_config(mf))
            )
        tq_sig = _track_quality_meta_signature(self.track_quality_cfg)
        checks.append(meta.get('track_quality') == tq_sig)
        ep_min = _endpoint_min_scaled(mf)
        meta_ep = meta.get('endpoint_min_scaled')
        if ep_min is None:
            checks.append(meta_ep is None)
        else:
            checks.append(
                meta_ep is not None and abs(float(meta_ep) - ep_min) < eps
            )
        if not all(checks):
            return None
        keys = np.load(trip_path)
        if keys.ndim != 2 or keys.shape[1] != 3:
            return None
        return set(map(tuple, keys.astype(np.int64).tolist()))

    def _save_triple_cache(self, scene_sig, triple_list):
        mf = self.motion_filter
        if not mf.get('use_cache', True):
            return
        bundle_dir = self._motion_cache_bundle_dir()
        os.makedirs(bundle_dir, exist_ok=True)
        meta_path = os.path.join(bundle_dir, '_motion_filter_meta.json')
        trip_path = os.path.join(bundle_dir, '_motion_filter_triples.npy')
        metric_name = mf.get('metric', 'endpoint')
        meta = {
            'scene_sig': scene_sig,
            'data_dir_signature_sha256': self._motion_cache_dir_signature(),
            'data_dir_absolute': os.path.abspath(self.data_dir),
            'cache_bundle': bundle_dir,
            'metric': metric_name,
            'threshold': float(mf['threshold']),
            'obs_len': self.obs_len,
            'pred_len': self.pred_len,
            'skip': self.skip,
            'traj_scale': self.traj_scale,
            'split': self.split,
            'style': 'v2_motion_filter_triples_autodl_temp',
        }
        if metric_name == 'pose_change':
            meta['pose_change'] = _pose_change_meta_signature(_pose_change_config(mf))
            meta['threshold_unit'] = 'm'
        meta['track_quality'] = _track_quality_meta_signature(self.track_quality_cfg)
        ep_min = _endpoint_min_scaled(mf)
        if ep_min is not None:
            meta['endpoint_min_scaled'] = ep_min
        arr = np.array(triple_list, dtype=np.int64) if triple_list else np.zeros((0, 3), dtype=np.int64)
        np.save(trip_path, arr)
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)

    def _load_data(self):
        """加载并切分所有场景数据"""
        split_dir = os.path.join(self.data_dir, self.split)
        traj_dir = os.path.join(split_dir, 'trajectories')
        skel_dir = os.path.join(split_dir, 'skeletons')

        if not os.path.exists(traj_dir):
            print(f"[Dataset] Warning: {traj_dir} not found. Using synthetic data.")
            self._create_synthetic_data()
            return

        scene_files = sorted(
            f
            for f in os.listdir(traj_dir)
            if f.endswith('.npy') and not f.endswith('_ped_ids.npy')
        )
        scene_sig = _scene_list_signature(scene_files)

        tq_cfg = self.track_quality_cfg
        tq_use = tq_cfg is not None
        mf_use = self._motion_filter_active()
        triple_cached = None
        if mf_use:
            if 'threshold' not in self.motion_filter:
                raise KeyError(
                    "motion_filter.enabled 为 True 时必须在 YAML 的 data.motion_filter 中配置 "
                    "threshold、metric、apply_splits（见 configs/jrdb_v2_randgcn_e2e.yaml）"
                )
            triple_cached = self._try_load_triple_cache(scene_sig)

        if mf_use:
            metric = self.motion_filter.get('metric', 'endpoint')
            if metric == 'pose_change':
                pose_cfg = _pose_change_config(self.motion_filter)
                thr = float(pose_cfg['threshold_m'])
            else:
                thr = float(self.motion_filter['threshold'])
        else:
            metric, thr = 'endpoint', 0.0
            pose_cfg = None
        total_candidates = 0
        triples_to_save = []

        mf = self.motion_filter
        if mf_use and metric == 'pose_change' and pose_cfg is None:
            pose_cfg = _pose_change_config(mf)
        ep_min = _endpoint_min_scaled(mf, split=self.split) if mf_use else None

        for scene_idx, scene_file in enumerate(scene_files):
            traj_data = np.load(os.path.join(traj_dir, scene_file))

            skel_file = os.path.join(skel_dir, scene_file)
            skel_data = None
            if os.path.exists(skel_file):
                skel_data = np.load(skel_file)

            num_frames = traj_data.shape[0]
            max_peds = traj_data.shape[1]

            for start in range(0, num_frames - self.seq_len + 1, self.skip):
                end = start + self.seq_len

                for ped_id in range(max_peds):
                    traj_window = traj_data[start:end, ped_id, :]

                    if np.any(np.isnan(traj_window)):
                        continue
                    if np.all(traj_window == 0) or np.all(traj_window == -1):
                        continue
                    if not tq_use and np.any(traj_window < 0):
                        continue

                    skel_full = None
                    if skel_data is not None:
                        skel_full = skel_data[start:end, ped_id, :, :]

                    if tq_use and not _window_passes_track_quality(
                        traj_window,
                        skel_full,
                        tq_cfg,
                        JRDBTrajectoryDataset._parse_skel_window_pixels,
                    ):
                        continue

                    if np.any(traj_window < 0):
                        continue

                    origin = traj_window[0:1, :].astype(np.float32)
                    traj_scaled = (traj_window.astype(np.float32) - origin)
                    if self.traj_scale != 1.0:
                        traj_scaled = traj_scaled / self.traj_scale

                    if mf_use:
                        total_candidates += 1
                        if metric == 'pose_change':
                            mval = 0.0
                            if skel_full is not None and not np.all(skel_full == 0):
                                    sk_xy, sk_cf = self._parse_skel_window_pixels(
                                        skel_full, squeeze_time=True,
                                    )
                                    if sk_xy is not None:
                                        mval = _motion_metric_pose_change(
                                            sk_xy, sk_cf, pose_cfg,
                                            self.obs_len, self.pred_len,
                                        )
                        else:
                            mval = _motion_metric_full_window(traj_scaled, metric)
                        key = (scene_idx, start, ped_id)
                        if triple_cached is not None:
                            if key not in triple_cached:
                                continue
                        else:
                            if mval < thr:
                                continue
                            if ep_min is not None:
                                ep_val = _motion_metric_full_window(
                                    traj_scaled, 'endpoint',
                                )
                                if ep_val < ep_min:
                                    continue
                            triples_to_save.append(key)

                    has_skel = False
                    skel_kp = np.zeros((self.obs_len, 17, 2), dtype=np.float32)
                    skel_conf = np.zeros((self.obs_len, 17), dtype=np.float32)
                    pred_skel_kp = np.zeros((self.pred_len, 17, 2), dtype=np.float32)
                    pred_skel_conf = np.zeros((self.pred_len, 17), dtype=np.float32)

                    if skel_data is not None:
                        skel_window = skel_data[start:start + self.obs_len, ped_id, :, :]
                        if not np.all(skel_window == 0):
                            skel_kp = skel_window[..., :2].astype(np.float32)
                            skel_conf = skel_window[..., 2].astype(np.float32)
                            skel_kp = self._squeeze_skel_to_tv2(skel_kp)
                            skel_conf = np.asarray(skel_conf, dtype=np.float32)
                            skel_conf = np.squeeze(skel_conf)
                            if skel_conf.ndim == 3 and skel_conf.shape[-2] == 1:
                                skel_conf = skel_conf[:, 0, :]
                            elif skel_conf.ndim == 3 and skel_conf.shape[-1] == 1:
                                skel_conf = skel_conf[:, :, 0]
                            skel_kp = skel_kp - origin[None, None, :2]
                            if self.traj_scale != 1.0:
                                skel_kp = skel_kp / self.traj_scale
                            has_skel = True

                        skel_pred_window = skel_data[start + self.obs_len:end, ped_id, :, :]
                        if not np.all(skel_pred_window == 0):
                            pred_skel_kp = skel_pred_window[..., :2].astype(np.float32)
                            pred_skel_conf = skel_pred_window[..., 2].astype(np.float32)
                            pred_skel_kp = self._squeeze_skel_to_tv2(pred_skel_kp)
                            pred_skel_conf = np.asarray(pred_skel_conf, dtype=np.float32)
                            pred_skel_conf = np.squeeze(pred_skel_conf)
                            if pred_skel_conf.ndim == 3 and pred_skel_conf.shape[-2] == 1:
                                pred_skel_conf = pred_skel_conf[:, 0, :]
                            elif pred_skel_conf.ndim == 3 and pred_skel_conf.shape[-1] == 1:
                                pred_skel_conf = pred_skel_conf[:, :, 0]
                            pred_skel_kp = pred_skel_kp - origin[None, None, :2]
                            if self.traj_scale != 1.0:
                                pred_skel_kp = pred_skel_kp / self.traj_scale

                    tw_obs = traj_scaled[:self.obs_len].astype(np.float32)
                    tw_pred = traj_scaled[self.obs_len:].astype(np.float32)

                    self.samples.append({
                        'obs_traj': tw_obs,
                        'pred_traj': tw_pred,
                        'skeleton_kp': skel_kp,
                        'skeleton_conf': skel_conf,
                        'pred_skeleton_kp': pred_skel_kp,
                        'pred_skeleton_conf': pred_skel_conf,
                        'has_skeleton': has_skel,
                        'window_origin': origin.reshape(2).astype(np.float32),
                        'scene_id': Path(scene_file).stem,
                        'window_start': int(start),
                        'ped_id': int(ped_id),
                    })

        if mf_use:
            kept = len(self.samples)
            dropped = total_candidates - kept
            src = 'cache' if triple_cached is not None else 'recomputed'
            mname = mf.get('metric', 'endpoint')
            thr_unit = 'm' if mname == 'pose_change' else 'scaled'
            bdir = self._motion_cache_bundle_dir()
            tq_note = ' | track_quality=on' if tq_use else ''
            ep_note = (
                f' | endpoint_min_scaled={ep_min}' if ep_min is not None else ''
            )
            print(
                f"[Data] split={self.split} | motion_filter ({src}) | "
                f"windows total={total_candidates} | kept={kept} | dropped={dropped} "
                f"| thr({mname})={thr}{thr_unit}{ep_note}{tq_note} "
                f"| motion_cache_dir={bdir}"
            )
            if triple_cached is None:
                self._save_triple_cache(scene_sig, triples_to_save)
        elif tq_use:
            print(
                f"[Dataset] {self.split}: {len(self.samples)} samples "
                f"(track_quality: 全程在画面内、ID/轨迹不中断)"
            )
        else:
            print(f"[Dataset] {self.split}: {len(self.samples)} samples loaded")

        if mf_use and len(self.samples) < 50:
            print(
                "[Dataset] Warning: Very few samples after motion filter; "
                "consider lowering threshold (see version_II.md §10)."
            )

    def _create_synthetic_data(self):
        """生成合成数据用于调试"""
        print("[Dataset] Generating synthetic data for debugging...")
        np.random.seed(42)

        for _ in range(1000 if self.split == 'train' else 200):
            start_pos = np.random.randn(2).astype(np.float32) * 5
            velocity = np.random.randn(2).astype(np.float32) * 0.5
            noise = np.random.randn(self.seq_len, 2).astype(np.float32) * 0.05

            traj = np.zeros((self.seq_len, 2), dtype=np.float32)
            for t in range(self.seq_len):
                traj[t] = start_pos + velocity * t + noise[t]

            skel_kp = np.zeros((self.obs_len, 17, 2), dtype=np.float32)
            skel_conf = np.ones((self.obs_len, 17), dtype=np.float32) * 0.8
            pred_skel_kp = np.zeros((self.pred_len, 17, 2), dtype=np.float32)
            pred_skel_conf = np.ones((self.pred_len, 17), dtype=np.float32) * 0.8
            for t in range(self.obs_len):
                center = traj[t]
                offsets = np.random.randn(17, 2).astype(np.float32) * 0.3
                skel_kp[t] = center + offsets
            for t in range(self.pred_len):
                center = traj[self.obs_len + t]
                offsets = np.random.randn(17, 2).astype(np.float32) * 0.3
                pred_skel_kp[t] = center + offsets

            self.samples.append({
                'obs_traj': traj[:self.obs_len],
                'pred_traj': traj[self.obs_len:],
                'skeleton_kp': skel_kp,
                'skeleton_conf': skel_conf,
                'pred_skeleton_kp': pred_skel_kp,
                'pred_skeleton_conf': pred_skel_conf,
                'has_skeleton': True,
                'window_origin': np.zeros(2, dtype=np.float32),
            })

        print(f"[Dataset] {self.split}: {len(self.samples)} synthetic samples")

    @staticmethod
    def _parse_skel_conf(skel_conf):
        cf = np.asarray(skel_conf, dtype=np.float32)
        cf = np.squeeze(cf)
        if cf.ndim == 3 and cf.shape[-2] == 1:
            cf = cf[:, 0, :]
        elif cf.ndim == 3 and cf.shape[-1] == 1:
            cf = cf[:, :, 0]
        return cf

    @classmethod
    def _parse_skel_window_pixels(cls, skel_window, squeeze_time=True):
        """
        解析 (T, 17, 3) 或含多余维的骨骼窗 → 像素 (T,17,2) 与 (T,17) 置信度。
        用于 pose_change 过滤（整 obs+pred 窗，不做 traj_scale）。
        """
        if skel_window is None or np.all(skel_window == 0):
            return None, None
        raw = np.asarray(skel_window, dtype=np.float32)
        if squeeze_time and raw.ndim >= 4:
            raw = np.squeeze(raw)
        if raw.ndim == 3 and raw.shape[-1] >= 3:
            kp = cls._squeeze_skel_to_tv2(raw[..., :2])
            cf = cls._parse_skel_conf(raw[..., 2])
        elif raw.ndim == 3 and raw.shape[-1] == 2:
            kp = cls._squeeze_skel_to_tv2(raw)
            cf = np.ones(kp.shape[:2], dtype=np.float32)
        else:
            return None, None
        if kp.ndim != 3 or kp.shape[-1] != 2 or kp.shape[1] < 17:
            return None, None
        if cf.ndim != 2:
            cf = cf.reshape(kp.shape[0], -1)
        if cf.shape[1] < 17:
            pad = np.zeros((cf.shape[0], 17 - cf.shape[1]), dtype=np.float32)
            cf = np.concatenate([cf, pad], axis=1)
        return kp[:, :17, :], cf[:, :17]

    @staticmethod
    def _squeeze_skel_to_tv2(skel_kp):
        """将骨骼数组规整为 (T, V, 2)"""
        x = np.asarray(skel_kp, dtype=np.float32)
        x = np.squeeze(x)
        while x.ndim > 3:
            if x.shape[-1] in (2, 3) and x.shape[-2] == 2:
                x = x[..., 0, :]
                continue
            if x.shape[0] == 1:
                x = x[0]
                continue
            if x.shape[-1] == 1:
                x = x[..., 0]
                continue
            if x.ndim == 4 and x.shape[1] == 1:
                x = x[:, 0, :, :]
                continue
            if x.ndim == 4 and x.shape[2] == 1:
                x = x[:, :, 0, :]
                continue
            if x.ndim == 4 and x.shape[1] in (1, 2) and x.shape[2] == 17 and x.shape[-1] in (2, 3):
                x = x[:, 0, :, :]
                continue
            if x.ndim == 4 and x.shape[-1] in (2, 3):
                t, a, b, c = x.shape
                if a * b == 17:
                    x = x.reshape(t, 17, c)
                    continue
            break
        if x.ndim == 3 and x.shape[-1] == 3:
            x = x[..., :2].copy()
        return x

    def _augment(self, obs, pred, skel_kp, pred_skel_kp=None):
        """数据增强: 随机旋转和翻转"""
        skel_kp = self._squeeze_skel_to_tv2(skel_kp)
        if skel_kp.ndim != 3 or skel_kp.shape[-1] != 2:
            raise ValueError(
                f"skeleton_kp 形状应为 (T_obs, V, 2)，当前 {np.asarray(skel_kp).shape}；"
                "请检查 skeletons/*.npy 是否为 (F,P,17,3) 且与 trajectories 的 P 一致。"
            )

        theta = np.random.uniform(0, 2 * np.pi)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]], dtype=np.float32)

        obs = obs @ R.T
        pred = pred @ R.T
        t, v, two = skel_kp.shape
        if two != 2:
            raise ValueError(f"skeleton 最后一维应为 2 (x,y)，当前 shape={skel_kp.shape}")
        skel_kp = skel_kp.reshape(-1, 2) @ R.T
        skel_kp = skel_kp.reshape(t, v, 2)

        if pred_skel_kp is not None:
            pred_skel_kp = self._squeeze_skel_to_tv2(pred_skel_kp)
            tp, vp, _ = pred_skel_kp.shape
            pred_skel_kp = pred_skel_kp.reshape(-1, 2) @ R.T
            pred_skel_kp = pred_skel_kp.reshape(tp, vp, 2)

        if np.random.random() > 0.5:
            obs[:, 0] = -obs[:, 0]
            pred[:, 0] = -pred[:, 0]
            skel_kp[:, :, 0] = -skel_kp[:, :, 0]
            if pred_skel_kp is not None:
                pred_skel_kp[:, :, 0] = -pred_skel_kp[:, :, 0]

        if pred_skel_kp is not None:
            return obs, pred, skel_kp, pred_skel_kp
        return obs, pred, skel_kp

    def _canonicalize_skel_batch_fields(self, skel_kp, skel_conf):
        T, V = self.obs_len, 17
        kp = np.asarray(skel_kp, dtype=np.float32)
        cf = np.asarray(skel_conf, dtype=np.float32)
        kp = self._squeeze_skel_to_tv2(kp)
        while kp.ndim > 3:
            kp = np.squeeze(kp)
        while cf.ndim > 2:
            cf = np.squeeze(cf)
        if kp.shape != (T, V, 2):
            if kp.size == T * V * 2:
                kp = kp.reshape(T, V, 2)
            else:
                raise ValueError(
                    f"skeleton_kp 无法规整为 ({T},{V},2)，当前 shape={kp.shape}"
                )
        if cf.shape != (T, V):
            if cf.size == T * V:
                cf = cf.reshape(T, V)
            else:
                raise ValueError(
                    f"skeleton_conf 无法规整为 ({T},{V})，当前 shape={cf.shape}"
                )
        return kp, cf

    def _canonicalize_pred_skel_batch_fields(self, skel_kp, skel_conf):
        T, V = self.pred_len, 17
        kp = np.asarray(skel_kp, dtype=np.float32)
        cf = np.asarray(skel_conf, dtype=np.float32)
        kp = self._squeeze_skel_to_tv2(kp)
        while kp.ndim > 3:
            kp = np.squeeze(kp)
        while cf.ndim > 2:
            cf = np.squeeze(cf)
        if kp.shape != (T, V, 2):
            if kp.size == T * V * 2:
                kp = kp.reshape(T, V, 2)
            else:
                raise ValueError(
                    f"pred_skeleton_kp 无法规整为 ({T},{V},2)，当前 shape={kp.shape}"
                )
        if cf.shape != (T, V):
            if cf.size == T * V:
                cf = cf.reshape(T, V)
            else:
                raise ValueError(
                    f"pred_skeleton_conf 无法规整为 ({T},{V})，当前 shape={cf.shape}"
                )
        return kp, cf

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        obs = sample['obs_traj'].copy()
        pred = sample['pred_traj'].copy()
        skel_kp = sample['skeleton_kp'].copy()
        skel_conf = sample['skeleton_conf'].copy()
        pred_skel_kp = sample.get('pred_skeleton_kp', np.zeros((self.pred_len, 17, 2), np.float32)).copy()
        pred_skel_conf = sample.get('pred_skeleton_conf', np.zeros((self.pred_len, 17), np.float32)).copy()

        last_obs = obs[-1:, :].copy()
        obs = obs - last_obs
        pred = pred - last_obs

        if self.augment:
            obs, pred, skel_kp, pred_skel_kp = self._augment(obs, pred, skel_kp, pred_skel_kp)

        skel_kp, skel_conf = self._canonicalize_skel_batch_fields(skel_kp, skel_conf)
        pred_skel_kp, pred_skel_conf = self._canonicalize_pred_skel_batch_fields(
            pred_skel_kp, pred_skel_conf,
        )

        wo = sample.get('window_origin')
        if wo is None:
            wo = np.zeros(2, dtype=np.float32)
        else:
            wo = np.asarray(wo, dtype=np.float32).reshape(2)

        return {
            'obs_traj': torch.from_numpy(obs),
            'pred_traj': torch.from_numpy(pred),
            'skeleton_kp': torch.from_numpy(skel_kp),
            'skeleton_conf': torch.from_numpy(skel_conf),
            'pred_skeleton_kp': torch.from_numpy(pred_skel_kp),
            'pred_skeleton_conf': torch.from_numpy(pred_skel_conf),
            'has_skeleton': sample['has_skeleton'],
            'last_obs_pos': torch.from_numpy(last_obs.squeeze(0)),
            'window_origin': torch.from_numpy(wo.copy()),
            'domain': self.domain_name,
            'scene_id': sample.get('scene_id', ''),
            'window_start': int(sample.get('window_start', -1)),
            'ped_id': int(sample.get('ped_id', -1)),
        }


def _worker_init_fn_for_seed(worker_id, base_seed):
    if base_seed is None:
        return
    s = int(base_seed) + int(worker_id)
    np.random.seed(s)
    import random as _r
    _r.seed(s % (2 ** 32))


def _resolve_motion_filter_cfg(data_cfg, split, motion_filter_enabled_override=None):
    """合并 track_quality；val/test 可单独使用 motion_filter_eval。"""
    mf_src = data_cfg.get('motion_filter') or {}
    if split in ('val', 'test') and data_cfg.get('motion_filter_eval'):
        mf_src = data_cfg['motion_filter_eval']
    mf = dict(mf_src)
    if motion_filter_enabled_override is not None:
        mf['enabled'] = bool(motion_filter_enabled_override)
    tq = dict(data_cfg.get('track_quality') or {})
    if tq and 'track_quality' not in mf:
        mf = {**mf, 'track_quality': tq}
    elif tq:
        mf['track_quality'] = {**(mf.get('track_quality') or {}), **tq}
    return mf


def _make_jrdb_dataset(data_cfg, data_dir, split, motion_filter):
    return JRDBTrajectoryDataset(
        data_dir=data_dir,
        split=split,
        obs_len=data_cfg['obs_len'],
        pred_len=data_cfg['pred_len'],
        skip=data_cfg.get('skip', 1),
        min_ped=data_cfg.get('min_ped', 1),
        augment=(split == 'train'),
        traj_scale=data_cfg.get('traj_scale', _DEFAULT_TRAJ_SCALE),
        motion_filter=motion_filter,
        track_quality=data_cfg.get('track_quality'),
    )


def build_dataloader(cfg, split='train', motion_filter_enabled_override=None):
    """构建 DataLoader；支持 data_dirs 多域合并（ConcatDataset）。"""
    data_cfg = cfg['data']
    train_cfg = cfg['training']
    mf = _resolve_motion_filter_cfg(data_cfg, split, motion_filter_enabled_override)

    train_seed = train_cfg.get('seed')
    generator = None
    worker_init_fn = None
    if split == 'train' and train_seed is not None:
        g = torch.Generator()
        g.manual_seed(int(train_seed))
        generator = g
        worker_init_fn = lambda wid: _worker_init_fn_for_seed(wid, int(train_seed))

    data_dirs = data_cfg.get('data_dirs')
    if data_dirs:
        parts = [
            _make_jrdb_dataset(data_cfg, str(d), split, mf)
            for d in data_dirs
        ]
        dataset = ConcatDataset(parts)
        print(
            f"[DataLoader] split={split} | merged {len(parts)} domains | "
            f"total_samples={len(dataset)}"
        )
    else:
        dataset = _make_jrdb_dataset(
            data_cfg, data_cfg['data_dir'], split, mf,
        )

    loader = DataLoader(
        dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=(split == 'train'),
        num_workers=data_cfg.get('num_workers', 4),
        pin_memory=True,
        drop_last=(split == 'train'),
        generator=generator,
        worker_init_fn=worker_init_fn,
    )

    return loader
