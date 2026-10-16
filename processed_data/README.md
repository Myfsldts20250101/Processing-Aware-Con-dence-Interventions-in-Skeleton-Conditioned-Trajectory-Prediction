# 处理后轨迹与骨架数据

加载器读取 `data/jrdb_processed/`，对应配置 `configs/jrdb_v2_decoupled_z.yaml` 中的 `data.data_dir: ./data/jrdb_processed`。本目录仅提供说明，不另存数组副本。

## 数据来源与处理

原始数据来自 JRDB：<https://jrdb.erc.monash.edu/>。本包不含原始图像和视频。

| 项目 | 实现或定义 |
|---|---|
| 轨迹导出 | `data/preprocess/export_jrdb_stitched_labels_to_npy.py`；读取 `labels_2d_stitched` JSON，将框 `[u,v,w,h]` 的中心写为像素坐标 |
| 骨架导出 | `data/preprocess/export_jrdb_pose_coco_to_skeleton_npy.py`；COCO-17 关节格式 |
| 场景划分 | `data/jrdb_scene_splits.json` 与 `configs/jrdb_scene_splits.json`；使用 `official` 的 train / val / test 划分 |
| 窗口设置 | 观测 8 帧，预测 12 帧，`skip=1` |
| 坐标与指标 | 加载后使用 agent-centric 相对坐标，`traj_scale=1000`；指标单位为缩放后的图像平面单位 |
| 加载与预处理 | `data/jrdb_dataset.py` 及 `models/ctr_gcn.py` 的 `SkeletonPreprocessor`；包括髋部中心、无效关节 xy 置零和 confidence 掩码 |
| 诊断输入赋值 | `diagnostics/eval_nj_cell.py` |

公开仓副本中的 `provenance/public_gd_cvae_extract/data/preprocess_jrdb.py` 是占位脚手架，不用于上述转换。

## 文件布局与窗口名册

```text
data/jrdb_processed/{train,val,test}/{trajectories,skeletons}/*.npy
```

这些文件是场景级数组。核心诊断的 15,160 个目标窗口由 `protocols/diagnostic_BV/diagnostic_set_BV.json` 的 `target_keys` 指定，键解析见 `diagnostics/e123_common.py` 的 `batch_window_keys`，donor 对应关系见冻结配对文件。

`data/indices_endpoint_0.5_background/` 保留 29,158 个测试窗口的历史 moving 索引，不用于替代核心诊断名册。文件哈希见 `provenance/SHA256SUMS.txt`。

## 使用条件

处理后数组衍生自 JRDB，再分发须遵守 JRDB 官方条款。代码的 Apache-2.0 许可不覆盖这些数组或模型权重。
