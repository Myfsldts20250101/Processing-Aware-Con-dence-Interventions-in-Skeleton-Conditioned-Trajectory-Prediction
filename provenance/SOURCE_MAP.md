# 文件来源

初次打包日期：2026-10-15，服务器暂存目录为 `work/back/mva_confidence_diagnostic/`。初次打包从 `/root/work` 复制，未跟随指向 `/root/autodl-tmp` 的符号链接。后续本地补入的权重和绘图源另列于下方。

| 包内路径 | 来源 |
|---|---|
| `models/`、`utils/`、`data/jrdb_dataset.py`、`data/__init__.py` | `ctr_gcn_traj/ctr_gcn_traj/ctr_gcn_traj/` 对应目录 |
| `data/jrdb_scene_splits.json`、`configs/jrdb_scene_splits.json`、`configs/jrdb_v2_decoupled_z.yaml` | 同上 `configs/` |
| `configs/gd_cvae_full_public.yaml` | `gd-cvae/configs/gd_cvae_full.yaml` |
| `data/jrdb_processed/` | 执行目录 `data/jrdb_processed/` |
| `data/preprocess/*.py` | `ctr_gcn_traj/scripts/export_jrdb_stitched_labels_to_npy.py`、`export_jrdb_pose_coco_to_skeleton_npy.py` |
| `data/indices_endpoint_0.5_background/` | `gd-cvae/data/indices/`（测试 moving 29,158，背景材料） |
| `diagnostics/*.py` | 执行目录同名文件 |
| `diagnostics/runners/` | `ctr_gcn_traj/scripts/` 中 U1F 与 N_J 的 py/sh |
| `protocols/diagnostic_BV/`、`protocols/donor_maps/`、`protocols/freeze/` | `eval_logs/e123/` 下同名目录 |
| `statistics/u1f_lock_Jt_v1/` | `eval_logs/u1f_lock_Jt_v1/` |
| `statistics/nj_same_assign_v1/` | `eval_logs/nj_same_assign_v1/` |
| `figures/u1f_joint_response_delta/`、`figures/nj_seed_donor_range/` | 上述两个实验目录中的图 |
| `archived_analyses/hst_posthoc_bhst_u1f_20261004/` | `eval_logs/posthoc_bhst_u1f_20261004/` |
| `environment.yml`、`LICENSE` | `gd-cvae/` |
| `requirements-lock.txt` | `conda_envs_backup/ctr_gcn_traj.requirements.txt`（2026-07-30 导出） |
| `checkpoints/gd_cvae_full_public_single.pt` | `gd-cvae/checkpoints/gd_cvae_full.pt` |
| `provenance/public_gd_cvae_extract/` | 公开仓源码副本；`preprocess_jrdb.py` 与 `eval.py` 不是四条件入口 |

未复制：`.git`、`__pycache__`、TensorBoard 事件、训练中间日志、消融权重压缩包、原始 JRDB 图像与视频、HST 权重。

SHA-256：`provenance/SHA256SUMS.txt`（打包完成后对包内文件计算，不含该清单自身）。

## 2026-10-15 补入八种子 B 权重

来源：`J:/2026/paper/source/best-8/`。保留各种子原目录名，复制到 `checkpoints/`；8/8 SHA-256 与诊断脚本登记一致。此记录只确认文件身份与完整性，不表示前向推理已重跑。

## 2026-10-15 补入绘图源

`figures/绘图源/` 完整复制自 `J:/2026/paper/论文02/输出/MVA_投稿整理_20261015/绘图源/`。源文件与脚本保持原样，逐文件 SHA-256 一致；未执行重绘。用法与输出路径见 `figures/README.md`。

## 统计复算材料

`statistics/repro/` 已包含核心统计复算程序、压缩输入表和冻结参考结果。该子包的输入哈希与来源连接记录见其中的 `derived_input_hashes.json`、`source_input_hashes.json` 和 `source_join_audit.json`。

## 2026-10-16 路径适配

修改批量入口、历史汇总和加载器缓存位置；新输出默认隔离于 `local_runs/`。原脚本备份在 `J:/2026/UGV_Controller/.codex_mva_stage_20261015/paths_before_20261016/`。模型结构、四条件定义、权重、冻结协议及归档结果未改动。新增 `diagnostics/smoke_inference.py` 用于小规模加载与推理检查。
