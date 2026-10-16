# 骨架条件轨迹预测：四条件诊断复现包

对应论文：*Processing-Aware Confidence Interventions in Skeleton-Conditioned Trajectory Prediction*。

本包提供模型 B 在 O、F、N_J、N 四种输入条件下的固定权重诊断材料，包括执行代码、处理后数据、八种子权重、冻结评估协议、逐窗结果和绘图源文件。

拟沿用的分发仓库：

<https://github.com/Myfsldts20250101/Skeleton-Enhanced-Pedestrian-Trajectory-Prediction-for-Human-Robot-Collaboration-via-Gated-Fusion-an>

本地整理未向该仓库推送文件。

## 文件目录

| 路径 | 内容 |
|---|---|
| `models/`、`data/`、`utils/`、`configs/jrdb_v2_decoupled_z.yaml` | 诊断使用的模型、加载器、工具和配置 |
| `diagnostics/` | 四条件推理、F 回放、输入检查及汇总脚本 |
| `checkpoints/` | 训练种子 42–49 的八个 B 权重及哈希登记 |
| `protocols/` | 15,160 个目标窗口、五份 donor 配对及冻结记录 |
| `data/jrdb_processed/` | 处理后的轨迹与 COCO-17 骨架数组 |
| `statistics/nj_same_assign_v1/` | 40 个种子–donor 组合的四条件逐窗结果 |
| `statistics/u1f_lock_Jt_v1/` | 较早的 O/F 锁定处理实验结果 |
| `statistics/repro/` | 核心统计复算程序、输入表及参考结果 |
| `figures/` | 历史图、投稿图源文件和重绘脚本 |
| `archived_analyses/` | B–HST 事后联合统计 |
| `provenance/` | 文件来源、完整性清单、核验记录及公开仓源码副本 |

## 使用步骤

1. 核对 `provenance/SHA256SUMS.txt`。八个 B 权重的文件哈希已与原诊断脚本登记值逐一核对，详见 `checkpoints/REGISTRY.md`。
2. 准备运行环境。`requirements-lock.txt` 是 2026-07-30 导出的历史 pip 清单（Python 3.12.3，`torch==2.11.0`）；`environment.yml` 来自公开 `gd-cvae`，仅列宽泛依赖。两者均未在新环境中完成安装验证，历史 GPU、驱动和 CUDA 组合未记录。
3. 确认 `data/jrdb_processed/{train,val,test}/{trajectories,skeletons}/`。数据字段和导出程序见 `processed_data/README.md`。
4. 使用 `protocols/diagnostic_BV/diagnostic_set_BV.json` 及 `protocols/donor_maps/donor_pairs_B_d42.json` 至 `d46.json`。八个训练种子共用这五份冻结配对，复现时保持名册和配对不变。
5. 按 `diagnostics/README.md` 运行单格推理，设置 `PYTHONPATH` 和 `CTR_GCN_TRAJ_MOTION_CACHE_ROOT`。
6. 按 `statistics/repro/README.md` 复算核心统计；按 `figures/README.md` 重绘投稿图。总体统计先在种子内平均五个 donor 结果，再报告八种子均值和样本标准差；窗口加权与场景等权分别报告。

## 复现范围与限制

本包支持核心 B 四条件诊断。文件哈希核验已完成；本地整理尚未完成模型加载、前向推理及绘图环境验证。核心 NJ 批量入口及历史汇总脚本已改为包内路径，新增输出默认写入 `local_runs/`。历史 U1F 回放仍需要本包未包含的 E2/U1 参考表。

历史 F 响应分位图保留十组汇总，未提供原始 80,000 行抽样记录。HST 材料仅包含事后统计，缺少完整前向推理所需的权重、代码和环境。原始 JRDB 图像与视频、消融权重、训练日志和 `.git` 未纳入。各项状态见 `provenance/GAPS.md`。

## 许可

随包代码按 `LICENSE` 所列 Apache-2.0 分发。该许可不覆盖 JRDB 衍生数组、八个 B 权重或 `gd_cvae_full_public_single.pt`；相关材料须分别遵守其适用条款。

`data/indices_endpoint_0.5_background/` 是包含 29,158 个测试窗口的历史背景索引，核心诊断使用的名册为 15,160 窗。
