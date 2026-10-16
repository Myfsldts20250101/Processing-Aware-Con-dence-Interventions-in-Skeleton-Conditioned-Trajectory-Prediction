# B 检查点登记

本文核心诊断使用训练种子 42–49 的八个 **B**（解耦 z）检查点。推理为 fusion 分支、确定性 prior-mean、`most_likely`、K=1。选择规则与文中验证集 moving minADE@20 一致。目录名和参考 SHA-256 来自实际运行脚本 `diagnostics/runners/run_u1f_40.sh` 与 `run_nj_40.sh`。

八个文件于 2026-10-15 从 `J:/2026/paper/source/best-8/` 按原目录结构复制至 `checkpoints/`。本包文件的 SHA-256 均与原诊断脚本登记值一致；尚未执行本地模型加载或前向推理。

| train_seed | model | 原相对路径 | 已核验 SHA-256 | 本包内文件 |
|---|---|---|---|---|
| 42 | B | `checkpoints/jrdb_v2_decoupled_z_s42_20260619_1432/best.pt` | `94cb5326200fa2544340c87d59200b9e15f397c66fe99e39a6727f885c6ce07f` | 已核验 |
| 43 | B | `checkpoints/jrdb_v2_decoupled_z_s43_paired3/best.pt` | `7767867582a9dff71ceb787da149023c419e4faec4d984b542e777a79c6339fe` | 已核验 |
| 44 | B | `checkpoints/jrdb_v2_decoupled_z_s44_paired3/best.pt` | `15d12a1fa535de7b713a726dc342ac01a449b73eab09b7033196c8cf77393374` | 已核验 |
| 45 | B | `checkpoints/jrdb_v2_decoupled_z_s45_paired5/best.pt` | `eb2f367fe9aa6e197180562978ebc7053fc2c8f0bfed3ddad4feece5e8142d27` | 已核验 |
| 46 | B | `checkpoints/jrdb_v2_decoupled_z_s46_paired5/best.pt` | `83d3a816765a1b55a8b9b911840d817ae2592aa610bad1735a59992f52cbdde0` | 已核验 |
| 47 | B | `checkpoints/jrdb_v2_decoupled_z_s47_paired8/best.pt` | `5257aad963453a7f971137a9504b95d970b716beb3cb0e8df99e5f6771966d60` | 已核验 |
| 48 | B | `checkpoints/jrdb_v2_decoupled_z_s48_paired8/best.pt` | `bf59925031a14672858518fe0268eea37408b59f38eff4a47e73e3755673eb38` | 已核验 |
| 49 | B | `checkpoints/jrdb_v2_decoupled_z_s49_paired8/best.pt` | `39525c1aa7f612bb96787bad7bc727fd6e8b8ed48eb7018e4cfdde2547b59465` | 已核验 |

检查点对应的配置副本为 `configs/jrdb_v2_decoupled_z.yaml`。公开仓的 `configs/gd_cvae_full_public.yaml` 仅作公开版本对照，不用于本轮诊断。

## 公开单检查点

`checkpoints/gd_cvae_full_public_single.pt` 来自 `gd-cvae/checkpoints/gd_cvae_full.pt`。大小 11,398,351 字节，SHA-256 `de38823faabb95a5ce976d1e1b9b590b5491b43ee507386107b37a17a675889f`。它是公开主模型 B 的单个权重，不是种子 42–49 的诊断检查点集合。消融权重（concat、no_gate、no_skeleton、shared_z 等）未纳入本包。
