# 诊断代码

可执行入口使用本包根目录的 `models/`、`data/`、`utils/`、`configs/`。这些文件来自实际诊断目录 `ctr_gcn_traj/ctr_gcn_traj/ctr_gcn_traj/`，导入名是 `models.full_model`、`models.fusion`、`models.cvae`，与公开仓 `gd_cvae.py` 不同。公开仓副本在 `provenance/public_gd_cvae_extract/`，与诊断执行版本分开存放。

## 模块

| 文件 | 作用 |
|---|---|
| `eval_nj_cell.py` | 四条件正式单格：在同一窗口评估 O / F / N_J / N |
| `eval_nj_stageA.py` | N_J 阶段 A |
| `eval_u1f_cell.py` | O/F lock-\(J_t\) 单格；F 的 confidence 赋值与 `predict_locked` |
| `eval_nf_same_session_diag.py` | 同会话 N/F 诊断（历史 2 格，不是 40 格主表） |
| `u1_common.py` | N：整幅 confidence 换为 donor，坐标保留 target |
| `e1_rev1_common.py` | donor map 加载、ADE/FDE、哈希 |
| `e123_common.py` | 窗口键 |
| `e2ext_common.py` | `u1_common` 引用的掩码比较 |
| `construct_u1f_f_input_gate.py` | F 输入闸门 |
| `replay_u1f_g4g5_preproc_sanitize.py` | 预处理与 sanitizer 回放 |
| `analyze_u1f_40.py` | U1F 40 格分析 |

O：target 原始坐标与 confidence，走官方预处理。  
F：与 N_J 使用同一套按 `c > 0.1` 选出的 `c_J`，回放并锁定 O 的处理后 xy 与掩码结构，只改指定 confidence 通道。  
N_J：`J_t = (c_t > 0.1)`，`c_J = where(J_t, c_d, c_t)`，再按官方规则预处理。阈值是严格大于 0.1。  
N：整幅 confidence 赋为 donor，坐标仍是 target，再按官方规则预处理。

推理是 fusion 分支上的确定性 prior-mean（`most_likely`，K=1）。各条件分别使用其条件先验均值，不共用同一个 z。

## 单格命令

以下命令使用已纳入本包的 s42 权重及 d42 配对，在 Linux shell 中执行。其他种子与权重路径见 `checkpoints/REGISTRY.md`；本地尚未完成模型加载或推理验证。

```bash
cd mva_confidence_diagnostic
export PYTHONPATH=".:diagnostics"
export CTR_GCN_TRAJ_MOTION_CACHE_ROOT="$PWD/motion_filter_cache"
python diagnostics/eval_nj_cell.py \
  --config configs/jrdb_v2_decoupled_z.yaml \
  --checkpoint checkpoints/jrdb_v2_decoupled_z_s42_20260619_1432/best.pt \
  --train_seed 42 \
  --donor_assign_seed 42 \
  --diagnostic_set_json protocols/diagnostic_BV/diagnostic_set_BV.json \
  --donor_map_json protocols/donor_maps/donor_pairs_B_d42.json \
  --out_dir "$PWD/local_runs/nj_s42_d42" \
  --gpu 0 \
  --expected_ckpt_sha256 94cb5326200fa2544340c87d59200b9e15f397c66fe99e39a6727f885c6ce07f
```

可通过 `CTR_GCN_TRAJ_MOTION_CACHE_ROOT` 指定缓存目录。默认缓存为包内 `local_runs/motion_filter_cache/`。

配置里的 `data_dir` 是 `./data/jrdb_processed`，相对于当前工作目录。

## 批量推理与统计

Linux 批量入口：`bash diagnostics/runners/run_nj_40.sh`。默认读取 `protocols/` 和 `checkpoints/`，结果写入 `local_runs/nj_same_assign_v1/`。可通过 `PYTHON`、`GPU`、`OUT_ROOT`、`TRAIN_SEEDS` 和 `DONOR_SEEDS` 指定运行参数。完整协议仍为八种子、五 donor、每格 15,160 窗；子集运行不构成完整复现。

历史汇总脚本以自身位置定位包根目录，默认读取 `statistics/`，输出写入 `local_runs/`。可用 `DIAGNOSTIC_INPUT_ROOT` 和 `DIAGNOSTIC_OUTPUT_ROOT` 分别指定输入和输出。例如汇总新生成的 NJ 结果：

```bash
DIAGNOSTIC_INPUT_ROOT="$PWD/local_runs/nj_same_assign_v1" python diagnostics/runners/45_nj_aggregate.py
```

历史 `run_u1f_40.sh` 需要原始 E2/U1 参考 CSV，本核心包未提供这些表。脚本会检查依赖并提示缺失；它不作为自包含四条件推理入口。

## 模型加载与冒烟检查

```bash
python -B diagnostics/smoke_inference.py --train-seeds 42 --max-targets 8
```

入口使用测试加载器、冻结 d42 配对和 s42 权重，执行 O/F/N_J/N 的确定性 K=1 推理，并检查严格权重加载、形状、有限值、重复推理和锁定处理回放。保持原加载器批次大小，从首个含有效目标–donor 配对的批次选取至多八个目标窗口。检查完整批次的前向结果，仅保存选定目标窗口；不用于估计论文效应。

结果写入 `local_runs/smoke/SMOKE_REPORT.json`。可用 `--train-seeds 42,43,44,45,46,47,48,49` 检查全部权重。输出不写入论文正式统计目录，不生成正式完成标记。

2026-10-16 本地尝试因缺少 PyTorch 和 PyYAML 返回 `BLOCKED`，模型加载及前向推理尚未执行。
