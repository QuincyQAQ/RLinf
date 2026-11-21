# Go2 A40：A2、B1、B2 三任务实验

[English](README.md)

本目录只保留 A2 全向移动、B1 粗糙地形、B2 楼梯的实验入口、配置、权重和结果，来自已经核验的 A40 五任务归档。原五任务目录和 GitHub 分支保持原样。

表格和逐回合评测集中在 `reports/`。 [完整 CSV](reports/A40_A2_B1_B2_20261121.csv) 有 13 行、19 列，包含三个任务的主实验、B1 固定 policy 补测和 B2 的 MSE 分支；训练 seed 与评测 seed 分列。

| 路径 | 内容 |
|---|---|
| `reports/A40_A2_B1_B2_20261121.csv` | 成功计数、跟踪成功率、速度误差、失败终止率、存活时间、实际通过率和 seed |
| `reports/three_task_final_20261121.json` | 三个任务主实验的协议与结果 |
| `reports/completed_b1_confirmatory_14036295.json` | B1 新评测 seed 的三组补测与合并统计 |
| `reports/raw/`、`reports/raw_index.json` | 28 份逐回合评测的集中副本和原路径索引；包含对应地形几何文件 |
| `outputs/` | 为运行兼容而保留的原评测路径、命令、配置和日志 |
| `benchmark/logs/rsl_rl/` | 普通 PPO 权重与配置；RLinf 权重在对应 `outputs/` 下 |
| `a40/`、`config/` | 三任务入口、容器工具与配置 |
| `RLinf/`、`benchmark/`、`IsaacLab/` | 固定提交的公共依赖子模块 |
| `provenance.json`、`integrity.json` | 来源、筛选范围、范围调整记录与文件校验值 |

`reports/raw/` 是原始文件的逐字节副本。`outputs/` 中仍保留一份，是为了让原评测命令和训练脚本继续找到原来的路径。两者由离线核验检查一致性。

## 实验设置与结果

三组均使用训练 seed 0：完整普通 PPO 源权重、继续普通 PPO、继续 RLinf-Huber。续训从相同源权重开始，各追加 1000 次更新，使用新优化器和新环境。主评测 seed 为 23456、34567，分别评测后按回合数合并。

| 任务 | 源训练更新 | 续训更新 | 并行环境 | 原 PPO 跟踪成功率 | 继续普通 PPO | 继续 RLinf-Huber |
|---|---:|---:|---:|---:|---:|---:|
| A2 全向移动 | 1500 | 1000 | 2048 | 1887/2063（91.4687%） | 1917/2054（93.3301%） | 2001/2049（97.6574%） |
| B1 粗糙地形 | 3000 | 1000 | 2048 | 2045/2048（99.8535%） | 2040/2048（99.6094%） | 2044/2048（99.8047%） |
| B2 楼梯 | 2200 | 1000 | 1024 | 2045/2048（99.8535%） | 2048/2048（100%） | 2048/2048（100%） |

B2 的实际楼梯通过率分别为 **438/2048（21.3867%）**、**918/2048（44.8242%）**、**1568/2048（76.5625%）**；补充 RLinf-MSE 分支为 **958/2048（46.7773%）**。跟踪和楼梯实际通过的同回合交集在 CSV 中独立列出。

B1 另用评测 seed 45678 固定权重补测，各组 2048 回合，三组均为 2044/2048。该补测与主实验分行保存。B1 使用有效的 3000 更新恢复权重，三组统一使用 ±100 的动作／历史限幅；恢复来源为作业 14032252。

Actor 与 Critic 均接收 235 维观测，使用 ELU 隐藏层 `[512,256,128]`，关闭观测归一化。Actor 输出 12 维关节动作。物理频率为 200 Hz，policy 运行频率为 50 Hz，rollout 24 步，5 个更新 epoch、4 个 minibatch。RLinf-Huber 使用学习率 `3e-4`、熵系数 `0.003`、目标 KL `0.005`、Huber delta 10。B2 续训保留后期姿态奖励权重。

跟踪成功要求没有失败终止，且达标步比例达到任务阈值。A2／B1／B2 的线速度阈值依次为 0.25／0.35／0.40 m/s，偏航角速度阈值为 0.50／0.60／0.70 rad/s，达标步比例为 80%／75%／70%。B2 实际通过要求沿 +x 越过完整楼梯、四足在远端落地至少 0.2 秒并在该回合结束前没有失败终止。

## 获取代码与核验结果

用 Git 获取独立分支及固定版本的公共源码，再用 Python 3.11 恢复生成文件和复算结果。

```bash
git clone --single-branch --branch experiment/go2-a40-a2-b1-b2-20261121 \
  --recurse-submodules https://github.com/QuincyQAQ/RLinf.git go2-a40-three-tasks
cd go2-a40-three-tasks
python3.11 tools/restore_runtime.py
python3.11 tools/audit_results.py
```

`restore_runtime.py` 将 `artifacts/benchmark_runtime.tar.gz` 中的生成文件恢复到公共源码旁边，并恢复 17 份 checkpoint 的原修改时间。重复执行会验证已有文件；遇到被改动的文件会停止。已有本地目录可以直接运行这两条 Python 命令。

核验包括文件哈希、三个依赖提交、任务范围、原始逐回合统计、成功判定、汇总与原始记录的一致性、集中副本及 CSV 复算。通过后会输出 `"status": "passed"`、`"raw_evaluations": 28`、`"episodes": 31771`、`"csv_rows": 13`。记录写入 `validation/subset_audit.json`。重新生成结果表：

```bash
python3.11 reports/export_results.py
```

## 后续运行入口

将本目录作为新的工作目录，按 `environment_lock.json` 准备原容器、第三方依赖和资源。`a40/hpc_environment.sh` 会将当前目录挂载至 `/workspace/go2`。可复用服务器已验证的容器、资产和 overlay 安装，公共源码固定在 `provenance.json` 记录的提交。

计划在 GPU 上重跑实验时，使用以下命令：

```bash
sbatch a40/fullppo_huber.sbatch A2
sbatch a40/b1_recovered.sbatch
sbatch a40/b2_continuation.sbatch
```

A2、B1 检查原配置并验证训练更新。B2 复用相同协议的检查记录；环境有变化时先用 `a40/b2_continuation_gate.sbatch` 检查。B1 固定权重补测入口为 `a40/b1_confirmatory_eval.sbatch`。具体命令、环境变量和 checkpoint 在各 `outputs/.../command.json` 及 `protocol.json` 中。

## 来源与维护

本分支以 A40 在 2026-11-21 的快照为依据。`provenance.json` 记录原归档提交、三个固定依赖提交和范围调整；`integrity.json` 保存交付文件的 SHA256。公共源码使用 Git 子模块，生成的普通 PPO 文件使用单独归档，克隆后按上述命令恢复。

范围调整只移除无关任务和历史中途切换入口，筛选配置，并把独立探针的默认任务改成 A2。奖励、控制、PPO 数值参数、权重和已有评测值保持原样。GitHub 的 CPU 工作流恢复文件、复算结果并检查脚本语法。维护要求见 `AGENTS.md`，许可证与第三方归属见 `LICENSE`、`NOTICE`。

## Limitations

- 本次仅整理现有代码与记录并运行离线检查，没有新申请 GPU、执行训练、物理仿真或真机动作；重新训练不能保证权重与成绩逐位相同。
- 三个公共依赖保留固定提交的完整源码，因此上游源码内部仍有其他机器人、环境和任务的注册实现；本目录的实验配置、启动任务、结果表和权重仅保留 A2、B1、B2。
- 完整容器、已安装第三方包、资产缓存和全部历史中间权重不在本目录。保留了 17 份必要权重、版本锁定和资源清单。B2 历史曲线应使用 `outputs/b2_postbaseline_14025574/curve/` 中记录的显式权重路径，不能对权重子集重新计算曲线抽样位置。
- 每种方法只有训练 seed 0；新增评测 seed 是固定 policy 的评测重复。B1 基线含已记录的数值修复；Huber loss 和超参数也有差异，因此不能将结果差异全部归因于框架。
