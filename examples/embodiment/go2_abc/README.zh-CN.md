# Go2 ABC 实验

[English](README.md)

在远程 Slurm A40 节点运行 Go2 全向移动（A）、粗糙地形（B）和楼梯（C）任务。每个任务包含已训练的 PPO 基线、普通 PPO 续训和 RLinf-Huber 续训。

两个续训组从同一任务的基线权重出发，各自新建优化器，训练 seed 默认为 0，增加 1000 轮更新。A、B 使用 2048 个并行环境，C 使用 1024 个。基线组只评测已保存权重；续训组训练后评测。评测 seed 为 23456 和 34567。

## 准备输入

在 Slurm 集群上进入干净的仓库根目录，将示例路径替换为已准备好的运行环境和共享资产路径。

```bash
export GO2_ABC_RUNTIME=/path/to/prepared-runtime
export GO2_ABC_SHARED_RUNTIME=/path/to/shared-runtime
export GO2_ABC_IMAGE="$GO2_ABC_SHARED_RUNTIME/containers/isaac-lab_2.3.2.sif"
export GO2_ABC_CONTAINER=singularity
```

运行环境包含 Isaac Lab 2.3.2、Isaac Sim 5.1、RSL-RL 3.1.2、benchmark、脚本、依赖包和资产。仿真任务与普通 PPO 使用固定版本的[多任务 benchmark](https://github.com/viktoriageorrg04/bsc-thesis-multi-task-RL-env)。运行环境中的 `benchmark` 和 `IsaacLab` 目录须保留 Git checkout，HEAD 与[清单](manifest.json)一致。其他集群须将这些变量设为各自兼容的运行环境、共享资产、镜像和容器程序路径。公开仓库本身不能独立重建这套运行环境。

按清单准备 A 任务的基线权重。

```bash
python -m examples.embodiment.go2_abc.prepare_data --task A
```

B、C 任务使用对应的权重和目标目录。启动器会按清单校验所选基线的哈希。

## 提交作业

在远程仓库根目录提交 A 任务的 RLinf-Huber 续训：

```bash
sbatch --gres gpu:a40:1 --cpus-per-task 4 --mem 64G --time 06:00:00 examples/embodiment/go2_abc/run_a40.sbatch A_rlinf
```

基线评测使用 `A_baseline`，普通 PPO 续训使用 `A_ppo`。准备 B 或 C 的基线后，将实验名中的 A 换成 B 或 C。其他集群须按实际情况调整分区、QoS、账户和资源。提交后的作业在分配的 A40 计算节点启动容器并运行。提交作业不代表运行完成；用 `squeue -j <job-id>` 或 `sacct -j <job-id>` 检查状态，并查看日志。

历史 A、B 续训在正式运行前还分别做了 64 和 2048 环境、各两轮的预检。此入口直接运行完整配置，不包含这些预检。

每组实验都要求 `data/go2_abc/runs/<experiment>/` 输出目录尚不存在。普通 PPO 还需要确保运行环境中的 `benchmark/logs/rsl_rl/<experiment>` 路径尚不存在。启动器会拒绝覆盖已有结果。

运行目录包含各阶段的 `console.log` 和评测结果；续训组还会导出 `policy.pt`。普通 PPO 原始训练日志和 checkpoint 保存在 `$GO2_ABC_RUNTIME/benchmark/logs/rsl_rl/<experiment>/`。Slurm 默认输出文件为 `slurm-<jobid>.out`。

## 配置实验

RLinf 续训共用 [config/_common.yaml](config/_common.yaml)，任务覆盖配置分别为 [A_rlinf.yaml](config/A_rlinf.yaml)、[B_rlinf.yaml](config/B_rlinf.yaml) 和 [C_rlinf.yaml](config/C_rlinf.yaml)。普通 PPO 使用 [A_ppo.json](config/A_ppo.json)、[B_ppo.json](config/B_ppo.json) 和 [C_ppo.json](config/C_ppo.json)；基线评测使用 [A_baseline.json](config/A_baseline.json)、[B_baseline.json](config/B_baseline.json) 和 [C_baseline.json](config/C_baseline.json)。

配置保留 actor、critic 各 235 维观测、12 维动作、ELU 隐藏层 `[512, 256, 128]`、24 步 rollout 和五轮 PPO 更新。RLinf 使用 Huber 价值损失、学习率 `3e-4`、熵系数 `0.003` 和解析 KL 目标 `0.005`；普通 PPO 使用自适应 KL 目标 `0.01`。B 保留数值保护，C 续训使用后期姿态奖励权重。

## 结果与离线检查

A40 实验均使用评测种子 23456、34567，按实际回合数合并。训练种子 0 的历史结果来自 `e8815cac`（C 基线和普通 PPO 为 `0e8129b9`）。在 `994a4268` 上，六个续训组改用训练种子 2，各训练 1000 次更新。三个基线组各自沿用原任务的冻结权重，各任务两份评测报告都与历史文件逐字节一致。

| 任务 | 组别 | 跟踪成功率（seed 0） | 跟踪成功率（seed 2） | 仿真楼梯通过率（seed 0） | 仿真楼梯通过率（seed 2） |
| --- | --- | --- | --- | --- | --- |
| A：全向移动 | PPO 基线 | 91.47%（1887/2063） | 91.47%（1887/2063） | — | — |
| A：全向移动 | 普通 PPO 续训 | 93.33%（1917/2054） | 90.90%（1878/2066） | — | — |
| A：全向移动 | RLinf-Huber 续训 | 96.73%（1981/2048） | 98.73%（2022/2048） | — | — |
| B：粗糙地形 | PPO 基线 | 99.85%（2045/2048） | 99.85%（2045/2048） | — | — |
| B：粗糙地形 | 普通 PPO 续训 | 99.61%（2040/2048） | 99.76%（2043/2048） | — | — |
| B：粗糙地形 | RLinf-Huber 续训 | 99.85%（2045/2048） | 99.90%（2046/2048） | — | — |
| C：楼梯 | PPO 基线 | 99.85%（2045/2048） | 99.85%（2045/2048） | 21.39%（438/2048） | 21.39%（438/2048） |
| C：楼梯 | 普通 PPO 续训 | 100%（2048/2048） | 99.80%（2044/2048） | 44.82%（918/2048） | 33.50%（686/2048） |
| C：楼梯 | RLinf-Huber 续训 | 100%（2048/2048） | 99.80%（2044/2048） | 45.17%（925/2048） | 41.94%（859/2048） |

在相同 A40 配置下，`e8815cac` 的两次 C 组 RLinf-Huber 训练（seed 0）得到相同结果。`994a4268` 再次复现了全部 20 个检查点的 340 个张量、2048 条逐回合评测记录和两份汇总。

如需对九份模型权重做可选离线检查，可下载权重，使用 Python 3.11、PyTorch 和 RLinf 依赖校验哈希、策略加载及解析 KL。权重可能来自不同的历史训练，用于离线回放；此检查不代表新的训练成绩。下载需要联网；检查在 CPU 上运行，无需 A40：

```bash
python -m examples.embodiment.go2_abc.prepare_data
python -m examples.embodiment.go2_abc.audit
```

## 运动控制

[运动控制说明](MOTION.zh-CN.md)介绍 ABC policy 的观测拼接、电机换算和 SDK2 接口。[Go2 六方向真机控制](../go2_sport/README.zh-CN.md)使用独立的 SportClient 链路。ABC policy 的部署、感知和独立停止仍需真机验证。
