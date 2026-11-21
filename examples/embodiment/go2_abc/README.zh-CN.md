# Go2 ABC 实验

[English](README.md)

运行全向移动、粗糙地形、楼梯三个任务。每个任务包括完整 PPO 基线、继续普通 PPO、继续 RLinf-Huber，共九组。

| 任务 | 基线训练轮数 | 每个续训组增加轮数 | 并行环境数 | 基线跟踪成功率 | 继续 PPO | 继续 RLinf-Huber |
|---|---:|---:|---:|---:|---:|---:|
| A：全向移动 | 1500 | 1000 | 2048 | 91.4687% | 93.3301% | 97.6574% |
| B：粗糙地形 | 3000 | 1000 | 2048 | 99.8535% | 99.6094% | 99.8047% |
| C：楼梯 | 2200 | 1000 | 1024 | 99.8535% | 100% | 100% |

C 在仿真中的楼梯通过率依次为 21.3867%、44.8242%、76.5625%。[结果表](results.csv)还包含速度误差、同回合跟踪与通过均成功的比例。训练 seed 为 0；评测 seed 为 23456、34567，分别评测后按回合数合并。

## 准备输入

复用[已准备的 A40 运行环境](https://github.com/QuincyQAQ/RLinf/tree/b405f67382ce63af2401a2a9ff5b0b3c7f28ccb7)，包括 Isaac Lab 2.3.2、Isaac Sim 5.1、RSL-RL 3.1.2、依赖包和资产。仿真任务、普通 PPO 来自固定版本的[多任务 benchmark](https://github.com/viktoriageorrg04/bsc-thesis-multi-task-RL-env)，控制与奖励沿用原实现。

在 RLinf 仓库根目录下载九份权重和 18 份评测记录。已有归档可加上 `--source /path/to/frozen-runtime`，从本地复制。

```bash
python -m examples.embodiment.go2_abc.prepare_data
python -m examples.embodiment.go2_abc.audit
```

CPU 检查涵盖 33 个输入文件的哈希、权重严格加载、解析 KL，以及 18,454 个回合与 CSV 的一致性。输入和运行产物统一放在 `data/go2_abc/`。

## 运行

设置已有运行环境的路径，预览一组命令。`GO2_ABC_CONTAINER` 指向已安装的 Singularity 或 Apptainer。

```bash
export GO2_ABC_RUNTIME=/path/to/prepared-runtime
export GO2_ABC_CONTAINER=singularity
bash examples/embodiment/go2_abc/container.sh --experiment C_rlinf
```

进入已分配 A40 的 Slurm 作业后，加 `--execute` 执行。实验名为 `A_baseline`、`A_ppo`、`A_rlinf`，B、C 同理。基线组评测已完成训练的权重；两个续训组从同一份基线权重出发，各自新建优化器，训练后评测两个 seed。日志和 checkpoint 存入 `data/go2_abc/runs/<experiment>/`。

## 配置与合并

配置保留 actor、critic 各 235 维观测、12 维动作、ELU 隐藏层 `[512, 256, 128]`、24 步 rollout、五轮 PPO 更新。RLinf 使用 Huber 价值损失、学习率 `3e-4`、熵系数 `0.003`、解析 KL 目标 `0.005`；普通 PPO 沿用自适应 KL 目标 `0.01`。B 保留原实验的数值保护，C 续训采用后期姿态奖励权重。

分支从上游 RLinf `dcf00fd6` 建立，通用 KL 调度单独提交；Go2 使用 `RLINF_EXT_MODULE` 注册。更新框架可直接 rebase 到上游。[清单](manifest.json)记录实验代码版本、输入哈希与权重路径。

观测拼接、电机换算与 SDK2 接口见[运动控制说明](MOTION.zh-CN.md)。

## Limitations

表中成绩来自 RLinf `5aa29ec` 上已完成的 A40 实验；接入上游后的代码已做 CPU 检查，尚未重新用 GPU 训练。比较采用一个训练 seed 和上述 PPO 配置。
