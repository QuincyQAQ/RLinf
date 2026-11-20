# Go2 A40 五任务实验与复现

本说明用于恢复 A40 实际运行的代码和权重，离线复算五个 Go2 任务的结果，并按保存的配置重放评测或续训。统一数据表是 [29 行完整 CSV](reports/A40_five_tasks_complete_20261121.csv)，包含主实验、补测、验证选型和最终预留集评测。

## 源码依据

代码以 A40 工作目录 `/hpc2hdd/home/qli736/Unitree_GO2_A40_reference_20261118` 在 2026-11-21 的只读快照为准。本机同步前发现 `a40/split_independent.py` 和两个测试文件有差异，已备份本机版本后采用 A40 版本。三个依赖仓库的受版本管理文件与本机一致。

| 部分 | 固定提交／版本 | 用途 |
|---|---|---|
| RLinf | `5aa29ecb43ef01e7865b41762856b25a1228702a` | 原生 RLinf PPO、Go2 policy（策略网络）与损失函数 |
| 原 benchmark | `743ac04d9543b04b7854f1100bfa0cc69c993106` | 原任务、奖励、普通 PPO、成功定义 |
| Isaac Lab | `37ddf626871758333d6ed89cf64ad702aef127d0` | Isaac Lab 2.3.2 环境和接口 |
| 仿真 | Isaac Sim 5.1.0／PhysX | GPU 物理仿真 |
| 官方容器 | `nvcr.io/nvidia/isaac-lab:2.3.2` | 学校 Singularity CE 4.1.3 运行 |
| PyTorch／TorchVision | `2.7.0+cu128`／`0.22.0+cu128` | 学习器运行时 |
| RSL-RL／TensorDict | `3.1.2`／`0.8.3` | 普通 PPO 与批量张量 |
| NumPy／Gymnasium | `1.26.4`／`1.2.1` | 数值与环境接口 |

`canonical_source.tar.gz` 保存 113 个 A40 外层源文件，保持原始字节。`canonical_manifest.json` 固定源码版本和 SHA256，`selection.json` 列出随分支交付的文件。30 个权重文件保存在 `checkpoints/`；逐回合评测和生成的配置保存在 `evidence.tar.gz`；训练日志保存在 `training_logs.tar.gz`。`reports/` 中的汇总由这些原始结果派生。

## 实验设置

主实验先完成普通 PPO，再从同一任务的最终保存权重分别追加 1000 次普通 PPO 或 RLinf PPO 更新。两个分支重新创建优化器和环境。训练 seed 均为 0，主评测 seed 均为 23456 和 34567，每个 seed 至少完成 1024 回合。

| 任务 | 含义 | 并行环境 | 源训练更新数 | 续训更新数 | 每条续训分支新增环境步 |
|---|---|---:|---:|---:|---:|
| A1 | 前进 | 2048 | 3000 | 1000 | 49,152,000 |
| A2 | 全向移动 | 2048 | 1500 | 1000 | 49,152,000 |
| B1 | 粗糙地形 | 2048 | 3000 | 1000 | 49,152,000 |
| B2 | 楼梯 | 1024 | 2200 | 1000 | 24,576,000 |
| C2 | 沟壑 | 2048 | 3000 | 1000 | 49,152,000 |

Actor 和 Critic 均使用 235 维观测、ELU 隐藏层 `[512, 256, 128]`，Actor 输出 12 维关节动作，关闭观测归一化。物理频率为 200 Hz，策略频率为 50 Hz，每次 rollout 为 24 步，PPO 更新 5 个 epoch、4 个 minibatch，gamma 为 0.99，GAE lambda 为 0.95。

RLinf-Huber 是本实验的配置名称：RLinf 原生 PPO 的 Critic 使用 Huber 损失，delta 为 10，学习率和价值学习率均为 `3e-4`，熵系数为 `0.003`，目标 KL 为 `0.005`，裁剪范围为 0.2。学习率按解析 KL 调整，调度器为 `torch_constant`。B2 额外保留相同参数的 MSE 分支。

B1 使用已修复的源 policy，动作在写入 ActionManager 和上一动作历史前统一限幅至 ±100；普通 PPO、RLinf 和评测采用相同处理，奖励和终止条件保持原定义。原始最终权重出现非有限值后，恢复过程最终从有效的 `model_2850.pt` 连同优化器继续 149 次更新，得到有效的 3000 更新源权重。对应历史恢复作业为 14032252，GPU 实际更新探针为 14032253。

B2 源训练保留原来的前 35% 姿态奖励退火，续训两侧均使用后期权重：高度项 -3.5、姿态项 -2。训练更新预算以日志中的完成计数为准；分块 runner 的文件标签 `2112`、`960` 等不能直接当作总训练更新数。

## 成功与误差的口径

跟踪成功要求回合没有失败终止，且达标时间步比例达到任务阈值。一个达标步同时满足平面线速度误差与偏航角速度误差阈值。

| 任务 | 线速度误差上限 | 偏航角速度误差上限 | 达标步比例下限 |
|---|---:|---:|---:|
| A1 | 0.20 m/s | 0.40 rad/s | 85% |
| A2 | 0.25 m/s | 0.50 rad/s | 80% |
| B1 | 0.35 m/s | 0.60 rad/s | 75% |
| B2 | 0.40 m/s | 0.70 rad/s | 70% |
| C2 | 0.45 m/s | 0.75 rad/s | 65% |

线速度误差是平面速度差的二范数，角速度误差是偏航角速度差的绝对值，先在回合内取平均，再按回合数合并。成功率使用成功回合总数除以总回合数。失败终止率单独统计，存活但跟踪不达标的回合仍可能不成功。

B2、C2 另外报告依据地形几何与接触证据判定的实际障碍通过率，以及同一回合中跟踪成功且障碍通过的交集。实际通过使用原地形网格、机器人与足端运动、落地和接触证据判定，保留对应 `.geometry.json` 与逐回合 `passage_evidence`。

## 已完成结果

下表合并评测 seed 23456 和 34567。普通 PPO 指原项目 RSL-RL PPO，续训预算与 RLinf 一致。

| 任务 | 原 PPO 跟踪成功率 | 继续普通 PPO | 继续 RLinf-Huber | 实际障碍通过率：基线／PPO／RLinf |
|---|---:|---:|---:|---|
| A1 | 99.8047%（2044/2048） | 100%（2048/2048） | 99.9023%（2046/2048） | |
| A2 | 91.4687%（1887/2063） | 93.3301%（1917/2054） | 97.6574%（2001/2049） | |
| B1 | 99.8535%（2045/2048） | 99.6094%（2040/2048） | 99.8047%（2044/2048） | |
| B2 | 99.8535%（2045/2048） | 100%（2048/2048） | 100%（2048/2048） | 21.3867%／44.8242%／76.5625% |
| C2 | 99.3164%（2034/2048） | 99.5117%（2038/2048） | 99.3164%（2034/2048） | 0.0977%／0%／0% |

完整误差、成功计数、存活时间、达标步比例和失败终止率见 CSV。B2 的 RLinf-MSE 实际通过率为 46.7773%（958/2048）。

A1 固定 policy 补测使用 seed 45678，各 1024 回合。A1 调参从既有 4000 更新策略出发，两个 RLinf 候选各追加 500 更新，学习率降至 `1e-4`，目标 KL 为 `0.003`，熵系数分别为 `0.001` 和 `0`。验证 seed 56789 选出熵系数为 0 的候选后冻结选择，再评测预留 seed 67890、78901。最终 RLinf 为 2047/2048，普通 PPO 4000 更新为 2048/2048，普通 PPO 4500 更新为 2035/2048。验证集成绩与预留集成绩在 CSV 中分开。

B1 固定 policy 补测使用 seed 45678，各 2048 回合，三组均为 2044/2048。合并原结果后，基线为 4089/4096，普通 PPO 续训为 4084/4096，RLinf 续训为 4088/4096。原结果始终保留。

| 实验 | 已完成作业 |
|---|---|
| 主实验 A1／A2／B1／B2／C2 | 14029742／14029743／14032301／14025574／14029748 |
| A1 固定 policy 补测 | 14032589 |
| B1 固定 policy 补测 | 14036295 |
| A1 调参及预留集评测 | 14034565 |

## 离线复算

在独立分支的 RLinf 仓库根目录，用 Python 3.11 运行归档核验。这个命令只读取现有文件，检查源码和数据哈希、逐回合统计、最终汇总及 CSV。

```bash
python3.11 examples/embodiment/go2_a40_reference/reproduce.py audit \
  --output /tmp/go2-a40-audit.json
```

安装 CPU PyTorch 后，追加 `--checkpoints` 可以检查 30 份权重的键、维度、有限值、正标准差及 CPU 前向输出。A40 现成容器还使用 Go2Policy 严格加载了全部 30 份权重，记录在 `verification/container_checkpoint_check.json`。

```bash
python3.11 examples/embodiment/go2_a40_reference/reproduce.py audit \
  --checkpoints --output /tmp/go2-a40-audit-with-weights.json
```

重新生成同一张 CSV，输出为 29 行、19 列，采用 Excel 可直接读取的 UTF-8 BOM 编码。

```bash
python3.11 examples/embodiment/go2_a40_reference/reproduce.py export \
  --output /tmp/A40_five_tasks_complete_20261121.csv
```

本次交付检查使用已有源码、配置、日志和权重，没有提交新的 Slurm 作业。详细机器可读结果保存在 `verification/`。空目录恢复后的 32 项原始测试、新归档工具的 7 项测试，以及现成容器中的 4 项策略检查均通过。

## 恢复工作目录

以下命令在空目录中恢复与 A40 相同的层级，包含 `a40/`、`config/`、三个固定版本仓库、`outputs/` 和训练权重。RLinf 从本地分支仓库复制并检出固定提交，benchmark 与 Isaac Lab 从记录的上游仓库获取。

```bash
python3.11 examples/embodiment/go2_a40_reference/reproduce.py restore \
  --workspace ../go2-a40-replay
```

已有三个源码仓库时，可以用 `--reference-workspace <已有工作目录>` 离线复制它们。恢复过程验证依赖仓库的每个记录文件，保留原权重修改时间，解压源训练完成日志，并拒绝覆盖已有非空目录。恢复后可以运行原来的 CPU 检查。

```bash
cd ../go2-a40-replay
PYTHONPATH=RLinf python3.11 -m unittest discover -s tests -v
python3.11 a40/reference.py check
```

## 配置运行环境与重放实验

容器挂载工作目录为 `/workspace/go2`。在 A40 主机复用已经验证的 `containers/`、`assets/`、`.python_overlay`、`.extra_overlay`、`.rlinf_overlay` 和 `.usd_converter`，或按归档的 `a40/prepare_hpc_environment.sh`、`a40/prepare_usd_isolation.sh`、`a40/prepare_rlinf_runtime.py` 及锁定依赖重新安装。容器 CPU 检查使用 `bash a40/hpc_environment.sh verify`，USD 资源使用 `a40/prepare_assets.py` 准备。

实际使用的 SIF 的 SHA256 为 `6ea64e2483d9b38495e9056c6544bb428bfa2aa0786218b33d58dc57c1da3b4d`，完整环境依据见 `environment_lock.json`。14 个官方资源文件共 44,720,015 字节，其路径和校验记录包含在归档 `validation/asset_mirror.json` 中。

下面是已有实验的后续重放入口。执行这些命令会申请计算资源，应在计划重跑时使用；本次归档未执行它们。

```bash
sbatch a40/fullppo_huber.sbatch A1
sbatch a40/fullppo_huber.sbatch A2
sbatch a40/b1_recovered.sbatch
sbatch a40/b2_continuation.sbatch
sbatch a40/fullppo_huber.sbatch C2
```

A1、A2、B1、C2 会按原入口执行配置检查、64 环境及原环境数实际更新检查，再运行主实验和最终评测。B1 使用归档的有效 3000 更新源 policy。B2 使用已归档且与当前协议相符的 GPU 更新检查记录；在不同环境重新验证门禁时，先使用 `a40/b2_continuation_gate.sbatch`。分区、账户、QoS 与节点排除规则在 `.sbatch` 文件中设置。

单独重放评测时，使用归档对应阶段的 `command.json`：其中保存了 checkpoint、任务、环境数、回合数和 `GO2_EVAL_SEED`。B1 同时保留 `GO2_B1_NUMERIC_GUARD=1`。B2 三点历史曲线的重放应使用 `curve/0`、`curve/1`、`curve/2` 保存的显式 checkpoint 路径。主实验中两个评测 seed 分别执行，汇总时才按回合数合并。

## Limitations

- 这份交付验证了已保存记录的离线复算与权重完整性，未重新运行 GPU 训练、物理仿真或真机。重新训练会受 GPU 数值计算、仿真与随机过程影响，不能保证各回合结果、权重的逐位数值或成功率的小数末位完全一致。
- A40 当前快照及三个固定提交可以恢复代码，但早期作业没有对每次 Python 导入保存完整源码哈希，无法逆向证明所有历史进程加载的每一个字节。已保留实际命令、生成配置、门禁、日志和最终权重作为运行证据。
- 每组只有训练 seed 0。补测 seed 只衡量同一个已训练 policy 的评测波动。RLinf-Huber 同时采用不同的价值损失和超参数，普通 PPO 也可以使用 Huber，因此不能把全部差异单独归因于框架。
- B1 原始最终权重无效，表中基线为记录过恢复过程的有效策略。此前失败日志保留；动作历史限幅属于显式修复，不能称为完全未修改的原版从头成功复现。
- C2 的跟踪成功率接近 99%，实际跨沟通过率仍接近零。B2 实际通过率与跟踪成功率独立报告，不能用跟踪结果替代障碍通过或真机部署结论。
- 完整 SIF、安装后的第三方包、全部训练中间权重与资源缓存没有上传。分支保存镜像哈希、版本和资源清单，并包含主表所用源权重、最终权重及 B2 三点曲线权重。B2 原完整 checkpoint 历史较长，对归档权重子集重新计算历史位置会改变曲线抽样，应使用保存的三个显式路径。
- HKUST CPU MuJoCo／NPU、robot_lab 与硬件控制是独立项目，本分支的结论和核验不覆盖这些项目。
