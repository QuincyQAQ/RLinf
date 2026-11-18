# Go2 Mjlab 与 RLinf PPO

使用 RLinf 标准 PPO worker 训练宇树官方 `Unitree-Go2-Flat` 任务。任务源码固定为 Unitree RL Mjlab 提交 `1425b15f73bd4095f0df53709d7c389c3eb9e790`。CPU 执行仿真，配置的加速器执行 policy 推理和 PPO 更新。昇腾配置使用 8–15 号设备和 64 个 CPU 环境 worker。

## 训练配置

配置默认使用 `env.train.physics_backend=rlinf_mujoco_warp`。RLinf 的 `MjWarpWorld` 复用 Mjlab 初始化的模型和数据，推进同一份物理状态；奖励、观测、地形、传感器、随机化和重置仍由 Mjlab 管理。设置为 `mjlab` 可对照 Mjlab 原有的物理步进。原有五任务 `mujoco_warp` 环境继续使用原配置。

配置为 4096 个环境、每次采样 24 个控制步、10001 次更新、每次训练五轮 PPO、每轮四个全局 minibatch，每 100 次更新保存模型。全局 minibatch 为 24576，八个 actor 每个处理 3072 个样本。每个环境 worker 运行 64 个环境，使用两个 Torch 线程。

Actor 和 critic 都使用 512/256/128 的 ELU 隐藏层，分别接收 47/74 维观测，启用 epsilon 为 0.01 的观测归一化。Policy 输出 12 维高斯关节动作，初始标准差为 1.0。PPO 使用系数为 1.0 的裁剪平方价值损失，裁剪系数为 0.2，熵系数为 0.01，gamma 为 0.99，GAE lambda 为 0.95，最大梯度范数为 1.0。学习率从 0.001 开始，按解析高斯 KL 自适应调整，KL 目标为 0.01，学习率范围为 0.00001–0.01。`torch_constant` 保留每个 minibatch 更新时调整的学习率，不表示学习率固定不变。

观测、奖励、随机化、指令课程、接触传感器和终止条件均来自上游任务。初始回合长度也按官方训练入口随机化。该配置建立独立实验，从随机初始化开始；不要加载旧的 45/60 维观测模型。

## 昇腾环境安装

验证服务器使用 Linux aarch64、glibc 2.34、CANN 8.5、PyTorch 2.7.1、TorchNPU 2.7.1.post2 和 Python 3.10。PyPI 的 ARM Warp 1.12 wheel 要求 glibc 2.35，在此机器无法加载。锁定的 conda-forge 同版本 CPU 构建已验证可用，无需修改系统。

先准备 `conda` 和 `uv`。在 RLinf 仓库根目录，将任务源码克隆到相邻目录，并固定提交：

```bash
git clone https://github.com/unitreerobotics/unitree_rl_mjlab.git ../unitree_rl_mjlab
git -C ../unitree_rl_mjlab checkout 1425b15f73bd4095f0df53709d7c389c3eb9e790
```

创建 CPU Warp 环境，再建立独立 Python 运行环境。Pip 锁文件不包含 Warp，由 conda 提供兼容构建：

```bash
GO2_RUNTIME_DIR="$PWD/outputs/go2_mjlab_runtime"
conda create --prefix "$GO2_RUNTIME_DIR/warp_cpu" --file examples/embodiment/go2_mjlab/ascend-conda-explicit.lock -y
uv venv --system-site-packages --python "$GO2_RUNTIME_DIR/warp_cpu/bin/python" .venv-mjlab-ascend
uv pip install --python .venv-mjlab-ascend/bin/python --no-deps -r examples/embodiment/go2_mjlab/ascend-requirements.lock
uv pip install --python .venv-mjlab-ascend/bin/python --no-deps -e ../unitree_rl_mjlab
```

这些锁文件面向上述昇腾环境。自动安装目标 `--env go2_mjlab` 和 Docker stage 覆盖 NVIDIA 主机；此昇腾配置使用上面的 CPU Warp 安装步骤。

## 启动

指定匹配的 CANN 初始化脚本和 Python，先跑单设备测试：

```bash
export GO2_PYTHON="$PWD/.venv-mjlab-ascend/bin/python"
export CANN_ENV_FILE=/usr/local/Ascend/ascend-toolkit/set_env.sh
bash examples/embodiment/go2_ascend/run_native.sh go2_mjlab_ppo_ascend_smoke
```

该配置在 8 号设备上运行 16 个环境，训练三次更新。然后使用完整的 4096 个环境验证八设备运行：

```bash
bash examples/embodiment/go2_ascend/run_native.sh go2_mjlab_ppo_ascend \
  runner.max_epochs=3 runner.max_steps=3 runner.save_interval=1 \
  runner.logger.log_path="$PWD/outputs/go2_mjlab/eight_smoke" \
  runner.logger.experiment_name=eight_smoke
```

确认测试进程已退出后，启动 10001 次更新的实验：

```bash
bash examples/embodiment/go2_ascend/run_native.sh go2_mjlab_ppo_ascend
```

设备编号不同时，修改 `examples/embodiment/config/go2_mjlab_ppo_ascend.yaml` 中的 placement。后台辅助脚本 `launch_training.py` 不解析继承的 Hydra 配置，此任务使用 `run_native.sh` 启动。

停止原训练进程后，可从已保存的 `global_step_*` 目录恢复：

```bash
checkpoint="$PWD/outputs/go2_mjlab/native_ascend/go2_mjlab/checkpoints/global_step_100"
bash examples/embodiment/go2_ascend/run_native.sh go2_mjlab_ppo_ascend runner.resume_dir="$checkpoint"
```

## 输出与验证

默认输出目录为 `outputs/go2_mjlab/native_ascend`。TensorBoard 事件保存在 `tensorboard/`，可恢复的模型、优化器、学习率调度器和归一化统计保存在 `go2_mjlab/checkpoints/`。恢复训练会重置物理状态，并按已保存的更新次数恢复基于时间的指令课程。

已完成上游 RSL-RL 单 NPU 对照。此前原生 RLinf 检查使用 `physics_backend=mjlab`：单 NPU 和八 NPU 的平地任务均从随机初始化完成三次更新，并恢复到总计五次更新；八 NPU 使用 4096 个环境。那些恢复检查核对了 Adam 步数和归一化样本数。

使用 `physics_backend=rlinf_mujoco_warp` 的八 NPU B2 台阶短测采用 1024 个环境，从随机初始化完成三次更新，再恢复至累计五次更新。Checkpoint 检查确认 Adam 步数为 60 和 100，调度器计数为 3 和 5。CPU 对照覆盖 Flat 和五个 benchmark 任务，包括随机化重置；两个后端在已测轨迹中的观测和奖励一致。这些检查证明执行流程可用，不代表已经收敛或通过真机验证。日志和 checkpoint 检查结果另存于各实验记录中。

本目录的 `launch_native.py`、`inspect_checkpoint.py`、`watch_main_task.py` 分别保存解析后的启动配置、核对 B2 checkpoint 计数和独立评估 B2 checkpoint，所需路径见各脚本的 CLI 帮助。B2 watcher 检查 100、500、1000、2200 次更新，跳过恢复点之前的 checkpoint，只评估主任务，不填充跨任务矩阵。

重启 watcher 会复用 checkpoint、配置和种子匹配的已完成评测。回合配额是最小数量，最后一个并行批次完成的所有回合均计入汇总。每个评测进程的超时为两小时，可用 `--evaluation-timeout` 按秒调整。

任务和上述 PPO 参数对齐上游，采样、GAE、优化与保存由 RLinf 执行。RLinf 在每批 rollout 优化前汇总各 actor 的归一化统计；RSL-RL 在每个环境步后更新本地统计。RLinf 使用终止前观测计算超时回合的 bootstrap 值，上游 RSL 包装器使用保存的 transition value。分布式打乱顺序和环境种子也有差异，因此不宣称与官方训练逐步数值一致。

## 官方对照验证

`validate_upstream.py` 独立检查官方环境，也可运行单设备 RSL-RL 学习器。输出目录应为空：

```bash
"$GO2_PYTHON" examples/embodiment/go2_mjlab/validate_upstream.py \
  --num-envs 4 --physics-device cpu --output outputs/go2_mjlab/cpu_probe
source "$CANN_ENV_FILE"
ASCEND_RT_VISIBLE_DEVICES=8 "$GO2_PYTHON" examples/embodiment/go2_mjlab/validate_upstream.py \
  --num-envs 4 --learner-device npu:0 --updates 3 --output outputs/go2_mjlab/upstream_npu_probe
```

工具保存解析后的任务配置、算法配置和 `result.json`。可选学习器属于 RSL-RL 对照，不是原生 RLinf PPO。`run_upstream.sh` 保留了固定版本的 NVIDIA 官方基线入口，默认解释器为 `.venv-mjlab/bin/python`，启动前会检查依赖版本。

## 跨任务评估

独立的五任务版本使用 235 维观测，操作步骤见[评估说明](BENCHMARK.zh-CN.md)。上面的 Flat 训练保留原配置。


## 五任务 RSL-RL baseline

使用 RSL-RL 5.0.1 的 PPO，为五个任务分别从随机初始化训练一个专项模型。任务和奖励沿用已有 MuJoCo/Mjlab 配方，CPU 进程负责仿真，每个任务使用一张加速卡。本实验不调用 RLinf PPO worker。B2 保留 2200 次更新的姿态奖励调度和有界高斯分布；其余四项使用各自已有的任务配方。五项均使用 1024 个环境，每次采样 24 步，每批数据优化 5 遍、分为 4 个 minibatch，训练种子统一为 42。

先按上文激活昇腾运行环境，再选择五张空闲卡和一个尚不存在的输出目录：

```bash
source "$CANN_ENV_FILE"
export PYTHONPATH="$PWD:$PWD/../unitree_rl_mjlab${PYTHONPATH:+:$PYTHONPATH}"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$GO2_PYTHON" \
  examples/embodiment/go2_mjlab/run_rsl_baselines.py \
  --python "$GO2_PYTHON" --output outputs/go2_rsl_baselines \
  --devices 8 9 10 11 12 --updates 2200 --num-envs 1024 --workers 32
```

每个任务使用 32 个 CPU 进程，每进程 1 个 Torch 线程；每完成 100 次更新保存一次 checkpoint。在 1200、1500、1800、2200 次更新时，各模型只评估对应任务。中途每次至少收集 64 个完整回合，最终至少 256 个。评测统一使用种子 43，与训练种子 42 分开。评测按任务并行，每个模型只使用一个评测种子。这些检查只提供单种子结果，不代表原项目完整的交叉评测矩阵。

通过 `suite_status.json` 查看每个任务的训练和评测状态。每个任务分别保存 `metrics.jsonl`、TensorBoard 事件和 `checkpoints/global_step_<count>/model.pt`；后者包含 RSL-RL 模型及优化器状态，可用于恢复训练。旁边的 `full_weights.pt` 将冻结的观测归一化合入 actor 和 critic 的第一层，供已有评测器使用；它是导出文件，不能替代原生恢复 checkpoint。恢复时物理状态会重置。单独恢复已停止的专项模型时，向 `rsl_baseline.py` 传入 `--resume <model.pt>`，并保持原任务、输出目录、环境数、进程数、种子和设备设置。

baseline 本身已经使用 PPO，后续切换到 RLinf PPO 比较的是不同实现下继续训练的效果，不能称为“首次加入 PPO”的收益。此实验采用已安装的 RSL-RL 5 模型结构和 MuJoCo 物理引擎，不宣称原 Isaac Lab 实验的数值复现。
