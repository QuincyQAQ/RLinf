# Go2：RLinf 原生 PPO 实验

本实验使用仓库现有的 `train_embodied_agent.py`、`EmbodiedRunner`、`EmbodiedFSDPActor`、标准 Rollout/Env worker 和注册名 `actor_critic` 的原生 PPO。只提供 Go2 环境、policy 和配置，不加载 `go2_rsl_extension`，不调用旧 RSL-RL 训练循环。

## 与兼容实验的区别

兼容实验于 2026-11-15 在第 37600 次更新保存完整检查点后停止。服务器目录仍为 `/data1/qli736/go2_rlinf`，暂停记录是 `experiments/from_scratch/PAUSED.json`，原恢复命令在其中；整套检查点另打包为 `backups/paused_37600.tar.gz`，并备份到本机历史实验目录。不要删除旧目录；原生实验采用另一个源码部署目录。

原生实验从随机初始化开始，不加载第 37600 次权重。它使用原生 clipped PPO + clipped Huber value loss、GAE 和常数学习率 .001；兼容实验是平方价值损失和解析 KL 自适应学习率。PPO 并非官方承诺最优的一组超参数，这里是本仓库 Go2 原生配置的昇腾资源版本。

保持网络 45/60→512/256/128→12、4096 环境、24 步采样、5 epochs、4 minibatches、全局 minibatch 24576、四卡每卡 microbatch 6144、gamma .99、lambda .95、clip .2 和 entropy .01。物理卡 8–11，仿真线程 160。

标准 runner 要求保存间隔可被评估间隔整除。因此正式实验每 1000 次评估和保存，目标 50000 次；不再使用兼容 runner 的独立 100/1000 调度。

## 依赖与运行

依赖、CANN 设置和编译方法与本目录 README 相同。可用 `GO2_PYTHON` 指定已安装匹配 Torch/torch_npu 的 Python，`CANN_ENV_FILE` 指定 CANN 初始化脚本。已有兼容实验的隔离 venv 可以复用；不要修改其依赖。入口主动清除 `RLINF_EXT_MODULE`。

从仓库根目录执行：

```bash
# 单卡短训练，含短回合评估和保存。
bash examples/embodiment/go2_ascend/run_native.sh go2_mujoco_ppo_ascend_smoke
# 同配置从第3次恢复到第5次，写独立输出。
bash examples/embodiment/go2_ascend/run_native.sh go2_mujoco_ppo_ascend_smoke \
  runner.resume_dir=$PWD/outputs/go2_native/single_smoke/single_smoke/checkpoints/global_step_3 \
  runner.max_epochs=5 runner.max_steps=5 \
  runner.logger.log_path=$PWD/outputs/go2_native/single_resume \
  runner.logger.experiment_name=single_resume
# 四卡、4096环境、160线程的3次规模验证。
bash examples/embodiment/go2_ascend/run_native.sh go2_mujoco_ppo_ascend_four_smoke
# 正式启动前查看命令与配置，不启动训练。
.venv/bin/python examples/embodiment/go2_ascend/launch_training.py --native --dry-run
# 后台从零训练；已有活动作业时不要重复执行。
.venv/bin/python examples/embodiment/go2_ascend/launch_training.py --native
.venv/bin/python examples/embodiment/go2_ascend/status.py --output-dir $PWD/outputs/go2_native/from_scratch
```

标准配置输出到仓库的 `outputs/go2_native/from_scratch`。启动器的 `--output-dir` 可改输出目录，`--native --resume` 从最近带 DCP 元数据的检查点恢复。原生 checkpoint 恢复模型、优化器、调度器和保存的 RNG；标准 Env worker 不恢复 MuJoCo 物理状态、每环境 RNG 或本实验的速度课程，环境课程重新开始。这与旧兼容 worker 的课程恢复不同。

## 评估与结论边界

内置评估固定前进 .4m/s，16 环境、20秒、确定性动作、关闭噪声和随机化。训练中保留噪声、随机化和推扰。奖励曲线只是一个指标；未在环境中定义任务成功率，不可将回合长度或奖励增量冒充成功率。

第1000次与第34000次之间奖励增幅小，既可能因早期已经学会基础任务，也可能是后续提升有限；单一指令和两点比较不能区分。需要保存未训练模型/早期模型的固定评估，再对后续检查点使用相同指令、种子和成功标准。不能预先保证改成原生 PPO 会获得更大提升。

本文件描述配置与验收流程；实际运行的验证结果及部署提交另记在实验目录，不将计划描述为已完成。

独立随机初始化策略的评估使用官方评估入口：`bash examples/embodiment/go2_ascend/run_native.sh --eval`。它是独立随机策略基线，不声称与正式训练第0次的参数逐位相同。

## 使用八个 NPU

八设备配置使用 8–15 号设备，保留 4096 个环境、160 个仿真线程和 24576 的全局 minibatch。每个 actor 每个 minibatch 处理 3072 个样本，每轮 PPO 共四个全局 minibatch。恢复前先确认设备可用，并停止原训练进程。

将 `checkpoint` 设为已保存的 `global_step_*` 目录，然后从仓库根目录运行：

```bash
checkpoint=/path/to/global_step_15000
bash examples/embodiment/go2_ascend/run_native.sh go2_mujoco_ppo_ascend_eight \
  runner.resume_dir="$checkpoint" \
  runner.logger.log_path="$PWD/outputs/go2_native/eight"
```

使用上述命令启动八设备配置。后台启动辅助脚本只读取独立 YAML，不解析 Hydra defaults。改变 actor 数量后，分布式 checkpoint 必须同时恢复权重、优化器和调度器。将短程恢复测试与长期训练分别记录；设备启动成功不代表训练指标正常。
