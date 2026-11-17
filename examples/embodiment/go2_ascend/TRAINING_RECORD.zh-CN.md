# Go2：RLinf + Ascend 从零训练复现记录

2026-11-14 已在 hkust（AI-NPU-2）启动正式训练：8 张 Ascend NPU（物理编号 8–15）、4096 个 MuJoCo 环境、248 个仿真线程，目标 50000 次 PPO 更新。刚检查完成 124 次，已生成第 100 次完整检查点；尚未完成长训练，也尚未验收最终步态。

这是 RLinf 的实际采样、FSDP/HCCL 优化、权重同步及原生检查点链路。物理仿真在 CPU MuJoCo 上，策略采样和学习在 NPU 上；不是 Isaac Sim，也没有用旧 RSL-RL 训练循环代替 RLinf。

## 文件与目录

服务器根目录 `/data1/qli736/go2_rlinf`：

| 路径 | 内容 |
|---|---|
| `RLinf/` | Go2 接入源码快照，提交 `3148620a7c535d5afacb5ec1b09912a489cacb2b` |
| `.venv/` | 独立依赖 overlay，不修改旧训练环境 |
| `setup/` | 本目录中的配置、扩展、启动与检查脚本 |
| `experiments/from_scratch.log` | 正式训练控制台日志 |
| `experiments/from_scratch/` | 指标、TensorBoard、worker 状态与课程状态 |
| `experiments/from_scratch/from_scratch/checkpoints/global_step_N/` | 原生 RLinf 模型、优化器、调度器检查点及环境课程 |
| `launch.json`、`training.pid` | 启动参数、源码摘要、驱动 PID |
| `experiments/diagnostic_first_launch/` | 首次启动的调度异常诊断记录 |

本地 `rlinf_source_3148620.tar.gz` 是对应源码快照；`evidence/` 保存运行证据与依赖冻结版本。旧训练模型和旧实验没有被覆盖。

## 与旧 Ascend 训练的参数对照

旧基线见 `../../ascend_training/training_8npu.json`、`TRAINING_8NPU.md`。

| 项目 | 本次设置（与旧基线一致） |
|---|---|
| 环境、训练量 | 4096 环境；50000 次更新；每环境每次采样 24 步 |
| PPO batch | 全局 98304 条采样；5 epochs；4 minibatches；每 minibatch 全局 24576、每卡 3072 |
| 网络 | Actor 45 维、Critic 60 维、12 动作；512/256/128 ELU；初始标准差 1 |
| 优化 | Adam；初始 LR .001；解析高斯 KL 自适应，目标 .01，范围 1e-5 至 .01 |
| 损失 | PPO clip .2；clipped squared value loss，系数 1；entropy .01；grad norm 1 |
| GAE | gamma .99、lambda .95；全局优势归一化，Bessel 标准差，epsilon 1e-8 |
| 物理和奖励 | 原 16 项奖励；dt .005、decimation 4；20 秒回合；训练噪声、随机化与推扰启用 |
| 资源 | 物理 NPU 8–15；248 个仿真线程；Torch OMP 4、OpenBLAS 1；被动等待 |
| 保存与评估 | 每 100 次保存，每 1000 次评估 |

种子为 42，Actor/Rollout rank 使用 42+rank。已对照旧 RSL-RL：seed42 初始化参数最大误差为 0。训练 XML 的旧/新 SHA256 相同：`89e31dc37e57bd6a0d66649fe5f7975db0394c3b2a89f31a06898d3d0687bb78`。

参数接近不意味着轨迹逐位相同：旧训练分为 8 个环境池，本次一个统一环境 worker 管理 4096 环境和全局课程，8 个 Actor 和 8 个 Rollout worker 分布于 8 张卡。随机流、数据打乱顺序及框架执行顺序不同。旧实验曾从少卡扩展到八卡；本次直接八卡随机初始化。评估频率相同，但本次固定前进 .4m/s、16 环境、20 秒，旧评估为三个指令、10 秒，不能直接混合比较奖励。

## 为什么需要独立扩展入口

必须使用 `run_stage.sh` → `train_go2.py`，不能直接换成标准训练入口。`go2_rsl_extension.py` 注册专用策略和 loss，并提供兼容 Actor/Env/Rollout；`train_go2.py` 实例化这些 worker 和 runner。

扩展补齐旧基线的解析 Gaussian KL 自适应学习率、平方价值损失及全局优势归一化，保存采样时旧均值/标准差，并记录八卡同步误差。采样通信、GAE、FSDP/HCCL 梯度更新、权重同步和 DCP 保存恢复仍由 RLinf 完成。

默认 RLinf runner 要求保存间隔能被评估间隔整除，`100/1000` 会在首轮报错。首次失败记录已保留；当前 `Go2Runner` 分别调度保存和评估，保持原两个间隔，随后重新从随机初始化启动。没有修改 RLinf 核心仓库来绕过约束。

## 安装与启动

已验证平台：aarch64，Python 3.10.22，Torch 2.6.0+cpu、torch_npu 2.6.0.post5、CANN 8.5、MuJoCo 3.3.7、Ray 2.48.0。Torch 版本的 `+cpu` 后缀不代表本次在 CPU 学习；worker 实际设备为 `npu:0`，各自映射物理卡 8–15。驱动、CANN、可用 NPU 和现有匹配的 Torch/torch_npu 是前提，本清单不是裸机驱动安装器。

在 hkust 上复用当前安装即可；新建同布局环境时：

```bash
ROOT=/data1/qli736/go2_rlinf
mkdir -p "$ROOT/RLinf" "$ROOT/setup" "$ROOT/experiments"
# 将本目录内容复制到 setup，并将源码快照解压到 RLinf。
tar -xzf "$ROOT/setup/rlinf_source_3148620.tar.gz" -C "$ROOT/RLinf"
/data1/qli736/miniconda3/envs/lerobot/bin/python -m venv --system-site-packages "$ROOT/.venv"
"$ROOT/.venv/bin/python" -m pip install -r "$ROOT/setup/requirements-ascend.txt" -c "$ROOT/setup/constraints-ascend.txt"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd "$ROOT/RLinf"
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 "$ROOT/.venv/bin/python" -m rlinf.envs.sim.go2_mujoco.build
```

编译必须在目标架构重新执行，不能复制本机 x86 的 `.so`。正式复现以 `evidence/pip-freeze.txt` 的实际版本为准。官方 RLinf 已有 Ascend accelerator manager 和实验性安装支持，本次复用了其 NPU/FSDP/HCCL 链路，未执行全套官方安装器。

脚本和 YAML 中路径针对当前服务器。更换目录须同时调整配置、`run_stage.sh` 和辅助脚本的根路径；`launch_training.py --root` 本身不会重写 YAML。

先依次运行验证（脚本会同步等待结束，输出目录需空）：

```bash
bash /data1/qli736/go2_rlinf/setup/run_stage.sh single_smoke
bash /data1/qli736/go2_rlinf/setup/run_stage.sh single_resume
bash /data1/qli736/go2_rlinf/setup/run_stage.sh eight_smoke
bash /data1/qli736/go2_rlinf/setup/run_stage.sh eight_resume
bash /data1/qli736/go2_rlinf/setup/run_stage.sh final_scale_smoke
```

正式启动命令如下。**当前已经运行，不要重复启动**；脚本有驱动 PID 和目录覆盖检查。正式配置没有加载旧成功权重，`initial_checkpoint`、`resume_dir` 和 `ckpt_path` 均为空。

```bash
/data1/qli736/go2_rlinf/.venv/bin/python /data1/qli736/go2_rlinf/setup/launch_training.py
```

该命令创建独立后台会话，SSH 断开后继续运行。启动脚本关闭 Ray 的批量 worker 预启动，避免按服务器 640 个逻辑 CPU 创建大量空闲 worker；不停止其他用户的 Ray 或 NPU 作业。

## 看进度与恢复

本机查询：

```bash
ssh hkust '/data1/qli736/go2_rlinf/.venv/bin/python /data1/qli736/go2_rlinf/setup/status.py'
ssh hkust 'tail -n 30 /data1/qli736/go2_rlinf/experiments/from_scratch/metrics.log'
```

`status.py` 显示 PID、完成更新数、最近速度、粗略 ETA、同步误差、课程和最新检查点。ETA 根据最近 50 个记录估算，会随课程及评估变化。124 次更新时约为 45 小时，仅为早期估计。

仅在训练确实退出后，从最近完整检查点恢复：

```bash
ssh hkust '/data1/qli736/go2_rlinf/.venv/bin/python /data1/qli736/go2_rlinf/setup/launch_training.py --resume'
```

恢复加载模型、Adam、调度器以及 RLinf 保存的 RNG，并恢复环境全局课程及 common_step。MuJoCo 物理状态、每环境 C++ RNG 和进行中的回合会重新开始，不能称为轨迹无缝恢复。必须保留整个 `global_step_N` 目录，不能仅给 `full_weights.pt`。脚本以末尾写出的 `environment_curriculum.json` 作为检查点可用标记。

## 已完成的验证与限制

- 单 NPU：3 次训练/评估/保存，再由第 3 次恢复至第 5 次。
- 八 NPU：同样完成 3→5 次训练恢复；各 rank 实际映射 8–15，同步误差为 0。
- DCP 内容检查：单卡及八卡 Adam step 均由 60 恢复至 100，scheduler 由 3 恢复至 5。
- 正式规模：4096 环境、248 仿真线程、八卡 3 次更新及保存完成；课程状态进入检查点。
- 数学对照：初始参数、平方价值损失及梯度、高斯 KL、LR 边界和独立保存/评估间隔检查通过，见 `math_verification.json`。
- 正式长训练已经超过第 100 次保存，但尚未到第一次第 1000 次评估；不能据此保证 50000 次收敛或与旧步态等质。

运行证据见 `evidence/`，脚本 `verify_checkpoints.py` 可检查原生 DCP 的优化器恢复；`verify_math.py` 是本机与旧 RSL-RL 依赖的数学对照，运行它需要额外的旧 RSL-RL 环境，正式训练不依赖旧训练循环。

## CPU 资源调整：248 → 320

用户要求使用整机一半 CPU（640 逻辑 CPU 的一半）。从第 400 次完整检查点恢复时，正式配置的 `env.train.threads` 改为 320；原 248 线程配置及 launch 元数据在服务器 `setup/cpu_resize_248_to_320/` 保留。其余 PPO、环境数量和八 NPU 设置不变。前文的 248 是初始训练配置与基线对照，当前正式 YAML 以 320 为准。320 是仿真并行线程数，不是独占或持续跑满 320 核的保证；整体利用率仍受同步采样、通信和串行部分限制。

调整前近期约 3.14 秒/更新，实际仿真平均约 16 个核。调整后应排除启动和恢复耗时，使用稳定阶段的 `time/step` 对比，不能根据线程数直接宣称提速。

## 资源对比：四 NPU、160 仿真线程

按用户要求，在第 1000 次完整检查点后从八 NPU/320 线程切换为物理 NPU 8–11、160 个仿真线程（640 逻辑 CPU 的四分之一）。全局 4096 环境、24 步采样、global minibatch 24576、5 epochs/4 minibatches 不变；每卡 micro batch 从 3072 调为 6144，保持 gradient accumulation=1。原配置和启动信息保存在服务器 `setup/resize_8npu320_to_4npu160/`。

恢复仍使用完整原生 DCP，包含模型、优化器和 scheduler；框架支持 world size 变化时重新分片。现存四个 rank 恢复其保存的 RNG，但 world size 变化、样本划分及环境重置会改变随机轨迹，不承诺逐位复现八卡结果。评价性能时排除恢复启动和第 1000 次评估，按后续稳定窗口对比。八卡此前约 3.14 秒/更新。使用四卡后，12–15 不再由这个 Go2 作业占用。

四卡初测：完成第 1016 次更新，排除启动后的 14 次平均约 2.58 秒/更新（采样 1.56、训练 .63、同步 .38 秒），相比此前八卡约 3.14 秒缩短约 18%。仅早期样本，需继续观察稳定窗口；四卡参数同步误差均为 0。
