# Go2 昇腾训练：一份仓库，两处部署

本目录提供完整的 Go2 MuJoCo + RLinf 训练入口、PPO 扩展、配置、启动、恢复和监控脚本。物理仿真运行在 CPU，policy 采样和更新运行在昇腾 NPU。本机维护源码，GitHub 保存提交，hkust 部署指定提交；三处都是同一份代码的副本。

当前正式配置是四张 NPU（物理 8–11）、160 个仿真线程、4096 环境和 50000 次目标更新。单卡、八卡和恢复链路此前已验证，最终训练步态仍待验收。历史记录见 [TRAINING_RECORD.zh-CN.md](TRAINING_RECORD.zh-CN.md)，其中的服务器路径对应已启动的旧快照；新入口使用下面的命令。

## 安装

先提供匹配的昇腾驱动、CANN、Torch 2.6.0 和 torch_npu 2.6.0.post5。使用包含该栈的 Python 创建项目虚拟环境，在仓库根目录执行：

```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r examples/embodiment/go2_ascend/requirements-ascend.txt \
  -c examples/embodiment/go2_ascend/constraints-ascend.txt
source /usr/local/Ascend/ascend-toolkit/set_env.sh
.venv/bin/python -m rlinf.envs.sim.go2_mujoco.build
```

CANN 路径按实际安装位置替换；安装栈需要 ACL Python bindings 时，应确保 Ray 进程也能导入。MuJoCo 扩展必须在目标 CPU 架构重新编译。从零训练不需要旧权重或旧 RSL-RL 训练循环。

## 启动与查看进度

脚本根据自身位置寻找仓库，可以从其他目录通过绝对路径调用。`GO2_PYTHON` 可指定已有 Python，默认使用仓库 `.venv/bin/python`；`CANN_ENV_FILE` 可指定要加载的 CANN 环境脚本。卡号、环境数和 PPO batch 配置在 YAML 中，默认四卡配置要求物理 8–11 可用。

在仓库根目录执行：

```bash
.venv/bin/python examples/embodiment/go2_ascend/launch_training.py --dry-run
.venv/bin/python examples/embodiment/go2_ascend/launch_training.py
.venv/bin/python examples/embodiment/go2_ascend/status.py
```

启动器创建独立后台会话，检查重复驱动和非空输出目录，并记录 Git 提交、工作区是否有改动、源码摘要和实际配置。默认输出到仓库内 `outputs/go2_ascend/from_scratch`。用 `--output-dir /path/to/run` 指定其他目录，监控和恢复时传入相同参数。

训练退出后，从最近完整检查点恢复：

```bash
.venv/bin/python examples/embodiment/go2_ascend/launch_training.py --resume
```

须保留完整检查点。恢复模型、优化器、调度器、RLinf 保存的 RNG 和环境课程；物理状态及每环境 RNG 重新初始化。改变卡数也会改变随机轨迹。

必须使用这里的入口，才能启用旧基线对应的平方价值损失、解析 KL 自适应学习率、全局优势归一化及独立保存/评估间隔。全局 minibatch 为 24576，四卡每卡 microbatch 为 6144。更改卡数时，应相应调整 microbatch，保持每个本地 minibatch 只有一个 microbatch。

前台验证和 Hydra 参数覆盖：

```bash
bash examples/embodiment/go2_ascend/run_stage.sh single_smoke
bash examples/embodiment/go2_ascend/run_stage.sh from_scratch --cfg job
```

smoke 与 resume 配置各有输出目录，仍保留已验证的单卡 8 或八卡 8–15 放置。`verify_math.py` 的数学对照另需旧 RSL-RL 包，正式训练不依赖它。

## 两台机器怎么同步

本机修改后提交并推送 GitHub，在服务器新目录检出指定提交，安装依赖、编译和验证后启动。不要在正在运行的实验目录直接 pull 或覆盖源码。

hkust 当前训练继续使用 `3148620a7c535d5afacb5ec1b09912a489cacb2b` 快照和当时记录的外部扩展。本次只把这些逻辑纳入仓库，改造路径解析和启动记录；迁移入口经过本地验证，尚未替换正在运行的 NPU 作业。日志和权重是运行产物，不是第二套代码。
