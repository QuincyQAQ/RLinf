# Go2 固定 seed 复现归档

[English](README.md)

本目录保存 A40 实际使用的辅助源码、启动脚本、输入哈希与复现证据，训练版本固定为 `e8815cac3b3abecedf13e9310aa9661baa6d1043`。分支包含完整 RLinf 实现、Go2 环境、解析 KL 调度、观测与控制接口，以及 A/B/C 配置。

## 已验证结果

C 的两次独立 RLinf-Huber 续训，作业 14060127、14061035，均从同一份冻结 PPO 基线权重出发，训练 seed 0、续训 1000 轮。20 份 checkpoint 的全部张量完全一致；2048 条评测记录和两份汇总完全一致：跟踪成功率 100%，楼梯通过率 925/2048（45.1660%）。评测 seed 为 23456、34567。[evidence](evidence) 保存报告与逐回合数据。

A、B 六项均已完成。按基线、继续 PPO、继续 RLinf-Huber 排列，跟踪成功率为 A：91.4687%、93.3301%、96.7285%；B：99.8535%、99.6094%、99.8535%。B 使用 [manifest.json](manifest.json) 中的重提作业。基线组评测冻结权重；普通 PPO、RLinf-Huber 各续训 1000 轮。两任务均使用 2048 个环境，保留原任务配置及 B 的数值保护。

## 恢复代码并运行

从本分支独立克隆并检出固定训练版本，再恢复启动脚本：

```bash
export GO2_SNAPSHOT="$PWD/artifacts/go2_seed0_20261122"
git clone --depth 1 --branch repro/go2-abc-seed0-verified-20261122 \
  https://github.com/QuincyQAQ/RLinf.git ../go2-seed0-run
git -C ../go2-seed0-run fetch --depth 1 origin e8815cac3b3abecedf13e9310aa9661baa6d1043
git -C ../go2-seed0-run checkout --detach e8815cac3b3abecedf13e9310aa9661baa6d1043
cd ../go2-seed0-run
tar -xzf "$GO2_SNAPSHOT/launchers.tar.gz"
python -m examples.embodiment.go2_abc.prepare_data
```

使用[已准备的 A40 环境](https://github.com/QuincyQAQ/RLinf/tree/b405f67382ce63af2401a2a9ff5b0b3c7f28ccb7)，包括 Isaac Lab 2.3.2、Isaac Sim 5.1、RSL-RL 3.1.2。另建运行环境副本，将辅助源码恢复到副本中。设置 `GO2_ABC_PROJECT` 为新工作目录，`GO2_ABC_RUNTIME` 为环境副本，`GO2_ABC_IMAGE`、`GO2_ABC_CONTAINER` 分别为容器镜像和 Singularity 路径。共用已有资产与依赖包时，设置 `GO2_ABC_SHARED_RUNTIME` 为其所在环境目录。

```bash
tar -xzf "$GO2_SNAPSHOT/runtime_sources.tar.gz" -C "$GO2_ABC_RUNTIME"
bash examples/embodiment/go2_abc/container.sh --python \
  /workspace/project/data/go2_abc/AB_six_20261122/dispatch.py cpu-audit
```

CPU 预检核验输入并预览六项命令。分配到 A40 后，续训组自动完成短程更新检查、正式续训和评测；基线组仅评测。每项单独提交并使用独立日志路径，例如：

```bash
sbatch --export=ALL --job-name=go2_A_rlinf_seed0 \
  --output=data/go2_abc/AB_six_20261122/A_rlinf/slurm-%j.out \
  --error=data/go2_abc/AB_six_20261122/A_rlinf/slurm-%j.err \
  data/go2_abc/AB_six_20261122/run.sbatch A_rlinf
```

六项名称为 `A_baseline`、`A_ppo`、`A_rlinf`、`B_baseline`、`B_ppo`、`B_rlinf`。归档的 `submit.sh` 保留学校服务器原始提交流程；在其他安装目录使用前，要修改其中的环境文件路径。输出目录使用新目录，seed、权重哈希、环境数量及依赖版本保持与 [manifest.json](manifest.json) 一致。

重新验证 C 的两次训练时，将归档内 `verify_seed0.py` 复制到 `data/go2_abc/C_repeat/`。在分配好的 A40 作业中，以 `bash examples/embodiment/go2_abc/container.sh --python /workspace/project/data/go2_abc/C_repeat/verify_seed0.py` 为命令前缀，分别执行 `run --stage full --repeat 1`、`run --stage full --repeat 2`，最后执行 `compare --stage full`，逐项比较全部保存权重和评测回合。

## Limitations

目前已确认 C 在记录环境中的逐项复现，A、B 各完成一次运行；独立训练重复一致性目前只验证了 C。历史 C 的 76.5625% 属于另一份模型，已复现其评测，尚未复现其训练成绩。只固定 seed 不能保证跨硬件或软件版本完全一致。本归档包含源码与证据；容器二进制、仿真资产和依赖包仍需准备好的运行环境。辅助文件中的验证标记记录原环境状态，不能替代新安装的检查。
