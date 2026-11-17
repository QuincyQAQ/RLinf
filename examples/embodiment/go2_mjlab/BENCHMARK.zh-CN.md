# Go2 的 Mjlab 跨任务评估

用 RLinf 分别训练五类专项策略，再让每个策略接受同样的五项测试，组成跨任务矩阵。这是[参考项目](https://github.com/viktoriageorrg04/bsc-thesis-multi-task-RL-env/tree/743ac04d9543b04b7854f1100bfa0cc69c993106)的 `rlinf-mjlab-cross-task-v1` 移植版，结果与原 Isaac Lab 矩阵、旧的 Flat 策略地形测试分开保存。

## 任务和观测

五个配置沿用 `go2_mjlab_benchmark_a1_ascend` 的命名格式，中间的 `a1` 可换成 `a2`、`b1`、`b2` 或 `c2`，依次对应平地前进、平地全向、粗糙地形、阶梯和跨沟。Actor 和 Critic 均接收 235 维观测：机体线速度 3 维、角速度 3 维、重力投影 3 维、指令 3 维、相对关节位置 12 维、相对关节速度 12 维、上次动作 12 维、高度扫描 187 维。动作是 12 维关节偏移，缩放系数为 0.25。新模型从头初始化，旧的 47/74 维 Flat 配置和检查点保留原接口。

新任务继承 RLinf PPO、隐藏层宽度为 512、256、128、使用 ELU 激活函数的网络、归一化 epsilon 0.01 和 10001 次更新预算，关闭初始回合长度随机化和 Flat 指令课程。这里的更新次数是新版本配置，不是参考项目的训练时长。独立训练使用不同的 Actor、环境种子，并分别保存输出。

## 先做短验证

先按 [README.zh-CN.md](README.zh-CN.md) 准备环境，并确认设备可用。短验证配置使用 8 号卡、16 个环境、3 次 PPO 更新，每次保存检查点：

```bash
bash examples/embodiment/go2_ascend/run_native.sh go2_mjlab_benchmark_a1_smoke
```

原训练占用 8 号卡期间不要运行。五项任务已在本机通过 CPU 仿真短测，A1 也已在 hkust 通过 CPU 验证；设备空闲后再验证此版本的加速器训练。当前没有启动新的长训练。

短验证通过后，专项训练使用独立输出目录：

```bash
bash examples/embodiment/go2_ascend/run_native.sh go2_mjlab_benchmark_a1_ascend \
  runner.logger.log_path=outputs/go2_cross_task/A1_forward_seed0 \
  actor.seed=0 env.train.seed=0
```

正式配置使用 8–15 号卡和 4096 个环境，应等原任务释放设备后再启动。

## 评估和导出

传入保存的完整模型状态和解析后的训练配置；只含 Hydra defaults 的配置文件不够。每个进程只评估一项任务。下面用 A1 模型测试阶梯任务：

```bash
PYTHONPATH="$PWD" "$GO2_PYTHON" examples/embodiment/go2_mjlab/evaluate_benchmark.py \
  --checkpoint /path/to/full_weights.pt \
  --training-config /path/to/resolved_config.yaml \
  --eval-task B2_stairs --num-envs 64 --num-episodes 1024 \
  --output outputs/go2_cross_task/evaluation_step1000
```

对五项任务分别运行，矩阵同一行使用相同检查点、训练配置和评估种子。评估器拒绝 47 维旧模型，也不会把不同权重的结果混入一行。每个单元格保存完整回合记录和评估协议，行汇总保存在 `summary.json`。请求少于 1024 个回合时标记为短测，默认不进入正式矩阵；向量环境可能在最后一批多收集若干回合。

```bash
PYTHONPATH="$PWD" "$GO2_PYTHON" examples/embodiment/go2_mjlab/export_benchmark_matrix.py \
  --results-root outputs/go2_cross_task/evaluation_step1000
```

导出 JSON、CSV 和 Markdown。没有测量的单元格保留 null，不用零代替，也不复制其他模型的结果。每个专项策略应独立训练 3 个种子，再计算各单元格的样本标准差。本版提供五个专项任务；MTL 行需要另外实现并训练多任务策略。

## 成功条件与差异

某一步同时满足线速度、偏航误差门槛，且没有失败终止，才计为成功步。整回合没有失败终止，并且成功步比例达标，才算回合成功；起步阶段也计入。这不是 RMSE 门槛，也不是到达终点或穿过全部地形的判据。

| 任务 | 平面速度误差 m/s | 偏航速度误差 rad/s | 成功步比例下限 |
|---|---:|---:|---:|
| A1 | 0.20 | 0.40 | 0.85 |
| A2 | 0.25 | 0.50 | 0.80 |
| B1 | 0.35 | 0.60 | 0.75 |
| B2 | 0.40 | 0.70 | 0.70 |
| C2 | 0.45 | 0.75 | 0.65 |

本版使用 MuJoCo 物理、固定版本的 Unitree Mjlab 机器人、执行器、奖励和启动随机化，移植参考项目的任务指令范围、10 秒指令重采样、成功门槛和机身接触地面的失败判据。A1、B2、C2 直接采样偏航速度，A2、B1 使用航向指令。回合均为 20 秒。B1 使用 Mjlab 随机粗糙地形，B2 使用 4–16 cm 倒金字塔阶梯；C2 用盒体构造 2 m 中央平台周围的方形沟隙，宽 0–20 cm、深 1 m，替代原项目网格生成器。这些差异意味着结果不能当作原项目数值的直接复现。

Actor 高度扫描减去 0.5 偏移后裁剪至 [-1,1]，坐标系和关节顺序沿用 Mjlab。评估时关闭观测噪声、外部推力和地形自适应，从生成地形的全部 10 个难度行中采样，保留启动随机化。跟踪误差与参考代码一样在步进后读取，回合结束时包含自动重置后的状态。上述设置属于本版协议，与参考评估继承的课程、随机化行为存在差异。矩阵非对角单元格仍衡量跨任务迁移，不代表每个专项模型都训练过所有任务。
