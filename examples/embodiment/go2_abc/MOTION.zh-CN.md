# Go2 ABC 策略回放

[English](MOTION.md)

CPU 回放加载 Go2 权重，把策略输出的 12 个关节位置偏移转换为 SDK 顺序的目标角度和力矩。[Sport 示例](../go2_sport/README.zh-CN.md)可以向机器人的原厂步态控制器发送机身速度指令。

## 在 CPU 上回放

在仓库根目录运行以下命令。传入对应任务的 RLinf actor 权重 `full_weights.pt` 或 RSL-RL 检查点，并指定一个新的输出文件。

```bash
python -m examples.embodiment.go2_abc.deploy \
  --task A --checkpoint path/to/policy.pt \
  --frames examples/embodiment/go2_abc/frames.example.jsonl \
  --output commands.jsonl
```

如果权重按 `models/<任务>/<组别>.pt` 存放，可以用 `--group rlinf --data path/to/data` 替代 `--checkpoint path/to/policy.pt`。组别支持 `baseline`、`ppo` 和 `rlinf`。省略 `--data` 时，程序读取 `GO2_ABC_DATA`，默认目录为仓库下的 `data/go2_abc`。

示例输入有四帧合成的中立姿态数据。输出每行包含时间戳、目标角度（rad）和电机力矩（Nm）。电机模型每 5 ms 计算一次，策略每四帧更新一次动作。位置和速度增益分别为 25 和 0.5，力矩上限为 23.5 Nm，转速上限为 30 rad/s。[控制参数](control.json)

## 采集输入

输入字段和单位见[示例帧](frames.example.jsonl)。策略的 235 维输入包含 12 个关节的位置和速度、IMU 数据、上次动作、速度指令、实测机身线速度和 187 个地形高度值。机身线速度和地形高度由感知系统提供。高度网格随机身偏航角旋转，间距为 0.1 m。按 y 从 −0.5 m 到 0.5 m 排列各行，每行按 x 从 −0.8 m 到 0.8 m 写入 17 个采样值。机身高度与地面高度使用同一世界坐标系，各数据源使用同一单调时钟记录时间戳。

如需查看真机关节和 IMU 数据，先安装 [Unitree SDK2 Python](https://github.com/unitreerobotics/unitree_sdk2_python/tree/814556d15970dd2ecf1c9984e845ca02ab07e206)。在新进程中运行下列命令，填入连接 Go2 的网卡名。

```bash
python -m examples.embodiment.go2_abc.sdk2 --interface <interface>
```

程序订阅 `rt/lowstate`，最多等待一秒，读到新鲜且通过校验的数据后输出样本并断开。原厂控制器继续控制电机。ABC 示例不会发送电机指令。
