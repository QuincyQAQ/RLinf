# Go2 运动控制

[English](MOTION.md)

九组 ABC policy 共用观测和电机接口。速度命令进入 235 维观测，policy 输出 12 个关节位置偏移，再由原实验的 DC 电机模型换算成 SDK2 力矩指令。

此前通过现场验收的[六方向原厂控制](../go2_sport/README.zh-CN.md)使用 SportClient，运行入口和验收记录单独保存。

## 控制接口

适配器保留仿真的关节顺序、默认姿态、`0.25` 动作缩放、`25/0.5` 控制增益，以及 `23.5 Nm / 30 rad/s` 力矩与转速参数。电机控制周期为 5 ms，每四步推理一次。SDK 指令关闭位置 PD，避免重复施加增益。

观测包含 48 个本体状态与命令值，以及 187 个地形值。外部感知模块须提供机身坐标系线速度，以及 `scan_xy()` 返回的网格对应的地面高度。网格只随偏航角旋转；地面高度与机身高度采用同一世界坐标系。高度差减去 `0.5 m`，再裁剪到 `[-1, 1]`。传感时间戳使用控制主机的单调时钟。[参数文件](control.json)记录来源版本。

## CPU 回放

准备好 [ABC 输入](README.zh-CN.md#准备输入)后，用提供的合成静止姿态数据检查控制链。以下命令不会建立 SDK2 通道。

```bash
python -m examples.embodiment.go2_abc.deploy \
  --task A --group rlinf \
  --frames examples/embodiment/go2_abc/frames.example.jsonl \
  --output /tmp/go2-A-rlinf-commands.jsonl
```

输出包含四帧按 SDK 顺序排列的关节目标和力矩。任务可选 A/B/C，组别可选 baseline/ppo/rlinf，覆盖九份权重。离线回放实测数据时，按示例字段替换 JSONL 输入。

## 接入 SDK2

在控制电脑安装固定版本的 [Unitree SDK2 Python](https://github.com/unitreerobotics/unitree_sdk2_python/tree/814556d15970dd2ecf1c9984e845ca02ab07e206)。`Go2SDK2(interface)` 订阅通过校验的关节与 IMU 状态，默认只读。

集成 `run_control_loop()` 时，传入提供实测感知数据和速度命令的两个回调。发布力矩须显式设置 `enable_motion=True`。操作员须提前释放内置控制器，让机器人处于原实验默认姿态，并将首帧速度命令设为零。适配器检查状态时效、拒绝不完整地形数据，出错后关闭通道；若已发布过指令，清理时请求零力矩。

## Limitations

CPU 测试用原 Isaac Lab 函数核对观测和电机输出，用 SDK 替身检查接口调用。真机部署、传感器标定、地形与速度估计，以及独立停止和恢复仍需硬件验证；零力矩清理尚未通过物理急停验证。
