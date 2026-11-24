# Go2 六方向真机控制

[English](README.md)

通过 RLinf 控制 Go2 前进、后退、左移、右移、左转和右转。宇树原厂控制器负责步态与关节控制。

2026-11-16 的六方向运动和停稳已获现场操作者确认，见[历史验收归档](https://github.com/QuincyQAQ/RLinf/blob/a253e5e657299103fa5e766c24b0f05eb5fb854f/artifacts/go2_sport_20261116/README.zh-CN.md)。本次迁移保留驱动行为，并检查当前 RLinf 接口兼容性；迁移版本尚未重新做真机测试。

## 代码位置

目录遵循仓库 [AGENTS.md](../../../AGENTS.md) 和[新增机器人说明](../../../docs/source-zh/rst_source/extending/new_robot.rst)。

| 位置 | 内容 |
|---|---|
| [rlinf/robotics/parts/mobility/unitree_go2.py](../../../rlinf/robotics/parts/mobility/unitree_go2.py) | 原厂 SDK2 状态订阅、速度指令、超时停止请求 |
| [rlinf/robotics/robots/unitree_go2.py](../../../rlinf/robotics/robots/unitree_go2.py) | 机器人组合与硬件配置 |
| [rlinf/envs/real/go2/](../../../rlinf/envs/real/go2/) | 注册环境 `Go2DriveEnv-v1` |
| [run.py](run.py) | 只读检查与单方向手动测试入口 |
| [历史验收归档](https://github.com/QuincyQAQ/RLinf/tree/a253e5e657299103fa5e766c24b0f05eb5fb854f/artifacts/go2_sport_20261116) | 原始 JSON、脚本快照与 SHA-256 清单 |
| [tests/unit_tests/test_robotics.py](../../../tests/unit_tests/test_robotics.py) | 驱动、六方向及退出路径的 SDK 替身测试 |

## 使用

按[安装与运行说明](../../../docs/source-zh/rst_source/examples/embodied/go2.rst)准备依赖。在仓库根目录预览单个方向，`<interface>` 替换为连接 Go2 的网卡名称：

```bash
python -m examples.embodiment.go2_sport.run --interface <interface> --direction forward --dry-run
```

预览不连接硬件。六个方向名为 `forward`、`backward`、`left`、`right`、`turn_left`、`turn_right`。实际运动前须在现场核实站立状态、场地和遥控器，再移除 `--dry-run` 并输入 `GO`。单次运行只测一个方向，默认指令持续时间为 1 秒；`--seconds 2` 可改为 2 秒。日志写入 `outputs/go2_sport/`。

## 与 ABC 运控的关系

本目录发送三维机身速度指令，不加载训练权重。[go2_abc](../go2_abc/MOTION.zh-CN.md) 则加载 ABC policy，输出 12 关节动作并换算电机力矩；其真机验证单独记录。

同一进程内请先启动适配器，让它确认 SDK2 绑定的网卡。YAML 中的 `enable_motion` 须填写不带引号的布尔值。如果 `Move` 返回时已超过主机侧的截止时间，驱动会报错并请求停车。连接期间，SDK 未确认停车请求时，看门狗会在 200 毫秒后重试。指令或状态超时后，即使发送零速度或重置环境，驱动也会保持故障状态。请在现场确认机器人已停稳，再重新连接并发送运动指令。

## Limitations

六方向归档属于现场功能确认，距离、转角和停止距离的量化验收尚未完成。SDK 速度估计与停止请求均不能代替现场观察；进程退出、网络断开等情况下仍需独立停止手段。环境奖励为零，尚未提供真机强化学习任务。
