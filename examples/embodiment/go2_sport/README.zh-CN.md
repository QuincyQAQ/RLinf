# Go2 六方向控制

[English](README.md)

Go2 EDU 的 Sport 控制器接收机身速度指令，支持前进、后退、左右横移和左右转向。将网卡名称填入以下命令即可预览前进指令：

```bash
python -m examples.embodiment.go2_sport.run --interface <interface> --direction forward --dry-run
```

预览不会建立 SDK 通道。安装步骤、只读状态检查和现场操作见[真机指南](../../../docs/source-zh/rst_source/examples/embodied/go2.rst)。现场请备好遥控器并观察停稳情况。指令或状态超时后，确认机器人停稳，再重新连接。

如需在 CPU 上用已训练策略生成关节目标，请参阅 [ABC 回放](../go2_abc/MOTION.zh-CN.md)。
