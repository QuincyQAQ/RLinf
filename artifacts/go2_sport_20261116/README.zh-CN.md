# Go2 六方向验收归档

[English](README.md)

[basic_control_acceptance_20261116.json](basic_control_acceptance_20261116.json) 汇总了 2026-11-16 的六方向功能验收。前进、后退、左移、右移、左转、右转的 `operator_confirmed_motion_and_stop` 均为 `true`，依据是操作者在现场观察后的反馈。

六份原始运行报告按汇总中的 `report` 字段保存在同一目录。部分报告写于操作者反馈前，仍保留“等待确认”的原文字段；现场确认结果以汇总记录为准。JSON 文件逐字复制，哈希及源路径见 [manifest.json](manifest.json)。

`source_snapshot/` 保存迁移时旧目录中的脚本副本，包括已停用的早期入口，以 `.py.txt` 留档；这些文件不保证是每次实测时的精确脚本版本。当前可运行入口和目录说明见 [go2_sport](../../examples/embodiment/go2_sport/README.zh-CN.md)。

## 验证范围

这是 RLinf 调用原厂 SportClient 的历史真机结果。`quantitative_acceptance_complete` 为 `false`，距离与转角的量化验收尚未完成；`migration_hardware_retest` 为 `false`，本次代码迁移没有重新运行真机实验。ABC policy 的关节控制验证不包含在这份记录中。
