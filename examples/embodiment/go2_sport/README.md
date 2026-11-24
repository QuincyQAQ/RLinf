# Go2 six-direction control

[简体中文](README.zh-CN.md)

The Go2 EDU Sport controller accepts body velocity commands for forward, backward, left, right, and turns in both directions. To preview a forward command, supply the robot network interface:

```bash
python -m examples.embodiment.go2_sport.run --interface <interface> --direction forward --dry-run
```

The preview opens no SDK channels. For installation, read-only state checks, and the on-site procedure, see the [hardware guide](../../../docs/source-en/rst_source/examples/embodied/go2.rst). Keep the remote available and observe the stop on site. After a command or state timeout, confirm the robot is stationary before reconnecting.

To generate joint targets from a trained policy on CPU, use [ABC replay](../go2_abc/MOTION.md).
