# Go2 Six-Direction Hardware Control

[中文](README.zh-CN.md)

Use RLinf to move the Go2 forward, backward, sideways left or right, and turn left or right. The Unitree controller handles gait and joint control.

The operator confirmed motion and stopping in all six directions on 2026-11-16; see the [historical acceptance archive](https://github.com/QuincyQAQ/RLinf/blob/a253e5e657299103fa5e766c24b0f05eb5fb854f/artifacts/go2_sport_20261116/README.md). This migration preserves the driver behavior and checks compatibility with this RLinf revision. The migrated version has not been retested on hardware.

## Code Locations

The layout follows [AGENTS.md](../../../AGENTS.md) and [Adding a Robot](../../../docs/source-en/rst_source/extending/new_robot.rst).

| Location | Purpose |
|---|---|
| [rlinf/robotics/parts/mobility/unitree_go2.py](../../../rlinf/robotics/parts/mobility/unitree_go2.py) | SDK2 state reception, velocity commands, and timeout stop requests |
| [rlinf/robotics/robots/unitree_go2.py](../../../rlinf/robotics/robots/unitree_go2.py) | Robot composition and hardware configuration |
| [rlinf/envs/real/go2/](../../../rlinf/envs/real/go2/) | Registered `Go2DriveEnv-v1` environment |
| [run.py](run.py) | Read-only checks and manual single-direction tests |
| [Historical acceptance archive](https://github.com/QuincyQAQ/RLinf/tree/a253e5e657299103fa5e766c24b0f05eb5fb854f/artifacts/go2_sport_20261116) | Original JSON reports, script snapshots, and SHA-256 manifest |
| [tests/unit_tests/test_robotics.py](../../../tests/unit_tests/test_robotics.py) | Tests with a mocked SDK for the driver, all six directions, and cleanup |

## Usage

Follow the [installation and run guide](../../../docs/source-en/rst_source/examples/embodied/go2.rst). From the repository root, preview one direction, replacing `<interface>` with the network interface attached to the Go2:

```bash
python -m examples.embodiment.go2_sport.run --interface <interface> --direction forward --dry-run
```

The preview does not connect hardware. Direction names are `forward`, `backward`, `left`, `right`, `turn_left`, and `turn_right`. For motion, verify standing posture, clear space, and a ready remote on site, then remove `--dry-run` and enter `GO`. Each run tests one direction, with a default command duration of one second; use `--seconds 2` for two seconds. Reports go to `outputs/go2_sport/`.

## Relationship to ABC Control

This example sends three body velocity values without loading policy weights. [go2_abc](../go2_abc/MOTION.md) loads an ABC policy, produces twelve joint actions, and converts them to motor torque. Its hardware validation is tracked separately.

While connected, the watchdog retries an unconfirmed stop after 200 ms. Even after the SDK acknowledges a retry, motion requires a new connection. Confirm the robot has stopped on site.

## Limitations

The archive records operator-confirmed functionality. Quantitative distance, angle, and stopping-distance acceptance remains incomplete. SDK velocity estimates and stop requests do not replace direct observation; process exit and network loss require an independent stop mechanism. The environment returns zero reward and does not provide a real-robot RL task.
