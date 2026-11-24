# Replay a Go2 ABC policy

[简体中文](MOTION.zh-CN.md)

The CPU replay loads a Go2 checkpoint and converts its twelve joint-position offsets into joint targets and torques in SDK order. The [Sport example](../go2_sport/README.md) sends body velocity commands through the robot's built-in gait controller.

## Replay on CPU

From the repository root, pass the RLinf actor’s `full_weights.pt` or an RSL-RL checkpoint, and select its task. Use a new path for the output file.

```bash
python -m examples.embodiment.go2_abc.deploy \
  --task A --checkpoint path/to/policy.pt \
  --frames examples/embodiment/go2_abc/frames.example.jsonl \
  --output commands.jsonl
```

For weights arranged as `models/<task>/<group>.pt`, replace `--checkpoint path/to/policy.pt` with `--group rlinf --data path/to/data`. Groups are `baseline`, `ppo`, and `rlinf`. When `--data` is omitted, replay uses `GO2_ABC_DATA` or the repository's `data/go2_abc` directory.

The bundled file contains four synthetic neutral-pose frames. Each output line contains the timestamp, joint targets in radians, and torques in Nm. Replay evaluates one policy action every four 5-ms samples. The motor model uses position and velocity gains of 25 and 0.5, a torque limit of 23.5 Nm, and a speed limit of 30 rad/s. [Control parameters](control.json)

## Record inputs

Use the field names and units in [the sample frames](frames.example.jsonl). The 235 policy inputs include twelve joint positions, twelve velocities, IMU data, the previous action, a body velocity command, measured body linear velocity, and 187 terrain heights. Supply body linear velocity and terrain heights from your perception system. Use a yaw-aligned grid spaced at 0.1 m. For each y from −0.5 to 0.5 m, list x from −0.8 to 0.8 m (17 values per row). Express terrain height and base height in the same world frame and timestamp all sources with the same monotonic clock.

To inspect joint and IMU data on a robot, install the [Unitree SDK2 Python package](https://github.com/unitreerobotics/unitree_sdk2_python/tree/814556d15970dd2ecf1c9984e845ca02ab07e206) and run the following command from a fresh process. Specify the interface attached to the Go2.

```bash
python -m examples.embodiment.go2_abc.sdk2 --interface <interface>
```

The command subscribes to `rt/lowstate`, waits up to one second for a fresh checksum-valid sample, logs it, and disconnects. The robot's built-in controller keeps motor ownership. The ABC example sends no motor commands.
