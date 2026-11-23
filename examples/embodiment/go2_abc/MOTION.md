# Go2 Motion Control

[简体中文](MOTION.zh-CN.md)

Use the nine ABC policies through a shared observation and motor interface. Velocity commands enter the 235-value observation; the policy outputs twelve joint-position offsets. The recorded DC-motor model converts them to SDK2 motor torques.

[Operator-confirmed six-direction control](../go2_sport/README.md) uses the original SportClient and has a separate entry point and acceptance archive.

## Control Contract

The adapter preserves the simulator's joint order, default posture, action scale `0.25`, gains `25/0.5`, and torque-speed limits `23.5 Nm / 30 rad/s`. Policy inference runs every four 5-ms motor steps. SDK messages disable position PD to avoid applying the gains twice.

The observation has 48 proprioceptive/command values and 187 terrain values. Provide measured body velocity and ground heights on the yaw-aligned grid returned by `scan_xy()`. Heights and base height share one world frame; the scan subtracts `0.5 m` and clips to `[-1, 1]`. Sensor timestamps use the control host's monotonic clock. [Parameters](control.json) pin the source versions.

## Replay on CPU

After preparing the [ABC inputs](README.md#prepare-inputs), replay the supplied synthetic neutral-pose frames. This command opens no SDK2 channels.

```bash
python -m examples.embodiment.go2_abc.deploy \
  --task A --group rlinf \
  --frames examples/embodiment/go2_abc/frames.example.jsonl \
  --output /tmp/go2-A-rlinf-commands.jsonl
```

The output contains four frames of SDK-ordered joint targets and torques. Select A/B/C and baseline/ppo/rlinf to use any of the nine weights. Replace the input with JSONL measurements using the sample's fields for recorded replay.

## Connect SDK2

Install [Unitree SDK2 Python](https://github.com/unitreerobotics/unitree_sdk2_python/tree/814556d15970dd2ecf1c9984e845ca02ab07e206) in the control computer's environment. `Go2SDK2(interface)` subscribes to checksum-valid joint/IMU state and defaults to read-only operation.

Integrate `run_control_loop()` with callbacks supplying measured perception and velocity commands. Sending torques requires `enable_motion=True`, an operator-released built-in controller, the recorded neutral pose and an initial zero velocity command. The adapter checks state age, rejects incomplete terrain data and closes its channels on errors. It requests zero torque during cleanup after publishing.

## Limitations

CPU tests compare observations and motor outputs with the original Isaac Lab functions and test SDK2 calls through vendor fakes. Physical deployment, sensor calibration, terrain/velocity estimation and independent stop/recovery still require hardware validation; zero-torque cleanup is not a verified physical emergency stop.
