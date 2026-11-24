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

One Sport or ABC adapter can connect per process. Start a new process to change the DDS network interface. Before sending each torque command, ABC checks the control deadline and the age of the joint state and perception samples used to compute it.

Pass measured perception and velocity callbacks to `run_control_loop()` for read-only inference. For motion, `Go2SDK2(interface, enable_motion=True, acquire_control=...)` requires an independent controller for the same robot. It owns the sole motor publisher and grants an exclusive `SupervisedControl` session. Acquisition must arrange a watchdog that takes over if the client disappears, and roll back if acquisition fails. This repository does not supply that hardware service. Keep the built-in controller engaged when using the read-only path.

When the motion duration expires, the policy keeps running with a zero velocity command. The supervisor must report fresh contact, body stability and motor ownership during both phases. Once all four feet support the upright, stationary robot for 0.5 continuous seconds, the client requests takeover. It then confirms another 0.5 seconds of measured support with the standing controller in charge before returning. That controller continues supporting the robot after client exit.

Each phase has a 3-second deadline and uses the host monotonic clock. A gap over 50 ms restarts the stability window. A timeout raises an error and requests independent takeover during cleanup.

A control fault relinquishes the policy session immediately. Cleanup closes the local subscription without sending a motor packet. Communication loss and process death require the independent watchdog to take over.

In a flat-ground MuJoCo test using the A baseline, zero velocity was requested at 0.6, 1.2 and 3.0 seconds of forward motion. All three runs reached the 0.5-second standing criterion within 0.61 seconds of the request, then maintained four-foot support throughout the final second after simulated takeover. This tested the stop sequence with simulated sensing and actuation.

## Limitations

CPU tests cover observation and motor parity, handover ordering, stale feedback and simulated faults. Contact and posture calibration, hardware takeover, link loss and recovery require validation on the target robot. The timing windows above are software acceptance criteria.
