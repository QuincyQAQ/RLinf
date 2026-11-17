> [Native RLinf PPO](NATIVE_PPO.md) — the compatibility run is paused at update 37600; use the native entry for the new experiment.

# Go2 training on Ascend

Train Go2 with CPU MuJoCo simulation and RLinf sampling, PPO updates, FSDP/HCCL synchronization and native checkpoints on Ascend. This repository contains the complete training implementation. The server is a deployment of a pinned source version, not a second implementation.

The active experiment uses four NPUs (physical IDs 8–11), 160 simulation threads, 4096 environments and 50000 target updates. Earlier validation covered one and eight NPUs and checkpoint restoration. Final training quality has not yet been accepted. The historical record is [TRAINING_RECORD.zh-CN.md](TRAINING_RECORD.zh-CN.md); its server paths describe the already-running snapshot, not the commands for this integrated entry.

## Install

Supply compatible Ascend drivers, CANN, Torch 2.6.0 and torch_npu 2.6.0.post5. Create a project virtual environment from a Python installation providing that stack, then run from the repository root:

```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r examples/embodiment/go2_ascend/requirements-ascend.txt \
  -c examples/embodiment/go2_ascend/constraints-ascend.txt
source /usr/local/Ascend/ascend-toolkit/set_env.sh
.venv/bin/python -m rlinf.envs.sim.go2_mujoco.build
```

The CANN location is an example; source the location installed on your host. Its Python bindings must be importable by Ray when required by the installed stack. Compile the simulation extension on the target CPU architecture. No checkpoint or old RSL-RL training loop is required to start fresh.

## Launch and monitor

Run from any directory by giving the script's path. `GO2_PYTHON` selects an existing Python; it defaults to this checkout's `.venv/bin/python`. `CANN_ENV_FILE` optionally selects a CANN setup script. Hardware placement, environment count and PPO batch sizes live in the YAML files. The checked-in four-NPU configuration requires physical IDs 8–11 to be available.

From the repository root:

```bash
.venv/bin/python examples/embodiment/go2_ascend/launch_training.py --dry-run
.venv/bin/python examples/embodiment/go2_ascend/launch_training.py
.venv/bin/python examples/embodiment/go2_ascend/status.py
```

The launcher starts a background session, rejects an existing live driver or a nonempty fresh output directory, and records the Git commit, dirty state, code hashes, placement and configuration. The default output is `outputs/go2_ascend/from_scratch` inside this checkout. Use `--output-dir /path/to/run` for another location; pass the same argument to the status script and resume command.

```bash
.venv/bin/python examples/embodiment/go2_ascend/launch_training.py --resume
```

Resume only after the driver exits. Keep the entire checkpoint directory. Model, optimizer, scheduler and saved RLinf RNG are restored; the environment curriculum is restored, while physical state and per-environment RNG restart. Changing world size changes random trajectories.

Use this entry rather than the generic training entry: the extension supplies the original baseline's squared value loss, analytic KL adaptive learning rate, global advantage normalization and independent save/evaluation intervals. Global minibatch is 24576, with microbatch 6144 on each of four NPUs. Before changing world size, adjust microbatch to keep one microbatch per local minibatch.

For direct foreground runs and Hydra overrides:

```bash
bash examples/embodiment/go2_ascend/run_stage.sh single_smoke
bash examples/embodiment/go2_ascend/run_stage.sh from_scratch --cfg job
```

Smoke and resume configurations use separate output directories. Their defaults retain the previously verified placement (one NPU 8, or eight NPUs 8–15). Mathematical comparison in `verify_math.py` additionally requires the old RSL-RL package; the training entry does not.

## One source of truth

Commit and push local changes, then clone/fetch that commit into a separate server checkout. Install dependencies and compile there, validate the configuration, then launch from that checkout. Do not update a running experiment's source directory.

The current hkust run continues using its original snapshot of `3148620a7c535d5afacb5ec1b09912a489cacb2b` plus the recorded external extension. It is intentionally left unchanged. This integration moves that logic into the repository and changes path resolution and launch metadata; the relocated entry has local validation, but has not replaced the active NPU job. Checkpoints and logs are runtime artifacts, not a second codebase.
