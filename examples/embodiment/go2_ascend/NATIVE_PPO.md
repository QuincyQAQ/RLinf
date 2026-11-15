# Go2 with native RLinf PPO

This experiment uses the standard `train_embodied_agent.py` entry, `EmbodiedRunner`, `EmbodiedFSDPActor`, rollout/env workers and native `actor_critic` loss. `run_native.sh` unsets `RLINF_EXT_MODULE`. It does not load the compatibility extension or an RSL-RL training loop.

The previous compatibility run was paused at update 37600 with a complete resumable checkpoint. Its original directory, launch configuration and resume command are retained in `/data1/qli736/go2_rlinf/experiments/from_scratch/PAUSED.json`; the complete checkpoint archive is also backed up locally. The new run uses a separate source deployment and starts from random initialization.

Native PPO uses the repository's clipped policy objective, clipped Huber value loss, GAE and constant .001 learning rate. The compatibility run used squared value loss and analytic-KL adaptive learning rate. This is an Ascend resource configuration for the existing Go2 recipe, not a claim that these are universally optimal official hyperparameters.

The full run keeps 4096 environments, 24 rollout steps, 5 epochs and 4 minibatches. Global minibatch is 24576, local microbatch is 6144 on four NPUs (physical 8–11), with 160 simulation threads. Network, rewards, gamma .99, lambda .95, clip .2 and entropy .01 remain unchanged. Native runner scheduling requires compatible evaluation/save intervals: both are 1000, with 50000 target updates.

Use the CANN setup and environment documented in the main README. `GO2_PYTHON` selects an existing matching Python; `CANN_ENV_FILE` selects its setup script. Compile the MuJoCo extension on the target architecture.

```bash
bash examples/embodiment/go2_ascend/run_native.sh go2_mujoco_ppo_ascend_smoke
bash examples/embodiment/go2_ascend/run_native.sh go2_mujoco_ppo_ascend_smoke \
  runner.resume_dir=$PWD/outputs/go2_native/single_smoke/single_smoke/checkpoints/global_step_3 \
  runner.max_epochs=5 runner.max_steps=5 \
  runner.logger.log_path=$PWD/outputs/go2_native/single_resume \
  runner.logger.experiment_name=single_resume
bash examples/embodiment/go2_ascend/run_native.sh go2_mujoco_ppo_ascend_four_smoke
.venv/bin/python examples/embodiment/go2_ascend/launch_training.py --native --dry-run
.venv/bin/python examples/embodiment/go2_ascend/launch_training.py --native
.venv/bin/python examples/embodiment/go2_ascend/status.py --output-dir $PWD/outputs/go2_native/from_scratch
```

The full run writes to `outputs/go2_native/from_scratch`. Use `--output-dir` for another location, and `--native --resume` only after the driver exits. Native checkpoints restore model, optimizer, scheduler and saved RNG; the standard environment worker restarts physical state, per-environment RNG and velocity curriculum. This differs from the compatibility worker's curriculum restoration.

Built-in evaluation uses deterministic actions, forward .4m/s, 16 environments, 20 seconds and no observation noise or physical randomization. Training retains noise, randomization and pushes. No task-success metric is currently defined; episode length and return must not be presented as success rate.

The small return increase between updates 1000 and 34000 can reflect early acquisition of the basic task or limited further improvement. A single command and two checkpoints do not distinguish these explanations. Evaluate untrained/early and later models with the same commands, seeds and success criteria. Native PPO does not guarantee a larger improvement. Actual validation results and deployment commit are recorded with the experiment; this document describes the procedure rather than claiming execution.
