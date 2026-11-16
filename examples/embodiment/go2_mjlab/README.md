# Go2 Mjlab with RLinf PPO

Train the official `Unitree-Go2-Flat` task with RLinf's standard PPO workers. The task comes from Unitree RL Mjlab commit `1425b15f73bd4095f0df53709d7c389c3eb9e790`. Physics runs on CPU; policy inference and PPO updates run on the configured accelerator. The Ascend configuration uses devices 8–15 and 64 CPU environment workers.

## Configuration

The recipe uses 4096 environments, 24 control steps per update, 10001 updates, five PPO epochs, four global minibatches, and a checkpoint every 100 updates. Global minibatch size is 24576; each of eight actors processes 3072 samples. Each environment worker runs 64 environments with two Torch threads.

Actor and critic use 512/256/128 ELU layers, 47/74 observation inputs, observation normalization with epsilon 0.01, and 12 Gaussian joint actions with initial standard deviation 1.0. PPO uses clipped squared value error with coefficient 1.0, clip 0.2, entropy coefficient 0.01, gamma 0.99, GAE lambda 0.95, and maximum gradient norm 1.0. The learning rate starts at 0.001 and adapts between 0.00001 and 0.01 using analytic Gaussian KL with a target of 0.01. `torch_constant` preserves these minibatch LR changes; it does not fix the learning rate.

The upstream task supplies observations, rewards, randomization, command curriculum, contact sensing, and termination conditions. Initial episode lengths are randomized as in the upstream training entry. This configuration starts a separate experiment from random initialization. Do not resume the older 45/60-observation model into it.

## Installation on Ascend

The verified server uses Linux aarch64, glibc 2.34, CANN 8.5, PyTorch 2.7.1, and TorchNPU 2.7.1.post2. Python is 3.10. The PyPI ARM Warp 1.12 wheel fails to load on this host because its native library requires glibc 2.35. The locked conda-forge CPU build of the same version works without changing the operating system.

Prepare `conda` and `uv`, then run from the RLinf repository root. Clone the pinned task beside the repository:

```bash
git clone https://github.com/unitreerobotics/unitree_rl_mjlab.git ../unitree_rl_mjlab
git -C ../unitree_rl_mjlab checkout 1425b15f73bd4095f0df53709d7c389c3eb9e790
```

Create the CPU Warp runtime and an isolated Python overlay. The pip lock excludes Warp so that it remains supplied by conda:

```bash
GO2_RUNTIME_DIR="$PWD/outputs/go2_mjlab_runtime"
conda create --prefix "$GO2_RUNTIME_DIR/warp_cpu" --file examples/embodiment/go2_mjlab/ascend-conda-explicit.lock -y
uv venv --system-site-packages --python "$GO2_RUNTIME_DIR/warp_cpu/bin/python" .venv-mjlab-ascend
uv pip install --python .venv-mjlab-ascend/bin/python --no-deps -r examples/embodiment/go2_mjlab/ascend-requirements.lock
uv pip install --python .venv-mjlab-ascend/bin/python --no-deps -e ../unitree_rl_mjlab
```

These locks target the stated Ascend host. The automatic `--env go2_mjlab` installer and Docker stage cover NVIDIA hosts; use the CPU Warp procedure above for this Ascend configuration.

## Run It

Select the matching CANN setup script and interpreter. First run the single-device smoke configuration:

```bash
export GO2_PYTHON="$PWD/.venv-mjlab-ascend/bin/python"
export CANN_ENV_FILE=/usr/local/Ascend/ascend-toolkit/set_env.sh
bash examples/embodiment/go2_ascend/run_native.sh go2_mjlab_ppo_ascend_smoke
```

This runs 16 environments for three updates on device 8. For a full-scale validation on eight devices, run:

```bash
bash examples/embodiment/go2_ascend/run_native.sh go2_mjlab_ppo_ascend \
  runner.max_epochs=3 runner.max_steps=3 runner.save_interval=1 \
  runner.logger.log_path="$PWD/outputs/go2_mjlab/eight_smoke" \
  runner.logger.experiment_name=eight_smoke
```

Stop the validation process before starting the 10001-update experiment:

```bash
bash examples/embodiment/go2_ascend/run_native.sh go2_mjlab_ppo_ascend
```

Change the placement in `examples/embodiment/config/go2_mjlab_ppo_ascend.yaml` if the host has different device indices. The background helper `launch_training.py` does not compose inherited Hydra configs; use `run_native.sh` for this recipe.

Resume after stopping the previous training process, with the saved `global_step_*` directory:

```bash
checkpoint="$PWD/outputs/go2_mjlab/native_ascend/go2_mjlab/checkpoints/global_step_100"
bash examples/embodiment/go2_ascend/run_native.sh go2_mjlab_ppo_ascend runner.resume_dir="$checkpoint"
```

## Outputs and Validation

The default output is `outputs/go2_mjlab/native_ascend`. TensorBoard events appear in `tensorboard/`; resumable model, optimizer, LR scheduler, and normalization state appear in `go2_mjlab/checkpoints/`. Physics state is reset when a run resumes. The time-based command curriculum resumes from the saved update count.

Validation on Ascend completed an upstream RSL-RL single-NPU reference, three native RLinf updates from random initialization followed by restore to five total updates, on both one NPU and eight NPUs (4096 environments on eight NPUs). Restore checks verify Adam steps and normalizer sample counts. These are execution checks, not convergence or real-robot validation. Independent experiment records contain logs and checkpoint verification results.

The task and listed PPO settings follow upstream, while RLinf owns rollout collection, GAE, optimization, and checkpointing. RLinf merges normalization moments across actors once per rollout before optimization; RSL-RL updates local moments after each environment step. RLinf bootstraps timeouts from terminal observations; the upstream RSL wrapper uses its stored transition value. Distributed shuffling and environment seeds also differ. These differences preclude a claim of step-by-step numerical equivalence.

## Run an Upstream Reference

`validate_upstream.py` checks the official environment and optional single-device RSL-RL learner independently of RLinf workers. Use an empty output directory:

```bash
"$GO2_PYTHON" examples/embodiment/go2_mjlab/validate_upstream.py \
  --num-envs 4 --physics-device cpu --output outputs/go2_mjlab/cpu_probe
source "$CANN_ENV_FILE"
ASCEND_RT_VISIBLE_DEVICES=8 "$GO2_PYTHON" examples/embodiment/go2_mjlab/validate_upstream.py \
  --num-envs 4 --learner-device npu:0 --updates 3 --output outputs/go2_mjlab/upstream_npu_probe
```

The tool writes resolved task and agent YAML plus `result.json`. The optional learner is explicitly an RSL-RL reference, not native RLinf PPO. `run_upstream.sh` retains the pinned NVIDIA upstream baseline launcher; its default interpreter is `.venv-mjlab/bin/python` and its dependencies are checked before launch.

## Cross-task benchmark

For the separate five-task, 235-observation variant, follow [the benchmark guide](BENCHMARK.md). The Flat recipe above retains its original configuration.
