# Go2 cross-task benchmark on Mjlab

Train five specialist policies with RLinf and evaluate each on the same five tasks. This is the `rlinf-mjlab-cross-task-v1` variant of [the reference benchmark](https://github.com/viktoriageorrg04/bsc-thesis-multi-task-RL-env/tree/743ac04d9543b04b7854f1100bfa0cc69c993106). Keep its results separate from both the reference Isaac Lab matrix and the existing Flat-policy terrain pilot.

## Task and observation contract

Use `go2_mjlab_benchmark_a1_ascend`, replacing `a1` with `a2`, `b1`, `b2`, or `c2` to select the other tasks. They select forward flat walking, omnidirectional flat walking, rough walking, stairs, and gap crossing respectively. Each actor and critic receives 235 values: body linear velocity (3), angular velocity (3), projected gravity (3), command (3), relative joint position (12), relative joint velocity (12), previous action (12), and terrain scan (187). Actions are 12 joint offsets with scale 0.25. These policies require new initialization. The existing 47/74 Flat configuration and its checkpoints retain their original interface.

The five-task configuration inherits RLinf PPO, its networks with hidden-layer widths of 512, 256, and 128 and ELU activations, normalization epsilon 0.01, and the 10001-update budget. It disables randomized initial episode lengths and the Flat command curriculum. This budget is a configuration choice, not the reference project's training duration. Set separate actor/environment seeds for independent training runs and keep each run in a separate output directory.

## Run a short validation

Use the installation in [README.md](README.md), and select an available accelerator before launching. The supplied Ascend short validation uses device 8, 16 environments, three PPO updates and checkpointing each update:

```bash
bash examples/embodiment/go2_ascend/run_native.sh go2_mjlab_benchmark_a1_smoke
```

Do not launch this while the original job owns device 8. CPU simulation checks have passed locally for all five tasks and on hkust for A1; accelerator training for this variant is pending a free device window. No new long training job has been started.

For a specialist run after short validation, use its configuration and a new output path:

```bash
bash examples/embodiment/go2_ascend/run_native.sh go2_mjlab_benchmark_a1_ascend \
  runner.logger.log_path=outputs/go2_cross_task/A1_forward_seed0 \
  actor.seed=0 env.train.seed=0
```

The production configurations use devices 8–15 and 4096 environments. Schedule them after the previous job releases those devices.

## Evaluate and export

Pass the complete saved model state and the resolved training configuration; a YAML file containing only Hydra defaults is insufficient. Run one task in each process. This example evaluates the A1 checkpoint on B2:

```bash
PYTHONPATH="$PWD" "$GO2_PYTHON" examples/embodiment/go2_mjlab/evaluate_benchmark.py \
  --checkpoint /path/to/full_weights.pt \
  --training-config /path/to/resolved_config.yaml \
  --eval-task B2_stairs --num-envs 64 --num-episodes 1024 \
  --output outputs/go2_cross_task/evaluation_step1000
```

Repeat for all five evaluation tasks. Keep the same checkpoint, configuration and evaluation seed within one row. The evaluator rejects incompatible 47-dimensional models and rows already associated with different weights. Each cell includes full episodes and a protocol record; each row gets `summary.json`. Requests below 1024 episodes are labeled smoke tests and excluded from the matrix by default. Vector batches may collect more episodes than requested.

```bash
PYTHONPATH="$PWD" "$GO2_PYTHON" examples/embodiment/go2_mjlab/export_benchmark_matrix.py \
  --results-root outputs/go2_cross_task/evaluation_step1000
```

The exporter writes JSON, CSV and Markdown. Missing cells remain null; it never fills untrained tasks with zero or copies another policy's results. Train three independent seeds for each specialist to estimate per-cell sample standard deviation. An MTL row requires a separately implemented and trained multi-task policy; this version supplies five specialist tasks.

## Success criteria and differences

Count a step as successful when both errors are within their thresholds and no failure terminates that step. An episode succeeds when it has no failure and enough successful steps over the full episode, including startup. This is not an RMSE threshold or a geometric goal-reaching test.

| Task | Linear error m/s | Yaw error rad/s | Minimum successful-step ratio |
|---|---:|---:|---:|
| A1 | 0.20 | 0.40 | 0.85 |
| A2 | 0.25 | 0.50 | 0.80 |
| B1 | 0.35 | 0.60 | 0.75 |
| B2 | 0.40 | 0.70 | 0.70 |
| C2 | 0.45 | 0.75 | 0.65 |

The variant uses MuJoCo physics, the pinned Unitree Mjlab robot, actuators, rewards and startup randomization. It ports the reference task command ranges, 10-second command resampling, the success thresholds and a base-contact failure criterion. A1/B2/C2 use direct yaw commands; A2/B1 use heading commands. All tasks have a 20-second horizon. B1 uses Mjlab random rough terrain; B2 uses 4–16 cm inverted pyramid stairs. C2 uses a box-generated square gap around a 2 m island, with width 0–20 cm and depth 1 m, rather than the reference mesh generator. These are explicit simulator and task differences, not numerical reproduction claims.

The actor scan clips relative heights with offset 0.5 to [-1,1]; its frame and joint ordering follow Mjlab. During evaluation, observation noise, external pushes and terrain adaptation are disabled; generated terrains sample all ten difficulty rows. Startup randomization remains active. Evaluation reads tracking errors after the step, including the state after automatic reset on a completed episode, as the reference evaluator does. These choices are recorded for this variant and differ from the reference evaluation's inherited curriculum/randomization behavior. Off-diagonal cells still measure transfer to another task; a cross-task matrix does not mean every specialist was trained on every task.
