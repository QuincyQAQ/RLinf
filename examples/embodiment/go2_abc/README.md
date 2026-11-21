# Go2 ABC Experiments

[简体中文](README.zh-CN.md)

Run the three Go2 tasks with the completed PPO policy, ordinary PPO continuation, and RLinf-Huber continuation: nine groups in total.

| Task | Source Updates | Added Updates per Continuation | Parallel Environments | Baseline Tracking | Continue PPO | Continue RLinf-Huber |
|---|---:|---:|---:|---:|---:|---:|
| A: omnidirectional | 1500 | 1000 | 2048 | 91.4687% | 93.3301% | 97.6574% |
| B: rough terrain | 3000 | 1000 | 2048 | 99.8535% | 99.6094% | 99.8047% |
| C: stairs | 2200 | 1000 | 1024 | 99.8535% | 100% | 100% |

C stair-passage rates in simulation are 21.3867%, 44.8242%, and 76.5625%, respectively. [Results](results.csv) include velocity errors and the fraction of episodes that pass both tracking and passage checks. Training seed is 0; evaluation seeds are 23456 and 34567, pooled by episode count.

## Prepare Inputs

Use the [prepared A40 runtime](https://github.com/QuincyQAQ/RLinf/tree/b405f67382ce63af2401a2a9ff5b0b3c7f28ccb7): Isaac Lab 2.3.2, Isaac Sim 5.1, RSL-RL 3.1.2, its package overlays and assets. The simulator tasks and ordinary PPO come from the pinned [multi-task benchmark](https://github.com/viktoriageorrg04/bsc-thesis-multi-task-RL-env); this example reuses their control and rewards.

From the RLinf repository root, download the nine weights and 18 evaluations, or add `--source /path/to/frozen-runtime` to copy them locally:

```bash
python -m examples.embodiment.go2_abc.prepare_data
python -m examples.embodiment.go2_abc.audit
```

The CPU audit checks 33 input hashes, strict weight loading, analytical KL, and all 18,454 episode records against the CSV. Inputs and run outputs live in `data/go2_abc/`.

## Run It

Set the runtime path and preview one group. `GO2_ABC_CONTAINER` selects the installed Singularity or Apptainer executable.

```bash
export GO2_ABC_RUNTIME=/path/to/prepared-runtime
export GO2_ABC_CONTAINER=singularity
bash examples/embodiment/go2_abc/container.sh --experiment C_rlinf
```

Inside an allocated A40 Slurm job, add `--execute` to run it. Experiment names are `A_baseline`, `A_ppo`, `A_rlinf`, and the corresponding B/C names. A baseline evaluates the saved completed policy; each continuation starts from that task's same source weights with a fresh optimizer, then evaluates both seeds. Logs and checkpoints go under `data/go2_abc/runs/<experiment>/`.

## Configuration and Integration

The configs retain 235 observations each for the actor and critic, 12 actions, ELU layers `[512, 256, 128]`, 24 rollout steps and five PPO epochs. RLinf uses Huber value loss, learning rate `3e-4`, entropy coefficient `0.003`, and analytical KL target `0.005`; ordinary PPO retains its adaptive KL target `0.01`. B retains the recorded numerical guard; C continuations use the late posture weights.

The branch starts from upstream RLinf `dcf00fd6`. The generic KL scheduler is a separate commit; Go2 registers through `RLINF_EXT_MODULE`. Rebase this branch onto upstream to update the framework. [Manifest](manifest.json) pins the recorded code, input hashes and checkpoint paths.

For observation assembly, motor conversion and SDK2 integration, see [Motion Control](MOTION.md).

## Limitations

These scores come from the recorded A40 runs on RLinf `5aa29ec`; the upstream integration has CPU checks and has not been retrained on a GPU. The comparison uses one training seed and the listed PPO configurations.
