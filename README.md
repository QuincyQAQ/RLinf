# Go2 A40: A2, B1 and B2 Experiments

[简体中文](README.zh-CN.md)

Use this archive to inspect and reproduce the recorded Go2 A2 omnidirectional, B1 rough-terrain and B2 stair experiments. It contains the selected experiment code, configurations, weights and measurements from the verified A40 five-task archive. The original archive directory and GitHub branch remain unchanged.

Tables and episode-level evaluations are collected in `reports/`. The [complete CSV](reports/A40_A2_B1_B2_20261121.csv) has 13 rows and 19 columns, covering the three main tasks, the B1 fixed-policy repeat and the B2 MSE branch. Training and evaluation seeds have separate columns.

| Path | Contents |
|---|---|
| `reports/A40_A2_B1_B2_20261121.csv` | Success counts, tracking success, velocity errors, failure rate, survival time, physical passage and seeds |
| `reports/three_task_final_20261121.json` | Protocols and main results for all three tasks |
| `reports/completed_b1_confirmatory_14036295.json` | B1 repeat on a separate evaluation seed, including pooled statistics |
| `reports/raw/`, `reports/raw_index.json` | Central copies and original paths for 28 episode-level evaluations, with geometry files |
| `outputs/` | Original runtime paths for evaluations, commands, configurations and logs |
| `benchmark/logs/rsl_rl/` | Ordinary PPO weights and configurations; RLinf weights remain under their `outputs/` runs |
| `a40/`, `config/` | Three-task launchers, container tools and configurations |
| `RLinf/`, `benchmark/`, `IsaacLab/` | Source submodules pinned to exact commits |
| `provenance.json`, `integrity.json` | Source records, scope changes and SHA256 checksums |

The copies in `reports/raw/` preserve the original bytes. Runtime copies remain in `outputs/` so recorded commands can resolve their paths. The offline audit compares both copies.

## Experiment Settings and Results

Each task has three groups: the completed ordinary PPO source policy, ordinary PPO continuation, and RLinf-Huber continuation. Training seed is 0. Both continuations start from the same source weights and run 1000 additional updates with fresh optimizers and environments. Main evaluations use seeds 23456 and 34567, pooled by episode count after separate runs.

| Task | Source Updates | Added Updates | Parallel Environments | Original PPO Tracking | Continue Ordinary PPO | Continue RLinf-Huber |
|---|---:|---:|---:|---:|---:|---:|
| A2 omnidirectional | 1500 | 1000 | 2048 | 1887/2063 (91.4687%) | 1917/2054 (93.3301%) | 2001/2049 (97.6574%) |
| B1 rough terrain | 3000 | 1000 | 2048 | 2045/2048 (99.8535%) | 2040/2048 (99.6094%) | 2044/2048 (99.8047%) |
| B2 stairs | 2200 | 1000 | 1024 | 2045/2048 (99.8535%) | 2048/2048 (100%) | 2048/2048 (100%) |

B2 physical stair-passage rates are **438/2048 (21.3867%)**, **918/2048 (44.8242%)** and **1568/2048 (76.5625%)**, respectively. The supplementary RLinf-MSE branch reaches **958/2048 (46.7773%)**. The CSV also reports the same-episode intersection of tracking success and physical passage.

The B1 fixed-policy repeat uses evaluation seed 45678 and 2048 episodes per group. All three groups reach 2044/2048. Its rows remain separate from the main experiment. B1 uses a valid recovered 3000-update source checkpoint from job 14032252, with the same ±100 action/history bound in all groups.

Actor and critic each receive 235-dimensional observations and use ELU hidden layers `[512,256,128]`, with observation normalization disabled. The actor outputs 12 joint actions. Physics runs at 200 Hz and the policy at 50 Hz. Rollouts contain 24 steps, followed by 5 update epochs and 4 minibatches. RLinf-Huber uses learning rate `3e-4`, entropy coefficient `0.003`, target KL `0.005` and Huber delta 10. B2 continuations retain the late posture reward weights.

Tracking success requires no failure termination and a minimum fraction of steps within both tracking thresholds. For A2/B1/B2, linear-velocity limits are 0.25/0.35/0.40 m/s, yaw-rate limits are 0.50/0.60/0.70 rad/s, and minimum step fractions are 80%/75%/70%. B2 physical passage requires clearing the complete stair flight along +x, landing all four feet beyond its far edge for at least 0.2 seconds, and finishing the episode without failure.

## Get the Code and Verify Results

Clone the dedicated branch with its pinned source submodules. Use Python 3.11 to restore generated files and audit the results.

```bash
git clone --single-branch --branch experiment/go2-a40-a2-b1-b2-20261121 \
  --recurse-submodules https://github.com/QuincyQAQ/RLinf.git go2-a40-three-tasks
cd go2-a40-three-tasks
python3.11 tools/restore_runtime.py
python3.11 tools/audit_results.py
```

The restore command extracts generated files from `artifacts/benchmark_runtime.tar.gz` and restores the original modification times of all 17 retained checkpoints. Repeating it verifies existing files and stops on conflicting changes. In an existing local workspace, run the two Python commands directly.

The audit checks file hashes, dependency commits, task scope, per-episode statistics, success rules, raw-to-summary consistency, central copies and CSV regeneration. A passing run prints `"status": "passed"`, `"raw_evaluations": 28`, `"episodes": 31771` and `"csv_rows": 13`. It writes its report to `validation/subset_audit.json`. Regenerate the table with:

```bash
python3.11 reports/export_results.py
```

## Run the Experiments

Prepare the recorded container, packages and assets using `environment_lock.json`. The container launcher mounts this project root at `/workspace/go2`. An A40 host can reuse the verified container, asset mirror and package overlays. Source commits are pinned in `provenance.json`.

Use these launchers when you plan a GPU rerun:

```bash
sbatch a40/fullppo_huber.sbatch A2
sbatch a40/b1_recovered.sbatch
sbatch a40/b2_continuation.sbatch
```

A2 and B1 check the original configuration and run a training-update check. B2 reuses the recorded check for the same protocol; check a changed runtime with `a40/b2_continuation_gate.sbatch` first. The B1 fixed-policy repeat uses `a40/b1_confirmatory_eval.sbatch`. Each run's `command.json` and `protocol.json` retain its arguments, environment variables and checkpoint paths.

## Provenance and Maintenance

This branch uses the A40 snapshot captured on 2026-11-21. `provenance.json` records the original archive commit, the three pinned source commits and scope changes. `integrity.json` records SHA256 checksums. Git submodules hold source code; the separate runtime archive holds generated ordinary PPO files restored by the commands above.

Scope changes remove unrelated task and midpoint-switch launchers, filter task configurations and set standalone probes to A2 by default. Rewards, control, PPO numerical parameters, weights and recorded measurements are preserved. The GitHub CPU workflow restores runtime files, recomputes results and checks script syntax. Follow `AGENTS.md` when maintaining the project. Licensing and third-party attribution are in `LICENSE` and `NOTICE`.

## Limitations

- This release packages existing code and records with offline verification. It allocates no GPU and reruns no training, simulation or physical robot action. Retraining need not reproduce weights or scores bit for bit.
- Pinned source submodules retain upstream implementations for other robots and tasks. Experiment configurations, launchers, result tables and weights in this archive cover A2, B1 and B2.
- The full container, installed packages, asset caches and complete intermediate checkpoint history are external. The archive retains 17 required weights, environment pins and the asset manifest. Replay B2 history using explicit checkpoint paths under `outputs/b2_postbaseline_14025574/curve/`; resampling positions from the retained subset changes the curve.
- Each method has training seed 0 only. Additional evaluation seeds repeat fixed policies. B1 includes the recorded numerical repair. Huber loss and hyperparameter differences prevent attributing every result difference to the framework alone.
