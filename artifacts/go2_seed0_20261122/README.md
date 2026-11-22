# Go2 Seed-0 Reproduction Snapshot

[简体中文](README.zh-CN.md)

This snapshot packages the A40 helper sources, job launchers, input hashes and evidence for training commit `e8815cac3b3abecedf13e9310aa9661baa6d1043`. The branch includes the full RLinf implementation, Go2 environments, analytical-KL scheduler, observation/control adapters and A/B/C configs.

## Verified Result

Two independent C RLinf-Huber continuations, jobs 14060127 and 14061035, each ran 1000 updates from the same frozen PPO baseline with training seed 0. All 20 saved checkpoint tensors matched exactly. All 2048 evaluation records and both summaries matched exactly: tracking 100%, stair passage 925/2048 (45.1660%). Evaluation seeds are 23456 and 34567. The reports and raw episodes are in [evidence](evidence).

The remaining A/B experiments were submitted as six separate jobs; IDs are in [jobs.tsv](evidence/AB_six_20261122/jobs.tsv). Baseline groups evaluate frozen policies; PPO and RLinf-Huber groups each add 1000 updates. Both tasks use 2048 environments and retain their original configurations, including B's numerical guard.

## Restore the Code

From this branch, make a clean worktree at the exact training commit and restore the archived launchers:

```bash
export GO2_SNAPSHOT="$PWD/artifacts/go2_seed0_20261122"
git worktree add --detach ../go2-seed0-run e8815cac3b3abecedf13e9310aa9661baa6d1043
cd ../go2-seed0-run
tar -xzf "$GO2_SNAPSHOT/launchers.tar.gz"
python -m examples.embodiment.go2_abc.prepare_data
```

Use the [prepared A40 runtime](https://github.com/QuincyQAQ/RLinf/tree/b405f67382ce63af2401a2a9ff5b0b3c7f28ccb7) with Isaac Lab 2.3.2, Isaac Sim 5.1 and RSL-RL 3.1.2. Restore the helper snapshot into a separate copy of that runtime. Set `GO2_ABC_PROJECT` to the new worktree, `GO2_ABC_RUNTIME` to that runtime copy, and `GO2_ABC_IMAGE` and `GO2_ABC_CONTAINER` to the prepared image and Singularity executable. When sharing the prepared assets and overlays, set `GO2_ABC_SHARED_RUNTIME` to their runtime directory.

```bash
tar -xzf "$GO2_SNAPSHOT/runtime_sources.tar.gz" -C "$GO2_ABC_RUNTIME"
bash examples/embodiment/go2_abc/container.sh --python \
  /workspace/project/data/go2_abc/AB_six_20261122/dispatch.py cpu-audit
```

The CPU audit verifies inputs and previews all six commands. Within an allocated A40 job, continuation groups run short update checks, full training and evaluation; baseline groups only evaluate. Submit the A/B script once per experiment, using distinct stdout/stderr paths, for example:

```bash
sbatch --export=ALL --job-name=go2_A_rlinf_seed0 \
  --output=data/go2_abc/AB_six_20261122/A_rlinf/slurm-%j.out \
  --error=data/go2_abc/AB_six_20261122/A_rlinf/slurm-%j.err \
  data/go2_abc/AB_six_20261122/run.sbatch A_rlinf
```

The six experiment names are `A_baseline`, `A_ppo`, `A_rlinf`, `B_baseline`, `B_ppo`, `B_rlinf`. The archived `submit.sh` is the original campus-specific submission receipt/script; edit its environment-file path before using it on another installation. Use fresh output directories, and retain the seed, checkpoint hashes, environment counts and dependency commits in [manifest.json](manifest.json).

For two new C runs, copy the archived `verify_seed0.py` into `data/go2_abc/C_repeat/` and run it in allocated A40 jobs with `run --stage full --repeat 1` and `run --stage full --repeat 2`, then `compare --stage full`. Use `bash examples/embodiment/go2_abc/container.sh --python /workspace/project/data/go2_abc/C_repeat/verify_seed0.py` as the command prefix. The comparison checks all saved weights and raw evaluation episodes.

## Limitations

Exact repetition has been established for C on the recorded runtime. A/B jobs are not yet validated. The historical 76.5625% C result is a separate model whose evaluation, but not training score, was reproduced. Seed alone does not guarantee equality across different hardware or software. The snapshot includes source and evidence; container binaries, simulator assets and package overlays remain prerequisites in the prepared runtime. The helper validation files describe the source runtime and do not replace checks on a new installation.
