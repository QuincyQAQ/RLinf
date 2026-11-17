# Go2 experiment backup

This branch preserves the current RLinf Go2 implementation and the external benchmark orchestration sources. It is an experiment snapshot, not a proposed upstream PR. Training runs on hkust are unaffected by this backup.

`go2_benchmark_control.tar.gz` contains the three control scripts, the experiment plan, the baseline provenance, launch records, CPU timing results, and the external experiment README/status snapshot. `manifest.json` records the SHA256 of each archived file. No checkpoints, credentials, environments, or simulator assets are included. Machine paths and process IDs are historical run metadata.

Restore the controller sources to a separate directory:

```bash
mkdir -p /tmp/go2-benchmark-control
tar -xzf experiment_snapshot/go2_benchmark_control.tar.gz -C /tmp/go2-benchmark-control
```

The active plan uses five tasks, three fine-tuning seeds sharing A1 checkpoint200, 1500 updates per specialist, and 1024 episodes per matrix cell. Seed42 must finish its five policies and25 evaluations before seed43 starts; seed44 follows. After the existing eight-device stair run, two disjoint four-device lanes use devices8–11 and12–15. See the archived plan for limits. Do not run an additional controller against an experiment that already has one.

For an upstream contribution, create a focused branch from the current upstream main. Separate simulation/PPO integration, benchmark evaluation, and physical robot support; omit this backup directory and machine-specific experiment archives.
