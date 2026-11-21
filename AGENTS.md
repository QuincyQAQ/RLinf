# Working on the Go2 A40 Three-Task Archive

This repository root is an experiment archive, on a dedicated branch of
QuincyQAQ/RLinf. The RLinf framework itself lives in the pinned RLinf/ submodule.

- Keep the experiment scope to A2, B1 and B2, training seed 0. Preserve original
  rewards, control, budgets, checkpoints and raw evidence. Record packaging or
  scope changes in provenance.json; never edit measurements to improve rankings.
- Read README.md or README.zh-CN.md before changing the project. Keep their
  commands, settings and results aligned. Put scope and verification caveats in
  one Limitations section per language.
- Dependencies are pinned Git submodules. Do not update or modify them while
  packaging results. Follow each dependency's AGENTS.md when explicitly changing it.
- Keep benchmark-generated files in artifacts/benchmark_runtime.tar.gz and restore
  them with tools/restore_runtime.py. Preserve checkpoint modification times,
  because the original B2 runner selects checkpoints by write time.
- Audit with Python 3.11: python3.11 tools/restore_runtime.py, then
  python3.11 tools/audit_results.py. Check shell syntax with bash -n.
- Refresh integrity.json only for reviewed file changes; preserve original
  A40 hashes separately in provenance.json. Verify all retained evidence and
  table regeneration after changes. CPU checks do not prove GPU or hardware results.
- Use Conventional Commits and the human submitter's Git identity with sign-off.
  Push only when requested. Do not merge this archive branch into RLinf main.
- This packaging task authorizes no new GPU allocation, training, simulator run,
  real-robot action, or scheduler mutation. Do not store credentials in the project.
