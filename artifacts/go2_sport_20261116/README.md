# Go2 Six-Direction Acceptance Archive

[中文](README.zh-CN.md)

[basic_control_acceptance_20261116.json](basic_control_acceptance_20261116.json) summarizes the functional checks on 2026-11-16. Forward, backward, left, right, turn-left, and turn-right each have `operator_confirmed_motion_and_stop: true`, based on the operator's on-site feedback.

The six original reports are stored beside the summary under their recorded `report` names. Some reports were written before operator feedback and still say that confirmation is pending; the summary records the subsequent confirmation. JSON files were copied byte for byte; [manifest.json](manifest.json) lists source paths and hashes.

`source_snapshot/` contains copies of the old scripts as found during migration, including disabled earlier entry points, archived as `.py.txt`. These are not guaranteed to be the exact script revisions used in every hardware run. The runnable entry point and directory map are in [go2_sport](../../examples/embodiment/go2_sport/README.md).

## Validation Scope

This is historical hardware evidence for RLinf calling the original SportClient. `quantitative_acceptance_complete` is `false`: distance and angle acceptance remains incomplete. `migration_hardware_retest` is `false`: migration did not rerun hardware experiments. This archive does not validate ABC policy joint control.
