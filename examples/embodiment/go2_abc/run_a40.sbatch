#!/usr/bin/env bash
# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

experiment="${1:?Expected an experiment such as A_rlinf}"
shift
case "$experiment" in
  A_baseline|A_ppo|A_rlinf|B_baseline|B_ppo|B_rlinf|C_baseline|C_ppo|C_rlinf) ;;
  *) echo "Unknown Go2 experiment: $experiment" >&2; exit 2 ;;
esac

cd "${SLURM_SUBMIT_DIR:?Submit this job from the RLinf repository root}"
test -f examples/embodiment/go2_abc/container.sh
exec bash examples/embodiment/go2_abc/container.sh --experiment "$experiment" --execute "$@"
