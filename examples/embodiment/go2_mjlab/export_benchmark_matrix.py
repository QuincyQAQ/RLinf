# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Aggregate completed cross-task results over independent training seeds."""

import argparse
import csv
import json
import statistics
from pathlib import Path

from rlinf.envs.sim.go2_mjlab.benchmark_protocol import TASKS


def aggregate(root: Path, include_smoke: bool = False) -> dict:
    """Keep missing cells empty and compute per-cell sample deviation."""
    values = {train: {task: [] for task in TASKS} for train in TASKS}
    per_seed = []
    seen = set()
    for manifest in sorted(root.glob("*/policy_manifest.json")):
        identity = json.loads(manifest.read_text())
        train = identity["train_task"]
        seed = identity["train_seed"]
        if train not in TASKS:
            raise ValueError(f"Unexpected training task {train}")
        if (train, seed) in seen:
            raise ValueError(f"Duplicate independent training seed: {train}/{seed}")
        seen.add((train, seed))
        seed_row = {"train_seed": f"{train}_s{seed}"}
        for task in TASKS:
            file = manifest.with_name(task + ".json")
            value = None
            if file.exists():
                result = json.loads(file.read_text())
                protocol = result["protocol"]
                if protocol["checkpoint_sha256"] != identity["checkpoint_sha256"]:
                    raise ValueError(f"Checkpoint mismatch in {file}")
                if include_smoke or protocol["evaluation_scope"] == "benchmark":
                    value = float(result["summary"]["success_rate"])
                    if not 0 <= value <= 1:
                        raise ValueError(f"Invalid success rate in {file}")
                    values[train][task].append(value)
            seed_row[task] = value
        if any(seed_row[task] is not None for task in TASKS):
            rates = [seed_row[task] for task in TASKS]
            seed_row["mean_over_eval_tasks"] = (
                statistics.mean(rates) if all(v is not None for v in rates) else None
            )
            per_seed.append(seed_row)
    means, stds, counts = [], [], []
    for train in TASKS:
        mean = {"train_task": train}
        std = {"train_task": train}
        count = {"train_task": train}
        for task, rates in values[train].items():
            mean[task] = statistics.mean(rates) if rates else None
            std[task] = statistics.stdev(rates) if len(rates) >= 2 else None
            count[task] = len(rates)
        macros = [
            r["mean_over_eval_tasks"]
            for r in per_seed
            if r["train_seed"].startswith(train + "_s")
            and r["mean_over_eval_tasks"] is not None
        ]
        mean["mean_over_eval_tasks"] = statistics.mean(macros) if macros else None
        std["mean_over_eval_tasks"] = (
            statistics.stdev(macros) if len(macros) >= 2 else None
        )
        means.append(mean)
        stds.append(std)
        counts.append(count)
    return {
        "metric": "success_rate",
        "variant": "rlinf-mjlab-cross-task-v1",
        "task_order": list(TASKS),
        "row_order": list(TASKS),
        "mean": means,
        "std": stds,
        "per_seed": per_seed,
        "seed_counts": counts,
        "includes_smoke": include_smoke,
        "missing_cells": "null means unmeasured; never interpreted as zero success",
        "macro_std": "sample deviation across independent training-seed macro averages",
    }


def main() -> None:
    """Write JSON, CSV and Markdown matrices without launching simulation."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--include-smoke", action="store_true")
    a = p.parse_args()
    if not a.results_root.is_dir():
        p.error("Results root must be an existing directory")
    result = aggregate(a.results_root, a.include_smoke)
    prefix = "smoke_matrix" if a.include_smoke else "matrix_success_rate"
    (a.results_root / (prefix + ".json")).write_text(
        json.dumps(result, indent=2) + "\n"
    )
    fields = ["train_task", *TASKS, "mean_over_eval_tasks"]
    with (a.results_root / (prefix + ".csv")).open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["mean"])
    lines = [
        "# Go2 Mjlab cross-task success matrix",
        "",
        "| " + " | ".join(fields) + " |",
        "|" + "---|" * len(fields),
    ]
    for row in result["mean"]:
        lines.append(
            "| "
            + " | ".join("—" if row[k] is None else str(row[k]) for k in fields)
            + " |"
        )
    lines += [
        "",
        "Missing cells are unmeasured. Smoke results are excluded by default.",
    ]
    (a.results_root / (prefix + ".md")).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
