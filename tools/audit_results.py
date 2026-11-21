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

"""Audit source pins, retained measurements and CSV without running simulation."""

import csv
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

from restore_runtime import digest

ROOT = Path(__file__).resolve().parents[1]
TASKS = {"A2", "B1", "B2"}
RUNS = {
    "A2": "fullppo_huber_A2_14029743",
    "B1": "fullppo_huber_B1_14032301",
    "B2": "b2_postbaseline_14025574",
}
SHORT = {"A2": "A2_omni", "B1": "B1_rough", "B2": "B2_stairs"}


def require(condition: bool, message: str) -> None:
    """Raise a descriptive error even when Python assertions are disabled."""
    if not condition:
        raise ValueError(message)


def verify_evaluation(data: dict) -> int:
    """Recalculate the original summary from every recorded episode."""
    episodes, summary = data["episodes"], data["summary"]
    count = len(episodes)
    if count == 0 or count != summary["num_episodes"]:
        raise ValueError("Episode count does not match the saved evaluation summary")
    expected = {
        "success_rate": sum(bool(v["episode_success"]) for v in episodes) / count,
        "failure_rate": sum(v["termination_reason"] == "failure" for v in episodes)
        / count,
    }
    for source, target in [
        ("success_step_ratio", "mean_success_step_ratio"),
        ("mean_lin_vel_error_xy", "mean_lin_vel_error"),
        ("mean_ang_vel_error_z", "mean_ang_vel_error"),
        ("alive_time_s", "mean_alive_time_s"),
    ]:
        expected[target] = sum(v[source] for v in episodes) / count
    if "physical_obstacle_passage_rate" in summary:
        passage = sum(bool(v["physical_obstacle_passage"]) for v in episodes)
        joint = sum(bool(v["joint_tracking_and_passage"]) for v in episodes)
        expected.update(
            physical_passage_successes=passage,
            physical_obstacle_passage_rate=passage / count,
            joint_tracking_and_passage_success_rate=joint / count,
        )
        for episode in episodes:
            if bool(episode["joint_tracking_and_passage"]) != (
                bool(episode["episode_success"])
                and bool(episode["physical_obstacle_passage"])
            ):
                raise ValueError(
                    "Tracking/passage intersection is inconsistent within an episode"
                )
    for key, value in expected.items():
        if not math.isfinite(value) or not math.isclose(
            value, summary[key], rel_tol=1e-11, abs_tol=1e-11
        ):
            raise ValueError(
                f"Evaluation summary mismatch for {key}: {summary[key]} != {value}"
            )
    return count


def main() -> None:
    """Check the complete published archive and record the observed counts."""
    integrity = json.loads((ROOT / "integrity.json").read_text())
    for relative, expected in integrity.items():
        require(digest(ROOT / relative) == expected, f"SHA256 mismatch: {relative}")
    metadata = json.loads((ROOT / "provenance.json").read_text())
    for name, info in metadata["repositories"].items():
        revision = subprocess.check_output(
            ["git", "-C", str(ROOT / name), "rev-parse", "HEAD"], text=True
        ).strip()
        require(revision == info["commit"], f"Dependency revision changed: {name}")
    sys.path.insert(0, str(ROOT / "a40"))
    from reference import SHORT as configured_short
    from reference import SOURCES, verify_source

    require(set(SOURCES) == set(configured_short) == TASKS, "Task scope changed")
    verify_source()
    count = episodes = 0
    ratios = {"A2": 0.80, "B1": 0.75, "B2": 0.70}
    for record in json.loads((ROOT / "reports/raw_index.json").read_text()):
        path = ROOT / record["original"]
        central = ROOT / record["central_copy"]
        require(
            path.read_bytes() == central.read_bytes(), f"Raw copy differs: {central}"
        )
        data = json.loads(path.read_text())
        episodes += verify_evaluation(data)
        count += 1
        for row in data["episodes"]:
            expected = (
                row["success_step_ratio"] >= ratios[record["task"]]
                and row["termination_reason"] != "failure"
            )
            require(
                bool(row["episode_success"]) == expected,
                f"Tracking rule mismatch: {path}",
            )
        if "physical_obstacle_passage_rate" in data["summary"]:
            require(
                path.with_suffix(".geometry.json").read_bytes()
                == central.with_suffix(".geometry.json").read_bytes(),
                f"Geometry copy differs: {path}",
            )
    final = json.loads((ROOT / "reports/three_task_final_20261121.json").read_text())
    require(set(final["results"]) == TASKS, "Summary task scope changed")
    for task, folder in RUNS.items():
        actual = json.loads((ROOT / "outputs" / folder / "result.json").read_text())
        require(
            actual == final["results"][task]["data"] and actual["status"] == "verified",
            f"Final report mismatch: {task}",
        )
        for lane, seeds in actual["results"].items():
            middle = "holdout_baseline" if task == "B2" and lane == "baseline" else lane
            if lane != "baseline":
                middle += "/evaluation"
            for seed, expected in seeds.items():
                path = (
                    ROOT
                    / "outputs"
                    / folder
                    / middle
                    / seed
                    / (SHORT[task] + "_s0")
                    / (SHORT[task] + ".json")
                )
                require(
                    json.loads(path.read_text())["summary"] == expected,
                    f"Final summary differs from raw episodes: {task}/{lane}/{seed}",
                )
    supplementary = json.loads(
        (ROOT / "reports/completed_b1_confirmatory_14036295.json").read_text()
    )
    original = json.loads(
        (ROOT / "outputs/b1_confirmatory_14036295/result.json").read_text()
    )
    require(
        supplementary == original and original["status"] == "verified",
        "B1 supplementary report changed",
    )
    for lane, expected in original["new_seed_results"].items():
        path = (
            ROOT
            / "outputs/b1_confirmatory_14036295"
            / lane
            / "B1_rough_s0/B1_rough.json"
        )
        require(
            json.loads(path.read_text())["summary"] == expected,
            f"B1 supplementary raw summary changed: {lane}",
        )
    spec = importlib.util.spec_from_file_location(
        "three_task_export", ROOT / "reports/export_results.py"
    )
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory)
        for name in [
            "three_task_final_20261121.json",
            "completed_b1_confirmatory_14036295.json",
        ]:
            (target / name).write_bytes((ROOT / "reports" / name).read_bytes())
        exporter.ROOT = target
        exporter.main()
        name = "A40_A2_B1_B2_20261121.csv"
        require(
            (target / name).read_bytes() == (ROOT / "reports" / name).read_bytes(),
            "Regenerated CSV differs",
        )
    with (ROOT / "reports/A40_A2_B1_B2_20261121.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    require(
        len(rows) == 13 and {r["任务"][:2] for r in rows} == TASKS, "CSV scope changed"
    )
    require(
        count == metadata["raw_evaluations"] and episodes == metadata["episodes"],
        "Evaluation inventory changed",
    )
    for entry in metadata["checkpoints"]:
        path = ROOT / entry["path"]
        require(
            path.stat().st_mtime_ns == entry["mtime_ns"],
            f"Restore checkpoint selection times: {path}",
        )
    report = dict(
        status="passed",
        tasks=sorted(TASKS),
        csv_rows=len(rows),
        raw_evaluations=count,
        episodes=episodes,
        checkpoints=len(metadata["checkpoints"]),
        files_hashed=len(integrity),
        new_gpu_allocations=0,
        new_simulation_runs=0,
        new_training_runs=0,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    (ROOT / "validation/subset_audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
