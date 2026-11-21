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

"""Verify the ABC grid against its weights and episode records on CPU."""

import argparse
import csv
import json
import math
import os
from pathlib import Path

from examples.embodiment.go2_abc.prepare_data import DEFAULT_DATA, HERE, digest
from rlinf.utils.logging import get_logger


def audit(data: Path) -> dict:
    """Check input hashes, pooled metrics, policy loading, and analytical KL."""
    import torch
    from torch.nn import functional as F

    from examples.embodiment.go2_abc.policy import Go2Policy

    manifest = json.loads((HERE / "manifest.json").read_text())
    expected = {
        f"{task}_{group}" for task in "ABC" for group in ("baseline", "ppo", "rlinf")
    }
    experiments = manifest["experiments"]
    assert len(experiments) == 9 and {x["id"] for x in experiments} == expected
    for item in manifest["files"]:
        assert digest(data / item["path"]) == item["sha256"], item["path"]
    rows = list(csv.DictReader((HERE / "results.csv").open(encoding="utf-8-sig")))
    assert len(rows) == 9, "Expected exactly nine result rows"
    generator = torch.Generator().manual_seed(0)
    observations = torch.randn(8, 235, generator=generator)
    total = 0
    for experiment, row in zip(experiments, rows, strict=True):
        task, group = experiment["task"], experiment["group"]
        assert experiment["training_seed"] == 0
        assert experiment["evaluation_seeds"] == [23456, 34567]
        episodes = []
        for seed in experiment["evaluation_seeds"]:
            raw = json.loads((data / f"raw/{task}/{group}/{seed}.json").read_text())
            assert raw["summary"] == experiment["summaries"][str(seed)]
            assert len(raw["episodes"]) == raw["summary"]["num_episodes"]
            episodes.extend(raw["episodes"])
        count = len(episodes)
        total += count
        threshold = {"A": 0.80, "B": 0.75, "C": 0.70}[task]
        for episode in episodes:
            tracking = (
                episode["termination_reason"] != "failure"
                and episode["success_step_ratio"] >= threshold
            )
            assert episode["episode_success"] == tracking
        successes = sum(x["episode_success"] for x in episodes)
        assert int(row["跟踪成功回合数"]) == successes
        assert int(row["评测回合数"]) == count
        values = {
            "跟踪成功率（%）": 100 * successes / count,
            "平均线速度误差（m/s）": sum(x["mean_lin_vel_error_xy"] for x in episodes)
            / count,
            "平均偏航角速度误差（rad/s）": sum(
                x["mean_ang_vel_error_z"] for x in episodes
            )
            / count,
            "失败终止率（%）": 100
            * sum(x["termination_reason"] == "failure" for x in episodes)
            / count,
            "平均达标步比例（%）": 100
            * sum(x["success_step_ratio"] for x in episodes)
            / count,
            "平均存活时间（s）": sum(x["alive_time_s"] for x in episodes) / count,
        }
        if task == "C":
            passages = sum(x["physical_obstacle_passage"] for x in episodes)
            assert int(row["实际障碍通过回合数"]) == passages
            for episode in episodes:
                assert episode["joint_tracking_and_passage"] == (
                    episode["episode_success"] and episode["physical_obstacle_passage"]
                )
            values["实际障碍通过率（%）"] = 100 * passages / count
            values["同回合跟踪且通过率（%）"] = (
                100 * sum(x["joint_tracking_and_passage"] for x in episodes) / count
            )
        for key, value in values.items():
            assert math.isclose(float(row[key]), value, abs_tol=1e-9), (
                experiment["id"],
                key,
            )
        policy = Go2Policy(str(data / experiment["model"]), 235, 235)
        state = policy.state_dict()
        assert all(torch.isfinite(x).all() for x in state.values())
        assert (policy.std > 0).all()
        action, rollout = policy.predict_action_batch(
            {"states": observations}, mode="eval"
        )
        expected_action = observations
        for index in (0, 2, 4, 6):
            expected_action = F.linear(
                expected_action,
                state[f"actor.{index}.weight"],
                state[f"actor.{index}.bias"],
            )
            if index != 6:
                expected_action = F.elu(expected_action)
        torch.testing.assert_close(action[:, 0], expected_action, rtol=0, atol=0)
        output = policy.default_forward(rollout["forward_inputs"])
        torch.testing.assert_close(
            output["kl_divergence"], torch.zeros(8), rtol=0, atol=0
        )
        policy.actor[-1].bias.data.add_(0.01)
        shifted = policy.default_forward(rollout["forward_inputs"])["kl_divergence"]
        assert (shifted > 0).all(), "Analytical KL did not detect a changed policy"
    return {
        "experiments": 9,
        "inputs": len(manifest["files"]),
        "episodes": total,
        "device": "cpu",
        "status": "passed",
    }


def main() -> None:
    """Run the CPU audit without starting a simulator or a Ray cluster."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path(os.environ.get("GO2_ABC_DATA", DEFAULT_DATA))
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = audit(args.data)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
    get_logger().info(json.dumps(result))


if __name__ == "__main__":
    main()
