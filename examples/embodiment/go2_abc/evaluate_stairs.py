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

"""Evaluate a trained Go2 C checkpoint on two simulated stair trials."""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
RUNTIME = PROJECT.parent / "public-evaluator"
TASK_ID = "MTL-Velocity-Rough-Unitree-Go2-B2-StairClimb-v0"
TRAIN_NAME = "B2_stairs_s0"
EVAL_NAME = "B2_stairs"
MIN_EPISODES = 1024


def export_policy(checkpoint: Path, initial: Path, destination: Path) -> None:
    """Verify actor and critic updates and export weights for RSL-RL inference."""
    import torch

    from rlinf.models.embodiment.go2_policy import Go2Policy

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    baseline = torch.load(initial, map_location="cpu", weights_only=True)[
        "model_state_dict"
    ]
    model = Go2Policy()
    for label, weights in (("baseline", baseline), ("trained", state)):
        model.load_state_dict(weights, strict=True)
        if not all(torch.isfinite(value).all() for value in weights.values()):
            raise ValueError(f"Nonfinite {label} policy weights")
        if not (model.std > 0).all():
            raise ValueError(f"The {label} policy standard deviations must be positive")
    for prefix in ("actor.", "critic."):
        if not any(
            not torch.equal(value, baseline[name])
            for name, value in state.items()
            if name.startswith(prefix)
        ):
            raise ValueError(f"No update to {prefix}")
    torch.save(
        {
            "model_state_dict": state,
            "iter": 1000,
            "optimizer_state_dict": torch.optim.Adam(model.parameters()).state_dict(),
            "infos": {
                "inference_export_only": True,
                "native_checkpoint": str(checkpoint),
            },
        },
        destination,
    )


def evaluation_args(policy: Path, output: Path, asset_root: Path | None) -> list[str]:
    """Build the C evaluation command for one seed."""
    kit = "--/app/extensions/excluded/0=omni.kit.telemetry"
    if asset_root is not None:
        kit += f" --/persistent/isaac/asset_root/cloud={asset_root.resolve()}"
    return [
        "--task",
        TASK_ID,
        "--checkpoint",
        str(policy),
        "--headless",
        "--num_envs",
        "256",
        "--num_episodes",
        str(MIN_EPISODES),
        "--report_train_task",
        TRAIN_NAME,
        "--eval_task",
        EVAL_NAME,
        "--output_dir",
        str(output),
        f"--kit_args={kit}",
        "env.rewards.base_height.func=rlinf.envs.sim.isaaclab.tasks.go2:terrain_base_height_l2",
    ]


def read_episodes(directory: Path) -> list[dict]:
    """Cross-check the task report and summary before pooling their episodes."""
    folder = directory / TRAIN_NAME
    report = json.loads((folder / f"{EVAL_NAME}.json").read_text())
    rows, summary = report["episodes"], report["summary"]
    if json.loads((folder / "summary.json").read_text())[EVAL_NAME] != summary:
        raise ValueError(f"Evaluation summaries disagree in {folder}")
    if (
        not isinstance(rows, list)
        or len(rows) < MIN_EPISODES
        or summary["num_episodes"] != len(rows)
        or (summary["train_task"], summary["eval_task"]) != (TRAIN_NAME, TASK_ID)
    ):
        raise ValueError(f"Incomplete or unexpected stair evaluation in {folder}")
    if any(
        (row["task_id"], row["family_id"]) != (3, 1)
        or type(row["episode_success"]) is not bool
        or type(row["physical_obstacle_passage"]) is not bool
        for row in rows
    ):
        raise ValueError(f"Invalid stair episode in {folder}")
    tracking = sum(row["episode_success"] for row in rows)
    passage = sum(row["physical_obstacle_passage"] for row in rows)
    if (
        summary["required_obstacle_metrics_complete"] is not True
        or summary["physical_measured_episodes"] != len(rows)
        or summary["physical_passage_successes"] != passage
        or not math.isclose(
            summary["success_rate"], tracking / len(rows), rel_tol=1e-12, abs_tol=1e-12
        )
        or not math.isclose(
            summary["physical_obstacle_passage_rate"],
            passage / len(rows),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(f"Evaluation counts or rates disagree in {folder}")
    return rows


def main() -> None:
    """Export the native weights, verify each seed and report pooled episode counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint", type=Path, help="RLinf step 1,000 full_weights.pt"
    )
    parser.add_argument("output", type=Path, help="New directory for policy and scores")
    parser.add_argument("--asset-root", type=Path, help="Local Isaac Sim 5.1 assets")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("run inside a Slurm GPU allocation")

    from rlinf.utils.logging import get_logger

    entry = RUNTIME / "a40/evaluate_continuation.py"
    baseline = Path(os.environ.get("GO2_ABC_DATA", PROJECT / "data/go2_abc"))
    output = args.output.resolve()
    output.mkdir(parents=True)
    policy = output / "policy.pt"
    export_policy(
        args.checkpoint.resolve(),
        (baseline / "models/C/baseline.pt").resolve(),
        policy,
    )

    episodes = tracking = passage = 0
    for seed in (23456, 34567):
        directory = output / str(seed)
        argv = evaluation_args(policy, directory, args.asset_root)
        imports = [str(PROJECT), str(RUNTIME / "benchmark")]
        if os.environ.get("PYTHONPATH"):
            imports.append(os.environ["PYTHONPATH"])
        env = dict(
            os.environ, GO2_EVAL_SEED=str(seed), PYTHONPATH=os.pathsep.join(imports)
        )
        subprocess.run([sys.executable, str(entry), *argv], env=env, check=True)
        rows = read_episodes(directory)
        episodes += len(rows)
        tracking += sum(row["episode_success"] for row in rows)
        passage += sum(row["physical_obstacle_passage"] for row in rows)

    summary = {
        "num_episodes": episodes,
        "tracking_successes": tracking,
        "passage_successes": passage,
        "tracking_rate": tracking / episodes,
        "passage_rate": passage / episodes,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    get_logger().info(
        f"Tracking {tracking}/{episodes} ({tracking / episodes:.2%}), "
        f"stair passage {passage}/{episodes} ({passage / episodes:.2%})"
    )


if __name__ == "__main__":
    main()
