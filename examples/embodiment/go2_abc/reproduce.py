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

"""Validate both learners, reproduce task C, and compare recorded episodes."""

import json
import os
import subprocess
import sys
from pathlib import Path

from examples.embodiment.go2_abc.prepare_data import HERE, digest
from examples.embodiment.go2_abc.run import launch
from rlinf.utils.logging import get_logger


def check_weights(path: Path, initial: Path, native: bool) -> dict:
    """Require finite actor and critic updates and positive action deviations."""
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    if not native:
        state = state["model_state_dict"]
    original = torch.load(initial, map_location="cpu", weights_only=False)[
        "model_state_dict"
    ]
    assert all(torch.isfinite(value).all() for value in state.values()), path
    assert (state["std"] > 0).all(), path
    for prefix in ("actor.", "critic."):
        assert any(
            not torch.equal(value, original[key])
            for key, value in state.items()
            if key.startswith(prefix)
        ), f"Expected changed {prefix} weights in {path}"
    return {"checkpoint": str(path), "sha256": digest(path), "status": "passed"}


def main() -> None:
    """Keep two-update gates separate from the three reported experiments."""
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Task C reproduction requires an allocated A40 job")
    project = HERE.parents[2]
    runtime = Path(os.environ["GO2_ABC_RUNTIME"])
    data = Path(os.environ["GO2_ABC_DATA"])
    validation = data / "validation"
    validation.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    environment.update(
        RLINF_EXT_MODULE="examples.embodiment.go2_abc.extension",
        PYTHONPATH=os.pathsep.join(
            [
                str(project),
                str(runtime / "a40"),
                str(runtime / "benchmark"),
                os.environ.get("PYTHONPATH", ""),
            ]
        ),
        RAY_ADDRESS="local",
    )
    initial = data / "models/C/baseline.pt"
    status = {
        "job_id": os.environ["SLURM_JOB_ID"],
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project, text=True
        ).strip(),
        "task": "C",
        "training_seed": 0,
        "evaluation_seeds": [23456, 34567],
        "baseline_sha256": digest(initial),
        "stages": [],
        "status": "running",
    }

    def run(command: list[str], directory: Path) -> None:
        directory.mkdir(parents=True)
        environment["RAY_TMPDIR"] = f"/tmp/abc_gate_{directory.name}"
        (directory / "command.json").write_text(json.dumps(command, indent=2))
        with (directory / "console.log").open("w") as log:
            subprocess.run(
                command,
                cwd=project,
                env=environment,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=3600,
            )

    try:
        for group in ("ppo", "rlinf"):
            for count in (64, 1024):
                label = f"C_{group}_gate_{count}"
                get_logger().info("Validating %s: two updates", label)
                directory = validation / label
                if group == "ppo":
                    config = json.loads((HERE / "config/C_ppo.json").read_text())
                    argv = [
                        x.replace("${GO2_ABC_DATA}", str(data)) for x in config["argv"]
                    ]
                    for key, value in (
                        ("--num_envs", str(count)),
                        ("--max_iterations", "2"),
                        ("--experiment_name", label),
                        ("--run_name", label),
                    ):
                        argv[argv.index(key) + 1] = value
                    argv.append(
                        "--kit_args=--/persistent/isaac/asset_root/cloud="
                        f"{runtime}/assets/isaac_sim_5.1"
                        " --/app/extensions/excluded/0=omni.kit.telemetry"
                    )
                    run(
                        [
                            sys.executable,
                            str(runtime / "a40/train_continuation_rsl.py"),
                            *argv,
                        ],
                        directory,
                    )
                    path = max(
                        (runtime / f"benchmark/logs/rsl_rl/{label}").glob(
                            "*/model_*.pt"
                        ),
                        key=lambda p: p.stat().st_mtime_ns,
                    )
                else:
                    run(
                        [
                            sys.executable,
                            str(
                                project / "examples/embodiment/train_embodied_agent.py"
                            ),
                            "--config-path",
                            str(HERE / "config"),
                            "--config-name",
                            "C_rlinf",
                            f"env.train.total_num_envs={count}",
                            f"actor.micro_batch_size={count * 6}",
                            f"actor.global_batch_size={count * 6}",
                            "runner.max_epochs=2",
                            "runner.max_steps=2",
                            "runner.save_interval=1",
                            f"runner.logger.log_path={directory}",
                            f"runner.logger.experiment_name={label}",
                            f"env.train.validation_output={directory}/environment",
                        ],
                        directory,
                    )
                    path = (
                        directory
                        / label
                        / "checkpoints/global_step_2/actor/model_state_dict/full_weights.pt"
                    )
                status["stages"].append(
                    {"stage": label, **check_weights(path, initial, group == "rlinf")}
                )
                (validation / "status.json").write_text(json.dumps(status, indent=2))
        for group in ("baseline", "ppo", "rlinf"):
            get_logger().info("Reproducing C_%s", group)
            launch(f"C_{group}", execute=True)
            comparisons = []
            for seed in (23456, 34567):
                candidates = list(
                    (data / f"runs/C_{group}/evaluation/{seed}").rglob("*.json")
                )
                records = [json.loads(path.read_text()) for path in candidates]
                records = [
                    record
                    for record in records
                    if isinstance(record, dict) and "episodes" in record
                ]
                assert len(records) == 1, candidates
                actual = records[0]
                expected = json.loads((data / f"raw/C/{group}/{seed}.json").read_text())
                episodes = actual["episodes"]
                count = len(episodes)
                assert count >= 1024, count
                values = {
                    "episodes": count,
                    "tracking_successes": sum(e["episode_success"] for e in episodes),
                    "passage_successes": sum(
                        e["physical_obstacle_passage"] for e in episodes
                    ),
                    "linear_error": sum(e["mean_lin_vel_error_xy"] for e in episodes)
                    / count,
                    "angular_error": sum(e["mean_ang_vel_error_z"] for e in episodes)
                    / count,
                }
                comparisons.append(
                    {
                        "evaluation_seed": seed,
                        **values,
                        "episodes_exact_match": actual["episodes"]
                        == expected["episodes"],
                        "summary": actual["summary"],
                        "historical_summary": expected["summary"],
                    }
                )
            policy = (
                initial if group == "baseline" else data / f"runs/C_{group}/policy.pt"
            )
            status["stages"].append(
                {
                    "stage": f"C_{group}",
                    "policy_sha256": digest(policy),
                    "comparisons": comparisons,
                }
            )
            (validation / "status.json").write_text(json.dumps(status, indent=2))
        status["status"] = "completed"
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (validation / "status.json").write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
