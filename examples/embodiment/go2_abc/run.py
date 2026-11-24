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

"""Run one of the nine ABC experiments in the prepared A40 environment."""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

from examples.embodiment.go2_abc.prepare_data import (
    HERE,
    baseline_input,
    data_directory,
    verify_digest,
)
from rlinf.utils.logging import get_logger


def verify_evaluation(directory: Path, task: str, argv: list[str]) -> None:
    """Verify the benchmark's per-task report and its row in the summary."""
    eval_name = argv[argv.index("--eval_task") + 1]
    train_name = argv[argv.index("--report_train_task") + 1]
    gym_id = argv[argv.index("--task") + 1]
    minimum = int(argv[argv.index("--num_episodes") + 1])
    folder = directory / train_name
    report = json.loads((folder / f"{eval_name}.json").read_text())
    summary = report["summary"]
    if json.loads((folder / "summary.json").read_text())[eval_name] != summary:
        raise ValueError(f"Evaluation summaries disagree in {folder}")
    episodes = report["episodes"]
    if (
        not isinstance(episodes, list)
        or len(episodes) < minimum
        or summary["num_episodes"] != len(episodes)
    ):
        raise ValueError(
            f"Incomplete evaluation in {folder}: expected at least {minimum} episodes and a matching count"
        )
    if (summary["train_task"], summary["eval_task"]) != (train_name, gym_id):
        raise ValueError(f"Unexpected evaluation task in {folder}")
    task_id, family_id = {"A": (1, 0), "B": (2, 1), "C": (3, 1)}[task]
    if any(
        (episode["task_id"], episode["family_id"]) != (task_id, family_id)
        or type(episode["episode_success"]) is not bool
        for episode in episodes
    ):
        raise ValueError(f"Invalid tracking episode in {folder}")
    success_rate = sum(e["episode_success"] for e in episodes) / len(episodes)
    if not math.isclose(
        summary["success_rate"], success_rate, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError(f"Tracking success rate disagrees with episodes in {folder}")
    if task == "C":
        passages = [e["physical_obstacle_passage"] for e in episodes]
        if (
            any(type(passed) is not bool for passed in passages)
            or summary["required_obstacle_metrics_complete"] is not True
            or summary["physical_measured_episodes"] != len(episodes)
            or summary["physical_passage_successes"] != sum(passages)
            or not math.isclose(
                summary["physical_obstacle_passage_rate"],
                sum(passages) / len(episodes),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(f"Incomplete stair passage evaluation in {folder}")


def _export_rlinf_policy(checkpoint: Path, initial: Path, destination: Path) -> None:
    """Verify both networks were updated and export the policy for the benchmark."""
    import torch

    from examples.embodiment.go2_abc.policy import Go2Policy

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = Go2Policy(obs_dim=235, critic_obs_dim=235)
    model.load_state_dict(state, strict=True)
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Nonfinite native policy weights")
    if not (model.std > 0).all():
        raise ValueError("Native policy standard deviations must be positive")
    original = torch.load(initial, map_location="cpu", weights_only=True)[
        "model_state_dict"
    ]
    for prefix in ("actor.", "critic."):
        if not any(
            not torch.equal(value, original[key])
            for key, value in state.items()
            if key.startswith(prefix)
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


def launch(experiment: str, execute: bool, train_seed: int = 0) -> None:
    """Train a continuation when selected, then evaluate both fixed seeds."""
    runtime = Path(os.environ["GO2_ABC_RUNTIME"]).resolve()
    data = data_directory().resolve()
    project = HERE.parents[2]
    import rlinf

    if Path(rlinf.__file__).resolve().parent.parent != project:
        raise RuntimeError(f"Expected RLinf from {project}, found {rlinf.__file__}")
    task, group = experiment.split("_")
    if not 0 <= train_seed < 2**32:
        raise ValueError("Expected training seed in [0, 2**32)")
    run_name = experiment if train_seed == 0 else f"{experiment}_s{train_seed}"
    directory = data / "runs" / run_name
    env = dict(
        os.environ,
        GO2_ABC_DATA=str(data),
        RLINF_EXT_MODULE="examples.embodiment.go2_abc.extension",
    )
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(project),
            str(runtime / "a40"),
            str(runtime / "benchmark"),
            env.get("PYTHONPATH", ""),
        ]
    )
    env.update(
        RAY_ADDRESS="local",
        RAY_TMPDIR=f"/tmp/abc_{run_name}_{os.environ.get('SLURM_JOB_ID', 'plan')}",
    )
    if task == "B":
        env["GO2_B1_NUMERIC_GUARD"] = "1"
    initial = data / f"models/{task}/baseline.pt"
    policy = initial if group == "baseline" else directory / "policy.pt"
    ppo_log_dir = runtime / "benchmark/logs/rsl_rl" / run_name
    commands = []

    def run(command: list[str], stage: str) -> None:
        if "--headless" in command:
            command.append(
                "--kit_args=--/persistent/isaac/asset_root/cloud="
                f"{runtime}/assets/isaac_sim_5.1"
                " --/app/extensions/excluded/0=omni.kit.telemetry"
            )
        commands.append({"stage": stage, "argv": command})
        if execute:
            output = directory / stage
            output.mkdir(parents=True)
            (output / "command.json").write_text(
                json.dumps(
                    {"argv": command, "evaluation_seed": env.get("GO2_EVAL_SEED")},
                    indent=2,
                )
            )
            with (output / "console.log").open("w") as log:
                subprocess.run(
                    command,
                    env=env,
                    cwd=project,
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )

    def resolve(value: str) -> str:
        return value.replace("${GO2_ABC_DATA}", str(data)).replace(
            "${GO2_ABC_RUNTIME}", str(runtime)
        )

    if execute:
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError(
                "Training and evaluation require an allocated A40 Slurm job"
            )
        if directory.exists():
            raise FileExistsError(f"Use a fresh output directory: {directory}")
        if group == "ppo" and ppo_log_dir.exists():
            raise FileExistsError(f"Ordinary PPO output already exists: {run_name}")
        manifest = json.loads((HERE / "manifest.json").read_text())
        for name, key in (
            ("benchmark", "benchmark_commit"),
            ("IsaacLab", "isaaclab_commit"),
        ):
            revision = subprocess.check_output(
                ["git", "-C", str(runtime / name), "rev-parse", "HEAD"], text=True
            ).strip()
            if revision != manifest[key]:
                raise RuntimeError(
                    f"Expected {name} at {manifest[key]}, found {revision}"
                )
        verify_digest(initial, baseline_input(manifest, task)["sha256"])

    if group == "rlinf":
        overrides = []
        if train_seed:
            overrides = [
                f"actor.seed={train_seed}",
                f"env.train.seed={train_seed}",
                f"runner.logger.experiment_name={run_name}",
                f"env.train.validation_output={directory}/environment",
                f"env.train.video_cfg.video_base_dir={directory}/video",
            ]
        run(
            [
                sys.executable,
                str(project / "examples/embodiment/train_embodied_agent.py"),
                "--config-path",
                str(HERE / "config"),
                "--config-name",
                experiment,
                *overrides,
            ],
            "training",
        )
        if execute:
            checkpoint = (
                directory
                / "checkpoints/global_step_1000/actor/model_state_dict/full_weights.pt"
            )
            _export_rlinf_policy(checkpoint, initial, policy)
    elif group == "ppo":
        config = json.loads((HERE / f"config/{experiment}.json").read_text())
        env.update(config["env"])
        argv = list(map(resolve, config["argv"]))
        if train_seed:
            argv[argv.index("--seed") + 1] = str(train_seed)
            argv[argv.index("--run_name") + 1] = run_name
            argv[argv.index("--experiment_name") + 1] = run_name
        run(
            [sys.executable, resolve(config["entry"]), *argv],
            "training",
        )
        if execute:
            checkpoints = list(ppo_log_dir.glob("*/model_*.pt"))
            if not checkpoints or len({path.parent for path in checkpoints}) != 1:
                raise RuntimeError(
                    f"Expected one {experiment} training run with saved weights, found {checkpoints}"
                )
            # Repeated learn() calls in the posture schedule reuse iteration labels.
            last = max(checkpoints, key=lambda path: path.stat().st_mtime_ns)
            shutil.copyfile(last, policy)

    config = json.loads((HERE / f"config/{task}_baseline.json").read_text())
    for seed in (23456, 34567):
        env["GO2_EVAL_SEED"] = str(seed)
        argv = list(map(resolve, config["argv"]))
        argv[argv.index("--checkpoint") + 1] = str(policy)
        stage = directory / f"evaluation/{seed}"
        argv += ["--output_dir", str(stage)]
        run([sys.executable, resolve(config["entry"]), *argv], f"evaluation/{seed}")
        if execute:
            verify_evaluation(stage, task, argv)
    if execute:
        (directory / "commands.json").write_text(json.dumps(commands, indent=2))
    get_logger().info(
        json.dumps(
            {"experiment": experiment, "execute": execute, "commands": commands},
            indent=2,
        )
    )


def main() -> None:
    """Preview commands by default; --execute starts the selected experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        required=True,
        choices=[
            f"{task}_{group}"
            for task in "ABC"
            for group in ("baseline", "ppo", "rlinf")
        ],
    )
    parser.add_argument(
        "--train-seed",
        type=int,
        default=0,
        help="Continuation training seed (default: 0)",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    launch(args.experiment, args.execute, args.train_seed)


if __name__ == "__main__":
    main()
