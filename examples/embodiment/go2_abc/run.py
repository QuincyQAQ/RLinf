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
import os
import shutil
import subprocess
import sys
from pathlib import Path

from examples.embodiment.go2_abc.prepare_data import DEFAULT_DATA, HERE
from rlinf.utils.logging import get_logger


def launch(experiment: str, execute: bool) -> None:
    """Train a continuation when selected, then evaluate both fixed seeds."""
    runtime = Path(os.environ["GO2_ABC_RUNTIME"]).resolve()
    data = Path(os.environ.get("GO2_ABC_DATA", DEFAULT_DATA)).resolve()
    project = HERE.parents[2]
    import rlinf

    if Path(rlinf.__file__).resolve().parent.parent != project:
        raise RuntimeError(f"Expected RLinf from {project}, found {rlinf.__file__}")
    task, group = experiment.split("_")
    directory = data / "runs" / experiment
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
        RAY_TMPDIR=f"/tmp/abc_{experiment}_{os.environ.get('SLURM_JOB_ID', 'plan')}",
    )
    if task == "B":
        env["GO2_B1_NUMERIC_GUARD"] = "1"
    initial = data / f"models/{task}/baseline.pt"
    policy = initial
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
        if (
            group == "ppo"
            and (runtime / f"benchmark/logs/rsl_rl/{experiment}").exists()
        ):
            raise FileExistsError(f"Ordinary PPO output already exists: {experiment}")
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
        from examples.embodiment.go2_abc.audit import audit

        audit(data)

    if group == "rlinf":
        run(
            [
                sys.executable,
                str(project / "examples/embodiment/train_embodied_agent.py"),
                "--config-path",
                str(HERE / "config"),
                "--config-name",
                experiment,
            ],
            "training",
        )
        checkpoint = (
            directory
            / "checkpoints/global_step_1000/actor/model_state_dict/full_weights.pt"
        )
        policy = directory / "policy.pt"
        if execute:
            import torch

            from examples.embodiment.go2_abc.policy import Go2Policy

            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model = Go2Policy(obs_dim=235, critic_obs_dim=235)
            model.load_state_dict(state, strict=True)
            assert all(torch.isfinite(value).all() for value in state.values()), (
                "Nonfinite native policy weights"
            )
            assert (model.std > 0).all(), (
                "Native policy standard deviations must be positive"
            )
            original = torch.load(initial, map_location="cpu", weights_only=False)[
                "model_state_dict"
            ]
            for prefix in ("actor.", "critic."):
                assert any(
                    not torch.equal(value, original[key])
                    for key, value in state.items()
                    if key.startswith(prefix)
                ), f"No update to {prefix}"
            torch.save(
                {
                    "model_state_dict": state,
                    "iter": 1000,
                    "optimizer_state_dict": torch.optim.Adam(
                        model.parameters()
                    ).state_dict(),
                    "infos": {
                        "inference_export_only": True,
                        "native_checkpoint": str(checkpoint),
                    },
                },
                policy,
            )
    elif group == "ppo":
        config = json.loads((HERE / f"config/{experiment}.json").read_text())
        env.update(config["env"])
        run(
            [sys.executable, resolve(config["entry"]), *map(resolve, config["argv"])],
            "training",
        )
        policy = directory / "policy.pt"
        if execute:
            checkpoints = list(
                (runtime / f"benchmark/logs/rsl_rl/{experiment}").glob("*/model_*.pt")
            )
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
        argv += ["--output_dir", str(directory / f"evaluation/{seed}")]
        run([sys.executable, resolve(config["entry"]), *argv], f"evaluation/{seed}")
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
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    launch(args.experiment, args.execute)


if __name__ == "__main__":
    main()
