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

"""Start or resume Go2 training from this repository checkout."""

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf


def build_launch(config_name, output_dir=None, resume=False):
    """Resolve one configuration and prepare a persistent training command."""
    scripts = Path(__file__).resolve().parent
    repo = scripts.parents[2]
    config_path = scripts / f"{config_name}.yaml"
    cfg = OmegaConf.load(config_path)
    output = Path(output_dir or cfg.runner.logger.log_path).expanduser()
    if not output.is_absolute():
        output = repo / output
    output = output.resolve()
    command = [
        "bash",
        str(scripts / "run_stage.sh"),
        config_name,
        f"runner.logger.log_path={output}",
    ]
    checkpoint = None
    if resume:
        checkpoints = output / cfg.runner.logger.experiment_name / "checkpoints"
        candidates = [
            p
            for p in checkpoints.glob("global_step_*")
            if (p / "environment_curriculum.json").is_file()
        ]
        if not candidates:
            raise FileNotFoundError(f"No complete checkpoints in {checkpoints}")
        checkpoint = max(candidates, key=lambda p: int(p.name.rsplit("_", 1)[1]))
        if int(checkpoint.name.rsplit("_", 1)[1]) >= int(cfg.runner.max_steps):
            raise RuntimeError("Training has already reached its configured target")
        command.append(f"runner.resume_dir={checkpoint}")
    elif (
        cfg.actor.model.initial_checkpoint
        or cfg.runner.resume_dir
        or cfg.runner.ckpt_path
    ):
        raise ValueError(
            "Fresh training requires empty initial_checkpoint, resume_dir and ckpt_path"
        )
    metadata = {
        "repo": str(repo),
        "output_dir": str(output),
        "command": command,
        "resume": str(checkpoint) if checkpoint else None,
        "global_envs": int(cfg.env.train.total_num_envs),
        "rollout_steps": int(cfg.env.train.max_steps_per_rollout_epoch),
        "simulation_threads": int(cfg.env.train.threads),
        "target_updates": int(cfg.runner.max_steps),
        "experiment_name": str(cfg.runner.logger.experiment_name),
        "component_placement": OmegaConf.to_container(cfg.cluster.component_placement),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "extension_sha256": hashlib.sha256(
            (scripts / "go2_rsl_extension.py").read_bytes()
        ).hexdigest(),
        "entry_sha256": hashlib.sha256(
            (scripts / "train_go2.py").read_bytes()
        ).hexdigest(),
    }
    return output, command, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="from_scratch")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output, command, metadata = build_launch(args.config, args.output_dir, args.resume)
    if args.dry_run:
        print(json.dumps(metadata, indent=2))
        return
    pid_file = output / "training.pid"
    if pid_file.exists():
        pid = int(pid_file.read_text())
        cmdline = Path(f"/proc/{pid}/cmdline")
        if cmdline.exists() and b"train_go2.py" in cmdline.read_bytes():
            raise RuntimeError(f"Go2 training driver {pid} is already running")
    if not args.resume and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite experiment {output}")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "console.log").open("a" if args.resume else "x") as handle:
        process = subprocess.Popen(
            command,
            cwd=metadata["repo"],
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=os.environ.copy(),
        )
    metadata["pid"] = process.pid
    metadata["launched_utc"] = datetime.now(timezone.utc).isoformat()
    try:
        metadata["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=metadata["repo"], text=True
        ).strip()
        metadata["git_dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=metadata["repo"], text=True
            ).strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        metadata["git_commit"] = None
    pid_file.write_text(str(process.pid) + "\n")
    (output / "launch.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
