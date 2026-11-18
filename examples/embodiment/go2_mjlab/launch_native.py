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

"""Launch native RLinf with explicit paths and a saved resolved configuration."""

import argparse
import json
import os
import resource
import subprocess
import time
import uuid
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", default="go2_mjlab_benchmark_b2_posture_ascend")
    parser.add_argument("--max-steps", type=int, default=2200)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--cann-env", type=Path)
    parser.add_argument(
        "--override", action="append", default=[], help="Additional Hydra override"
    )
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    repo = root / "RLinf"
    if args.max_steps < 1 or args.save_interval < 1:
        parser.error("Step counts must be positive")
    if not args.python.is_file():
        parser.error("Interpreter is missing")
    if args.resume and not (args.resume / "actor/dcp_checkpoint/.metadata").is_file():
        parser.error("Resume needs a complete distributed checkpoint directory")
    overrides = [
        *args.override,
        f"runner.logger.log_path={output}",
        f"runner.max_steps={args.max_steps}",
        f"runner.max_epochs={args.max_steps}",
        f"runner.save_interval={args.save_interval}",
    ]
    if args.resume:
        overrides.append(f"runner.resume_dir={args.resume.resolve()}")
    with initialize_config_dir(
        config_dir=str(repo / "examples/embodiment/config"), version_base=None
    ):
        config = compose(config_name=args.config, overrides=overrides)
    output.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(config, output / "launch_config.yaml", resolve=True)
    environment = os.environ.copy()
    environment.pop("TORCH_DEVICE_BACKEND_AUTOLOAD", None)
    environment.update(
        GO2_PYTHON=str(args.python.absolute()),
        PYTHONPATH=str(repo) + ":" + str(root / "unitree_rl_mjlab"),
        RAY_ADDRESS="local",
        RAY_TMPDIR=f"/tmp/go2up_{uuid.uuid4().hex[:12]}",
        MPLCONFIGDIR="/tmp/go2_migration_mpl",
        OPENBLAS_NUM_THREADS="1",
    )
    if args.cann_env:
        environment["CANN_ENV_FILE"] = str(args.cann_env.resolve())
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, min(65536, hard)), hard))
    command = [
        "bash",
        str(repo / "examples/embodiment/go2_ascend/run_native.sh"),
        args.config,
        *overrides,
    ]
    with (output / "console.log").open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=repo,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    record = {
        "pid": process.pid,
        "started_unix": time.time(),
        "command": command,
        "source_root": str(root),
        "python": str(args.python),
        "config": args.config,
        "resume_dir": str(args.resume) if args.resume else None,
        "target_updates": args.max_steps,
        "num_envs": int(config.env.train.total_num_envs),
        "physics_backend": config.env.train.physics_backend,
    }
    (output / "launch.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
