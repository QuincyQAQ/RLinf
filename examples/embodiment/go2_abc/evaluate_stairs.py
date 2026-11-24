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

"""Evaluate a trained Go2 C checkpoint on the two reference stair trials."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
REFERENCE = PROJECT.parent / "go2-reference"
RUNTIME = PROJECT.parent / "public-evaluator"


def _reference_python(command: str, *args: str | Path) -> None:
    # Isolate the two checkouts' identically named Python packages.
    imports = """from pathlib import Path
import sys
from examples.embodiment.go2_abc import run
if Path(run.__file__).resolve() != Path("examples/embodiment/go2_abc/run.py").resolve():
    raise ImportError(f"Expected the Go2 reference in {Path.cwd()}, got {run.__file__}")
"""
    subprocess.run(
        [sys.executable, "-c", imports + command, *args], cwd=REFERENCE, check=True
    )


def evaluation_args(
    config: dict, policy: Path, output: Path, asset_root: Path | None
) -> list[str]:
    """Use the reference C protocol with the exported checkpoint and output paths."""
    argv = config["argv"].copy()
    argv[argv.index("--checkpoint") + 1] = str(policy)
    argv.extend(["--output_dir", str(output)])
    kit = "--/app/extensions/excluded/0=omni.kit.telemetry"
    if asset_root is not None:
        kit += f" --/persistent/isaac/asset_root/cloud={asset_root.resolve()}"
    return [*argv, f"--kit_args={kit}"]


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

    config = json.loads(
        (REFERENCE / "examples/embodiment/go2_abc/config/C_baseline.json").read_text()
    )
    entry = Path(config["entry"].replace("${GO2_ABC_RUNTIME}", str(RUNTIME)))
    baseline = Path(os.environ.get("GO2_ABC_DATA", PROJECT / "data/go2_abc"))
    output = args.output.resolve()
    output.mkdir(parents=True)
    policy = output / "policy.pt"
    _reference_python(
        "run._export_rlinf_policy(*map(Path, sys.argv[1:]))",
        args.checkpoint.resolve(),
        (baseline / "models/C/baseline.pt").resolve(),
        policy,
    )

    episodes = tracking = passage = 0
    for seed in (23456, 34567):
        directory = output / str(seed)
        argv = evaluation_args(config, policy, directory, args.asset_root)
        env = dict(os.environ, GO2_EVAL_SEED=str(seed))
        subprocess.run([sys.executable, str(entry), *argv], env=env, check=True)
        _reference_python(
            "run.verify_evaluation(Path(sys.argv[1]), 'C', sys.argv[2:])",
            directory,
            *argv,
        )
        report = json.loads((directory / "B2_stairs_s0/B2_stairs.json").read_text())
        rows = report["episodes"]
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
