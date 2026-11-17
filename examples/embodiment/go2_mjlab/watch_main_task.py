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

"""Evaluate only B2 checkpoints without changing the active training run."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    sys.path.insert(0, str(root / "RLinf"))
    import torch

    from rlinf.envs.sim.go2_mjlab.benchmark_metrics import summarize

    python = sys.executable
    driver = json.loads((output / "launch.json").read_text())["pid"]
    state = {"driver": driver, "steps": [100, 500, 1000, 2200], "events": []}
    environment = os.environ.copy()
    environment.update(
        TORCH_DEVICE_BACKEND_AUTOLOAD="0",
        PYTHONPATH=str(root / "RLinf") + ":" + str(root / "unitree_rl_mjlab"),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_WAIT_POLICY="PASSIVE",
        GOMP_SPINCOUNT="0",
    )

    def note(event, **data):
        state["events"].append(dict(event=event, unix=time.time(), **data))
        temporary = output / "watch_status.tmp"
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(output / "watch_status.json")

    def alive():
        stat = Path(f"/proc/{driver}/stat")
        return stat.exists() and stat.read_text().split()[2] != "Z"

    def cell(step, checkpoint, seed, episodes):
        destination = output / "evaluations" / f"step_{step}_seed_{seed}"
        destination.mkdir(parents=True)
        command = [
            python,
            str(root / "RLinf/examples/embodiment/go2_mjlab/evaluate_benchmark.py"),
            "--checkpoint",
            str(checkpoint),
            "--training-config",
            str(output / "launch_config.yaml"),
            "--eval-task",
            "B2_stairs",
            "--num-envs",
            "16",
            "--num-episodes",
            str(episodes),
            "--threads",
            "1",
            "--eval-seed",
            str(seed),
            "--output",
            str(destination),
        ]
        with (destination / "console.log").open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=root / "RLinf",
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=2700)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise RuntimeError(f"Evaluation timed out: {step}, seed {seed}")
        if returncode:
            raise RuntimeError(f"Evaluation exited {returncode}: {destination}")
        files = list(destination.rglob("B2_stairs.json"))
        assert len(files) == 1
        result = json.loads(files[0].read_text())
        assert len(result["episodes"]) == episodes
        return result

    note(
        "started",
        main_task="B2_stairs",
        diagnostic_episodes=64,
        final_episodes=1024,
        max_evaluation_processes=4,
    )
    try:
        resume = json.loads((output / "launch.json").read_text()).get("resume_dir")
        resume_step = int(Path(resume).name.split("_")[-1]) if resume else 0
        for step in state["steps"]:
            if step <= resume_step:
                note("evaluation_precedes_resume", step=step, resume_step=resume_step)
                continue
            checkpoint = (
                output
                / f"B2_stairs_posture/checkpoints/global_step_{step}/actor/model_state_dict/full_weights.pt"
            )
            deadline = time.time() + 24 * 3600
            while not checkpoint.exists():
                if not alive() or time.time() > deadline:
                    raise RuntimeError(f"Training stopped before checkpoint {step}")
                time.sleep(20)
            time.sleep(5)
            weights = torch.load(checkpoint, map_location="cpu", weights_only=False)
            assert all(torch.isfinite(value).all() for value in weights.values())
            episodes_per_seed = 256 if step == 2200 else 16
            note("evaluation_started", step=step, total_episodes=4 * episodes_per_seed)
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(
                    pool.map(
                        lambda seed: cell(step, checkpoint, seed, episodes_per_seed),
                        range(42, 46),
                    )
                )
            episodes = [episode for result in results for episode in result["episodes"]]
            summary = summarize(episodes)
            for key in ("stairs_pass_success", "stairs_tracking_and_pass_success"):
                summary[key + "_rate"] = sum(
                    episode[key] for episode in episodes
                ) / len(episodes)
            result = {
                "summary": summary,
                "episodes": episodes,
                "protocol": {
                    "checkpoint_step": step,
                    "task": "B2_stairs",
                    "formal": (step == 2200),
                    "seeds": list(range(42, 46)),
                    "checkpoint_sha256": results[0]["protocol"]["checkpoint_sha256"],
                },
            }
            (output / "evaluations" / f"step_{step}.json").write_text(
                json.dumps(result, indent=2)
            )
            note("evaluation_complete", step=step, summary=summary)
        note("complete")
    except Exception as error:
        note("failed", error=repr(error), training_not_modified=True)
        raise


if __name__ == "__main__":
    main()
