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

"""Supervise independent RSL-RL specialists and main-task evaluations."""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from rsl_baseline import atomic_json

from rlinf.envs.sim.go2_mjlab.benchmark_protocol import TASKS
from rlinf.utils.logging import get_logger

logger = get_logger()


def start(command: list[str], log: Path, environment: dict) -> subprocess.Popen:
    """Keep experiment output outside the source checkout."""
    with log.open("x") as output:
        return subprocess.Popen(
            command,
            stdout=output,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=environment,
        )


def main() -> None:
    """Assign one accelerator and one training seed to each requested task."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", required=True)
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASKS), default=list(TASKS))
    parser.add_argument("--updates", type=int, default=2200)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval-steps", nargs="+", type=int, default=[1200, 1500, 1800, 2200]
    )
    parser.add_argument("--eval-seed", type=int, default=43)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--final-eval-episodes", type=int, default=256)
    parser.add_argument("--eval-envs", type=int, default=64)
    parser.add_argument("--max-evaluations", type=int, default=5)
    parser.add_argument("--evaluation-timeout", type=int, default=7200)
    args = parser.parse_args()
    if len(args.devices) != len(args.tasks) or len(set(args.devices)) != len(
        args.devices
    ):
        parser.error("Each task requires its own distinct device")
    if len(set(args.tasks)) != len(args.tasks):
        parser.error("Task names must be distinct")
    if (
        min(
            args.updates,
            args.num_envs,
            args.workers,
            args.max_evaluations,
            args.evaluation_timeout,
            args.eval_episodes,
            args.final_eval_episodes,
            args.eval_envs,
        )
        < 1
    ):
        parser.error("Budgets must be positive")
    if any(s < 1 or s > args.updates for s in args.eval_steps):
        parser.error("Evaluation checkpoints must be within the training budget")
    args.output.mkdir(parents=True, exist_ok=False)
    scripts = Path(__file__).resolve().parent
    environment = os.environ.copy()
    environment.update(
        TORCH_DEVICE_BACKEND_AUTOLOAD="0",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        OMP_WAIT_POLICY="PASSIVE",
        GOMP_SPINCOUNT="0",
    )
    processes = {}
    records = {}
    evaluations = {}
    active = {}
    for task, device in zip(args.tasks, args.devices, strict=True):
        child_environment = dict(environment, ASCEND_RT_VISIBLE_DEVICES=device)
        command = [
            str(args.python),
            str(scripts / "rsl_baseline.py"),
            "--task",
            task,
            "--output",
            str(args.output / task),
            "--updates",
            str(args.updates),
            "--num-envs",
            str(args.num_envs),
            "--workers",
            str(args.workers),
            "--seed",
            str(args.seed),
            "--device",
            "npu:0",
        ]
        process = start(
            command, args.output / (task + ".console.log"), child_environment
        )
        processes[task] = process
        records[task] = {"pid": process.pid, "device": device, "command": command}
    atomic_json(
        args.output / "launch.json",
        {
            "pid": os.getpid(),
            "started_unix": time.time(),
            "training": records,
            "plan": vars(args)
            | {"python": str(args.python), "output": str(args.output)},
        },
    )
    stopped = False

    def request_stop(signum, frame):
        nonlocal stopped
        stopped = True
        for process in processes.values():
            if process.poll() is None:
                process.terminate()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    while True:
        now = time.time()
        for key, (process, started) in list(active.items()):
            code = process.poll()
            if code is None and now - started > args.evaluation_timeout:
                process.terminate()
                code = process.wait(timeout=30)
            if code is not None:
                evaluations[key].update(exit_code=code, finished_unix=now)
                del active[key]
        for step in args.eval_steps:
            for task in args.tasks:
                key = f"{task}/{step}"
                checkpoint = args.output / task / "checkpoints" / f"global_step_{step}"
                if key in evaluations or not (checkpoint / "checkpoint.json").exists():
                    continue
                if len(active) >= args.max_evaluations or stopped:
                    continue
                destination = args.output / "evaluations" / task / f"step_{step}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                episodes = (
                    args.final_eval_episodes
                    if step == args.updates
                    else args.eval_episodes
                )
                command = [
                    str(args.python),
                    str(scripts / "evaluate_benchmark.py"),
                    "--checkpoint",
                    str(checkpoint / "full_weights.pt"),
                    "--training-config",
                    str(args.output / task / "training_config.yaml"),
                    "--eval-task",
                    task,
                    "--num-envs",
                    str(args.eval_envs),
                    "--num-episodes",
                    str(episodes),
                    "--eval-seed",
                    str(args.eval_seed),
                    "--threads",
                    "1",
                    "--output",
                    str(destination),
                ]
                process = start(command, destination.with_suffix(".log"), environment)
                active[key] = (process, now)
                evaluations[key] = {
                    "pid": process.pid,
                    "started_unix": now,
                    "requested_episodes": episodes,
                    "eval_seed": args.eval_seed,
                    "output": str(destination),
                }
        training = {
            task: {**record, "exit_code": processes[task].poll()}
            for task, record in records.items()
        }
        for task in args.tasks:
            progress = args.output / task / "progress.json"
            if progress.exists():
                training[task]["progress"] = json.loads(progress.read_text())
        status = {
            "unix": now,
            "training": training,
            "evaluations": evaluations,
            "stopping": stopped,
        }
        atomic_json(args.output / "suite_status.json", status)
        if all(p.poll() is not None for p in processes.values()) and not active:
            missing = [
                f"{task}/{step}"
                for task in args.tasks
                for step in args.eval_steps
                if f"{task}/{step}" not in evaluations
            ]
            status.update(
                complete=not missing
                and all(e.get("exit_code") == 0 for e in evaluations.values())
                and all(p.returncode == 0 for p in processes.values()),
                missing_evaluations=missing,
            )
            atomic_json(args.output / "finished.json", status)
            logger.info(
                "RSL-RL baseline suite finished; complete=%s", status["complete"]
            )
            break
        time.sleep(5)


if __name__ == "__main__":
    main()
