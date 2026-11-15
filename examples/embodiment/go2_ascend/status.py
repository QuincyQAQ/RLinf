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

"""Report the current Go2 RLinf run without touching other users' processes."""

import argparse
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "outputs/go2_ascend/from_scratch",
    )
    args = parser.parse_args()
    run = args.output_dir.expanduser().resolve()
    launch = json.loads((run / "launch.json").read_text())
    command = Path(f"/proc/{launch['pid']}/cmdline")
    alive = command.exists() and b"train_go2.py" in command.read_bytes()
    report = {
        "driver_pid": launch["pid"],
        "running": alive,
        "target_updates": launch["target_updates"],
    }
    if not (run / "tensorboard").exists():
        report["phase"] = "starting workers"
        print(json.dumps(report, indent=2))
        return
    events = EventAccumulator(
        str(run / "tensorboard"), size_guidance={"scalars": 0}
    ).Reload()
    tags = events.Tags()["scalars"]
    if "train/actor/lr" in tags:
        rows = sorted(
            {r.step: r for r in events.Scalars("train/actor/lr")}.values(),
            key=lambda r: r.step,
        )
        completed = rows[-1].step + 1
        report["completed_updates"] = completed
        report["control_transitions"] = (
            completed * launch["global_envs"] * launch["rollout_steps"]
        )
        report["learning_rate"] = rows[-1].value
        if len(rows) >= 3:
            window = rows[-min(50, len(rows)) :]
            seconds = (window[-1].wall_time - window[0].wall_time) / (len(window) - 1)
            report["recent_seconds_per_update"] = seconds
            report["estimated_hours_remaining"] = (
                (launch["target_updates"] - completed) * seconds / 3600
            )
        for tag in [
            "train/actor/policy_sync_error",
            "env/episode_len",
            "env/lin_vel_cmd_level",
            "eval/return",
        ]:
            if tag in tags:
                values = events.Scalars(tag)
                if values:
                    report[tag] = values[-1].value
    state = run / "training_env_state.json"
    if state.exists():
        report["environment"] = json.loads(state.read_text())
    checkpoints = run / launch["experiment_name"] / "checkpoints"
    candidates = list(checkpoints.glob("global_step_*"))
    if candidates:
        report["latest_checkpoint"] = str(
            max(candidates, key=lambda p: int(p.name.rsplit("_", 1)[1]))
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
