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

"""Replay policy sensor frames offline or connect an external perception provider."""

import argparse
import json
import math
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from rlinf.utils.logging import get_logger

from .control import Go2Control, JointState, Perception, vector
from .prepare_data import DEFAULT_DATA
from .sdk2 import Go2SDK2


def run_control_loop(
    controller: Go2Control,
    transport: Go2SDK2,
    perception: Callable[[], Perception],
    command: Callable[[], list[float]],
    *,
    duration: float = 10.0,
) -> None:
    """Run the supplied state/perception sources; a default transport only observes.

    Perception must supply synchronized body velocity and the measured terrain
    grid. The caller connects the transport after arranging its robot startup.
    """
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Expected a finite positive control duration")
    started = deadline = time.monotonic()
    first = True
    try:
        while time.monotonic() - started < duration:
            state, sensors, velocity = transport.read(), perception(), command()
            now = time.monotonic()
            if first and transport.enable_motion:
                q = vector(state.q, 12, "SDK initial pose")[controller.sdk_indices]
                if (
                    np.max(np.abs(q - controller.default_q))
                    > controller.spec["initial_pose_tolerance"]
                ):
                    raise ValueError(
                        "Expected the robot at the recorded neutral pose before joint control"
                    )
                if np.any(vector(velocity, 3, "initial velocity command") != 0):
                    raise ValueError(
                        "Expected zero velocity command for the first control sample"
                    )
            result = controller.step(state, sensors, velocity, now)
            if transport.enable_motion:
                transport.write(result["torque_sdk"])
            first = False
            deadline += controller.spec["physics_dt"]
            remaining = deadline - time.monotonic()
            if remaining < -controller.spec["physics_dt"]:
                raise TimeoutError("Control loop missed its physics-step deadline")
            time.sleep(max(0.0, remaining))
    finally:
        transport.close()


def main() -> None:
    """Replay timestamped JSONL sensor frames without opening an SDK2 channel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=list("ABC"))
    parser.add_argument("--group", required=True, choices=["baseline", "ppo", "rlinf"])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.frames.resolve() == args.output.resolve():
        raise ValueError("Expected a separate output path for replay commands")
    controller = Go2Control(
        args.data / f"models/{args.task}/{args.group}.pt", args.task
    )
    count = 0
    with args.frames.open() as frames, args.output.open("x") as output:
        for line in frames:
            frame = json.loads(line)
            result = controller.step(
                JointState(**frame["state"]),
                Perception(**frame["perception"]),
                frame["command"],
                frame["time"],
            )
            output.write(
                json.dumps(
                    {
                        "time": frame["time"],
                        "target_q_sdk": result["target_q_sdk"].tolist(),
                        "torque_sdk": result["torque_sdk"].tolist(),
                    }
                )
                + "\n"
            )
            count += 1
    if not count:
        raise ValueError("Expected at least one sensor frame")
    get_logger().info("Replayed %s frames on CPU; SDK2 channels opened: 0", count)


if __name__ == "__main__":
    main()
