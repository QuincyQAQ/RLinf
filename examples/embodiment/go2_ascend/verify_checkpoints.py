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

"""Inspect native DCP optimizer and scheduler restoration from the completed tests."""

import argparse
import json
import tempfile
from pathlib import Path

import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


def inspect(path):
    with tempfile.NamedTemporaryFile(suffix=".pt") as temporary:
        dcp_to_torch_save(str(path / "actor/dcp_checkpoint"), temporary.name)
        data = torch.load(temporary.name, map_location="cpu", weights_only=False)[
            "fsdp_checkpoint"
        ]
    steps = []

    def collect(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "step":
                    steps.append(float(child))
                else:
                    collect(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                collect(child)

    collect(data["optimizers"])
    return {"adam_steps": sorted(set(steps)), "schedulers": data["lr_schedulers"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "outputs/go2_ascend",
    )
    args = parser.parse_args()
    root = args.runs_dir.resolve()
    report = {}
    for stage in ("single", "eight"):
        before = root / f"{stage}_smoke/{stage}_smoke/checkpoints/global_step_3"
        after = root / f"{stage}_resume/{stage}_resume/checkpoints/global_step_5"
        a, b = inspect(before), inspect(after)
        assert a["adam_steps"] == [60.0] and b["adam_steps"] == [100.0]
        assert a["schedulers"][0]["last_epoch"] == 3
        assert b["schedulers"][0]["last_epoch"] == 5
        report[stage] = {"step_3": a, "step_5": b}
    (root / "checkpoint_restore_verification.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
