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

"""Inspect finite policy weights and DCP optimizer/scheduler counters."""

import argparse
import json
import tempfile
from pathlib import Path

import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


def inspect(path):
    state = torch.load(
        path / "actor/model_state_dict/full_weights.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert all(torch.isfinite(value).all() for value in state.values())
    with tempfile.TemporaryDirectory() as temporary:
        destination = Path(temporary) / "checkpoint.pt"
        dcp_to_torch_save(str(path / "actor/dcp_checkpoint"), str(destination))
        data = torch.load(destination, map_location="cpu", weights_only=False)[
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
    return {
        "checkpoint": str(path),
        "finite": True,
        "actor_shape": list(state["actor.0.weight"].shape),
        "adam_steps": sorted(set(steps)),
        "schedulers": data["lr_schedulers"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--expected-updates", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inspect(args.checkpoint)
    if args.expected_updates is not None:
        assert result["adam_steps"] == [float(args.expected_updates * 20)], result
        assert result["schedulers"][0]["last_epoch"] == args.expected_updates, result
    payload = json.dumps(result, indent=2, default=str)
    if args.output:
        args.output.write_text(payload)
    print(payload)
