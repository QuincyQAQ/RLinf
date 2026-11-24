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

"""Check the nine ABC policy inputs and their PPO interfaces on CPU."""

import argparse
import json
from pathlib import Path

from examples.embodiment.go2_abc.prepare_data import HERE, data_directory, verify_digest
from rlinf.utils.logging import get_logger


def audit(data: Path) -> dict:
    """Check input hashes, policy loading, and analytical KL."""
    import torch
    from torch.nn import functional as F

    from examples.embodiment.go2_abc.policy import Go2Policy

    manifest = json.loads((HERE / "manifest.json").read_text())
    expected = {
        f"models/{task}/{group}.pt"
        for task in "ABC"
        for group in ("baseline", "ppo", "rlinf")
    }
    files = manifest["files"]
    if len(files) != len(expected) or {item["path"] for item in files} != expected:
        raise ValueError(
            "Expected exactly one manifest entry for each of the nine Go2 policies"
        )
    generator = torch.Generator().manual_seed(0)
    observations = torch.randn(8, 235, generator=generator)
    for item in files:
        path = data / item["path"]
        verify_digest(path, item["sha256"])
        policy = Go2Policy(str(path), 235, 235)
        state = policy.state_dict()
        if not all(torch.isfinite(x).all() for x in state.values()):
            raise ValueError(f"Nonfinite policy parameters: {path}")
        if not (policy.std > 0).all():
            raise ValueError(f"Policy standard deviations must be positive: {path}")
        action, rollout = policy.predict_action_batch(
            {"states": observations}, mode="eval"
        )
        expected_action = observations
        for index in (0, 2, 4, 6):
            expected_action = F.linear(
                expected_action,
                state[f"actor.{index}.weight"],
                state[f"actor.{index}.bias"],
            )
            if index != 6:
                expected_action = F.elu(expected_action)
        torch.testing.assert_close(action[:, 0], expected_action, rtol=0, atol=0)
        output = policy.default_forward(rollout["forward_inputs"])
        torch.testing.assert_close(
            output["kl_divergence"], torch.zeros(8), rtol=0, atol=0
        )
        policy.actor[-1].bias.data.add_(0.01)
        shifted = policy.default_forward(rollout["forward_inputs"])["kl_divergence"]
        if not (torch.isfinite(shifted) & (shifted > 0)).all():
            raise ValueError(
                f"Expected finite positive analytical KL after changing the policy: {path}"
            )
    return {
        "models": len(files),
        "device": "cpu",
        "status": "passed",
    }


def main() -> None:
    """Run the CPU audit without starting a simulator or a Ray cluster."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=data_directory())
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = audit(args.data)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
    get_logger().info(json.dumps(result))


if __name__ == "__main__":
    main()
