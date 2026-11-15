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

"""Check initialization and PPO equations against the previous Go2 setup."""

import json
from pathlib import Path

import torch
from go2_rsl_extension import (
    RslGo2Policy,
    adaptive_learning_rate,
    gaussian_kl,
    rsl_ppo_loss,
)
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict
from train_go2 import progress_flags


def main():
    torch.set_num_threads(4)
    cfg = {
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "activation": "elu",
        "init_noise_std": 1.0,
        "actor_obs_normalization": False,
        "critic_obs_normalization": False,
        "noise_std_type": "scalar",
    }
    torch.manual_seed(42)
    obs = TensorDict(
        {"policy": torch.zeros(1, 45), "critic": torch.zeros(1, 60)}, batch_size=[1]
    )
    original = ActorCritic(obs, {"policy": ["policy"], "critic": ["critic"]}, 12, **cfg)
    torch.manual_seed(42)
    adapted = RslGo2Policy()
    error = max(
        (original.state_dict()[k] - v).abs().max().item()
        for k, v in adapted.state_dict().items()
    )
    assert error == 0
    logp = torch.zeros(2, 1, requires_grad=True)
    value = torch.zeros(2, 1, requires_grad=True)
    loss, _ = rsl_ppo_loss(
        logprobs=logp,
        old_logprobs=torch.zeros_like(logp),
        advantages=torch.tensor([[1.0], [-1.0]]),
        values=value,
        prev_values=torch.zeros_like(value),
        returns=torch.tensor([[20.0], [-20.0]]),
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        value_clip=0.2,
    )
    assert float(loss) == 400
    loss.backward()
    torch.testing.assert_close(value.grad, torch.tensor([[-20.0], [20.0]]))
    assert adaptive_learning_rate(0.001, 0.03, 0.01) == 0.001 / 1.5
    assert adaptive_learning_rate(0.001, 0.001, 0.01) == 0.0015
    assert adaptive_learning_rate(1e-5, 1.0, 0.01) == 1e-5
    assert adaptive_learning_rate(1e-2, 0.001, 0.01) == 1e-2
    mean = torch.randn(5, 12)
    std = torch.rand(5, 12) + 0.5
    old_mean = torch.randn(5, 12)
    old_std = torch.rand(5, 12) + 0.5
    expected = torch.distributions.kl_divergence(
        torch.distributions.Normal(old_mean, old_std),
        torch.distributions.Normal(mean, std),
    ).sum(-1)
    actual = gaussian_kl(old_mean, old_std, mean, std)
    # RSL-RL adds 1e-5 inside the logarithm; account for that explicit difference.
    correction = torch.log1p(1e-5 * old_std / std).sum(-1)
    torch.testing.assert_close(actual, expected + correction)
    for step, expected_flags in [
        (1, (False, False)),
        (100, (True, False)),
        (1000, (True, True)),
        (50000, (True, True)),
    ]:
        assert progress_flags(step, 50000, 100, 1000) == expected_flags
    report = {
        "seed42_initialization_max_error": error,
        "large_error_squared_value_loss": float(loss),
        "squared_value_gradient_checked": True,
        "gaussian_kl_checked": True,
        "adaptive_learning_rate_bounds_checked": True,
        "independent_intervals_checked": True,
    }
    (Path(__file__).parent / "math_verification.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
