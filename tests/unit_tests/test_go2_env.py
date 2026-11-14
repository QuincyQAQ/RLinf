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

"""Real Go2 simulation and asymmetric policy integration contracts."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.envs import get_env_cls
from rlinf.envs.action_utils import prepare_actions
from rlinf.models import get_model
from rlinf.models.embodiment.go2_policy.go2_policy import Go2Policy


@pytest.fixture
def env():
    pytest.importorskip("rlinf.envs.sim.go2_mujoco.batched_go2")
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "group_size": 1,
            "threads": 2,
            "auto_reset": True,
            "ignore_terminations": False,
            "randomize": False,
            "noise": False,
            "max_episode_steps": 2,
        }
    )
    instance = get_env_cls("go2_mujoco")(cfg, num_envs=4)
    yield instance
    instance.close()
    instance.close()


def test_go2_timeout_preserves_terminal_observation(env):
    obs, _ = env.reset()
    assert obs["states"].shape == (4, 45)
    assert obs["critic_states"].shape == (4, 60)
    actions = torch.zeros(4, 1, 12)
    _, _, terms, truncs, _ = env.chunk_step(actions)
    assert not terms.any() and not truncs.any()
    obs, reward, terms, truncs, infos = env.chunk_step(actions)
    assert reward.shape == terms.shape == truncs.shape == (4, 1)
    assert truncs.all() and not terms.any()
    info = infos[0]
    assert info["_final_observation"].all()
    assert (info["final_info"]["episode"]["episode_len"] == 2).all()
    assert (env.elapsed_steps == 0).all()
    assert not torch.equal(
        obs[0]["critic_states"], info["final_observation"]["critic_states"]
    )
    assert all(torch.isfinite(x).all() for x in obs[0].values())


def test_go2_partial_reset_and_action_passthrough(env):
    env.reset()
    env.step(np.zeros((4, 12)))
    before = env.task.state.copy()
    env.reset(np.array([1], dtype=np.int32))
    assert env.elapsed_steps.tolist() == [1, 0, 1, 1]
    np.testing.assert_array_equal(env.task.state[[0, 2, 3]], before[[0, 2, 3]])
    actions = torch.linspace(-2, 2, 48).reshape(4, 1, 12)
    prepared = prepare_actions(actions, "go2_mujoco", "go2_policy", 1, 12)
    np.testing.assert_array_equal(prepared, actions.numpy())
    with pytest.raises(ValueError, match="Expected actions"):
        env.chunk_step(torch.zeros(4, 2, 12))


def test_go2_rollout_recompute_and_privileged_critic():
    policy = get_model(
        OmegaConf.create(
            {
                "model_type": "go2_policy",
                "precision": "32",
                "is_lora": False,
                "load_to_device": False,
                "obs_dim": 45,
                "action_dim": 12,
                "num_action_chunks": 1,
                "initial_checkpoint": None,
            }
        )
    )
    obs = {"states": torch.randn(5, 45), "critic_states": torch.randn(5, 60)}
    actions, rollout = policy.predict_action_batch(obs)
    recomputed = policy(forward_inputs=rollout["forward_inputs"])
    torch.testing.assert_close(recomputed["logprobs"], rollout["prev_logprobs"])
    torch.testing.assert_close(recomputed["values"], rollout["prev_values"])
    assert actions.shape == (5, 1, 12)
    assert recomputed["logprobs"].shape == (5, 12)
    altered = {**obs, "critic_states": obs["critic_states"] + 2}
    a, original = policy.predict_action_batch(obs, mode="eval")
    b, changed = policy.predict_action_batch(altered, mode="eval")
    torch.testing.assert_close(a, b)
    assert not torch.equal(original["prev_values"], changed["prev_values"])
    (recomputed["values"].square().mean() - recomputed["logprobs"].mean()).backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in policy.parameters()
    )


def test_go2_rsl_checkpoint_import(tmp_path):
    original = Go2Policy()
    path = tmp_path / "model.pt"
    torch.save({"model_state_dict": original.state_dict()}, path)
    restored = Go2Policy(str(path))
    obs = {"states": torch.randn(3, 45), "critic_states": torch.randn(3, 60)}
    a, old = original.predict_action_batch(obs, mode="eval")
    b, new = restored.predict_action_batch(obs, mode="eval")
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(old["prev_values"], new["prev_values"], rtol=0, atol=0)


def test_go2_generic_rollout_sampling():
    policy = Go2Policy()
    obs = {"states": torch.randn(4, 45), "critic_states": torch.randn(4, 60)}
    deterministic, _ = policy.predict_action_batch(obs, do_sample=False)
    with torch.no_grad():
        mean = policy.actor(obs["states"])[:, None]
    torch.testing.assert_close(deterministic, mean, rtol=0, atol=0)
    repeated, _ = policy.predict_action_batch(obs, do_sample=False)
    torch.testing.assert_close(deterministic, repeated, rtol=0, atol=0)
    sampled, _ = policy.predict_action_batch(obs, do_sample=True)
    assert not torch.equal(sampled, mean)
