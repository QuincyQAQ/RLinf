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


def test_mjlab_timeout_preserves_terminal_observation():
    pytest.importorskip("mjlab")
    pytest.importorskip("src.tasks")
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "group_size": 1,
            "threads": 2,
            "auto_reset": True,
            "ignore_terminations": False,
            "max_episode_steps": 2,
            "max_steps_per_rollout_epoch": 24,
            "video_cfg": {"save_video": False},
        }
    )
    instance = get_env_cls("go2_mjlab")(cfg, num_envs=4)
    try:
        obs, _ = instance.reset()
        assert obs["states"].shape == (4, 47)
        assert obs["critic_states"].shape == (4, 74)
        instance.chunk_step(torch.zeros(4, 1, 12))
        obs, _, terms, truncs, infos = instance.chunk_step(torch.zeros(4, 1, 12))
        assert truncs.all() and not terms.any()
        assert not torch.equal(
            infos[0]["final_observation"]["critic_states"], obs[0]["critic_states"]
        )
        instance.set_global_step(5001)
        assert instance.task.common_step_counter == 5001 * 24
    finally:
        instance.close()


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


def test_go2_normalized_mjlab_observations_and_resume(tmp_path):
    policy = Go2Policy(obs_dim=47, critic_obs_dim=74, obs_normalization=True)
    obs = {"states": torch.randn(16, 47) * 3 + 2, "critic_states": torch.randn(16, 74)}
    policy.prepare_ppo_update(obs)
    torch.testing.assert_close(
        policy.actor_normalizer.mean, obs["states"].mean(0, keepdim=True)
    )
    torch.testing.assert_close(
        policy.actor_normalizer.variance,
        obs["states"].var(0, unbiased=False, keepdim=True),
    )
    assert policy.actor_normalizer.count == 16
    actions, rollout = policy.predict_action_batch(obs)
    output = policy(forward_inputs=rollout["forward_inputs"])
    torch.testing.assert_close(output["logprobs"], rollout["prev_logprobs"])
    torch.testing.assert_close(output["values"], rollout["prev_values"])
    assert output["kl_divergence"].abs().max() < 2e-4
    path = tmp_path / "policy.pt"
    torch.save(policy.state_dict(), path)
    restored = Go2Policy(obs_dim=47, critic_obs_dim=74, obs_normalization=True)
    restored.load_state_dict(torch.load(path, weights_only=True))
    expected, _ = policy.predict_action_batch(obs, mode="eval")
    actual, _ = restored.predict_action_batch(obs, mode="eval")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actions.shape == (16, 1, 12)


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


def test_go2_analytic_kl_preserves_old_distribution():
    policy = Go2Policy()
    obs = {"states": torch.randn(4, 45), "critic_states": torch.randn(4, 60)}
    _, sampled = policy.predict_action_batch(obs)
    inputs = sampled["forward_inputs"]
    before = policy(forward_inputs=inputs)["kl_divergence"]
    torch.testing.assert_close(before, torch.zeros(4), rtol=0, atol=0)
    with torch.no_grad():
        policy.std.add_(0.2)
    torch.testing.assert_close(inputs["old_std"], torch.ones(4, 12))
    expected = torch.distributions.kl_divergence(
        torch.distributions.Normal(inputs["old_mean"], inputs["old_std"]),
        torch.distributions.Normal(inputs["old_mean"], torch.full((4, 12), 1.2)),
    ).sum(-1)
    torch.testing.assert_close(policy(forward_inputs=inputs)["kl_divergence"], expected)


def test_go2_ascend_launch_is_independent_of_working_directory(tmp_path, monkeypatch):
    import runpy
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    launcher = repo / "examples/embodiment/go2_ascend/launch_training.py"
    build = runpy.run_path(str(launcher))["build_launch"]
    monkeypatch.chdir(tmp_path)
    output, command, metadata = build("from_scratch")
    assert output == repo / "outputs/go2_ascend/from_scratch"
    assert metadata["repo"] == str(repo)
    assert metadata["simulation_threads"] == 160
    assert metadata["component_placement"]["actor,rollout"] == "8-11"
    assert metadata["global_envs"] == 4096
    assert metadata["resume"] is None
    assert str(repo / "examples/embodiment/go2_ascend/run_stage.sh") in command
    assert not (tmp_path / "outputs").exists()


def test_go2_ascend_resume_selects_latest_complete_checkpoint(tmp_path):
    import runpy
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    build = runpy.run_path(
        str(repo / "examples/embodiment/go2_ascend/launch_training.py")
    )["build_launch"]
    checkpoints = tmp_path / "from_scratch/checkpoints"
    for step in (100, 200, 300):
        path = checkpoints / f"global_step_{step}"
        path.mkdir(parents=True)
        if step != 300:
            (path / "environment_curriculum.json").write_text("{}")
            (path / "actor/dcp_checkpoint").mkdir(parents=True)
            (path / "actor/dcp_checkpoint/.metadata").write_text("metadata")
    output, command, metadata = build("from_scratch", tmp_path, resume=True)
    expected = checkpoints / "global_step_200"
    assert output == tmp_path
    assert metadata["resume"] == str(expected)
    assert f"runner.resume_dir={expected}" in command
    # An incomplete newer save must not displace the latest complete checkpoint.
    (checkpoints / "global_step_50000").mkdir()
    (checkpoints / "global_step_50000/environment_curriculum.json").write_text("{}")
    (checkpoints / "global_step_50000/actor/dcp_checkpoint").mkdir(parents=True)
    (checkpoints / "global_step_50000/actor/dcp_checkpoint/.metadata").write_text(
        "metadata"
    )
    with pytest.raises(RuntimeError, match="already reached"):
        build("from_scratch", tmp_path, resume=True)


def test_go2_native_launch_uses_standard_entry_and_dcp_resume(tmp_path, monkeypatch):
    import runpy
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    build = runpy.run_path(
        str(repo / "examples/embodiment/go2_ascend/launch_training.py")
    )["build_launch"]
    monkeypatch.chdir(tmp_path)
    output, command, metadata = build("go2_mujoco_ppo_ascend", tmp_path, native=True)
    assert output == tmp_path
    assert command[1].endswith("run_native.sh")
    assert metadata["algorithm_loss"] == "actor_critic"
    assert metadata["model_type"] == "go2_policy"
    assert metadata["extension_sha256"] is None
    assert metadata["resume"] is None
    checkpoint = tmp_path / "from_scratch/checkpoints/global_step_1000"
    (checkpoint / "actor/dcp_checkpoint").mkdir(parents=True)
    (checkpoint / "actor/dcp_checkpoint/.metadata").write_text("metadata")
    # Native checkpoints do not contain the compatibility worker's curriculum file.
    _, command, metadata = build(
        "go2_mujoco_ppo_ascend", tmp_path, resume=True, native=True
    )
    assert metadata["resume"] == str(checkpoint)
    assert f"runner.resume_dir={checkpoint}" in command
    with pytest.raises(FileNotFoundError, match="No complete checkpoints"):
        build("from_scratch", tmp_path, resume=True)


def test_go2_native_eval_cli_resolves_default_config(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    script = repo / "examples/embodiment/go2_ascend/run_native.sh"
    result = subprocess.run(
        ["bash", str(script), "--eval", "--cfg", "job", "--resolve"],
        cwd=tmp_path,
        env={**os.environ, "GO2_PYTHON": sys.executable},
        capture_output=True,
        text=True,
        check=True,
    )
    cfg = OmegaConf.create(result.stdout)
    assert cfg.runner.only_eval
    assert cfg.rollout.model.model_type == "go2_policy"
    assert cfg.algorithm.loss_type == "actor_critic"
