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


def test_mjlab_b2_posture_schedule_updates_live_rewards_and_restores_on_resume():
    pytest.importorskip("mjlab")
    pytest.importorskip("src.tasks")
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    directory = Path(__file__).resolve().parents[2] / "examples/embodiment/config"
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        config = compose(config_name="go2_mjlab_benchmark_b2_posture_ascend")
    assert config.env.train.total_num_envs == 1024
    assert config.runner.max_steps == config.runner.max_epochs == 2200
    assert config.env.train.seed == config.actor.seed == 42
    assert config.actor.global_batch_size == 1024 * 24 // 4
    assert config.actor.micro_batch_size * 8 == config.actor.global_batch_size
    assert config.runner.resume_dir is None
    assert config.actor.model.initial_checkpoint is None
    instance = get_env_cls("go2_mjlab")(config.env.train, num_envs=2)
    try:
        instance.reset()
        manager = instance.task.reward_manager
        for step, height, orientation in (
            (0, -10.0, -5.0),
            (24, -10.0, -5.0),
            (25, -10.0 + 6.5 * 25 / 770, -5.0 + 3.0 * 25 / 770),
            (750, -10.0 + 6.5 * 750 / 770, -5.0 + 3.0 * 750 / 770),
            (770, -10.0 + 6.5 * 750 / 770, -5.0 + 3.0 * 750 / 770),
            (775, -3.5, -2.0),
            (2199, -3.5, -2.0),
            (25, -10.0 + 6.5 * 25 / 770, -5.0 + 3.0 * 25 / 770),
        ):
            instance.set_global_step(step)
            assert manager.get_term_cfg("base_height").weight == pytest.approx(height)
            assert manager.get_term_cfg("flat_orientation_l2").weight == pytest.approx(
                orientation
            )
            reward = manager.compute(instance.task.step_dt)
            expected = sum(
                term.func(instance.task, **term.params)
                * term.weight
                * instance.task.step_dt
                for term in (
                    manager.get_term_cfg(name) for name in manager.active_terms
                )
            )
            torch.testing.assert_close(reward, expected)
    finally:
        instance.close()


def test_go2_reference_distribution_safety_applies_to_rollout_and_update():
    policy = get_model(
        OmegaConf.create(
            {
                "model_type": "go2_policy",
                "precision": "32",
                "is_lora": False,
                "load_to_device": False,
                "obs_dim": 235,
                "critic_obs_dim": 235,
                "action_dim": 12,
                "num_action_chunks": 1,
                "distribution_safety_patch": True,
            }
        )
    )
    with torch.no_grad():
        policy.std.copy_(torch.tensor([float("nan"), float("inf"), -1.0, 10.0] * 3))
    obs = {"states": torch.zeros(2, 235), "critic_states": torch.zeros(2, 235)}
    action, result = policy.predict_action_batch(obs)
    assert torch.isfinite(action).all()
    torch.testing.assert_close(policy.std, torch.tensor([1.0, 5.0, 1.0e-6, 5.0] * 3))
    recomputed = policy.default_forward(result["forward_inputs"])
    torch.testing.assert_close(recomputed["logprobs"], result["prev_logprobs"])
    assert torch.isfinite(recomputed["entropy"]).all()
    (-recomputed["logprobs"].mean() + recomputed["values"].square().mean()).backward()
    assert all(
        torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None
    )


@pytest.mark.parametrize(
    "task", [None, "A1_forward", "A2_omni", "B1_rough", "B2_stairs", "C2_gap"]
)
def test_mjlab_shared_upstream_world_preserves_task_rollouts(task):
    pytest.importorskip("mjlab")
    pytest.importorskip("src.tasks")
    from rlinf.envs.sim.mujoco_warp.world import MjWarpWorld

    cfg = OmegaConf.create(
        {
            "seed": 42,
            "group_size": 1,
            "threads": 1,
            "auto_reset": True,
            "ignore_terminations": False,
            "max_episode_steps": 2,
            "max_steps_per_rollout_epoch": 24,
            "is_eval": False,
            "video_cfg": {"save_video": False},
            "benchmark_task": task,
            "benchmark_recipe": "b2_reference_v2"
            if task == "B2_stairs"
            else "mjlab_v1",
        }
    )
    legacy = get_env_cls("go2_mjlab")(cfg, num_envs=2)
    shared_cfg = OmegaConf.merge(cfg, {"physics_backend": "rlinf_mujoco_warp"})
    shared = get_env_cls("go2_mjlab")(shared_cfg, num_envs=2)
    try:
        assert isinstance(shared.task.sim.world, MjWarpWorld)
        assert shared.task.sim.world.data is shared.task.sim.simulation.wp_data
        assert shared.task.sim.world.model is shared.task.sim.simulation.wp_model
        torch.manual_seed(123)
        expected, _ = legacy.reset()
        torch.manual_seed(123)
        actual, _ = shared.reset()
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        actions = torch.linspace(-0.2, 0.2, 24).reshape(2, 1, 12)
        for step in range(4):
            torch.manual_seed(1000 + step)
            expected = legacy.chunk_step(actions)
            torch.manual_seed(1000 + step)
            actual = shared.chunk_step(actions)
            for key in actual[0][0]:
                torch.testing.assert_close(
                    actual[0][0][key], expected[0][0][key], rtol=0, atol=0
                )
            for index in (1, 2, 3):
                torch.testing.assert_close(
                    actual[index], expected[index], rtol=0, atol=0
                )
            if actual[3].any():
                for key in actual[4][0]["final_observation"]:
                    torch.testing.assert_close(
                        actual[4][0]["final_observation"][key],
                        expected[4][0]["final_observation"][key],
                        rtol=0,
                        atol=0,
                    )
        world = shared.task.sim.world
        before = world.qpos.clone()
        world.qpos[0, 0] += 0.01
        assert shared.task.sim.data.qpos[0, 0] == world.qpos[0, 0]
        world.qpos.copy_(before)
    finally:
        legacy.close()
        shared.close()


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


@pytest.mark.parametrize(
    "task", ["A1_forward", "A2_omni", "B1_rough", "B2_stairs", "C2_gap"]
)
def test_benchmark_failure_overrides_tracking(task):
    from rlinf.envs.sim.go2_mjlab.benchmark_metrics import Tracker, summarize

    tracker = Tracker(task, 2, 0.02)
    tracker.update([0, 0], [0, 0], [False, False], [False, False])
    rows = tracker.update([0, 0], [0, 0], [True, False], [True, True])
    assert [r["episode_success"] for r in rows] == [False, True]
    assert rows[0]["termination_reason"] == "failure"
    assert summarize(rows)["success_rate"] == 0.5
    assert summarize(rows)["failure_rate"] == 0.5
    assert not tracker.count.any()


def test_benchmark_partial_reset_keeps_other_episode():
    from rlinf.envs.sim.go2_mjlab.benchmark_metrics import Tracker

    tracker = Tracker("A1_forward", 2, 0.02)
    tracker.update([0, 0], [0, 0], [False, False], [False, False])
    tracker.reset_buffers([1])
    assert tracker.count.tolist() == [1, 0]
    rows = tracker.update([0, 0], [0, 0], [False, False], [True, True])
    assert [r["alive_time_s"] for r in rows] == [0.04, 0.02]


def test_benchmark_missing_results_are_not_zero_success():
    from rlinf.envs.sim.go2_mjlab.benchmark_metrics import summarize
    from rlinf.envs.sim.go2_mjlab.benchmark_protocol import validate_policy_interface

    with pytest.raises(ValueError, match="No completed episodes"):
        summarize([])
    with pytest.raises(ValueError, match="235"):
        validate_policy_interface(47, 12)


def _benchmark_example_module(filename):
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[2] / "examples/embodiment/go2_mjlab" / filename
    )
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "task,short",
    [
        ("A1_forward", "a1"),
        ("A2_omni", "a2"),
        ("B1_rough", "b1"),
        ("B2_stairs", "b2"),
        ("C2_gap", "c2"),
    ],
)
def test_rsl_baseline_accepts_native_task_config(task, short):
    module = _benchmark_example_module("rsl_baseline.py")
    cfg = module.compose_task(
        task, 1024, 42, f"go2_mjlab_benchmark_{short}_native_2200_ascend"
    )
    assert cfg.env.train.benchmark_task == task
    assert cfg.env.train.benchmark_recipe == (
        "b2_reference_v2" if short == "b2" else "specialist_reference_v1"
    )
    assert cfg.env.train.physics_backend == "rlinf_mujoco_warp"
    assert cfg.actor.model.obs_normalization is False
    assert cfg.env.train.total_num_envs == 1024
    assert cfg.env.train.seed == cfg.actor.seed == 42


def test_rsl_baseline_rejects_mismatched_task_config():
    module = _benchmark_example_module("rsl_baseline.py")
    with pytest.raises(ValueError, match="expected 'A1_forward'"):
        module.compose_task(
            "A1_forward", 1024, 42, "go2_mjlab_benchmark_a2_native_2200_ascend"
        )


def test_rsl_export_preserves_normalized_actor_and_critic():
    models = pytest.importorskip("rsl_rl.models")
    from tensordict import TensorDict

    exporter = _benchmark_example_module("rsl_baseline.py").canonical_weights
    batch = TensorDict(
        {"states": torch.randn(32, 235), "critic_states": torch.randn(32, 235)},
        batch_size=[32],
    )
    groups = {"actor": ["states"], "critic": ["critic_states"]}
    actor = models.MLPModel(
        batch,
        groups,
        "actor",
        12,
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    )
    critic = models.MLPModel(
        batch,
        groups,
        "critic",
        1,
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    actor.update_normalization(batch)
    critic.update_normalization(batch)
    actor.eval()
    critic.eval()
    policy = Go2Policy(obs_dim=235, critic_obs_dim=235, obs_normalization=False)
    policy.load_state_dict(exporter(actor, critic), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(
            policy.actor(batch["states"]), actor(batch), atol=2e-6, rtol=2e-5
        )
        torch.testing.assert_close(
            policy.critic(batch["critic_states"]), critic(batch), atol=2e-6, rtol=2e-5
        )
    assert not any("normalizer" in key for key in policy.state_dict())


def test_rsl_b2_distribution_bounds_invalid_scalar_std():
    pytest.importorskip("rsl_rl.modules.distribution")
    cls = _benchmark_example_module("rsl_distribution.py").BoundedGaussianDistribution
    distribution = cls(12, init_std=1.0, std_type="scalar")
    with torch.no_grad():
        distribution.std_param[0] = -1.0
        distribution.std_param[1] = float("nan")
        distribution.std_param[2] = 10.0
    distribution.update(torch.zeros(4, 12))
    assert torch.isfinite(distribution.log_prob(torch.zeros(4, 12))).all()
    assert (distribution.std_param >= 1e-6).all()
    assert (distribution.std_param <= 5.0).all()


def test_benchmark_frozen_actor_preserves_policy_normalization():
    frozen = _benchmark_example_module("evaluate_benchmark.py").FrozenActor
    policy = Go2Policy(obs_dim=235, critic_obs_dim=235, obs_normalization=True)
    states = torch.randn(8, 235)
    policy.actor_normalizer.update(states)
    actor = frozen(policy.state_dict()).eval()
    with torch.no_grad():
        expected = policy.actor(policy.actor_normalizer(states))
        torch.testing.assert_close(actor(states), expected)
    with pytest.raises(ValueError, match="235"):
        frozen(Go2Policy(obs_dim=47).state_dict())


@pytest.mark.parametrize(
    "count,saved_seed,error",
    [(16, 42, None), (17, 42, None), (15, 42, "at least"), (16, 43, "eval_seed")],
)
def test_benchmark_watcher_reuses_complete_batches(tmp_path, count, saved_seed, error):
    import hashlib
    import json

    module = _benchmark_example_module("watch_main_task.py")
    checkpoint = tmp_path / "weights.pt"
    config = tmp_path / "config.yaml"
    checkpoint.write_bytes(b"checkpoint identity")
    config.write_bytes(b"evaluation configuration")
    result = {
        "episodes": [{"episode_success": False}] * count,
        "protocol": {
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "training_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "eval_seed": saved_seed,
            "eval_task": "B2_stairs",
            "requested_episodes": 16,
        },
    }
    (tmp_path / "B2_stairs.json").write_text(json.dumps(result))
    if error:
        with pytest.raises(ValueError, match=error):
            module.read_saved_cell(tmp_path, checkpoint, config, 42, 16)
    else:
        assert module.read_saved_cell(tmp_path, checkpoint, config, 42, 16) == result


def test_benchmark_matrix_excludes_smoke_and_preserves_missing(tmp_path):
    import json

    module = _benchmark_example_module("export_benchmark_matrix.py")
    for seed, value in [(0, 0.5), (1, 1.0)]:
        root = tmp_path / f"A1_forward_s{seed}"
        root.mkdir()
        (root / "policy_manifest.json").write_text(
            json.dumps(
                {
                    "train_task": "A1_forward",
                    "train_seed": seed,
                    "checkpoint_sha256": "same",
                }
            )
        )
        (root / "A1_forward.json").write_text(
            json.dumps(
                {
                    "protocol": {
                        "checkpoint_sha256": "same",
                        "evaluation_scope": "smoke",
                    },
                    "summary": {"success_rate": value},
                }
            )
        )
    result = module.aggregate(tmp_path)
    assert result["mean"][0]["A1_forward"] is None
    result = module.aggregate(tmp_path, include_smoke=True)
    assert result["mean"][0]["A1_forward"] == 0.75
    assert result["mean"][0]["A2_omni"] is None
    assert result["std"][0]["A1_forward"] == pytest.approx(0.35355339)
    assert result["std"][0]["mean_over_eval_tasks"] is None


def test_b2_reference_tracking_separates_vertical_and_roll_pitch_motion():
    from types import SimpleNamespace

    from rlinf.envs.sim.go2_mjlab import b2_reference_rewards as rewards

    command = torch.tensor([[0.6, 0.0, 0.0]])
    data = SimpleNamespace(
        root_com_lin_vel_b=torch.tensor([[0.6, 0.0, 3.0]]),
        root_com_ang_vel_b=torch.tensor([[2.0, 1.0, 0.0]]),
    )
    env = SimpleNamespace(
        scene={"robot": SimpleNamespace(data=data)},
        command_manager=SimpleNamespace(get_command=lambda _: command),
    )
    torch.testing.assert_close(
        rewards.track_lin_vel_xy_exp(env, 0.5, "twist"), torch.ones(1)
    )
    torch.testing.assert_close(
        rewards.track_ang_vel_z_exp(env, 0.5, "twist"), torch.ones(1)
    )
    torch.testing.assert_close(rewards.lin_vel_z_l2(env), torch.tensor([9.0]))
    torch.testing.assert_close(rewards.ang_vel_xy_l2(env), torch.tensor([5.0]))


def test_b2_reference_air_time_is_landing_reward_with_command_gate():
    from types import SimpleNamespace

    from rlinf.envs.sim.go2_mjlab.b2_reference_rewards import feet_air_time

    sensor = SimpleNamespace(
        data=SimpleNamespace(last_air_time=torch.tensor([[0.50, 0.20, 0.80, 0.70]])),
        compute_first_contact=lambda _: torch.tensor([[True, True, False, False]]),
    )
    command = torch.tensor([[0.6, 0.0, 0.0]])
    env = SimpleNamespace(
        scene={"feet": sensor},
        step_dt=0.02,
        command_manager=SimpleNamespace(get_command=lambda _: command),
    )
    torch.testing.assert_close(
        feet_air_time(env, "twist", "feet", 0.35), torch.zeros(1), atol=1e-7, rtol=0
    )
    command.zero_()
    sensor.data.last_air_time.fill_(0.8)
    assert feet_air_time(env, "twist", "feet", 0.35).item() == 0.0


def test_b2_reference_counts_net_body_contacts_excluding_feet():
    from types import SimpleNamespace

    from rlinf.envs.sim.go2_mjlab.b2_reference_rewards import undesired_contacts

    force = torch.zeros(1, 12, 3, 3)
    force[0, 0, 0, 0] = 2.0  # FL thigh.
    force[0, 1, 0, 0] = 2.0  # Opposing forces on FL calf must cancel.
    force[0, 2, 0, 0] = -2.0
    force[0, 4, 2, 0] = 0.6  # FR calf's two geoms jointly exceed1N.
    force[0, 5, 2, 0] = 0.6
    env = SimpleNamespace(
        scene={"legs": SimpleNamespace(data=SimpleNamespace(force_history=force))}
    )
    assert undesired_contacts(env, "legs", 1.0).item() == 2


def test_benchmark_frozen_actor_supports_reference_unnormalized_policy():
    frozen = _benchmark_example_module("evaluate_benchmark.py").FrozenActor
    policy = Go2Policy(obs_dim=235, critic_obs_dim=235, obs_normalization=False)
    actor = frozen(policy.state_dict()).eval()
    states = torch.randn(8, 235)
    with torch.no_grad():
        torch.testing.assert_close(actor(states), policy.actor(states), rtol=0, atol=0)
    state = policy.state_dict()
    state["actor.0.weight"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        frozen(state)


@pytest.mark.parametrize(
    "task,tracking,airtime,orientation,height",
    [
        ("A1_forward", 1.5, 0.25, -2.5, None),
        ("A2_omni", 1.5, 0.25, -2.5, -4.0),
        ("B1_rough", 2.0, 0.25, -5.0, -10.0),
        ("B2_stairs", 1.5, 0.04, -2.0, -3.5),
        ("C2_gap", 2.0, 0.08, -5.0, -10.0),
    ],
)
def test_specialist_reference_uses_each_tasks_reward_recipe(
    task, tracking, airtime, orientation, height
):
    pytest.importorskip("mjlab")
    pytest.importorskip("src.tasks")
    from rlinf.envs.sim.go2_mjlab.benchmark import make_benchmark_cfg

    cfg = make_benchmark_cfg(task, recipe="specialist_reference_v1")
    rw = cfg.rewards
    assert rw["track_lin_vel_xy_exp"].weight == tracking
    assert rw["feet_air_time"].weight == airtime
    assert rw["flat_orientation_l2"].weight == orientation
    if height is None:
        assert "base_height" not in rw
        assert rw["stand_still"].weight == -0.15
    else:
        assert rw["base_height"].weight == height
    if task.startswith("A"):
        assert "undesired_contacts" not in rw
        assert cfg.scene.terrain.terrain_generator is None


def test_specialist_gap_height_ignores_missed_rays():
    from types import SimpleNamespace

    from rlinf.envs.sim.go2_mjlab.specialist_reference import safe_base_height

    hits = torch.tensor(
        [
            [[0.0, 0.0, 0.1], [0.0, 0.0, float("inf")]],
            [[0.0, 0.0, float("nan")], [0.0, 0.0, float("inf")]],
        ]
    )
    env = SimpleNamespace(
        scene={
            "scan": SimpleNamespace(data=SimpleNamespace(hit_pos_w=hits)),
            "robot": SimpleNamespace(
                data=SimpleNamespace(
                    root_link_pos_w=torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]])
                )
            ),
        }
    )
    torch.testing.assert_close(
        safe_base_height(env, 0.37, "scan"), torch.tensor([0.0009, 0.0])
    )
