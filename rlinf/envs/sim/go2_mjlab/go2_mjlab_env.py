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

"""Official Unitree Go2 Flat task exposed through RLinf's chunk interface."""

from typing import Any

import gymnasium as gym
import numpy as np
import torch


class Go2MjlabEnv(gym.Env):
    """Run the pinned upstream Mjlab task on CPU physics workers."""

    def __init__(
        self,
        cfg: Any,
        num_envs: int = 1,
        seed_offset: int = 0,
        total_num_processes: int = 1,
        worker_info: Any = None,
        record_metrics: bool = True,
    ) -> None:
        import src.tasks  # noqa: F401
        from mjlab.tasks.registry import load_env_cfg

        from .backend import TerminalObservationEnv

        super().__init__()
        if not cfg.auto_reset or cfg.ignore_terminations or cfg.group_size != 1:
            raise ValueError(
                "Mjlab Go2 requires auto_reset, terminations and group_size=1."
            )
        if cfg.video_cfg.save_video:
            raise ValueError("Mjlab CPU adapter requires video_cfg.save_video=false.")
        self.benchmark_tracker = None
        self.cfg = cfg
        self.num_envs = num_envs
        self.auto_reset = True
        self.group_size = 1
        self.num_group = num_envs
        self.is_start = True
        torch.set_num_threads(int(cfg.threads))
        benchmark_task = cfg.get("benchmark_task")
        self.shared_policy_observation = cfg.get("benchmark_recipe") in (
            "b2_reference_v2",
            "specialist_reference_v1",
        )
        if benchmark_task is None:
            task_cfg = load_env_cfg("Unitree-Go2-Flat")
            actor_dim, critic_dim = 47, 74
        else:
            from .benchmark import make_benchmark_cfg

            task_cfg = make_benchmark_cfg(
                benchmark_task,
                evaluation=bool(cfg.is_eval),
                recipe=cfg.get("benchmark_recipe", "mjlab_v1"),
            )
            actor_dim, critic_dim = 235, 235
            from .benchmark_metrics import Tracker

            self.benchmark_tracker = Tracker(
                benchmark_task,
                num_envs,
                task_cfg.sim.mujoco.timestep * task_cfg.decimation,
            )
        task_cfg.scene.num_envs = num_envs
        task_cfg.seed = int(cfg.seed) + seed_offset
        task_cfg.episode_length_s = int(cfg.max_episode_steps) * (
            task_cfg.sim.mujoco.timestep * task_cfg.decimation
        )
        self.task = TerminalObservationEnv(task_cfg, device="cpu")
        physics_backend = cfg.get("physics_backend", "mjlab")
        if physics_backend == "rlinf_mujoco_warp":
            from .physics import RLinfWarpSimulation

            self.task.sim = RLinfWarpSimulation(self.task.sim)
        elif physics_backend != "mjlab":
            self.task.close()
            raise ValueError(
                "Go2 physics_backend must be mjlab or rlinf_mujoco_warp, "
                f"got {physics_backend!r}."
            )
        self.observation_space = gym.spaces.Dict(
            {
                "states": gym.spaces.Box(-np.inf, np.inf, (actor_dim,), np.float32),
                "critic_states": gym.spaces.Box(
                    -np.inf, np.inf, (critic_dim,), np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Box(-np.inf, np.inf, (12,), np.float32)
        self.returns = torch.zeros(num_envs)
        self.lengths = torch.zeros(num_envs, dtype=torch.int64)
        self._initialized_episode_lengths = False
        self.posture_reward_schedule = cfg.get("posture_reward_schedule")
        if self.posture_reward_schedule is not None:
            schedule = self.posture_reward_schedule
            if (
                not self.shared_policy_observation
                or schedule.interval < 1
                or schedule.total_updates < 1
                or not 0 <= schedule.start_fraction < schedule.end_fraction <= 1
                or round(schedule.start_fraction * schedule.total_updates)
                >= round(schedule.end_fraction * schedule.total_updates)
            ):
                raise ValueError(
                    "B2 posture scheduling requires the reference recipe, positive "
                    "interval/total_updates and distinct start/end updates."
                )
            self.set_global_step(0)

    def _wrap_obs(self, obs: dict) -> dict:
        critic = obs["actor"] if self.shared_policy_observation else obs["critic"]
        return {"states": obs["actor"].clone(), "critic_states": critic.clone()}

    @property
    def elapsed_steps(self) -> torch.Tensor:
        """Return episode lengths for each environment."""
        return self.task.episode_length_buf

    @property
    def info_logging_keys(self) -> list[str]:
        """Return extra environment logging keys."""
        return []

    def update_reset_state_ids(self) -> None:
        """Use upstream stochastic reset behavior."""

    def set_global_step(self, global_step: int) -> None:
        """Set the curriculum clock and posture weights for the next rollout."""
        steps = global_step * int(self.cfg.max_steps_per_rollout_epoch)
        self.task.common_step_counter = steps
        self.task._sim_step_counter = steps * self.task.cfg.decimation
        if self.posture_reward_schedule is not None:
            from .b2_reference import apply_posture_reward_schedule

            apply_posture_reward_schedule(
                self.task, self.posture_reward_schedule, global_step
            )

    def reset(self, env_idx: np.ndarray | None = None) -> tuple[dict, dict]:
        """Reset selected environments and return the complete observation batch."""
        ids = None if env_idx is None else torch.as_tensor(env_idx, dtype=torch.int64)
        self.task.capture_final_observation = False
        obs, info = self.task.reset(env_ids=ids)
        if (
            not self._initialized_episode_lengths
            and ids is None
            and self.cfg.get("randomize_initial_episode_length", False)
        ):
            self.task.episode_length_buf.copy_(
                torch.randint_like(
                    self.task.episode_length_buf, high=self.task.max_episode_length
                )
            )
            self._initialized_episode_lengths = True
        self.returns[ids if ids is not None else slice(None)] = 0
        self.lengths[ids if ids is not None else slice(None)] = 0
        if self.benchmark_tracker is not None:
            self.benchmark_tracker.reset_buffers(env_idx)
        self.is_start = True
        return self._wrap_obs(obs), {}

    def step(self, actions: torch.Tensor | np.ndarray) -> tuple:
        """Step the official task and preserve true terminal observations."""
        actions = torch.as_tensor(actions, dtype=torch.float32, device="cpu")
        self.task.final_observation = None
        self.task.capture_final_observation = True
        obs, reward, terminated, truncated, _ = self.task.step(actions)
        self.task.capture_final_observation = False
        terminated, truncated = terminated.clone(), truncated.clone()
        done = terminated | truncated
        self.returns += reward
        self.lengths += 1
        episode = {
            "return": torch.where(done, self.returns, 0).clone(),
            "episode_len": torch.where(done, self.lengths, 0).clone(),
        }
        info = {"episode": episode}
        if self.benchmark_tracker is not None:
            robot = self.task.scene["robot"]
            command = self.task.command_manager.get_command("twist")
            velocity = (
                robot.data.root_com_lin_vel_b
                if self.shared_policy_observation
                else robot.data.root_link_lin_vel_b
            )
            lin_error = torch.linalg.vector_norm(
                command[:, :2] - velocity[:, :2], dim=1
            )
            ang_error = torch.abs(command[:, 2] - robot.data.root_link_ang_vel_b[:, 2])
            info["benchmark_episodes"] = self.benchmark_tracker.update(
                lin_error.numpy(),
                ang_error.numpy(),
                terminated.numpy(),
                truncated.numpy(),
            )
        if done.any():
            if self.shared_policy_observation:
                self.task.final_observation["critic_states"] = (
                    self.task.final_observation["states"].clone()
                )
            info.update(
                final_observation=self.task.final_observation,
                final_info={"episode": episode},
                _final_observation=done,
                _final_info=done,
            )
        self.returns[done] = 0
        self.lengths[done] = 0
        self.is_start = False
        return self._wrap_obs(obs), reward.clone(), terminated, truncated, info

    def chunk_step(self, chunk_actions: torch.Tensor | np.ndarray) -> tuple:
        """Accept one 12-joint action per environment."""
        if chunk_actions.shape != (self.num_envs, 1, 12):
            raise ValueError(
                f"Expected [{self.num_envs}, 1, 12] actions, got {chunk_actions.shape}."
            )
        obs, reward, terminated, truncated, info = self.step(chunk_actions[:, 0])
        return [obs], reward[:, None], terminated[:, None], truncated[:, None], [info]

    def close(self) -> None:
        """Release the upstream environment."""
        self.task.close()
