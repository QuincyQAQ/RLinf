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
        self.cfg = cfg
        self.num_envs = num_envs
        self.auto_reset = True
        self.group_size = 1
        self.num_group = num_envs
        self.is_start = True
        torch.set_num_threads(int(cfg.threads))
        task_cfg = load_env_cfg("Unitree-Go2-Flat")
        task_cfg.scene.num_envs = num_envs
        task_cfg.seed = int(cfg.seed) + seed_offset
        task_cfg.episode_length_s = int(cfg.max_episode_steps) * (
            task_cfg.sim.mujoco.timestep * task_cfg.decimation
        )
        self.task = TerminalObservationEnv(task_cfg, device="cpu")
        self.observation_space = gym.spaces.Dict(
            {
                "states": gym.spaces.Box(-np.inf, np.inf, (47,), np.float32),
                "critic_states": gym.spaces.Box(-np.inf, np.inf, (74,), np.float32),
            }
        )
        self.action_space = gym.spaces.Box(-np.inf, np.inf, (12,), np.float32)
        self.returns = torch.zeros(num_envs)
        self.lengths = torch.zeros(num_envs, dtype=torch.int64)
        self._initialized_episode_lengths = False

    @staticmethod
    def _wrap_obs(obs: dict) -> dict:
        return {"states": obs["actor"].clone(), "critic_states": obs["critic"].clone()}

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
        """Restore the official time-based curriculum after checkpoint resume."""
        steps = global_step * int(self.cfg.max_steps_per_rollout_epoch)
        self.task.common_step_counter = steps
        self.task._sim_step_counter = steps * self.task.cfg.decimation

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
        if done.any():
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
