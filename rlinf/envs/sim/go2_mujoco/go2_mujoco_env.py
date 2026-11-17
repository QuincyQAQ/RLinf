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

"""CPU MuJoCo Go2 environment for RLinf single-action PPO rollouts."""

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch


class Go2MujocoEnv(gym.Env):
    """Expose the validated Go2 task through EnvWorker's batched contract."""

    def __init__(
        self,
        cfg: Any,
        num_envs: int = 1,
        seed_offset: int = 0,
        total_num_processes: int = 1,
        worker_info: Any = None,
        record_metrics: bool = True,
    ) -> None:
        from .task import Go2Env

        super().__init__()
        if not cfg.auto_reset or cfg.ignore_terminations:
            raise ValueError(
                "Go2 requires auto_reset=True and ignore_terminations=False."
            )
        if cfg.get("use_fixed_reset_state_ids", False) or cfg.get(
            "use_ordered_reset_state_ids", False
        ):
            raise ValueError("Go2 uses seeded stochastic resets, not reset-state IDs.")
        if total_num_processes != 1:
            raise ValueError(
                "Go2's initial integration supports one environment process; global curriculum is not yet distributed."
            )
        if num_envs < 1 or int(cfg.threads) < 1 or int(cfg.max_episode_steps) < 1:
            raise ValueError(
                "Go2 requires positive num_envs, threads and max_episode_steps."
            )
        if int(cfg.group_size) != 1:
            raise ValueError("Go2 PPO requires group_size=1.")
        self.cfg = cfg
        self.num_envs = num_envs
        self.seed = int(cfg.seed) + seed_offset
        self.is_start = True
        self.auto_reset = True
        self.group_size = 1
        self.num_group = num_envs
        self._closed = False
        xml = cfg.get("xml_path") or str(Path(__file__).with_name("go2_training.xml"))
        self.task = Go2Env(
            xml,
            num_envs,
            int(cfg.threads),
            self.seed,
            randomize=bool(cfg.randomize),
            noise=bool(cfg.noise),
            fixed_command=cfg.get("fixed_command"),
            max_episode_steps=int(cfg.max_episode_steps),
        )
        self.observation_space = gym.spaces.Dict(
            {
                "states": gym.spaces.Box(
                    -np.inf, np.inf, shape=(45,), dtype=np.float32
                ),
                "critic_states": gym.spaces.Box(
                    -np.inf, np.inf, shape=(60,), dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(12,), dtype=np.float32
        )

    @staticmethod
    def _wrap_obs(obs: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        return {
            "states": torch.from_numpy(obs["policy"].copy()),
            "critic_states": torch.from_numpy(obs["critic"].copy()),
        }

    @property
    def elapsed_steps(self) -> torch.Tensor:
        """Return the current episode lengths."""
        return torch.from_numpy(self.task.ages.copy())

    @property
    def info_logging_keys(self) -> list[str]:
        """Declare extra environment metric keys."""
        return []

    def update_reset_state_ids(self) -> None:
        """Keep seeded stochastic resets independent of rollout epoch boundaries."""

    def reset(self, env_idx: np.ndarray | None = None) -> tuple[dict, dict]:
        """Reset all or selected environments and return a full batch."""
        self.is_start = True
        return self._wrap_obs(self.task.reset(env_idx)), {}

    def step(self, actions: torch.Tensor | np.ndarray) -> tuple:
        """Execute one control step, preserving observations before auto-reset."""
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().float().cpu().numpy()
        obs, reward, done, info = self.task.step(actions)
        mask = torch.from_numpy(done.copy())
        episode = {
            "return": torch.from_numpy(
                np.where(done, info["episode_return"], 0).astype(np.float32)
            ),
            "episode_len": torch.from_numpy(np.where(done, info["episode_length"], 0)),
        }
        infos = {"episode": episode}
        if done.any():
            infos.update(
                final_observation=self._wrap_obs(info["final_observation"]),
                final_info={"episode": episode},
                _final_observation=mask,
                _final_info=mask,
            )
        self.is_start = False
        return (
            self._wrap_obs(obs),
            torch.from_numpy(reward),
            torch.from_numpy(info["terminated"]),
            torch.from_numpy(info["time_outs"]),
            infos,
        )

    def chunk_step(self, chunk_actions: torch.Tensor | np.ndarray) -> tuple:
        """Accept exactly one action per environment: [N, 1, 12]."""
        if chunk_actions.shape != (self.num_envs, 1, 12):
            raise ValueError(
                f"Expected actions [{self.num_envs}, 1, 12], got {chunk_actions.shape}."
            )
        obs, reward, terminated, truncated, info = self.step(chunk_actions[:, 0])
        return [obs], reward[:, None], terminated[:, None], truncated[:, None], [info]

    def close(self) -> None:
        """Release each native simulation model exactly once."""
        if not self._closed:
            del self.task
            self._closed = True
