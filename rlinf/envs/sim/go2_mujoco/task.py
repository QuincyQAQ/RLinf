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

"""MuJoCo adaptation of Unitree's Go2 task, with explicit engine differences in README."""

import numpy as np

from . import batched_go2

DT = 0.02
DEFAULT = np.array([0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5])
JOINT_NAMES = [
    f"{leg}_{joint}_joint"
    for joint in ["hip", "thigh", "calf"]
    for leg in ["FL", "FR", "RL", "RR"]
]
WEIGHTS = {
    "track_lin_vel_xy": 1.5,
    "track_ang_vel_z": 0.75,
    "base_linear_velocity": -2.0,
    "base_angular_velocity": -0.05,
    "joint_vel": -0.001,
    "joint_acc": -2.5e-7,
    "joint_torques": -2e-4,
    "action_rate": -0.1,
    "dof_pos_limits": -10.0,
    "energy": -2e-5,
    "flat_orientation_l2": -2.5,
    "joint_pos": -0.7,
    "feet_air_time": 0.1,
    "air_time_variance": -1.0,
    "feet_slide": -0.1,
    "undesired_contacts": -1.0,
}


def observations(state, command, actions, rng=None):
    q = state[:, 16:28] - DEFAULT
    v = state[:, 28:40]
    actor = [
        state[:, 10:13].copy(),
        state[:, 13:16].copy(),
        command.copy(),
        q.copy(),
        v.copy(),
        actions.copy(),
    ]
    if rng is not None:
        for idx, amplitude in [(0, 0.2), (1, 0.05), (3, 0.01), (4, 1.5)]:
            actor[idx] += rng.uniform(-amplitude, amplitude, actor[idx].shape)
    scales = [0.2, 1.0, 1.0, 1.0, 0.05, 1.0]
    actor = np.concatenate(
        [np.clip(x, -100, 100) * scale for x, scale in zip(actor, scales)], axis=1
    )
    critic = np.concatenate(
        [
            np.clip(state[:, 7:10], -100, 100),
            np.clip(state[:, 10:13], -100, 100) * 0.2,
            np.clip(state[:, 13:16], -100, 100),
            np.clip(command, -100, 100),
            np.clip(q, -100, 100),
            np.clip(v, -100, 100) * 0.05,
            np.clip(state[:, 40:52], -100, 100) * 0.01,
            np.clip(actions, -100, 100),
        ],
        axis=1,
    )
    return {"policy": actor.astype(np.float32), "critic": critic.astype(np.float32)}


def reward_terms(state, command, actions, previous_actions, limits):
    v, w = state[:, 7:10], state[:, 10:13]
    q, dq, tau, acc = state[:, 16:28], state[:, 28:40], state[:, 40:52], state[:, 52:64]
    contact = state[:, 64:68] > 1
    air, stance, first = state[:, 82:86], state[:, 86:90], state[:, 90:94]
    pos_penalty = np.linalg.norm(q - DEFAULT, axis=1)
    pos_penalty *= np.where(
        (np.linalg.norm(command, axis=1) > 0)
        | (np.linalg.norm(v[:, :2], axis=1) > 0.3),
        1.0,
        5.0,
    )
    return {
        "track_lin_vel_xy": np.exp(-np.square(command[:, :2] - v[:, :2]).sum(1) / 0.25),
        "track_ang_vel_z": np.exp(-np.square(command[:, 2] - w[:, 2]) / 0.25),
        "base_linear_velocity": np.square(v[:, 2]),
        "base_angular_velocity": np.square(w[:, :2]).sum(1),
        "joint_vel": np.square(dq).sum(1),
        "joint_acc": np.square(acc).sum(1),
        "joint_torques": np.square(tau).sum(1),
        "action_rate": np.square(actions - previous_actions).sum(1),
        "dof_pos_limits": (
            np.maximum(limits[:, 0] - q, 0) + np.maximum(q - limits[:, 1], 0)
        ).sum(1),
        "energy": np.abs(dq * tau).sum(1),
        "flat_orientation_l2": np.square(state[:, 13:15]).sum(1),
        "joint_pos": pos_penalty,
        "feet_air_time": ((air - 0.5) * first).sum(1)
        * (np.linalg.norm(command[:, :2], axis=1) > 0.1),
        "air_time_variance": np.minimum(air, 0.5).var(1, ddof=1)
        + np.minimum(stance, 0.5).var(1, ddof=1),
        "feet_slide": (
            np.linalg.norm(state[:, 68:80].reshape(-1, 4, 3)[:, :, :2], axis=2)
            * contact
        ).sum(1),
        "undesired_contacts": state[:, 80],
    }


class Go2Env:
    def __init__(
        self,
        xml,
        num_envs=4096,
        threads=32,
        seed=42,
        randomize=True,
        noise=True,
        fixed_command=None,
        curriculum_callback=None,
        max_episode_steps=1000,
    ):
        self.max_episode_steps = max_episode_steps
        self.n = num_envs
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        self.randomize = randomize
        self.curriculum_callback = curriculum_callback
        self.fixed_command = (
            None if fixed_command is None else np.asarray(fixed_command)
        )
        self.sim = batched_go2.Batch(xml, num_envs, threads, seed, randomize)
        self.limits = self.sim.limits()
        self.ids = np.arange(num_envs, dtype=np.int32)
        self.state = self.sim.reset(self.ids)
        self.actions = np.zeros((num_envs, 12))
        self.previous_actions = self.actions.copy()
        self.commands = np.zeros((num_envs, 3))
        self.command_timer = np.zeros(num_envs, dtype=np.int64)
        self.ages = np.zeros(num_envs, dtype=np.int64)
        self.push_timer = self.rng.integers(250, 501, num_envs)
        self.command_x = 0.1
        self.command_y = 0.1
        self.common_step = 0
        self.sums = {k: np.zeros(num_envs) for k in WEIGHTS}
        self.episode_returns = np.zeros(num_envs)
        self._resample(self.ids)

    def _resample(self, ids):
        count = len(ids)
        if self.fixed_command is not None:
            self.commands[ids] = self.fixed_command
        else:
            self.commands[ids] = self.rng.uniform(
                [-self.command_x, -self.command_y, -1.0],
                [self.command_x, self.command_y, 1.0],
                (count, 3),
            )
            self.commands[ids[self.rng.uniform(size=count) < 0.1]] = 0
        self.command_timer[ids] = 500

    def reset(self, ids=None):
        """Reset selected environments and their episode bookkeeping."""
        ids = self.ids if ids is None else np.asarray(ids, dtype=np.int32)
        self.state[ids] = self.sim.reset(ids)
        self.ages[ids] = 0
        self.actions[ids] = 0
        self.previous_actions[ids] = 0
        self.episode_returns[ids] = 0
        for sums in self.sums.values():
            sums[ids] = 0
        self.push_timer[ids] = self.rng.integers(250, 501, len(ids))
        self._resample(ids)
        return self.get_observations()

    def get_observations(self):
        return observations(
            self.state, self.commands, self.actions, self.rng if self.noise else None
        )

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (self.n, 12) or not np.isfinite(actions).all():
            raise ValueError("Invalid actions")
        self.previous_actions = self.actions.copy()
        self.actions = actions.copy()
        self.state = self.sim.step(actions)
        if not np.isfinite(self.state).all():
            raise FloatingPointError("Nonfinite MuJoCo state")
        terms = reward_terms(
            self.state, self.commands, self.actions, self.previous_actions, self.limits
        )
        reward = sum(WEIGHTS[k] * v for k, v in terms.items()) * DT
        if not np.isfinite(reward).all():
            raise FloatingPointError("Nonfinite reward")
        self.common_step += 1
        self.ages += 1
        self.command_timer -= 1
        self.push_timer -= 1
        timeout = self.ages >= self.max_episode_steps
        bad_orientation = np.arccos(np.clip(-self.state[:, 15], -1, 1)) > 0.8
        base_contact = self.state[:, 81] > 1.0
        failed = bad_orientation | base_contact
        dones = failed | timeout
        for k in WEIGHTS:
            self.sums[k] += terms[k] * WEIGHTS[k] * DT
        self.episode_returns += reward
        ids = np.flatnonzero(dones).astype(np.int32)
        info = {
            "time_outs": timeout.copy(),
            "terminated": failed.copy(),
            "final_observation": observations(self.state, self.commands, self.actions)
            if len(ids)
            else None,
            "episode_return": self.episode_returns.copy(),
            "episode_length": self.ages.copy(),
            "termination_counts": {
                "time_out": int(timeout.sum()),
                "bad_orientation": int(bad_orientation.sum()),
                "base_contact": int(base_contact.sum()),
            },
            "episode_returns": self.episode_returns[ids].tolist(),
            "episode_lengths_s": (self.ages[ids] * DT).tolist(),
            "mean_vx": float(self.state[:, 7].mean()),
            "mean_error_xy": float(
                np.linalg.norm(self.state[:, 7:9] - self.commands[:, :2], axis=1).mean()
            ),
            "reward_terms": {
                k: float((v * WEIGHTS[k]).mean()) for k, v in terms.items()
            },
            "lin_vel_cmd_level": self.command_x,
        }
        # All distributed ranks enter the global curriculum decision even if one shard has no reset.
        if self.fixed_command is None and self.common_step % 1000 == 0:
            if self.curriculum_callback is not None:
                promote = self.curriculum_callback(
                    float(self.sums["track_lin_vel_xy"][ids].sum()),
                    len(ids),
                    self.common_step,
                )
            else:
                promote = (
                    len(ids) > 0
                    and self.sums["track_lin_vel_xy"][ids].mean() / 20.0 > 1.5 * 0.8
                )
            if promote:
                self.command_x = min(1.0, self.command_x + 0.1)
                self.command_y = min(0.4, self.command_y + 0.1)
        if len(ids):
            self.state[ids] = self.sim.reset(ids)
            self.ages[ids] = 0
            self.actions[ids] = 0
            self.previous_actions[ids] = 0
            self.episode_returns[ids] = 0
            for s in self.sums.values():
                s[ids] = 0
            self._resample(ids)
        resample = np.flatnonzero(self.command_timer <= 0)
        if len(resample):
            self._resample(resample)
        if self.randomize:
            push_ids = np.flatnonzero(self.push_timer <= 0)
            if len(push_ids):
                pushes = np.zeros((self.n, 3))
                pushes[push_ids, :2] = self.rng.uniform(-0.5, 0.5, (len(push_ids), 2))
                pushes[push_ids, 2] = 1
                self.sim.push(pushes)
                # Event manager runs before observations in Isaac Lab: expose the new base velocity now.
                q = self.state[push_ids, 3:7]
                w, x, y, z = q.T
                rotation = np.stack(
                    [
                        1 - 2 * (y * y + z * z),
                        2 * (x * y - z * w),
                        2 * (x * z + y * w),
                        2 * (x * y + z * w),
                        1 - 2 * (x * x + z * z),
                        2 * (y * z - x * w),
                        2 * (x * z - y * w),
                        2 * (y * z + x * w),
                        1 - 2 * (x * x + y * y),
                    ],
                    axis=1,
                ).reshape(-1, 3, 3)
                old_world = np.einsum(
                    "nij,nj->ni", rotation, self.state[push_ids, 7:10]
                )
                new_world = old_world.copy()
                new_world[:, :2] = pushes[push_ids, :2]
                self.state[push_ids, 7:10] = np.einsum(
                    "nji,nj->ni", rotation, new_world
                )
                foot_velocity = self.state[push_ids, 68:80].reshape(-1, 4, 3)
                self.state[push_ids, 68:80] = (
                    foot_velocity + (new_world - old_world)[:, None, :]
                ).reshape(-1, 12)
                self.push_timer[push_ids] = self.rng.integers(250, 501, len(push_ids))
        return self.get_observations(), reward.astype(np.float32), dones, info
