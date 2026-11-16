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

"""Episode metrics with the pinned benchmark's full-episode semantics."""

import numpy as np

from .benchmark_protocol import TASKS


class Tracker:
    def __init__(self, task, num_envs, step_dt):
        self.task = task
        self.cfg = TASKS[task]
        self.num_envs = num_envs
        self.step_dt = step_dt
        self.reset_buffers()

    def reset_buffers(self, env_ids=None):
        if env_ids is not None:
            self.count[env_ids] = 0
            self.good[env_ids] = 0
            self.lin[env_ids] = 0
            self.ang[env_ids] = 0
            self.failed[env_ids] = False
            return
        self.count = np.zeros(self.num_envs, np.float32)
        self.good = np.zeros(self.num_envs, np.float32)
        self.lin = np.zeros(self.num_envs, np.float32)
        self.ang = np.zeros(self.num_envs, np.float32)
        self.failed = np.zeros(self.num_envs, bool)

    def update(self, lin_error, ang_error, terminated, truncated):
        lin = np.asarray(lin_error, np.float32)
        ang = np.asarray(ang_error, np.float32)
        term = np.asarray(terminated, bool)
        trunc = np.asarray(truncated, bool)
        if any(x.shape != (self.num_envs,) for x in (lin, ang, term, trunc)):
            raise ValueError("Expected one scalar per vector environment")
        if (
            not np.isfinite(lin).all()
            or not np.isfinite(ang).all()
            or (lin < 0).any()
            or (ang < 0).any()
        ):
            raise ValueError("Tracking errors must be finite nonnegative scalars")
        self.count += 1
        self.good += (
            (~term) & (lin <= self.cfg["eps_lin"]) & (ang <= self.cfg["eps_ang"])
        ).astype(np.float32)
        self.lin += lin
        self.ang += ang
        self.failed |= term
        rows = []
        for i in np.flatnonzero(term | trunc):
            steps = float(self.count[i])
            ratio = float(self.good[i]) / steps
            failure = bool(self.failed[i])
            rows.append(
                {
                    "episode_success": ratio >= self.cfg["min_success_ratio"]
                    and not failure,
                    "success_step_ratio": ratio,
                    "mean_lin_vel_error_xy": float(self.lin[i]) / steps,
                    "mean_ang_vel_error_z": float(self.ang[i]) / steps,
                    "alive_time_s": steps * self.step_dt,
                    "termination_reason": "failure" if failure else "time_out",
                    "task_id": list(TASKS).index(self.task),
                    "family_id": self.cfg["family_id"],
                }
            )
            self.count[i] = self.good[i] = self.lin[i] = self.ang[i] = 0
            self.failed[i] = False
        return rows


def summarize(episodes):
    if not episodes:
        raise ValueError(
            "No completed episodes: missing results must not be represented as zero success"
        )
    count = len(episodes)
    return {
        "num_episodes": count,
        "success_rate": sum(bool(e["episode_success"]) for e in episodes) / count,
        "failure_rate": sum(e["termination_reason"] == "failure" for e in episodes)
        / count,
        "mean_alive_time_s": sum(e["alive_time_s"] for e in episodes) / count,
        "mean_success_step_ratio": sum(e["success_step_ratio"] for e in episodes)
        / count,
        "mean_lin_vel_error": sum(e["mean_lin_vel_error_xy"] for e in episodes) / count,
        "mean_ang_vel_error": sum(e["mean_ang_vel_error_z"] for e in episodes) / count,
    }
