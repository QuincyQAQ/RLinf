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

"""Pinned cross-task benchmark contract, without simulator imports."""

TASKS = {
    "A1_forward": {
        "gym_id": "MTL-Velocity-Flat-Unitree-Go2-A1-Forward-v0",
        "family_id": 0,
        "eps_lin": 0.20,
        "eps_ang": 0.40,
        "min_success_ratio": 0.85,
    },
    "A2_omni": {
        "gym_id": "MTL-Velocity-Flat-Unitree-Go2-A2-Omni-v0",
        "family_id": 0,
        "eps_lin": 0.25,
        "eps_ang": 0.50,
        "min_success_ratio": 0.80,
    },
    "B1_rough": {
        "gym_id": "MTL-Velocity-Rough-Unitree-Go2-B1-RoughWalk-v0",
        "family_id": 1,
        "eps_lin": 0.35,
        "eps_ang": 0.60,
        "min_success_ratio": 0.75,
    },
    "B2_stairs": {
        "gym_id": "MTL-Velocity-Rough-Unitree-Go2-B2-StairClimb-v0",
        "family_id": 1,
        "eps_lin": 0.40,
        "eps_ang": 0.70,
        "min_success_ratio": 0.70,
    },
    "C2_gap": {
        "gym_id": "MTL-Custom-Gap-Unitree-Go2-C2-v0",
        "family_id": 2,
        "eps_lin": 0.45,
        "eps_ang": 0.75,
        "min_success_ratio": 0.65,
    },
}
UPSTREAM_REVISION = "743ac04d9543b04b7854f1100bfa0cc69c993106"
UPSTREAM_URL = "https://github.com/viktoriageorrg04/bsc-thesis-multi-task-RL-env"
REFERENCE_OBS_DIM = 235
REFERENCE_ACTION_DIM = 12
REQUIRED_METRICS = (
    "success_rate",
    "failure_rate",
    "mean_alive_time_s",
    "mean_success_step_ratio",
    "mean_lin_vel_error",
    "mean_ang_vel_error",
    "num_episodes",
)


def validate_policy_interface(observation_dim, action_dim):
    if observation_dim != REFERENCE_OBS_DIM or action_dim != REFERENCE_ACTION_DIM:
        raise ValueError(
            f"Original benchmark requires {REFERENCE_OBS_DIM} observations and "
            f"{REFERENCE_ACTION_DIM} actions; received {observation_dim}/{action_dim}. "
            "Do not pad, truncate, or relabel an incompatible checkpoint."
        )
