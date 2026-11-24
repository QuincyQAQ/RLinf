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

"""Register the Go2 example through existing RLinf extension points."""


def build_policy(cfg, torch_dtype):
    """Construct the checkpoint-compatible policy."""
    from .policy import Go2Policy

    return Go2Policy(
        initial_checkpoint=cfg.get("initial_checkpoint"),
        obs_dim=235,
        critic_obs_dim=235,
        distribution_safety_patch=cfg.distribution_safety_patch,
    )


def register() -> None:
    """Register the model and three Isaac Lab tasks in each worker process."""
    from rlinf.envs.sim.isaaclab import REGISTER_ISAACLAB_ENVS
    from rlinf.models import register_model

    from .environment import Go2IsaacLabEnv
    from .settings import read_manifest

    register_model("go2_policy", build_policy, force=True)
    for recipe in read_manifest()["tasks"].values():
        REGISTER_ISAACLAB_ENVS[recipe["gym_id"]] = Go2IsaacLabEnv
