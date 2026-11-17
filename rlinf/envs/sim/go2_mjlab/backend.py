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

"""Mjlab reset hook preserving observations required for RLinf bootstrapping."""

import torch
from mjlab.envs import ManagerBasedRlEnv


class TerminalObservationEnv(ManagerBasedRlEnv):
    """Capture the Go2 task's single-frame observations before automatic reset."""

    capture_final_observation = False
    final_observation = None

    def _reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
        if self.capture_final_observation:
            self.sim.forward()
            self.sim.sense()
            # Extra terminal observations must not consume reset randomization RNG.
            with torch.random.fork_rng(devices=[]):
                self.final_observation = {
                    "states": self.observation_manager.compute_group(
                        "actor", update_history=True
                    ).clone(),
                    "critic_states": self.observation_manager.compute_group(
                        "critic", update_history=True
                    ).clone(),
                }
        super()._reset_idx(env_ids)
