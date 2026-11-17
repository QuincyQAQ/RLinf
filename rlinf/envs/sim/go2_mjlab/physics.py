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

"""Mjlab task simulation using RLinf's shared MuJoCo Warp step implementation."""

from typing import Any

from rlinf.envs.sim.mujoco_warp.world import MjWarpWorld


class RLinfWarpSimulation:
    """Borrow Mjlab physics arrays while retaining its sensors and randomization.

    Mjlab owns the model, data, resets and sensor graphs. RLinf advances the
    same arrays. CUDA graph replay remains owned by Mjlab when enabled.
    """

    def __init__(self, simulation: Any) -> None:
        self.simulation = simulation
        self.world = MjWarpWorld.from_existing(
            simulation.mj_model,
            simulation.wp_model,
            simulation.wp_data,
            simulation.device,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.simulation, name)

    def step(self) -> None:
        """Integrate shared arrays with the task's numerical-state guard."""
        if self.simulation.use_cuda_graph:
            self.simulation.step()
        else:
            with self.simulation.nan_guard.watch(self.simulation.data):
                self.world.physics_step()

    def forward(self) -> None:
        """Refresh shared kinematics without advancing physics time."""
        if self.simulation.use_cuda_graph:
            self.simulation.forward()
        else:
            self.world.forward()

    def expand_model_fields(self, fields: tuple[str, ...]) -> None:
        """Refresh the view when Mjlab replaces randomization arrays."""
        self.simulation.expand_model_fields(fields)
        self.world = MjWarpWorld.from_existing(
            self.simulation.mj_model,
            self.simulation.wp_model,
            self.simulation.wp_data,
            self.simulation.device,
        )
