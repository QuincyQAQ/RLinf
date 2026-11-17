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

"""Box-based square gap for the Mjlab cross-task benchmark variant."""

from dataclasses import dataclass

import mujoco
import numpy as np
from mjlab.terrains import SubTerrainCfg
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput


@dataclass(kw_only=True)
class SquareGapTerrainCfg(SubTerrainCfg):
    """Central 2m island and a difficulty-scaled surrounding gap."""

    gap_width_range: tuple[float, float] = (0.0, 0.2)
    platform_width: float = 2.0
    depth: float = 1.0

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del rng
        gap = self.gap_width_range[0] + difficulty * (
            self.gap_width_range[1] - self.gap_width_range[0]
        )
        cx, cy = self.size[0] / 2, self.size[1] / 2
        half = self.platform_width / 2
        outer = half + gap
        body = spec.body("terrain")
        geometries = []

        def box(x0, x1, y0, y1, top):
            geom = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=((x1 - x0) / 2, (y1 - y0) / 2, 0.05),
                pos=((x0 + x1) / 2, (y0 + y1) / 2, top - 0.05),
            )
            geometries.append(TerrainGeometry(geom=geom))

        box(0, self.size[0], 0, self.size[1], -self.depth)
        box(cx - half, cx + half, cy - half, cy + half, 0)
        box(0, cx - outer, 0, self.size[1], 0)
        box(cx + outer, self.size[0], 0, self.size[1], 0)
        box(cx - outer, cx + outer, 0, cy - outer, 0)
        box(cx - outer, cx + outer, cy + outer, self.size[1], 0)
        return TerrainOutput(origin=np.array([cx, cy, 0.0]), geometries=geometries)
