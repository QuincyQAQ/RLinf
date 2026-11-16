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

"""Five-task Mjlab variant of the pinned Go2 cross-task benchmark."""

import copy
import math

from .benchmark_protocol import TASKS


def make_benchmark_cfg(task: str, *, evaluation: bool = False):
    """Build a fresh 235-observation task without modifying the Flat recipe."""
    from mjlab.sensor import ContactMatch, ContactSensorCfg
    from src.tasks.velocity.config.go2.env_cfgs import unitree_go2_rough_env_cfg

    from .benchmark_terrain import SquareGapTerrainCfg

    if task not in TASKS:
        raise ValueError(f"Expected benchmark task in {tuple(TASKS)}, got {task!r}")
    cfg = unitree_go2_rough_env_cfg()
    terms = cfg.observations["actor"].terms
    actor = {
        "base_lin_vel": copy.deepcopy(cfg.observations["critic"].terms["base_lin_vel"]),
        **{
            name: terms[name]
            for name in (
                "base_ang_vel",
                "projected_gravity",
                "command",
                "joint_pos",
                "joint_vel",
                "actions",
                "height_scan",
            )
        },
    }
    actor["height_scan"].scale = 1.0
    actor["height_scan"].clip = (-1.0, 1.0)
    actor["height_scan"].params = {
        "sensor_name": "terrain_scan",
        "offset": 0.5,
        "miss_value": 1.0,
    }
    cfg.observations["actor"].terms = actor
    cfg.observations["critic"].terms = copy.deepcopy(actor)
    cfg.observations["actor"].enable_corruption = not evaluation
    cfg.observations["critic"].enable_corruption = False
    cfg.curriculum.pop("command_vel", None)
    cfg.episode_length_s = 20.0
    # Match benchmark failure semantics: base contact, not every non-foot contact.
    cfg.scene.sensors = tuple(
        s for s in cfg.scene.sensors if s.name != "nonfoot_ground_touch"
    ) + (
        ContactSensorCfg(
            name="base_ground_contact",
            primary=ContactMatch(
                mode="geom", pattern=r"base\d+_collision", entity="robot"
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
            history_length=4,
        ),
    )
    cfg.terminations.pop("fell_over", None)
    cfg.terminations["illegal_contact"].params = {
        "sensor_name": "base_ground_contact",
        "force_threshold": 1.0,
    }
    cmd = cfg.commands["twist"]
    cmd.resampling_time_range = (10.0, 10.0)
    cmd.heading_command = task in ("A2_omni", "B1_rough")
    cmd.rel_heading_envs = 1.0 if cmd.heading_command else 0.0
    cmd.rel_standing_envs = {"A1_forward": 0.20, "A2_omni": 0.05}.get(task, 0.0)
    cmd.ranges.lin_vel_x = {
        "A1_forward": (0.35, 1.0),
        "B2_stairs": (0.4, 0.9),
        "C2_gap": (0.4, 0.8),
    }.get(task, (-1.0, 1.0))
    cmd.ranges.lin_vel_y = (
        (-1.0, 1.0)
        if cmd.heading_command
        else ((-0.1, 0.1) if task == "C2_gap" else (0.0, 0.0))
    )
    cmd.ranges.ang_vel_z = (
        (-1.0, 1.0)
        if cmd.heading_command
        else ((-0.2, 0.2) if task == "C2_gap" else (0.0, 0.0))
    )
    cmd.ranges.heading = (-math.pi, math.pi) if cmd.heading_command else None
    terrain = cfg.scene.terrain
    if task.startswith("A"):
        terrain.terrain_type = "plane"
        terrain.terrain_generator = None
        cfg.curriculum.pop("terrain_levels", None)
        cfg.sim.nconmax = None
    else:
        generator = terrain.terrain_generator
        if task == "B1_rough":
            chosen = generator.sub_terrains["random_rough"]
        elif task == "B2_stairs":
            chosen = generator.sub_terrains["pyramid_stairs_inv"]
            chosen.step_height_range = (0.04, 0.16)
            terrain.max_init_terrain_level = 1
            cfg.events["reset_base"].params["pose_range"]["yaw"] = (-0.2, 0.2)
        else:
            chosen = SquareGapTerrainCfg()
            terrain.max_init_terrain_level = 0
        chosen.proportion = 1.0
        generator.sub_terrains = {task: chosen}
        generator.num_rows = 10
        generator.num_cols = 1
        cfg.sim.nconmax = 256
    if evaluation:
        cfg.events.pop("push_robot", None)
        cfg.curriculum.clear()
        if terrain.terrain_generator is not None:
            terrain.terrain_generator.curriculum = False
            terrain.max_init_terrain_level = None
    return cfg
