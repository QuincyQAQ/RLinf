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

"""Apply the benchmark B2 specialist recipe to the MuJoCo backend."""

import copy
from typing import Any

from . import b2_reference_rewards as rewards


def apply_posture_reward_schedule(env: Any, schedule: Any, global_step: int) -> None:
    """Apply reference posture weights before each 25-update training chunk."""
    progress = global_step // schedule.interval * schedule.interval
    start = round(schedule.start_fraction * schedule.total_updates)
    end = round(schedule.end_fraction * schedule.total_updates)
    alpha = min(1.0, max(0.0, (progress - start) / (end - start)))
    for name, early, late in (
        ("base_height", schedule.height_early, schedule.height_late),
        (
            "flat_orientation_l2",
            schedule.orientation_early,
            schedule.orientation_late,
        ),
    ):
        env.reward_manager.get_term_cfg(name).weight = (
            1.0 - alpha
        ) * early + alpha * late


def apply_b2_reference(cfg: Any) -> Any:
    """Replace all inherited reward terms with the B2 reference terms."""
    from mjlab.actuator import DcMotorActuatorCfg
    from mjlab.envs import mdp
    from mjlab.envs.mdp import dr
    from mjlab.managers import EventTermCfg, RewardTermCfg, SceneEntityCfg
    from mjlab.sensor import ContactMatch, ContactSensorCfg
    from mjlab.utils.noise import UniformNoiseCfg

    from .b2_reference_contacts import ForceThresholdContactSensorCfg

    cfg.rewards = {
        "track_lin_vel_xy_exp": RewardTermCfg(
            func=rewards.track_lin_vel_xy_exp,
            weight=1.5,
            params={"std": 0.5, "command_name": "twist"},
        ),
        "track_ang_vel_z_exp": RewardTermCfg(
            func=rewards.track_ang_vel_z_exp,
            weight=0.75,
            params={"std": 0.5, "command_name": "twist"},
        ),
        "lin_vel_z_l2": RewardTermCfg(func=rewards.lin_vel_z_l2, weight=-2.0),
        "ang_vel_xy_l2": RewardTermCfg(func=rewards.ang_vel_xy_l2, weight=-0.05),
        "dof_torques_l2": RewardTermCfg(func=mdp.joint_torques_l2, weight=-2.0e-5),
        "dof_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.003),
        "feet_air_time": RewardTermCfg(
            func=rewards.feet_air_time,
            weight=0.04,
            params={
                "sensor_name": "feet_ground_contact",
                "command_name": "twist",
                "threshold": 0.35,
            },
        ),
        "undesired_contacts": RewardTermCfg(
            func=rewards.undesired_contacts,
            weight=-0.7,
            params={"sensor_name": "legs_ground_contact", "threshold": 1.0},
        ),
        "flat_orientation_l2": RewardTermCfg(func=mdp.flat_orientation_l2, weight=-2.0),
        "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-2.0),
        "base_height": RewardTermCfg(
            func=rewards.base_height_l2,
            weight=-3.5,
            params={"target_height": 0.42, "sensor_name": "terrain_scan"},
        ),
    }
    sensors = []
    for sensor in cfg.scene.sensors:
        if sensor.name == "feet_ground_contact":
            sensor = ForceThresholdContactSensorCfg(
                name="feet_ground_contact",
                primary=ContactMatch(
                    mode="geom", pattern=r".*_foot_collision", entity="robot"
                ),
                secondary=ContactMatch(mode="body", pattern="terrain"),
                fields=("found", "force"),
                reduce="netforce",
                track_air_time=True,
                history_length=3,
            )
        elif sensor.name == "base_ground_contact":
            sensor.primary = ContactMatch(
                mode="body", pattern="base_link", entity="robot"
            )
            sensor.history_length = 3
        elif sensor.name == "terrain_scan":
            # Robot visual/collision geoms use groups 2/3; terrain uses group 0.
            sensor.include_geom_groups = (0,)
        sensors.append(sensor)
    sensors.append(
        ContactSensorCfg(
            name="legs_ground_contact",
            primary=ContactMatch(
                mode="geom", pattern=r".*_(thigh|calf[12])_collision", entity="robot"
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            history_length=3,
        )
    )
    cfg.scene.sensors = tuple(sensors)
    actor = cfg.observations["actor"].terms
    actor["base_lin_vel"].func = rewards.base_lin_vel
    actor["base_lin_vel"].params = {}
    actor["base_lin_vel"].noise = UniformNoiseCfg(n_min=-0.1, n_max=0.1)
    actor["base_ang_vel"].func = rewards.base_ang_vel
    actor["base_ang_vel"].params = {}
    cfg.observations["critic"].terms = copy.deepcopy(actor)
    # The adapter shares the sampled policy observation with the critic.
    cfg.observations["critic"].enable_corruption = False

    robot = copy.deepcopy(cfg.scene.entities["robot"])
    robot.init_state.pos = (0.0, 0.0, 0.4)
    robot.init_state.joint_pos = {
        ".*L_hip_joint": 0.1,
        ".*R_hip_joint": -0.1,
        "F[L,R]_thigh_joint": 0.8,
        "R[L,R]_thigh_joint": 1.0,
        ".*_calf_joint": -1.5,
    }
    robot.articulation.actuators = (
        DcMotorActuatorCfg(
            target_names_expr=(".*",),
            stiffness=25.0,
            damping=0.5,
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            armature=0.0,
            frictionloss=0.0,
        ),
    )
    cfg.scene.entities["robot"] = robot
    cfg.events.pop("push_robot", None)
    cfg.events.pop("encoder_bias", None)
    cfg.events.pop("base_com", None)
    cfg.events["foot_friction"].params.update(
        asset_cfg=SceneEntityCfg("robot", geom_names=".*_collision"), ranges=(0.6, 0.6)
    )
    cfg.events["base_mass"] = EventTermCfg(
        mode="startup",
        func=dr.body_mass,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "ranges": (-1.0, 3.0),
            "operation": "add",
        },
    )
    cfg.events["reset_base"].params["velocity_range"] = dict.fromkeys(
        ("x", "y", "z", "roll", "pitch", "yaw"), (0.0, 0.0)
    )
    if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.num_cols = 20
    return cfg
