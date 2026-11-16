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

"""Evaluate one trained 235-dimensional RLinf policy on one benchmark task."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn

from rlinf.envs.sim.go2_mjlab.benchmark_metrics import summarize
from rlinf.envs.sim.go2_mjlab.benchmark_protocol import (
    TASKS,
    UPSTREAM_REVISION,
    validate_policy_interface,
)
from rlinf.envs.sim.go2_mjlab.go2_mjlab_env import Go2MjlabEnv


class FrozenActor(nn.Module):
    """Run the checkpointed MLP and frozen observation moments on CPU."""

    def __init__(self, state: dict) -> None:
        super().__init__()
        validate_policy_interface(
            state["actor.0.weight"].shape[1], state["actor.6.weight"].shape[0]
        )
        layers = []
        for i in (0, 2, 4, 6):
            weight = state[f"actor.{i}.weight"]
            layer = nn.Linear(weight.shape[1], weight.shape[0])
            layer.load_state_dict({"weight": weight, "bias": state[f"actor.{i}.bias"]})
            layers.append(layer)
            if i < 6:
                layers.append(nn.ELU())
        self.network = nn.Sequential(*layers)
        self.register_buffer("mean", state["actor_normalizer.mean"])
        self.register_buffer("variance", state["actor_normalizer.variance"])
        if not all(torch.isfinite(v).all() for v in state.values()):
            raise ValueError("Checkpoint contains nonfinite values")
        if (self.variance < 0).any() or state["actor_normalizer.count"].item() <= 0:
            raise ValueError("Checkpoint requires trained observation moments")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Return deterministic action means with normalization epsilon 0.01."""
        return self.network((observations - self.mean) / (self.variance.sqrt() + 0.01))


def main() -> None:
    """Collect full episodes and export one canonical matrix cell."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--training-config", type=Path, required=True)
    p.add_argument("--eval-task", choices=tuple(TASKS), required=True)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--num-episodes", type=int, default=1024)
    p.add_argument("--eval-seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.num_envs < 1 or a.num_episodes < 1 or a.threads < 1:
        p.error("Environment, episode and thread counts must be positive")
    trained = OmegaConf.load(a.training_config)
    train_task = trained.env.train.benchmark_task
    if train_task not in TASKS:
        p.error("Training config must identify one of the five benchmark tasks")
    validate_policy_interface(
        trained.actor.model.obs_dim, trained.actor.model.action_dim
    )
    torch.set_num_threads(a.threads)
    state = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    policy = FrozenActor(state).eval()
    cfg = OmegaConf.create(
        {
            "benchmark_task": a.eval_task,
            "auto_reset": True,
            "ignore_terminations": False,
            "group_size": 1,
            "seed": a.eval_seed,
            "threads": a.threads,
            "max_episode_steps": 1000,
            "max_steps_per_rollout_epoch": 24,
            "is_eval": True,
            "video_cfg": {"save_video": False},
        }
    )
    row_dir = a.output / f"{train_task}_s{trained.actor.seed}"
    row_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "train_task": train_task,
        "train_seed": int(trained.actor.seed),
        "checkpoint_sha256": hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        "training_config_sha256": hashlib.sha256(
            a.training_config.read_bytes()
        ).hexdigest(),
        "eval_seed": a.eval_seed,
    }
    identity_path = row_dir / "policy_manifest.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError(
            "Output row already belongs to a different policy, config or evaluation seed"
        )
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    destination = row_dir / f"{a.eval_task}.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to replace prior evaluation: {destination}")
    env = Go2MjlabEnv(cfg, num_envs=a.num_envs)
    episodes = []
    traversal = None
    if a.eval_task == "B2_stairs":
        terrain = env.task.cfg.scene.terrain.terrain_generator
        stairs = terrain.sub_terrains["B2_stairs"]
        goal_x = terrain.size[0] / 2 - stairs.border_width + 0.3
        corridor_y = terrain.size[1] / 2 - stairs.border_width - 0.3
        traversal = {
            "crossed": torch.zeros(a.num_envs, dtype=torch.bool),
            "max_forward": torch.full((a.num_envs,), -torch.inf),
            "inside_corridor": torch.ones(a.num_envs, dtype=torch.bool),
            "goal_x": goal_x,
            "corridor_y": corridor_y,
        }
    try:
        obs, _ = env.reset()
        with torch.inference_mode():
            for _ in range(50000):
                if traversal is not None:
                    # Observe before stepping so automatic resets cannot fake progress.
                    relative = (
                        env.task.scene["robot"].data.root_link_pos_w
                        - env.task.scene.env_origins
                    )
                    traversal["max_forward"] = torch.maximum(
                        traversal["max_forward"], relative[:, 0]
                    )
                    traversal["inside_corridor"] &= (
                        relative[:, 1].abs() <= traversal["corridor_y"]
                    )
                    traversal["crossed"] |= (
                        (relative[:, 0] >= traversal["goal_x"])
                        & traversal["inside_corridor"]
                        & (env.task.episode_length_buf > 0)
                    )
                actions = policy(obs["states"])
                obs, _, terminated, truncated, info = env.step(actions)
                rows = info["benchmark_episodes"]
                if traversal is not None:
                    ids = torch.nonzero(terminated | truncated).flatten().tolist()
                    for i, row in zip(ids, rows, strict=True):
                        row["stairs_crossed"] = bool(traversal["crossed"][i])
                        row["stairs_pass_success"] = (
                            row["stairs_crossed"]
                            and row["termination_reason"] == "time_out"
                        )
                        row["stairs_tracking_and_pass_success"] = (
                            row["stairs_pass_success"] and row["episode_success"]
                        )
                        row["max_forward_from_terrain_origin_m"] = float(
                            traversal["max_forward"][i]
                        )
                    traversal["crossed"][ids] = False
                    traversal["inside_corridor"][ids] = True
                    traversal["max_forward"][ids] = -torch.inf
                episodes.extend(rows)
                if len(episodes) >= a.num_episodes:
                    break
        if len(episodes) < a.num_episodes:
            raise RuntimeError(
                f"Collected {len(episodes)} episodes, requested at least {a.num_episodes}"
            )
    finally:
        env.close()
    metrics = summarize(episodes)
    if traversal is not None:
        for name in ("stairs_pass_success", "stairs_tracking_and_pass_success"):
            metrics[name + "_rate"] = sum(e[name] for e in episodes) / len(episodes)
    protocol = {
        "variant": "rlinf-mjlab-cross-task-v1",
        "evaluation_scope": "smoke" if a.num_episodes < 1024 else "benchmark",
        "reference_revision": UPSTREAM_REVISION,
        "train_task": train_task,
        "train_seed": int(trained.actor.seed),
        "eval_task": a.eval_task,
        "eval_seed": a.eval_seed,
        "requested_episodes": a.num_episodes,
        "num_envs": a.num_envs,
        "checkpoint_sha256": hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        "training_config_sha256": hashlib.sha256(
            a.training_config.read_bytes()
        ).hexdigest(),
        "episode_seconds": 20,
        "actor_dim": 235,
        "policy": "deterministic mean",
        "observation_noise": False,
        "external_pushes": False,
        "terrain_curriculum": False,
        "startup_randomization": "Mjlab defaults",
        "metrics_timing": "post-step/current state, including reset state as in reference evaluator",
        "hardware_access": False,
    }
    if traversal is not None:
        protocol["stairs_traversal"] = {
            "goal_x_from_terrain_origin_m": traversal["goal_x"],
            "corridor_abs_y_max_m": traversal["corridor_y"],
            "definition": "Reach +x outer landing with 0.3m root clearance, without leaving staircase corridor before crossing, and survive full episode",
            "combined_success": "stairs_pass_success AND original benchmark tracking success",
            "timing": "pre-action positions; reset positions excluded; crossing on final action conservatively not counted",
            "reference_metric": False,
        }
    destination.write_text(
        json.dumps(
            {"summary": metrics, "episodes": episodes, "protocol": protocol}, indent=2
        )
        + "\n"
    )
    summary_path = row_dir / "summary.json"
    row = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    row[a.eval_task] = metrics
    summary_path.write_text(json.dumps(row, indent=2) + "\n")


if __name__ == "__main__":
    main()
