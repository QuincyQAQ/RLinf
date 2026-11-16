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
    try:
        obs, _ = env.reset()
        with torch.inference_mode():
            for _ in range(50000):
                actions = policy(obs["states"])
                obs, _, _, _, info = env.step(actions)
                episodes.extend(info["benchmark_episodes"])
                if len(episodes) >= a.num_episodes:
                    break
        if len(episodes) < a.num_episodes:
            raise RuntimeError(
                f"Collected {len(episodes)} episodes, requested at least {a.num_episodes}"
            )
    finally:
        env.close()
    metrics = summarize(episodes)
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
