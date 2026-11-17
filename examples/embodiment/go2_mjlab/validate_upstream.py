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

"""Validate the official Mjlab task before adapting it to RLinf workers."""

import argparse
import json
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path


def main() -> None:
    """Benchmark physics or run a single-device upstream PPO reference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--physics-device", default="cpu")
    parser.add_argument("--learner-device", default=None)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import src.tasks  # noqa: F401
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.os import dump_yaml

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Expected an empty output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    task = "Unitree-Go2-Flat"
    env_cfg = load_env_cfg(task)
    agent_cfg = load_rl_cfg(task)
    agent_cfg.logger = "tensorboard"
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = agent_cfg.seed
    dump_yaml(args.output / "env.yaml", asdict(env_cfg))
    dump_yaml(args.output / "agent.yaml", asdict(agent_cfg))
    report = {
        "task": task,
        "framework": "upstream RSL-RL reference, not RLinf PPO",
        "num_envs": args.num_envs,
        "physics_device": args.physics_device,
        "learner_device": args.learner_device,
        "versions": {
            name: version(name)
            for name in (
                "torch",
                "mjlab",
                "mujoco",
                "mujoco-warp",
                "warp-lang",
                "rsl-rl-lib",
            )
        },
    }
    started = time.perf_counter()
    env = ManagerBasedRlEnv(env_cfg, device=args.physics_device)
    try:
        obs, _ = env.reset()
        report["observation_shapes"] = {k: list(v.shape) for k, v in obs.items()}
        report["initialization_seconds"] = time.perf_counter() - started
        if args.learner_device is None:
            actions = torch.zeros(args.num_envs, 12, device=args.physics_device)
            for _ in range(2):
                env.step(actions)
            started = time.perf_counter()
            for _ in range(args.steps):
                obs, reward, terminated, truncated, _ = env.step(actions)
            report["steps"] = args.steps
            report["step_seconds"] = time.perf_counter() - started
            report["observations_finite"] = all(
                torch.isfinite(v).all().item() for v in obs.values()
            )
            report["rewards_finite"] = torch.isfinite(reward).all().item()
        else:
            wrapper = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            cfg = asdict(agent_cfg)
            runner = load_runner_cls(task)(
                wrapper, cfg, str(args.output), args.learner_device
            )
            started = time.perf_counter()
            runner.learn(
                num_learning_iterations=args.updates, init_at_random_ep_len=True
            )
            report["updates"] = args.updates
            report["training_seconds"] = time.perf_counter() - started
            report["checkpoints"] = sorted(
                p.name for p in args.output.glob("model_*.pt")
            )
    finally:
        env.close()
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
