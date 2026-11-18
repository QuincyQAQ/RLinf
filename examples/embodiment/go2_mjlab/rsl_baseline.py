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

"""Train a benchmark specialist with upstream RSL-RL PPO and CPU physics."""

import argparse
import copy
import json
import os
import random
import signal
import time
from collections import deque
from contextlib import ExitStack
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.envs.sim.go2_mjlab.benchmark_protocol import TASKS
from rlinf.utils.logging import get_logger

logger = get_logger()


def compose_task(task: str, environments: int, seed: int, config: str | None = None):
    """Resolve a task recipe shared with native RLinf or the legacy baseline."""
    directory = Path(__file__).resolve().parents[1] / "config"
    short = task.split("_")[0].lower()
    name = config or (
        "go2_mjlab_benchmark_b2_posture_ascend"
        if task == "B2_stairs"
        else f"go2_mjlab_benchmark_{short}_ascend"
    )
    with initialize_config_dir(version_base="1.1", config_dir=str(directory)):
        cfg = compose(
            config_name=name,
            overrides=[
                f"env.train.total_num_envs={environments}",
                f"env.train.seed={seed}",
                f"actor.seed={seed}",
                "env.train.threads=1",
                *([] if config else ["env.train.physics_backend=mjlab"]),
                "env.train.randomize_initial_episode_length=true",
            ],
        )
    if cfg.env.train.benchmark_task != task:
        raise ValueError(
            f"Config {name!r} selects {cfg.env.train.benchmark_task!r}, expected {task!r}"
        )
    return cfg


@torch.no_grad()
def canonical_weights(actor, critic) -> dict:
    """Fold frozen affine normalization into first layers for Go2 evaluation."""
    result = {}
    for prefix, model in (("actor", actor), ("critic", critic)):
        weights = {
            k: v.detach().cpu().clone() for k, v in model.mlp.state_dict().items()
        }
        normalizer = copy.deepcopy(model.obs_normalizer).cpu().eval()
        zeros = torch.zeros(1, model.obs_dim)
        offset = normalizer(zeros)
        scale = normalizer(torch.ones_like(zeros)) - offset
        first = weights["0.weight"]
        weights["0.bias"] += first @ offset.flatten()
        weights["0.weight"] = first * scale
        result.update({f"{prefix}.{k}": v for k, v in weights.items()})
    result["std"] = actor.distribution.std_param.detach().cpu().clone()
    if not all(torch.isfinite(v).all() for v in result.values()):
        raise ValueError("Refusing to export nonfinite baseline weights")
    return result


def atomic_json(path: Path, data: dict) -> None:
    """Publish a complete status file for readers outside the training process."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    """Run the unchanged upstream PPO update with explicit experiment budgets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=2200)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--config", help="Hydra config name shared with the native RLinf experiment"
    )
    args = parser.parse_args()
    if min(args.updates, args.num_envs, args.workers, args.save_interval) < 1:
        parser.error("Training budgets must be positive")
    if args.num_envs % args.workers:
        parser.error("num-envs must be divisible by workers")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to replace existing experiment: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401

        torch.get_device_module(args.device).set_device(args.device)
    import src.tasks  # noqa: F401
    from mjlab.tasks.registry import load_rl_cfg
    from rsl_env_pool import CpuEnvironmentPool
    from rsl_rl.algorithms import PPO
    from torch.utils.tensorboard import SummaryWriter

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    task_cfg = compose_task(args.task, args.num_envs, args.seed, args.config)
    OmegaConf.save(task_cfg, args.output / "training_config.yaml")
    agent = asdict(load_rl_cfg("Unitree-Go2-Flat"))
    agent["obs_groups"] = {"actor": ["states"], "critic": ["critic_states"]}
    agent["multi_gpu"] = None
    agent["actor"]["obs_normalization"] = bool(task_cfg.actor.model.obs_normalization)
    agent["critic"]["obs_normalization"] = bool(task_cfg.actor.model.obs_normalization)
    # Match MjlabOnPolicyRunner's handling of absent optional model components.
    for name in ("actor", "critic"):
        for optional in ("cnn_cfg", "distribution_cfg"):
            if agent[name].get(optional) is None:
                agent[name].pop(optional, None)
    if task_cfg.actor.model.get("distribution_safety_patch", False):
        agent["actor"]["distribution_cfg"]["class_name"] = (
            "rsl_distribution:BoundedGaussianDistribution"
        )
    OmegaConf.save(OmegaConf.create(agent), args.output / "rsl_agent.yaml")
    metadata = {
        "task": args.task,
        "learner": "rsl_rl.algorithms.PPO",
        "native_rlinf_ppo": False,
        "config": args.config,
        "physics": "CPU MuJoCo/Mjlab port; not an Isaac Lab reproduction",
        "num_envs": args.num_envs,
        "cpu_workers": args.workers,
        "threads_per_worker": 1,
        "device": args.device,
        "physical_npu": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "training_seed": args.seed,
        "target_updates": args.updates,
        "evaluation_updates": [1200, 1500, 1800, 2200],
        "versions": {n: version(n) for n in ("torch", "rsl-rl-lib", "mjlab")},
        "pid": os.getpid(),
        "started_unix": time.time(),
    }
    atomic_json(args.output / "run.json", metadata)
    with ExitStack() as cleanup:
        pool = CpuEnvironmentPool(
            OmegaConf.to_container(task_cfg.env.train, resolve=True),
            args.num_envs,
            args.workers,
        )
        cleanup.callback(pool.close)
        algorithm = PPO.construct_algorithm(
            pool.get_observations(), pool, copy.deepcopy(agent), args.device
        )
        writer = SummaryWriter(str(args.output / "tensorboard"))
        cleanup.callback(writer.close)
        completed = 0
        if args.resume:
            checkpoint = torch.load(
                args.resume, map_location=args.device, weights_only=False
            )
            algorithm.load(checkpoint, load_cfg=None, strict=True)
            completed = int(checkpoint["completed_updates"])
            algorithm.learning_rate = checkpoint["learning_rate"]
            torch.set_rng_state(checkpoint["torch_rng"].cpu())
            if args.device.startswith("npu"):
                torch.get_device_module(args.device).set_rng_state(
                    checkpoint["device_rng"].cpu()
                )
        stopped = False

        def request_stop(signum, frame):
            nonlocal stopped
            stopped = True

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)

        def save() -> None:
            destination = args.output / "checkpoints" / f"global_step_{completed}"
            destination.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                **algorithm.save(),
                "completed_updates": completed,
                "iter": completed - 1,
                "learning_rate": algorithm.learning_rate,
                "torch_rng": torch.get_rng_state(),
                "infos": metadata,
            }
            if args.device.startswith("npu"):
                checkpoint["device_rng"] = torch.get_device_module(
                    args.device
                ).get_rng_state()
            for filename, state in (
                ("model.pt", checkpoint),
                (
                    "full_weights.pt",
                    canonical_weights(algorithm.actor, algorithm.critic),
                ),
            ):
                temporary = destination / (filename + ".tmp")
                torch.save(state, temporary)
                temporary.replace(destination / filename)
            atomic_json(
                destination / "checkpoint.json", {"completed_updates": completed}
            )
            atomic_json(
                args.output / "latest_checkpoint.json", {"directory": str(destination)}
            )

        durations = deque(maxlen=50)
        returns, lengths = deque(maxlen=1000), deque(maxlen=1000)
        algorithm.train_mode()
        observations = pool.get_observations().to(args.device)
        try:
            with (args.output / "metrics.jsonl").open("a", buffering=1) as metrics:
                while completed < args.updates and not stopped:
                    begin = time.perf_counter()
                    with torch.inference_mode():
                        for step in range(agent["num_steps_per_env"]):
                            actions = algorithm.act(observations)
                            obs, rewards, dones, extras = pool.step(
                                actions, completed if step == 0 else None
                            )
                            observations = obs.to(args.device)
                            algorithm.process_env_step(
                                observations,
                                rewards.to(args.device),
                                dones.to(args.device),
                                extras,
                            )
                            returns.extend(extras["returns"])
                            lengths.extend(extras["lengths"])
                        algorithm.compute_returns(observations)
                    collection_seconds = time.perf_counter() - begin
                    losses = algorithm.update()
                    completed += 1
                    durations.append(time.perf_counter() - begin)
                    row = {
                        "completed_updates": completed,
                        "unix": time.time(),
                        "update_seconds": durations[-1],
                        "collection_seconds": collection_seconds,
                        "learning_rate": algorithm.learning_rate,
                        "mean_episode_return": float(np.mean(returns))
                        if returns
                        else None,
                        "mean_episode_length": float(np.mean(lengths))
                        if lengths
                        else None,
                        "losses": losses,
                        "remaining_seconds_estimate": (args.updates - completed)
                        * np.mean(durations),
                    }
                    metrics.write(json.dumps(row) + "\n")
                    for name, value in losses.items():
                        writer.add_scalar("Loss/" + name, value, completed)
                    for name in (
                        "mean_episode_return",
                        "mean_episode_length",
                        "learning_rate",
                    ):
                        if row[name] is not None:
                            writer.add_scalar("Train/" + name, row[name], completed)
                    atomic_json(args.output / "progress.json", row)
                    logger.info(
                        "%s update %s/%s, %.2fs",
                        args.task,
                        completed,
                        args.updates,
                        durations[-1],
                    )
                    if completed % args.save_interval == 0 or completed == args.updates:
                        save()
            if stopped and completed:
                save()
            atomic_json(
                args.output / "finished.json",
                {"completed_updates": completed, "stopped": stopped},
            )
        finally:
            writer.flush()


if __name__ == "__main__":
    main()
