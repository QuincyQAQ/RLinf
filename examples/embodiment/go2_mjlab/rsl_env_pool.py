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

"""CPU simulation processes supplying observations to an upstream learner."""

import multiprocessing as mp
import traceback
from multiprocessing.connection import Connection

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict


def serve(connection: Connection, config: dict, count: int, offset: int) -> None:
    """Own a CPU environment until the parent closes its connection."""
    from rlinf.envs.sim.go2_mjlab.go2_mjlab_env import Go2MjlabEnv

    env = None
    try:
        env = Go2MjlabEnv(OmegaConf.create(config), count, seed_offset=offset)
        observations, _ = env.reset()
        connection.send(("ready", {k: v.numpy() for k, v in observations.items()}))
        while True:
            command, payload = connection.recv()
            if command == "close":
                break
            actions, update = payload
            if update is not None:
                env.set_global_step(update)
            obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated | truncated
            completed = info["episode"]
            connection.send(
                (
                    "step",
                    {
                        "obs": {k: v.numpy() for k, v in obs.items()},
                        "reward": reward.numpy(),
                        "done": done.numpy(),
                        "timeouts": truncated.numpy(),
                        "returns": completed["return"][done].tolist(),
                        "lengths": completed["episode_len"][done].tolist(),
                    },
                )
            )
    except (EOFError, BrokenPipeError):
        pass
    except Exception:
        connection.send(("error", traceback.format_exc()))
    finally:
        if env is not None:
            env.close()
        connection.close()


class CpuEnvironmentPool:
    """Step independent CPU batches without initializing accelerators in children."""

    def __init__(self, config: dict, count: int, workers: int) -> None:
        if count < 1 or workers < 1 or count % workers:
            raise ValueError(
                "Environment count must be positive and divisible by workers"
            )
        self.num_envs, self.num_actions = count, 12
        self.device = "cpu"
        self.connections = []
        self.processes = []
        self.batch_size = count // workers
        context = mp.get_context("spawn")
        try:
            for index in range(workers):
                parent, child = context.Pipe()
                process = context.Process(
                    target=serve,
                    args=(child, config, self.batch_size, index * 10000),
                )
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            batches = [self.receive(c, "ready") for c in self.connections]
            self.observations = self.combine(batches)
        except Exception:
            self.close()
            raise

    @staticmethod
    def receive(connection: Connection, expected: str) -> dict:
        """Surface failed simulation workers in the learner process."""
        if not connection.poll(300):
            raise TimeoutError(
                "CPU simulation worker produced no reply for 300 seconds"
            )
        kind, data = connection.recv()
        if kind != expected:
            raise RuntimeError(
                f"Expected simulation reply {expected!r}, got {kind}: {data}"
            )
        return data

    def combine(self, batches: list[dict]) -> TensorDict:
        """Keep actor and critic observations in the upstream TensorDict contract."""
        return TensorDict(
            {
                k: torch.from_numpy(np.concatenate([b[k] for b in batches]))
                for k in batches[0]
            },
            batch_size=[self.num_envs],
        )

    def get_observations(self) -> TensorDict:
        """Return the latest CPU observation batch."""
        return self.observations

    def step(self, actions: torch.Tensor, update: int | None = None) -> tuple:
        """Apply one control step and collect completed-episode measurements."""
        parts = actions.detach().cpu().numpy().reshape(-1, self.batch_size, 12)
        for connection, part in zip(self.connections, parts, strict=True):
            connection.send(("step", (part, update)))
        batches = [self.receive(c, "step") for c in self.connections]
        self.observations = self.combine([b["obs"] for b in batches])
        arrays = {
            k: torch.from_numpy(np.concatenate([b[k] for b in batches]))
            for k in ("reward", "done", "timeouts")
        }
        return (
            self.observations,
            arrays["reward"],
            arrays["done"],
            {
                "time_outs": arrays["timeouts"],
                "returns": [v for b in batches for v in b["returns"]],
                "lengths": [v for b in batches for v in b["lengths"]],
            },
        )

    def close(self) -> None:
        """Release this pool's processes without affecting other experiments."""
        for connection in self.connections:
            try:
                connection.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        for connection in self.connections:
            connection.close()
