# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""External RLinf adapter for the preserved five-task Isaac Lab benchmark."""

import json
import queue
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from reference import ROOT, read_manifest
from runtime_probe import compare, normalize


def register() -> None:
    """Register the benchmark through RLinf's existing Isaac Lab registry."""
    from rlinf.envs.sim.isaaclab import REGISTER_ISAACLAB_ENVS
    from rlinf.config import SupportedModel
    # Env workers do not import the actor model builders, but action dispatch
    # still resolves the model type. Register its identity in every worker.
    SupportedModel.register('go2_policy',force=True)

    for task in read_manifest()["tasks"].values():
        REGISTER_ISAACLAB_ENVS[task["gym_id"]] = Go2IsaacLabEnv


class Go2IsaacLabProcessEnv(gym.Env):
    """Expose the original manager environment through RLinf's chunk interface."""

    def __init__(self, cfg: Any, num_envs: int, seed_offset: int = 0,
                 total_num_processes: int = 1, worker_info: Any = None) -> None:
        from isaaclab.app import AppLauncher

        if seed_offset != 0 or total_num_processes != 1:
            raise ValueError("This reference uses one unpartitioned environment worker per model")
        self.cfg, self.num_envs = cfg, num_envs
        self.auto_reset, self.is_start = True, True
        self.group_size, self.num_group = 1, num_envs
        manifest = read_manifest()
        recipe = manifest["tasks"][cfg.task_key]
        kit_args="--/persistent/isaac/asset_root/cloud=" + str(ROOT / "assets/isaac_sim_5.1")
        if cfg.task_key == 'C2' or cfg.get('disable_kit_telemetry',False):
            kit_args += ' --/app/extensions/excluded/0=omni.kit.telemetry'
        self.app = AppLauncher(headless=True, enable_cameras=False, device="cuda:0",
                               kit_args=kit_args).app
        import envs.registry  # noqa: F401
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab_tasks.utils import load_cfg_from_registry

        env_cfg = load_cfg_from_registry(recipe["gym_id"], "env_cfg_entry_point")
        env_cfg.seed = int(cfg.seed)
        env_cfg.scene.num_envs = num_envs
        errors = compare(normalize(env_cfg.to_dict()), manifest["common_env_expectations"])
        errors += compare(normalize(env_cfg.to_dict()), recipe["env_expectations"])
        if errors:
            self.app.close()
            raise ValueError("Original task configuration mismatch: " + "; ".join(errors))

        if cfg.get('b1_numeric_guard',False):
            assert cfg.task_key=='B1'
            from b1_numerical_boundary import configure
            configure(env_cfg)

        class TerminalObservationEnv(ManagerBasedRLEnv):
            capture_terminal = False
            terminal_observation = None

            def _reset_idx(inner, env_ids):
                if inner.capture_terminal and len(env_ids):
                    # Extra terminal sensing must not advance the task's RNG or history.
                    with torch.random.fork_rng(devices=[torch.device(inner.device).index]):
                        terminal = inner.observation_manager.compute(update_history=False)["policy"]
                        terminal=terminal.clone()
                        if hasattr(inner,'_b1_bad_physics'):
                            terminal[inner._b1_bad_physics]=0
                        inner.terminal_observation = terminal
                return super()._reset_idx(env_ids)

        self.task = TerminalObservationEnv(cfg=env_cfg)
        self.device = self.task.device
        if list(self.task.observation_manager.group_obs_dim["policy"]) != [235] or self.task.action_manager.total_action_dim != 12:
            raise ValueError("Expected the original 235-observation/12-action Go2 interface")
        self.observation_space = gym.spaces.Dict({name: gym.spaces.Box(-np.inf, np.inf, (235,), np.float32)
                                                 for name in ("states", "critic_states")})
        self.action_space = gym.spaces.Box(-np.inf, np.inf, (12,), np.float32)
        self.returns = torch.zeros(num_envs, device=self.device)
        self.lengths = torch.zeros(num_envs, dtype=torch.int64, device=self.device)
        self.initial_reset = True
        self.set_global_step(0)
        output = Path(cfg.validation_output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "effective_env.json").write_text(json.dumps({
            "task": cfg.task_key, "seed": cfg.seed, "num_envs": num_envs,
            "env": normalize(env_cfg.to_dict()), "joint_names": list(self.task.scene["robot"].joint_names),
            "learner": "RLinf EmbodiedFSDPActor PPO", "rsl_rl_runner_used": False,
            "observation_dim": 235, "action_dim": 12, "config_errors": errors,
            "terminal_capture": "pre-reset sensing with RNG restored and history update disabled"}, indent=2))

    @property
    def elapsed_steps(self) -> torch.Tensor:
        """Return the simulator's per-environment episode clock."""
        return self.task.episode_length_buf

    @property
    def info_logging_keys(self) -> list[str]:
        """Return task-specific scalar logging keys."""
        return []

    def update_reset_state_ids(self) -> None:
        """Keep the original stochastic terrain and robot reset behavior."""

    def set_global_step(self, global_step: int) -> None:
        """Apply the original B2 posture schedule at its 25-update boundaries."""
        if self.cfg.task_key != "B2":
            return
        global_step += int(self.cfg.get("continuation_start_iteration", 0))
        schedule = read_manifest()["b2_schedule"]
        total = read_manifest()["tasks"]["B2"]["max_iterations"]
        position = global_step // schedule["schedule_check_interval"] * schedule["schedule_check_interval"]
        alpha = min(1.0, max(0.0, position / round(total * schedule["end_fraction"])))
        for name, endpoints in (("base_height", schedule["height_weight"]),
                                ("flat_orientation_l2", schedule["orientation_weight"])):
            term = self.task.reward_manager.get_term_cfg(name)
            term.weight = endpoints[0] + alpha * (endpoints[1] - endpoints[0])
            self.task.reward_manager.set_term_cfg(name, term)

    @staticmethod
    def _wrap_obs(obs: dict) -> dict:
        return {"states": obs["policy"].clone(), "critic_states": obs["policy"].clone()}

    def reset(self, env_idx: np.ndarray | None = None, *, seed=None, env_ids=None) -> tuple[dict, dict]:
        """Reset once initially, preserving the upstream random initial episode ages."""
        if env_ids is not None:env_idx=env_ids
        ids = None if env_idx is None else torch.as_tensor(env_idx, device=self.device, dtype=torch.long)
        obs, _ = self.task.reset(env_ids=ids)
        if self.initial_reset:
            self.task.episode_length_buf.copy_(torch.randint_like(self.task.episode_length_buf,
                                                                 high=self.task.max_episode_length))
            self.initial_reset = False
        self.returns[ids if ids is not None else slice(None)] = 0
        self.lengths[ids if ids is not None else slice(None)] = 0
        self.is_start = True
        return self._wrap_obs(obs), {"_go2_elapsed_steps":self.elapsed_steps.clone()}

    def step(self, actions: torch.Tensor) -> tuple:
        """Advance unchanged PhysX dynamics and keep terminal observations for GAE."""
        if isinstance(actions,tuple):
            actions,global_step=actions
            self.set_global_step(global_step)
        self.task.terminal_observation = None
        self.task.capture_terminal = True
        try:
            obs, rewards, terminated, truncated, _ = self.task.step(actions.to(self.device))
        finally:
            self.task.capture_terminal = False
        terminated, truncated = terminated.clone(), truncated.clone()
        done = terminated | truncated
        self.returns += rewards
        self.lengths += 1
        episode = {"return": self.returns.clone(), "episode_len": self.lengths.clone()}
        info = {"episode": episode,"_go2_elapsed_steps":self.elapsed_steps.clone()}
        if done.any():
            terminal = self.task.terminal_observation
            info.update(final_observation={"states": terminal, "critic_states": terminal.clone()},
                        final_info={"episode": episode}, _final_observation=done, _final_info=done)
        self.returns[done] = 0
        self.lengths[done] = 0
        self.is_start = False
        return self._wrap_obs(obs), rewards.clone(), terminated, truncated, info

    def chunk_step(self, chunk_actions: torch.Tensor) -> tuple:
        """Accept exactly one original joint-position command per control step."""
        actions = torch.as_tensor(chunk_actions, dtype=torch.float32, device=self.device)
        if tuple(actions.shape) != (self.num_envs, 1, 12):
            raise ValueError(f"Expected [{self.num_envs},1,12] actions, found {tuple(actions.shape)}")
        obs, rewards, terminated, truncated, info = self.step(actions[:, 0])
        return [obs], rewards[:, None], terminated[:, None], truncated[:, None], [info]

    def close(self) -> None:
        """Close only this worker's simulator and application."""
        if not getattr(self,'_closed',False):
            self._closed=True
            self.task.close()


class Go2IsaacLabEnv(gym.Env):
    """Use RLinf's official spawned Isaac Lab transport so Kit owns a main thread."""
    def __init__(self,cfg,num_envs,seed_offset=0,total_num_processes=1,worker_info=None):
        from rlinf.envs.sim.isaaclab.venv import SubProcIsaacLabEnv
        def make_env():
            env=Go2IsaacLabProcessEnv(cfg,num_envs,seed_offset,total_num_processes,None)
            return env,env.app
        self.cfg=cfg;self.num_envs=num_envs;self.device='cuda:0'
        self.auto_reset=True;self.is_start=True;self.group_size=1;self.num_group=num_envs
        self.global_step=0;self._elapsed_steps=torch.zeros(num_envs,dtype=torch.long,device=self.device)
        self.observation_space=gym.spaces.Dict({name:gym.spaces.Box(-np.inf,np.inf,(235,),np.float32)
                                               for name in ('states','critic_states')})
        self.action_space=gym.spaces.Box(-np.inf,np.inf,(12,),np.float32)
        self.env=SubProcIsaacLabEnv(make_env)
        self.env.parent_remote.send('device')
        deadline=time.monotonic()+180
        while not self.env.parent_remote.poll(1):
            self._check_child(deadline)
        self.device=self.env.parent_remote.recv()

    def _check_child(self,deadline):
        if not self.env.isaac_lab_process.is_alive():
            raise RuntimeError('Spawned Isaac Lab process exited; inspect its simulator traceback')
        if time.monotonic()>deadline:raise TimeoutError('Spawned Isaac Lab operation exceeded 180 seconds')

    def _receive(self):
        deadline=time.monotonic()+180
        while True:
            try:return self.env.obs_queue.get(timeout=1)
            except queue.Empty:self._check_child(deadline)

    @property
    def elapsed_steps(self):return self._elapsed_steps

    @property
    def info_logging_keys(self):return []

    def update_reset_state_ids(self):pass

    def set_global_step(self,global_step):self.global_step=global_step

    def reset(self,env_idx=None):
        self.env.parent_remote.send('reset')
        ids=None if env_idx is None else torch.as_tensor(env_idx,dtype=torch.long,device=self.device)
        self.env.reset_idx.put((ids,None))
        obs,info=self._receive();self._elapsed_steps=info.pop('_go2_elapsed_steps')
        self.is_start=True
        return obs,info

    def step(self,actions):
        self.env.parent_remote.send('step')
        self.env.action_queue.put((actions,self.global_step))
        obs,reward,terminated,truncated,info=self._receive()
        self._elapsed_steps=info.pop('_go2_elapsed_steps');self.is_start=False
        return obs,reward,terminated,truncated,info

    def chunk_step(self,chunk_actions):
        if tuple(chunk_actions.shape)!=(self.num_envs,1,12):raise ValueError(chunk_actions.shape)
        obs,r,t,tr,info=self.step(chunk_actions[:,0])
        return [obs],r[:,None],t[:,None],tr[:,None],[info]

    def close(self):
        if not self.env.isaac_lab_process.is_alive():return
        self.env.parent_remote.send('close');self.env.isaac_lab_process.join(timeout=30)
        if self.env.isaac_lab_process.is_alive():
            self.env.isaac_lab_process.terminate();self.env.isaac_lab_process.join(timeout=10)
