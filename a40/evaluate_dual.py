#!/usr/bin/env python3
"""Run the untouched upstream evaluator with additional same-episode passage evidence.

Only evaluation is instrumented. The original policy, reward, resets, RNG,
tracking computations and EpisodeStatsTracker are used unchanged.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback
from types import MethodType

from passage_metrics import PassageTracker, VERSION, obstacle_from_mesh, summarize_passage

ROOT = Path(__file__).resolve().parents[1]


def capture_geometry(cfg, task, captured):
    """Observe meshes before their placement transform; do not regenerate or reseed."""
    generator = cfg.scene.terrain.terrain_generator
    if generator is None:
        raise RuntimeError("Passage evaluation requires the original generated terrain")
    base = generator.class_type

    class CapturedGenerator(base):
        def _add_sub_terrain(self, mesh, origin, row, col, sub_terrain_cfg):
            captured[(row, col)] = obstacle_from_mesh(
                mesh, origin, task, row, col, sub_terrain_cfg.function.__name__)
            return super()._add_sub_terrain(mesh, origin, row, col, sub_terrain_cfg)

    generator.class_type = CapturedGenerator


class PassageRuntime:
    """Read terminal kinematics before Isaac Lab's automatic reset, without extra sensing."""

    def __init__(self, env, captured):
        import torch

        self.env, self.terminal, self.in_step = env, None, False
        self.robot = env.scene["robot"]
        self.sensor = env.scene["contact_forces"]
        self.foot_ids, self.foot_names = self.robot.find_bodies(".*_foot", preserve_order=True)
        self.contact_ids, names = self.sensor.find_bodies(self.foot_names, preserve_order=True)
        if len(self.foot_ids) != 4 or names != self.foot_names:
            raise RuntimeError(f"Cannot align four robot feet with contact sensor: {self.foot_names}, {names}")
        self.tracker = PassageTracker(captured, env.num_envs, env.step_dt, env.device)
        self.old_step, self.old_reset = env.step, env._reset_idx

        def reset(inner, ids):
            if self.in_step and len(ids):
                if self.terminal is not None:
                    raise RuntimeError("Unexpected multiple autoresets in one control step")
                self.terminal = (torch.as_tensor(ids, device=env.device).clone(), self.snapshot())
            return self.old_reset(ids)

        def step(inner, action):
            before = self.snapshot()
            terrain = env.scene.terrain
            self.tracker.begin(before["root"], before["feet"], terrain.env_origins,
                               terrain.terrain_levels, terrain.terrain_types)
            self.terminal = None
            self.in_step = True
            try:
                result = self.old_step(action)
            finally:
                self.in_step = False
            after = self.snapshot()
            done = (result[2] | result[3]).nonzero().flatten()
            if done.numel():
                if self.terminal is None or not torch.equal(done, self.terminal[0]):
                    raise RuntimeError("Terminal capture and original done IDs differ")
                for name in after:
                    after[name][done] = self.terminal[1][name][done]
            self.tracker.observe(after["root"], after["feet"], after["force"], after["upright"])
            return result

        env._reset_idx = MethodType(reset, env)
        env.step = MethodType(step, env)

    def snapshot(self):
        import torch

        return {"root": self.robot.data.root_pos_w.clone(),
                "feet": self.robot.data.body_pos_w[:, self.foot_ids].clone(),
                "force": torch.linalg.vector_norm(self.sensor.data.net_forces_w[:, self.contact_ids], dim=-1),
                "upright": -self.robot.data.projected_gravity_b[:, 2].clone()}

    def close(self):
        self.env.step, self.env._reset_idx = self.old_step, self.old_reset


def augment_evaluator(module):
    """Attach the collector around the author's loop, without copying its success logic."""
    import torch

    original_make = module.gym.make
    original_loop, original_summary = module._run_eval_on_env, module._summarize
    original_tracker = module.EpisodeStatsTracker
    by_env = {}

    def make(task_id, *args, **kwargs):
        task = "B2" if "B2-StairClimb" in task_id else ("C2" if "C2-v0" in task_id else None)
        captured = {}
        if task:
            capture_geometry(kwargs["cfg"], task, captured)
        env = original_make(task_id, *args, **kwargs)
        if task:
            by_env[id(env.unwrapped)] = captured
        return env

    def loop(policy, env, eval_task_id, num_episodes, device):
        captured = by_env.get(id(env.unwrapped))
        if captured is None:
            return original_loop(policy, env, eval_task_id, num_episodes, device)
        if torch.device(env.unwrapped.device).type != "cuda":
            raise RuntimeError("Production passage evaluation must run on the allocated A40")
        runtime = PassageRuntime(env.unwrapped, captured)

        class DualTracker(original_tracker):
            def update(self, **kwargs):
                rows = super().update(**kwargs)
                done_ids = (kwargs["terminated"] | kwargs["truncated"]).nonzero().flatten()
                return runtime.tracker.finish(done_ids, rows)

        module.EpisodeStatsTracker = DualTracker
        try:
            episodes = original_loop(policy, env, eval_task_id, num_episodes, device)
        finally:
            runtime.close()
            module.EpisodeStatsTracker = original_tracker
        reported = module.args_cli.report_train_task or module.args_cli.task
        short = module.TASK_SHORT.get(reported, reported.replace("-", "_"))
        if "Unified" in reported or "AllTerrains" in reported:
            short = "MTL_unified"
        output = Path(module.args_cli.output_dir) / short
        output.mkdir(parents=True, exist_ok=True)
        report = {"metric_version": VERSION, "source": "actual generated or cached terrain mesh",
                  "task": next(iter(captured.values())).task,
                  "checkpoint": str(Path(module.args_cli.checkpoint).resolve()),
                  "gpu_runtime_verified": True,
                  "gpu_name": torch.cuda.get_device_name(env.unwrapped.device),
                  "foot_names": runtime.foot_names,
                  "step_dt": env.unwrapped.step_dt,
                  "hold_steps": runtime.tracker.hold_steps,
                  "terminal_state": "captured before autoreset, including final control step",
                  "benchmark_success_logic": "unmodified upstream EpisodeStatsTracker and evaluation loop",
                  "tiles": {f"{row},{col}": geometry.to_dict() for (row, col), geometry in captured.items()}}
        (output / (module.TASK_SHORT[eval_task_id] + ".geometry.json")).write_text(json.dumps(report, indent=2))
        return episodes

    def summary(episodes):
        result = original_summary(episodes)
        if episodes and "physical_obstacle_passage" in episodes[0]:
            result.update(summarize_passage(episodes))
        return result

    module.gym.make = make
    module._run_eval_on_env, module._summarize = loop, summary


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Isaac Sim passage evaluation requires an allocated Slurm GPU job")
    if '--task' in sys.argv and 'C2' in sys.argv[sys.argv.index('--task')+1]:
        extra=' --/app/extensions/excluded/0=omni.kit.telemetry'
        extra+=' --/persistent/isaac/asset_root/cloud='+str(ROOT/'assets/isaac_sim_5.1')
        for index,value in enumerate(sys.argv):
            if value.startswith('--kit_args='):
                sys.argv[index]+=extra
                break
            if value=='--kit_args':
                sys.argv[index+1]+=extra
                break
        else:
            sys.argv.append('--kit_args='+extra.strip())
    spec = importlib.util.spec_from_file_location("go2_upstream_eval", ROOT / "benchmark/scripts/evaluate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    code = 0
    try:
        augment_evaluator(module)
        module.main()
    except KeyboardInterrupt:
        code = 130
    except Exception:
        traceback.print_exc()
        code = 1
    finally:
        # Match the author's headless exit path; Kit shutdown can hang in headless mode.
        if not module.args_cli.headless:
            module.simulation_app.close()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
