#!/usr/bin/env python3
"""A40-only effective config/physics/learner checks; never used to rewrite the MDP."""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time

from reference import ROOT, BENCH, LAB, flag, source_recipe, verify_source


def normalize(value):
    if isinstance(value, dict):
        return {str(k): normalize(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [normalize(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def nested(data, path):
    for key in path.split("."):
        data = data[key]
    return data


def compare(actual, expected, path=""):
    errors = []
    for key, value in expected.items():
        try:
            found = nested(actual, key)
        except (KeyError, TypeError):
            errors.append(f"Missing {path}{key}")
            continue
        equal = math.isclose(found, value, rel_tol=1e-7, abs_tol=1e-10) if type(value) is float and type(found) in (int, float) else found == value
        if not equal:
            errors.append(f"{path}{key}: expected {value!r}, actual {found!r}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-key", choices=["A2", "B1", "B2"], required=True)
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--mode", choices=["config", "smoke", "capacity"], required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument('--disable-kit-telemetry',action='store_true')
    # Isaac Lab must start before importing environment/task modules.
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.task_key == 'C2' or args.disable_kit_telemetry:
        # Headless Kit telemetry crashed in platform-info collection on gpu3-9.
        args.kit_args += ' --/app/extensions/excluded/0=omni.kit.telemetry'
    manifest = verify_source()
    report = {"task": args.task_key, "seed": args.seed, "mode": args.mode,
              "status": "running", "physics_or_training_executed": False,
              "author_binary_environment_reproduction_claimed": False}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    app = None
    env = None
    try:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        report['cuda_visible_devices']=os.environ.get('CUDA_VISIBLE_DEVICES')
        report['cuda_available']=torch.cuda.is_available()
        report['cuda_device_count']=torch.cuda.device_count()
        if not report['cuda_available'] or "A40" not in torch.cuda.get_device_name(0):
            raise RuntimeError("Allocated A40 is not CUDA-usable: "
                               f"host={platform.node()}, mask={report['cuda_visible_devices']}, "
                               f"cuda_available={report['cuda_available']}, device_count={report['cuda_device_count']}")
        report["gpu"] = torch.cuda.get_device_name(0)
        report["gpu_total_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
        report["python"] = platform.python_version()
        report["torch_settings"] = {"matmul_allow_tf32": True, "cudnn_allow_tf32": True,
                                    "cudnn_deterministic": False, "cudnn_benchmark": False}
        packages = {}
        for name in manifest["selected_package_versions"]:
            try:
                packages[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                version_file = Path("/isaac-sim/VERSION")
                standalone = version_file.read_text().strip() if version_file.exists() else ""
                if name != "isaacsim" or not re.match(r"5\.1\.0(?:\D|$)", standalone):
                    raise
                # Official standalone images need not provide the pip metapackage.
                # Record the actual VERSION evidence instead of installing another simulator.
                report["isaac_sim_standalone_version"] = standalone
                packages[name] = "5.1.0.0"
        report["packages"] = packages
        mismatches = [f"{k}: {packages[k]} != {v}" for k, v in manifest["selected_package_versions"].items() if packages[k] != v]
        if sys.version_info[:2] != (3, 11):
            mismatches.append("Requires Python 3.11")
        if mismatches:
            raise RuntimeError("Package mismatch: " + "; ".join(mismatches))
        app = AppLauncher(args).app
        from pxr import Usd
        report["isaacsim_usd_origin"] = str(Path(Usd.__file__).resolve())
        report["isaacsim_usd_version"] = list(Usd.GetVersion())
        if any(Path(Usd.__file__).resolve().is_relative_to(ROOT / folder)
               for folder in (".asset_tools", ".usd_converter")):
            raise RuntimeError("Asset inspection USD wheel must not override Isaac Sim's USD runtime")
        sys.path.insert(0, str(BENCH))
        import isaaclab
        import isaaclab_tasks
        import isaaclab_assets
        import isaaclab_rl
        import envs.registry
        for module in (isaaclab, isaaclab_tasks, isaaclab_assets, isaaclab_rl):
            origin = Path(module.__file__).resolve()
            if not origin.is_relative_to(LAB):
                raise RuntimeError(f"Editable package comes from an old project: {origin}")
        from isaaclab_tasks.utils import load_cfg_from_registry
        recipe = manifest["tasks"][args.task_key]
        argv = source_recipe(args.task_key, args.seed)
        env_cfg = load_cfg_from_registry(recipe["gym_id"], "env_cfg_entry_point")
        agent_cfg = load_cfg_from_registry(recipe["gym_id"], "rsl_rl_cfg_entry_point")
        env_cfg.scene.num_envs = int(flag(argv, "--num_envs"))
        env_cfg.seed = agent_cfg.seed = args.seed
        agent_cfg.max_iterations = int(flag(argv, "--max_iterations"))
        agent_cfg.experiment_name = flag(argv, "--experiment_name")
        agent_cfg.run_name = flag(argv, "--run_name")
        env_dict, agent_dict = normalize(env_cfg.to_dict()), normalize(agent_cfg.to_dict())
        report.update(env=env_dict, agent=agent_dict, source_cli=argv)
        errors = compare(env_dict, manifest["common_env_expectations"], "env.")
        errors += compare(env_dict, recipe["env_expectations"], "env.")
        errors += compare(agent_dict, manifest["agent_expectations"], "agent.")
        if env_cfg.scene.num_envs != recipe["num_envs"] or env_cfg.seed != args.seed or agent_cfg.seed != args.seed:
            errors.append("Environment count or seed mismatch")
        if agent_cfg.max_iterations != recipe["max_iterations"]:
            errors.append("Iteration budget mismatch")
        if env_cfg.scene.terrain.terrain_generator is not None:
            report["terrain_subtypes"] = list(env_cfg.scene.terrain.terrain_generator.sub_terrains)
            if report["terrain_subtypes"] != recipe["terrain_subtypes"]:
                errors.append("Terrain subtypes mismatch")
        report["checks"] = {"errors": errors, "training_transition_budget": recipe["num_envs"] * 24 * recipe["max_iterations"]}
        if errors:
            raise RuntimeError("Effective configuration differs from source expectations: " + "; ".join(errors))
        if args.mode in {"smoke", "capacity"}:
            import gymnasium as gym
            from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
            from rsl_rl.runners import OnPolicyRunner
            # Compatibility test only; any test checkpoints stay under smoke_runner/.
            test_num_envs = 64 if args.mode == "smoke" else recipe["num_envs"]
            env_cfg.scene.num_envs = test_num_envs
            env = gym.make(recipe["gym_id"], cfg=env_cfg)
            raw = env.unwrapped
            obs_dims = list(raw.observation_manager.group_obs_dim["policy"])
            if obs_dims != [235] or raw.action_manager.total_action_dim != 12:
                raise RuntimeError(f"Observation/action contract mismatch: {obs_dims}, {raw.action_manager.total_action_dim}")
            report["observation_terms"] = normalize(raw.observation_manager.active_terms)
            report["joint_names"] = list(raw.scene["robot"].joint_names)
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            # RSL-RL 3.1.2 learn() expects a logger when storing its source state.
            smoke_log = args.report.parent / f"{args.mode}_runner"
            smoke_log.mkdir(exist_ok=False)
            runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=str(smoke_log), device=agent_cfg.device)
            initial={k:v.detach().cpu().clone() for k,v in runner.alg.policy.state_dict().items()}
            start = time.perf_counter()
            runner.learn(num_learning_iterations=2, init_at_random_ep_len=True)
            elapsed = time.perf_counter() - start
            final=runner.alg.policy.state_dict()
            changed=any(not torch.equal(initial[k],v.detach().cpu()) for k,v in final.items())
            finite=all(torch.isfinite(v).all().item() for v in final.values())
            if not changed or not finite:raise RuntimeError('PPO weights unchanged or nonfinite')
            report["physics_or_training_executed"] = True
            report[args.mode] = {"num_envs": test_num_envs, "ppo_updates": 2, "elapsed_seconds": elapsed,
                               "changed_weights":changed,"finite_weights":finite,
                               "transitions": test_num_envs * 24 * 2,
                               "transitions_per_second": test_num_envs * 24 * 2 / elapsed,
                               "formal_training": False,
                               "full_environment_count_tested": args.mode == "capacity"}
            env.close()
            env = None
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        args.report.write_text(json.dumps(normalize(report), indent=2, ensure_ascii=False))
        if env is not None:
            env.close()
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
