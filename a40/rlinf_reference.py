#!/usr/bin/env python3
"""Launch the original Isaac task with RLinf's native PPO workers and runner."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from reference import ROOT, read_manifest, stamp, verify_source, run_evaluation


def make_config(task, output, *, num_envs=None, updates=None):
    import yaml
    cfg = yaml.safe_load((ROOT / "RLinf/examples/embodiment/config/go2_mujoco_ppo_ascend.yaml").read_text())
    manifest = read_manifest()
    recipe = manifest["tasks"][task]
    count = num_envs or recipe["num_envs"]
    budget = updates or recipe["max_iterations"]
    cfg["env"]["train"] = {
        "env_type": "isaaclab", "task_key": task, "seed": 0, "total_num_envs": count,
        "init_params": {"id": recipe["gym_id"]}, "group_size": 1,
        "max_steps_per_rollout_epoch": 24, "max_episode_steps": 1000,
        "auto_reset": True, "ignore_terminations": False, "use_rel_reward": False,
        "reward_scale": 1.0, "use_fixed_reset_state_ids": False,
        "use_ordered_reset_state_ids": False, "is_eval": False, "rollout_epoch": 1,
        "validation_output": str(output / "environment"),
        "video_cfg": {"save_video": False, "video_base_dir": str(output / "video")}}
    cfg["env"]["eval"] = None
    cfg["actor"]["seed"] = 0
    cfg["actor"]["global_batch_size"] = count * 24 // 4
    cfg["actor"]["micro_batch_size"] = count * 24 // 4
    cfg["actor"]["model"].update(obs_dim=235, critic_obs_dim=235,
                                   initial_checkpoint=None, obs_normalization=False,
                                   distribution_safety_patch=False)
    cfg["actor"]["optim"]["lr_scheduler"] = "torch_constant"
    cfg["algorithm"].update(update_epoch=5, value_loss_type="mse", desired_kl=0.01, clip_ratio_c=None)
    cfg["cluster"]["component_placement"] = {"actor,env,rollout": "0"}
    cfg["runner"].update(max_epochs=budget, max_steps=budget, save_interval=50 if budget > 2 else 1,
                          val_check_interval=-1, resume_dir=None, ckpt_path=None)
    cfg["runner"]["logger"].update(log_path=str(output), experiment_name="isaaclab_" + task + "_s0")
    return cfg


def run_phase(task, output, python, num_envs=None, updates=None, initial_checkpoint=None):
    import yaml
    output.mkdir(parents=True)
    cfg = make_config(task, output, num_envs=num_envs, updates=updates)
    if initial_checkpoint is not None:
        cfg["actor"]["model"]["initial_checkpoint"] = str(initial_checkpoint)
    config_dir = output / "config"
    config_dir.mkdir()
    (config_dir / "experiment.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    command = [str(python), str(ROOT / "RLinf/examples/embodiment/train_embodied_agent.py"),
               "--config-path", str(config_dir), "--config-name", "experiment"]
    (output / "launch.json").write_text(json.dumps({"argv": command, "env_type": "isaaclab",
        "task": task, "seed": 0, "num_envs": cfg["env"]["train"]["total_num_envs"],
        "updates": cfg["runner"]["max_steps"], "learner": "RLinf native actor_critic PPO",
        "rsl_rl_runner_used": False, "initial_checkpoint": str(initial_checkpoint) if initial_checkpoint else None}, indent=2))
    with (output / "console.log").open("w") as log:
        environment = dict(os.environ)
        # Never discover another Slurm job through node-wide Ray discovery.
        environment["RAY_ADDRESS"] = "local"
        if "RAY_TMPDIR" in environment:
            ray_tmp = Path(environment["RAY_TMPDIR"]) / output.name
            ray_tmp.mkdir(parents=True, exist_ok=True)
            environment["RAY_TMPDIR"] = str(ray_tmp)
        phase_timeout = 1200 if output.name == "smoke" else (1800 if output.name == "capacity" else None)
        result = subprocess.run(command, cwd=ROOT / "RLinf", env=environment,
                                stdout=log, stderr=subprocess.STDOUT, timeout=phase_timeout)
    if result.returncode:
        raise RuntimeError(f"RLinf phase exited {result.returncode}; inspect {output / 'console.log'}")
    expected = cfg["runner"]["max_steps"]
    model = output / cfg['runner']['logger']['experiment_name'] / f"checkpoints/global_step_{expected}/actor/model_state_dict/full_weights.pt"
    if not model.is_file():
        raise RuntimeError("RLinf exited without the expected final native checkpoint: " + str(model))
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=list(read_manifest()["tasks"]), required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    verify_source()
    if args.plan:
        print(json.dumps(make_config(args.task, ROOT / "outputs/RLINF_PLAN"), indent=2))
        return
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("RLinf GPU execution requires a Slurm compute-node job")
    session = ROOT / "outputs" / ("rlinf_" + args.task + "_s0_" + stamp())
    session.mkdir(parents=True)
    (session / "progress.json").write_text(json.dumps({"stage": "config_check", "task": args.task}))
    probe = [str(args.python), str(ROOT / "a40/runtime_probe.py"), "--headless", "--mode", "config",
             "--task-key", args.task, "--seed", "0", "--report", str(session / "effective_config.json")]
    with (session / "config_check.log").open("w") as log:
        subprocess.run(probe, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
    config_report=json.loads((session/'effective_config.json').read_text())
    if config_report.get('status')!='passed':
        raise RuntimeError('Native configuration probe failed: '+str(config_report.get('error')))
    try:
        for stage, count, updates in (("smoke", 64, 2),
                                       ("capacity", read_manifest()["tasks"][args.task]["num_envs"], 2),
                                       ("training", None, None)):
            (session / "progress.json").write_text(json.dumps({"stage": stage, "task": args.task}))
            model = run_phase(args.task, session / stage, args.python, count, updates)
        # Convert the state-dict envelope only; reference evaluation never updates weights.
        import torch
        from rlinf.models.embodiment.go2_policy.go2_policy import Go2Policy
        state = torch.load(model, map_location="cpu", weights_only=False)
        policy = Go2Policy(obs_dim=235, critic_obs_dim=235, obs_normalization=False)
        policy.load_state_dict(state, strict=True)
        optimizer = torch.optim.Adam(policy.parameters(), lr=.001)
        checkpoint = session / "rlinf_policy_for_reference_eval.pt"
        torch.save({"model_state_dict": state, "optimizer_state_dict": optimizer.state_dict(),
                    "iter": read_manifest()["tasks"][args.task]["max_iterations"],
                    "infos": {"learner": "RLinf PPO", "inference_export_only": True,
                              "resumable_native_checkpoint": str(model)}}, checkpoint)
        (session / "progress.json").write_text(json.dumps({"stage": "main_task_evaluation", "task": args.task}))
        environment = dict(os.environ, GO2_ASSET_SOURCE="mirror")
        run_evaluation(args.task, 0, checkpoint, args.python, session, environment, [args.task])
        evaluation = json.loads((session / args.task / f"verified_{args.task}.json").read_text())
        (session / "progress.json").write_text(json.dumps({"stage": "training_and_tracking_eval_completed", "task": args.task,
            "checkpoint": str(model), "physical_passage_evaluation": evaluation["physical_evaluation_status"],
            "required_obstacle_metrics_complete": evaluation["required_obstacle_metrics_complete"]}, indent=2))
    except BaseException as exc:
        (session / "failure.json").write_text(json.dumps({"error": repr(exc)}, indent=2))
        raise


if __name__ == "__main__":
    main()
