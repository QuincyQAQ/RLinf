#!/usr/bin/env python3
"""Prepare and run the author's specialist recipes without porting the learner/MDP."""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmark"
LAB = ROOT / "IsaacLab"
MANIFEST = ROOT / "config/reference.json"
SOURCES = {'A2': 'train_a2_baseline_alice.sbatch', 'B1': 'train_b1_baseline_alice.sbatch', 'B2': 'train_b2_baseline_posture_alice.sbatch'}
SHORT = {'A2': 'A2_omni', 'B1': 'B1_rough', 'B2': 'B2_stairs'}


def read_manifest():
    return json.loads(MANIFEST.read_text())


def source_recipe(task: str, seed: int) -> list[str]:
    """Read the actual final train command, not prose examples or old port defaults."""
    path = BENCH / "scripts/slurm" / SOURCES[task]
    text = path.read_text()
    variables = {"SEED": str(seed), "PLOT_SMOOTHING": "25"}
    for name, line in re.findall(r'^([A-Z_]+)="([^\n]*)"$', text, re.M):
        default = re.fullmatch(r'\$\{[A-Z_]+:-([^}]*)\}', line)
        if name in {"TASK_ID", "NUM_ENVS", "MAX_ITERATIONS", "EXPERIMENT_NAME", "RUN_NAME_PREFIX"}:
            variables[name] = default.group(1) if default else line
    marker = "-p scripts/train.py"
    command = text[text.rindex(marker) + len(marker):].strip()
    if not command.endswith('"'):
        raise ValueError(f"Unexpected source command terminator: {path}")
    command = command[:-1].replace("\\\n", " ")
    # C2's source default is scratch. A2 initialization is an optional shell variable.
    command = command.replace("\\$PRETRAINED_ARGS", "")
    command = re.sub(r'\$\{([A-Z_]+)(?::-([^}]*))?\}', lambda m: variables.get(m[1], m[2] or m[0]), command)
    command = re.sub(r'\$([A-Z_]+)', lambda m: variables.get(m[1], m[0]), command)
    if "$" in command:
        raise ValueError(f"Unresolved source variable in {path}: {command}")
    return shlex.split(command)


def flag(args, name):
    return args[args.index(name) + 1]


def verify_source():
    manifest = read_manifest()
    errors = []
    for directory, revision in ((BENCH, manifest["benchmark_commit"]), (LAB, manifest["isaaclab_commit"])):
        result = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True, capture_output=True)
        if result.returncode or result.stdout.strip() != revision:
            errors.append(f"Unexpected revision: {directory}")
        # Include untracked .py files: task discovery can import these too.
        result = subprocess.run(["git", "-C", str(directory), "status", "--porcelain", "--untracked-files=all"], text=True, capture_output=True)
        changes = [line for line in result.stdout.splitlines() if not line.endswith(".pyc")]
        if result.returncode or changes:
            errors.append(f"Source changes need review in {directory}: {changes[:10]}")
    for task in SOURCES:
        for seed in manifest["training_seeds"]:
            if source_recipe(task, seed) != manifest["tasks"][task]["commands_by_seed"][str(seed)]:
                errors.append(f"Source/manifest CLI mismatch: {task}, seed {seed}")
    if errors:
        raise RuntimeError("\n".join(errors))
    return manifest


def training_argv(task: str, seed: int, python: Path):
    entry = ROOT / "a40/train_legacy_distribution.py" if task == "B1" else BENCH / "scripts/train.py"
    return [str(python), str(entry), *source_recipe(task, seed)]


def evaluation_argv(task: str, seed: int, checkpoint: Path, python: Path, output: Path, eval_task: str | None = None):
    m = read_manifest()
    target = eval_task or task
    # Published evaluator uses non-Legacy A1 even for a Legacy-trained A1 policy.
    evaluator = ROOT / "a40/evaluate_dual.py" if target in {"B2", "C2"} else BENCH / "scripts/evaluate.py"
    return [str(python), str(evaluator),
            "--task", m["tasks"][target]["eval_gym_id"],
            "--checkpoint", str(checkpoint.resolve()),
            "--report_train_task", f"{SHORT[task]}_s{seed}", "--headless",
            "--num_envs", "256", "--num_episodes", "1024",
            "--output_dir", str(output.resolve()), "--eval_task", SHORT[target]]


def stamp():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")


def asset_argv(argv, source):
    """Change only asset file access when explicitly selecting the prepared mirror."""
    result = list(argv)
    if source == "mirror":
        mirror = ROOT / "assets/isaac_sim_5.1"
        result.append("--kit_args=--/persistent/isaac/asset_root/cloud=" + str(mirror))
    return result


def run_recorded(argv, directory, env):
    argv = asset_argv(argv, env.get("GO2_ASSET_SOURCE", "cloud"))
    directory.mkdir(parents=True, exist_ok=False)
    record = {"argv": argv, "cwd": str(BENCH), "start": stamp(), "returncode": None,
              "asset_source": env.get("GO2_ASSET_SOURCE", "cloud")}
    (directory / "launch.json").write_text(json.dumps(record, indent=2))
    print(shlex.join(argv), flush=True)
    with (directory / "console.log").open("w") as log:
        result = subprocess.run(argv, cwd=BENCH, env=env, stdout=log, stderr=subprocess.STDOUT)
    record.update(returncode=result.returncode, end=stamp())
    (directory / "launch.json").write_text(json.dumps(record, indent=2))
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}). Inspect {directory / 'console.log'}")
    # Kit can swallow a Python exception and still return zero at shutdown.
    # Require the probe's explicit result before proceeding to long training.
    if any(str(value).endswith('/runtime_probe.py') for value in argv):
        report_path=Path(argv[argv.index('--report')+1])
        data=json.loads(report_path.read_text())
        if data.get('status')!='passed':
            raise RuntimeError(f"Runtime probe failed: {data.get('error',data.get('status'))}")


def run_training(task, seed, python, session, env):
    args = source_recipe(task, seed)
    probe = [str(python), str(ROOT / "a40/runtime_probe.py"), "--headless",
             "--task-key", task, "--seed", str(seed), "--mode", "config",
             "--report", str(session / task / "effective_config.json")]
    run_recorded(probe, session / task / "config_check", env)
    # Smoke is a separate process/run; formal source counts and initial seed are unchanged.
    smoke = [str(python), str(ROOT / "a40/runtime_probe.py"), "--headless",
             "--task-key", task, "--seed", str(seed), "--mode", "smoke",
             "--report", str(session / task / "smoke.json")]
    run_recorded(smoke, session / task / "smoke_check", env)
    capacity = [str(python), str(ROOT / "a40/runtime_probe.py"), "--headless",
                "--task-key", task, "--seed", str(seed), "--mode", "capacity",
                "--report", str(session / task / "capacity.json")]
    run_recorded(capacity, session / task / "capacity_check", env)
    log_root = BENCH / "logs/rsl_rl" / flag(args, "--experiment_name")
    run_suffix = "_" + flag(args, "--run_name")
    before = {p for p in log_root.glob("*") if p.name.endswith(run_suffix)}
    run_recorded(training_argv(task, seed, python), session / task / "training", env)
    new_runs = [p for p in set(log_root.glob("*")) - before
                if p.is_dir() and p.name.endswith(run_suffix)]
    if len(new_runs) != 1:
        raise RuntimeError(f"Cannot identify exactly one new run: {new_runs}")
    for name in ("env.yaml", "agent.yaml"):
        if not (new_runs[0] / "params" / name).is_file():
            raise RuntimeError(f"Original trainer did not save {name}")
    models = list(new_runs[0].glob("model_*.pt"))
    if not models:
        raise RuntimeError("Training exited without checkpoints")
    # Upstream repeated learn(25) calls can reuse iteration labels. Never invent model_2199.pt.
    latest = max(models, key=lambda p: p.stat().st_mtime_ns)
    (session / task / "artifacts.json").write_text(json.dumps({
        "run": str(new_runs[0]), "checkpoint": str(latest),
        "requested_updates": int(flag(args, "--max_iterations")),
        "checkpoint_number_is_not_assumed_to_equal_updates": True,
        "main_task_eval": "pending",
    }, indent=2))
    return latest


def run_evaluation(task, seed, checkpoint, python, session, env, targets):
    (session / task).mkdir(parents=True, exist_ok=True)
    for target in targets:
        run_recorded(evaluation_argv(task, seed, checkpoint, python, session / "results", target),
                     session / task / f"eval_{target}", env)
        artifact = session / "results" / f"{SHORT[task]}_s{seed}" / f"{SHORT[target]}.json"
        result = json.loads(artifact.read_text())
        episodes, summary = result["episodes"], result["summary"]
        if len(episodes) < 1024 or summary["num_episodes"] != len(episodes):
            raise RuntimeError(f"Insufficient/inconsistent evaluation episodes in {artifact}")
        successes = sum(bool(row["episode_success"]) for row in episodes)
        failures = sum(row["termination_reason"] == "failure" for row in episodes)
        if abs(summary["success_rate"] - successes / len(episodes)) > 1e-10:
            raise RuntimeError(f"Summary success rate inconsistent with episodes in {artifact}")
        if abs(summary["failure_rate"] - failures / len(episodes)) > 1e-10:
            raise RuntimeError(f"Summary failure rate inconsistent with episodes in {artifact}")
        physical = {
            "physical_obstacle_passage_rate": None,
            "physical_evaluation_status": "pending_not_measured" if target in {"B2", "C2"} else "not_applicable",
            "joint_tracking_and_passage_success_rate": None,
            "required_obstacle_metrics_complete": target not in {"B2", "C2"},
        }
        if target in {"B2", "C2"} and any("physical_obstacle_passage" in row for row in episodes):
            from passage_metrics import summarize_passage
            physical = summarize_passage(episodes)
            geometry = artifact.with_suffix(".geometry.json")
            if not geometry.is_file():
                raise RuntimeError("Passage evaluation did not save its actual terrain geometry")
            geometry_data = json.loads(geometry.read_text())
            if geometry_data.get("task") != target or not geometry_data.get("gpu_runtime_verified"):
                raise RuntimeError("Passage geometry was not verified on the requested GPU task")
            if Path(geometry_data.get("checkpoint", "")).resolve() != checkpoint.resolve():
                raise RuntimeError("Tracking and passage geometry refer to different checkpoints")
            for row in episodes:
                evidence = row["passage_evidence"]
                tile = geometry_data["tiles"].get(evidence["geometry_key"])
                if not tile or abs(tile["outer_radius"] - evidence["required_outer_radius_m"]) > 1e-6:
                    raise RuntimeError("Episode passage boundary does not match captured geometry")
            for key, value in physical.items():
                if summary.get(key) != value:
                    raise RuntimeError(f"Physical summary/episodes mismatch for {key} in {artifact}")
        (session / task / f"verified_{target}.json").write_text(json.dumps({
            "artifact": str(artifact), "successes": successes, "episodes": len(episodes),
            "failures": failures, "benchmark_tracking_success_rate": successes / len(episodes),
            **physical,
            "explicit_environment_eval_seed": None,
        }, indent=2))
    marker = session / task / "artifacts.json"
    if marker.exists():
        data = json.loads(marker.read_text())
        data["main_task_eval"] = "completed" if task in targets else "not_requested"
        marker.write_text(json.dumps(data, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["check", "plan", "train", "suite", "eval"])
    parser.add_argument("--task", choices=list(SOURCES), default="A2")
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--python", type=Path, default=ROOT / ".venv/bin/python")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--scope", choices=["main", "matrix"], default="main")
    parser.add_argument("--asset-source", choices=["cloud", "mirror"], default="cloud",
                        help="Keep original URLs by default; mirror changes only asset file access")
    parser.add_argument("--execute", action="store_true", help="Run on the selected A40 machine; default only prints the plan")
    args = parser.parse_args()
    args.python = args.python.resolve()
    manifest = verify_source()
    if args.mode == "check":
        print(json.dumps({"status": "source_and_cli_checks_passed", "runtime_verified": False,
                          "author_container_lock_available": False}, indent=2))
        return
    tasks = list(SOURCES) if args.mode in {"suite", "plan"} else [args.task]
    plan = {"profile": "author_slurm_defaults", "seed": args.seed, "scope": args.scope,
            "evaluation_reporting_requirements": manifest["evaluation"].get("reporting_requirements", {}),
            "asset_source": args.asset_source,
            "learner": "RSL-RL PPO", "physics": "Isaac Sim PhysX CUDA",
            "commands": {task: asset_argv(training_argv(task, args.seed, args.python), args.asset_source) for task in tasks},
            "source_unknowns": manifest["unknowns"]}
    if args.mode == "plan" or (args.mode in {"train", "suite"} and not args.execute):
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return
    if args.mode == "eval":
        if not args.checkpoint or not args.checkpoint.is_file():
            parser.error("eval requires an existing --checkpoint")
        targets = list(SOURCES) if args.scope == "matrix" else [args.task]
        if not args.execute:
            for target in targets:
                print(shlex.join(asset_argv(evaluation_argv(args.task, args.seed, args.checkpoint, args.python,
                                                ROOT / "outputs/EVAL_PLAN", target), args.asset_source)))
            return
    if args.asset_source == "mirror":
        receipt = ROOT / "validation/asset_mirror.json"
        if not receipt.is_file() or json.loads(receipt.read_text()).get("status") != "passed":
            parser.error("The selected asset mirror has not passed CPU preparation")
    if not args.python.is_file():
        parser.error(f"Python not found: {args.python}; prepare the isolated environment on A40 first")
    # Reject accidental launch on the local 3070. Probe also checks CUDA/Vulkan details.
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True)
    if "A40" not in gpu:
        parser.error(f"This prepared entrypoint expects an A40 machine, found {gpu.strip()}")
    (ROOT / "outputs").mkdir(exist_ok=True)
    array_id = os.environ.get("SLURM_ARRAY_JOB_ID", "")
    if array_id and args.mode != "train":
        parser.error("Managed array jobs must run one task/seed, not a suite or evaluation")
    lock_name = f".array_{args.task}_s{args.seed}.lock" if os.environ.get('SLURM_JOB_ID') else ".single_gpu.lock"
    run_lock = (ROOT / "outputs" / lock_name).open("w")
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("Another job already owns this workspace task/seed or exclusive GPU session")
    session = ROOT / "outputs" / f"{args.mode}_s{args.seed}_{stamp()}"
    session.mkdir(parents=True, exist_ok=False)
    (session / "plan.json").write_text(json.dumps(plan, indent=2))
    env = os.environ.copy()
    # The source Slurm launcher explicitly exports these same application settings.
    env["ACCEPT_EULA"] = "Y"
    env["PRIVACY_CONSENT"] = "Y"
    env["GO2_ASSET_SOURCE"] = args.asset_source
    # Do not change upstream random, TF32, OpenMP, noise or CPU threading settings here.
    env["PYTHONPATH"] = str(BENCH) + os.pathsep + env.get("PYTHONPATH", "")
    if args.mode == "eval":
        run_evaluation(args.task, args.seed, args.checkpoint, args.python, session, env, targets)
    else:
        for task in tasks:
            checkpoint = run_training(task, args.seed, args.python, session, env)
            targets = list(SOURCES) if args.scope == "matrix" else [task]
            run_evaluation(task, args.seed, checkpoint, args.python, session, env, targets)
    print(f"Completed session: {session}")


if __name__ == "__main__":
    main()
