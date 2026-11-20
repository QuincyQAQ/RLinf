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

"""Restore and audit the recorded A40 experiment without allocating a GPU."""

from __future__ import annotations

import argparse
import contextlib
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
FRAMEWORK = Path(__file__).resolve().parents[3]
TASKS = {
    "A1": "A1_forward",
    "A2": "A2_omni",
    "B1": "B1_rough",
    "B2": "B2_stairs",
    "C2": "C2_gap",
}
RUNS = {
    "A1": "fullppo_huber_A1_14029742",
    "A2": "fullppo_huber_A2_14029743",
    "B1": "fullppo_huber_B1_14032301",
    "B2": "b2_postbaseline_14025574",
    "C2": "fullppo_huber_C2_14029748",
}


def digest(path: Path) -> str:
    """Return a file's SHA256 digest."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def unpack(archive: Path, destination: Path) -> None:
    """Extract regular files after rejecting traversal, links and overwrites."""
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        seen = set()
        for member in members:
            relative = Path(member.name)
            target = destination / relative
            if relative.is_absolute() or ".." in relative.parts or not member.isfile():
                raise ValueError(
                    f"Expected a relative regular file, found {member.name!r}"
                )
            if member.name in seen or target.exists():
                raise FileExistsError(
                    f"Refusing to overwrite archive destination {target}"
                )
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError(f"Archive path escapes its destination: {member.name}")
            seen.add(member.name)
        for member in members:
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.extractfile(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
            target.chmod(member.mode & 0o777)


def verify_integrity(package: Path) -> int:
    """Check every published source, evidence, report and checkpoint blob."""
    entries = json.loads((package / "integrity.json").read_text())
    for relative, expected in entries.items():
        actual = digest(package / relative)
        if actual != expected:
            raise ValueError(f"SHA256 mismatch for {relative}: {actual} != {expected}")
    return len(entries)


def verify_evaluation(data: dict) -> int:
    """Recalculate the original summary from every recorded episode."""
    episodes, summary = data["episodes"], data["summary"]
    count = len(episodes)
    if count == 0 or count != summary["num_episodes"]:
        raise ValueError("Episode count does not match the saved evaluation summary")
    expected = {
        "success_rate": sum(bool(v["episode_success"]) for v in episodes) / count,
        "failure_rate": sum(v["termination_reason"] == "failure" for v in episodes)
        / count,
    }
    for source, target in [
        ("success_step_ratio", "mean_success_step_ratio"),
        ("mean_lin_vel_error_xy", "mean_lin_vel_error"),
        ("mean_ang_vel_error_z", "mean_ang_vel_error"),
        ("alive_time_s", "mean_alive_time_s"),
    ]:
        expected[target] = sum(v[source] for v in episodes) / count
    if "physical_obstacle_passage_rate" in summary:
        passage = sum(bool(v["physical_obstacle_passage"]) for v in episodes)
        joint = sum(bool(v["joint_tracking_and_passage"]) for v in episodes)
        expected.update(
            physical_passage_successes=passage,
            physical_obstacle_passage_rate=passage / count,
            joint_tracking_and_passage_success_rate=joint / count,
        )
        for episode in episodes:
            if bool(episode["joint_tracking_and_passage"]) != (
                bool(episode["episode_success"])
                and bool(episode["physical_obstacle_passage"])
            ):
                raise ValueError(
                    "Tracking/passage intersection is inconsistent within an episode"
                )
    for key, value in expected.items():
        if not math.isfinite(value) or not math.isclose(
            value, summary[key], rel_tol=1e-11, abs_tol=1e-11
        ):
            raise ValueError(
                f"Evaluation summary mismatch for {key}: {summary[key]} != {value}"
            )
    return count


def export_csv(package: Path, destination: Path) -> int:
    """Regenerate the 29-row table from retained final experiment results."""
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        for path in (package / "reports").glob("*.json"):
            shutil.copy2(path, output / path.name)
        script = package / "reports/export_five_task_csv.py"
        spec = importlib.util.spec_from_file_location("a40_export", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.ROOT = output
        with contextlib.redirect_stdout(io.StringIO()):
            module.main()
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output / "A40_five_tasks_complete_20261121.csv", destination)
    with destination.open(encoding="utf-8-sig", newline="") as stream:
        return len(list(csv.DictReader(stream)))


def verify_checkpoint_files(package: Path) -> list[dict]:
    """Load archived weights on CPU and require the recorded 235-to-12 policy."""
    import torch

    torch.set_num_threads(4)
    shapes = {"std": (12,)}
    for side, sizes in [
        ("actor", [235, 512, 256, 128, 12]),
        ("critic", [235, 512, 256, 128, 1]),
    ]:
        for index, (inputs, outputs) in enumerate(zip(sizes, sizes[1:])):
            shapes[f"{side}.{2 * index}.weight"] = (outputs, inputs)
            shapes[f"{side}.{2 * index}.bias"] = (outputs,)
    results = []
    for item in json.loads((package / "selection.json").read_text())["checkpoints"]:
        path = package / "checkpoints" / item["path"]
        saved = torch.load(path, map_location="cpu", weights_only=False)
        state = saved.get("model_state_dict", saved)
        if set(state) != set(shapes) or any(
            tuple(state[key].shape) != shape for key, shape in shapes.items()
        ):
            raise ValueError(
                f"Checkpoint differs from the frozen 235-observation policy schema: {item['path']}"
            )
        if not all(torch.isfinite(tensor).all() for tensor in state.values()):
            raise ValueError(f"Nonfinite checkpoint: {item['path']}")
        if not (state["std"] > 0).all():
            raise ValueError(f"Nonpositive standard deviation: {item['path']}")
        with torch.inference_mode():
            outputs = {}
            for side in ["actor", "critic"]:
                value = torch.zeros(4, 235)
                for index in range(4):
                    value = torch.nn.functional.linear(
                        value,
                        state[f"{side}.{2 * index}.weight"],
                        state[f"{side}.{2 * index}.bias"],
                    )
                    if index < 3:
                        value = torch.nn.functional.elu(value)
                outputs[side] = value
            actions, values = outputs["actor"], outputs["critic"]
        if (
            actions.shape != (4, 12)
            or not torch.isfinite(actions).all()
            or not torch.isfinite(values).all()
        ):
            raise ValueError(f"Invalid CPU policy output for {item['path']}")
        results.append(
            {
                "path": item["path"],
                "finite": True,
                "strict_schema": True,
                "positive_std": True,
                "saved_iteration": saved.get("iter"),
            }
        )
    return results


def audit(package: Path, checkpoints: bool = False) -> dict:
    """Verify frozen source, raw episodes, final summaries and exported table."""
    blobs = verify_integrity(package)
    inventory = json.loads((package / "canonical_manifest.json").read_text())
    selection = json.loads((package / "selection.json").read_text())
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        source, evidence = temporary / "source", temporary / "evidence"
        unpack(package / "canonical_source.tar.gz", source)
        unpack(package / "evidence.tar.gz", evidence)
        for item in selection["source"]:
            if digest(source / item["path"]) != item["sha256"]:
                raise ValueError(f"Frozen source mismatch for {item['path']}")
        for item in selection["evidence"]:
            if digest(evidence / item["path"]) != item["sha256"]:
                raise ValueError(f"Frozen evidence mismatch for {item['path']}")
        evaluations = episodes = 0
        for path in evidence.rglob("*.json"):
            data = json.loads(path.read_text())
            if isinstance(data, dict) and "episodes" in data and "summary" in data:
                episodes += verify_evaluation(data)
                evaluations += 1
                if (
                    "physical_obstacle_passage_rate" in data["summary"]
                    and not path.with_suffix(".geometry.json").is_file()
                ):
                    raise ValueError(f"Missing actual obstacle geometry for {path}")
                relative = path.relative_to(evidence)
                container_path = "/workspace/go2/" + str(relative)
                # Each evaluator receipt must identify this exact episode artifact.
                receipt = path.parent.parent / "verified.json"
                if receipt.is_file():
                    verified = json.loads(receipt.read_text())
                    if (
                        verified.get("artifact") != container_path
                        or verified.get("summary") != data["summary"]
                    ):
                        raise ValueError(
                            f"Evaluation receipt disagrees with raw episodes: {relative}"
                        )
        final = json.loads(
            (package / "reports/five_task_final_20261120.json").read_text()
        )
        for task, folder in RUNS.items():
            actual = json.loads(
                (evidence / "outputs" / folder / "result.json").read_text()
            )
            if (
                actual != final["results"][task]["data"]
                or actual["status"] != "verified"
            ):
                raise ValueError(f"Final report diverged from remote result for {task}")
            for lane, seeds in actual["results"].items():
                for seed, expected in seeds.items():
                    middle = (
                        "holdout_baseline"
                        if task == "B2" and lane == "baseline"
                        else lane
                    )
                    if lane != "baseline":
                        middle += "/evaluation"
                    path = (
                        evidence
                        / "outputs"
                        / folder
                        / middle
                        / seed
                        / (TASKS[task] + "_s0")
                        / (TASKS[task] + ".json")
                    )
                    if json.loads(path.read_text())["summary"] != expected:
                        raise ValueError(
                            f"Final lane summary mismatch: {task}/{lane}/{seed}"
                        )
        for folder, report in [
            ("a1_confirmatory_14032589", "completed_a1_confirmatory_14032589.json"),
            ("b1_confirmatory_14036295", "completed_b1_confirmatory_14036295.json"),
            ("a1_policy_tune_14034565", "completed_a1_policy_tune_14034565.json"),
        ]:
            actual = json.loads(
                (evidence / "outputs" / folder / "result.json").read_text()
            )
            expected = json.loads((package / "reports" / report).read_text())
            if actual != expected or actual["status"] != "verified":
                raise ValueError(f"Supplementary report mismatch: {folder}")
        csv_path = temporary / "regenerated.csv"
        rows = export_csv(package, csv_path)
        if (
            csv_path.read_bytes()
            != (package / "reports/A40_five_tasks_complete_20261121.csv").read_bytes()
        ):
            raise ValueError(
                "CSV differs from the table regenerated from final results"
            )
    return {
        "status": "passed",
        "source_authority": inventory["root"],
        "source_capture_time": inventory["captured_at"],
        "source_files": len(selection["source"]),
        "integrity_files": blobs,
        "evaluations": evaluations,
        "episodes": episodes,
        "csv_rows": rows,
        "checkpoints": verify_checkpoint_files(package)
        if checkpoints
        else "not requested",
        "gpu_allocations": 0,
        "new_simulation_runs": 0,
        "new_training_runs": 0,
    }


def restore(package: Path, workspace: Path, reference: Path | None = None) -> None:
    """Materialize the A40 directory layout and pinned source repositories."""
    verify_integrity(package)
    if workspace.exists() and any(workspace.iterdir()):
        raise FileExistsError(
            f"Expected an empty workspace, found existing files in {workspace}"
        )
    workspace.mkdir(parents=True, exist_ok=True)
    inventory = json.loads((package / "canonical_manifest.json").read_text())
    remotes = {
        "RLinf": str(FRAMEWORK),
        "benchmark": "https://github.com/viktoriageorrg04/bsc-thesis-multi-task-RL-env.git",
        "IsaacLab": "https://github.com/isaac-sim/IsaacLab.git",
    }
    for name, info in inventory["repositories"].items():
        origin = str(reference / name) if reference else remotes[name]
        target = workspace / name
        subprocess.run(
            ["git", "clone", "--no-hardlinks", "--no-checkout", origin, str(target)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(target), "checkout", "--detach", info["commit"]],
            check=True,
        )
        for item in info["tracked"]:
            path = workspace / item["path"]
            if "sha256" in item and digest(path) != item["sha256"]:
                raise ValueError(f"Pinned dependency checkout mismatch: {item['path']}")
    unpack(package / "canonical_source.tar.gz", workspace)
    unpack(package / "evidence.tar.gz", workspace)
    unpack(package / "training_logs.tar.gz", workspace / "archived_logs")
    for compressed in (workspace / "archived_logs").rglob("*.gz"):
        destination = workspace / compressed.relative_to(
            workspace / "archived_logs"
        ).with_suffix("")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(compressed, "rb") as source, destination.open("xb") as output:
            shutil.copyfileobj(source, output)
    for item in json.loads((package / "selection.json").read_text())["checkpoints"]:
        destination = workspace / item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(package / "checkpoints" / item["path"], destination)
        os.utime(destination, ns=(item["mtime_ns"], item["mtime_ns"]))
    shutil.copytree(package / "reports", workspace / "reports", dirs_exist_ok=True)
    shutil.copy2(package / "README.md", workspace / "README.md")


def main() -> None:
    """Run an explicit offline audit, CSV export or workspace restoration."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("audit")
    check.add_argument("--checkpoints", action="store_true")
    check.add_argument("--output", type=Path, required=True)
    export = commands.add_parser("export")
    export.add_argument("--output", type=Path, required=True)
    materialize = commands.add_parser("restore")
    materialize.add_argument("--workspace", type=Path, required=True)
    materialize.add_argument("--reference-workspace", type=Path)
    args = parser.parse_args()
    if args.command == "audit":
        result = audit(PACKAGE, args.checkpoints)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {k: v for k, v in result.items() if k != "checkpoints"}, indent=2
            )
        )
    elif args.command == "export":
        print(
            json.dumps(
                {"rows": export_csv(PACKAGE, args.output), "output": str(args.output)}
            )
        )
    else:
        restore(PACKAGE, args.workspace.resolve(), args.reference_workspace)
        print(
            json.dumps(
                {
                    "workspace": str(args.workspace),
                    "gpu_allocations": 0,
                    "training_started": False,
                }
            )
        )


if __name__ == "__main__":
    main()
