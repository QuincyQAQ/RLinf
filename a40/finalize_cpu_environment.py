#!/usr/bin/env python3
"""Run CPU-only final checks in one container invocation; never launches Isaac Sim."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "validation"


def run(arguments, output=None):
    command = [sys.executable, *arguments]
    print("CPU CHECK:", arguments, flush=True)
    if output is None:
        subprocess.run(command, cwd=ROOT, check=True, timeout=300)
    else:
        with (VALIDATION / output).open("w") as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                           check=True, timeout=300)


def main():
    run(["a40/check_container_environment.py", "--verify", "--report",
         str(VALIDATION / "container_verify.json")])
    run(["a40/audit_container_dependencies.py"])
    run(["-m", "pip", "freeze"], "container_pip_freeze.txt")
    run(["-m", "pip", "inspect"], "container_pip_inspect.json")
    run(["a40/reference.py", "check"], "cluster_source_check.json")
    run(["-m", "unittest", "discover", "-s", "tests", "-v"], "cluster_offline_tests.txt")
    run(["a40/reference.py", "plan", "--seed", "0", "--python",
         str(ROOT / "a40/container_python.sh")], "cluster_seed0_plan.json")
    audit = json.loads((VALIDATION / "container_dependency_audit.json").read_text())
    issues = audit["active_environment"]["issues"]
    status = "cpu_environment_prepared_with_known_image_conflicts" if issues else "cpu_environment_prepared"
    report = {"status": status, "finished_unix": time.time(),
              "image": "nvcr.io/nvidia/isaac-lab:2.3.2",
              "gpu_used": False, "job_submitted": False,
              "simulation_started": False, "training_started": False,
              "core_package_verification": "passed", "offline_tests": "passed",
              "global_pip_check_clean": audit["global_pip_check_clean"],
              "unresolved_base_image_dependency_issues": audit["baseline_image"]["issues"],
              "unresolved_training_dependency_issues": issues,
              "overlay_introduced_dependency_issues": audit["introduced_issues"],
              "remaining": ["compute-node NVIDIA driver and Vulkan compatibility",
                            "asset resolution in the simulator", "effective environment configuration",
                            "64-env PPO smoke", "full-count PPO capacity check",
                            *(["base-image global dependency metadata conflicts remain documented"] if issues else [])]}
    (VALIDATION / "HPC_SETUP_STATUS.json").write_text(json.dumps(report, indent=2))
    mirror_file = VALIDATION / "asset_mirror.json"
    if mirror_file.exists():
        mirror = json.loads(mirror_file.read_text())
        if mirror["status"] == "passed":
            prestart = {"status": "non_simulation_preparation_complete", "finished_unix": time.time(),
                        "asset_files": mirror["file_count"], "asset_bytes": mirror["total_bytes"],
                        "core_status": status, "unresolved_dependency_issues": issues,
                        "simulation_started": False, "training_started": False,
                        "gpu_job_submitted": False, "remaining_requires_compute_node": report["remaining"]}
            (VALIDATION / "prestart_status.json").write_text(json.dumps(prestart, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
