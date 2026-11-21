#!/usr/bin/env python3
"""Compare unmodified-image and overlay pip metadata without starting simulation.

The official image includes libraries for other tasks with unresolved metadata.
Do not replace its USD/IK/web stack just to make global pip check green. Preserve
the exact diagnostics and fail if this workspace introduces additional issues.
This comparison is not a claim that the existing image issues are all harmless.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def snapshot(environment):
    result = subprocess.run([sys.executable, "-m", "pip", "check"], env=environment,
                            capture_output=True, text=True, timeout=120)
    if result.returncode not in (0, 1) or result.stderr.strip():
        raise RuntimeError("pip check could not run: " + result.stderr)
    lines = sorted(line.strip() for line in result.stdout.splitlines()
                   if line.strip() and line.strip() != "No broken requirements found.")
    # Exit 1 without a dependency diagnostic is not an audited dependency check.
    if result.returncode == 1 and not lines:
        raise RuntimeError("pip check failed without dependency diagnostics")
    return {"return_code": result.returncode, "issues": lines, "stdout": result.stdout}


def main():
    baseline_environment = dict(os.environ)
    overlays = {(ROOT / name).resolve() for name in (".python_overlay", ".extra_overlay")}
    baseline_environment["PYTHONPATH"] = os.pathsep.join(
        path for path in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if path and Path(path).resolve() not in overlays)
    exporter = "/isaac-sim/exts/isaacsim.asset.exporter.urdf/pip_prebundle"
    if exporter not in baseline_environment["PYTHONPATH"].split(os.pathsep):
        baseline_environment["PYTHONPATH"] += os.pathsep + exporter
    baseline = snapshot(baseline_environment)
    active = snapshot(dict(os.environ))
    introduced = sorted(set(active["issues"]) - set(baseline["issues"]))
    report = {"status": "failed" if introduced else "no_new_dependency_conflicts",
              "baseline_image": baseline, "active_environment": active,
              "introduced_issues": introduced, "global_pip_check_clean": not active["issues"],
              "remaining_image_issues_resolved": False if active["issues"] else True,
              "original_image_modified": False,
              "optional_exporter_isolated": exporter not in sys.path,
              "resolution": "Standalone converter venv and exclusion of its optional prebundle from locomotion Python; original image retained",
              "scope": "CPU package metadata comparison only; no simulation or GPU validation",
              "gpu_used": False, "simulation_started": False}
    validation = ROOT / "validation"
    (validation / "container_dependency_audit.json").write_text(json.dumps(report, indent=2))
    (validation / "container_base_pip_check.txt").write_text(baseline["stdout"])
    (validation / "container_pip_check.txt").write_text(active["stdout"])
    print(json.dumps(report, indent=2))
    if introduced:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
