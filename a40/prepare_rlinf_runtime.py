#!/usr/bin/env python3
"""Resolve missing RLinf dependencies using installed core packages, then isolate additions."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    overlay = ROOT / ".rlinf_overlay"
    overlay.mkdir(exist_ok=True)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(overlay) + os.pathsep + environment.get("PYTHONPATH", "")
    report = ROOT / "validation/rlinf_dependency_resolution.json"
    subprocess.run([sys.executable, "-m", "pip", "install", "--dry-run", "--report", str(report),
                    "-r", "config/rlinf-runtime.in", "-c", "config/constraints.txt"],
                   env=environment, cwd=ROOT, check=True, timeout=1200)
    resolution = json.loads(report.read_text())
    selected = [row["metadata"]["name"] + "==" + row["metadata"]["version"]
                for row in resolution["install"]]
    protected = {"torch", "torchvision", "numpy", "tensordict", "rsl-rl-lib", "gymnasium"}
    if any(row.split("==")[0].lower().replace("_", "-") in protected for row in selected):
        raise RuntimeError("RLinf dependencies would replace a protected training package: " + repr(selected))
    lock = ROOT / "config/rlinf-runtime.additions.lock.txt"
    lock.write_text("\n".join(selected) + "\n")
    if selected:
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--upgrade", "--target",
                        str(overlay), "-r", str(lock)], env=environment, check=True, timeout=2400)
    subprocess.run([sys.executable, "-m", "pip", "check"], env=environment, check=True, timeout=120)
    (ROOT / "validation/rlinf_dependency_overlay.json").write_text(json.dumps({
        "status": "passed", "finished_unix": time.time(), "added_packages": selected,
        "baseline_overlays_modified": False, "simulation_started": False}, indent=2))


if __name__ == "__main__":
    main()
