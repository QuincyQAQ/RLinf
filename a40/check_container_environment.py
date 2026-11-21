#!/usr/bin/env python3
"""CPU-only package and source-path validation; never starts SimulationApp or CUDA work."""
import argparse
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    expected = json.loads((ROOT / "config/reference.json").read_text())["selected_package_versions"]
    packages, missing = {}, []
    for name in expected:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
    standalone = None
    version_file = Path("/isaac-sim/VERSION")
    if version_file.exists():
        standalone = version_file.read_text().strip()
    release = re.match(r"5\.1\.0(?:\D|$)", standalone or "") is not None
    errors = []
    tool_path_leaked = any((ROOT / ".asset_tools").resolve() == Path(path).resolve()
                          or (ROOT / ".usd_converter").resolve() in Path(path).resolve().parents
                          for path in sys.path if path)
    if tool_path_leaked:
        errors.append("Isolated asset USD tools leaked into the training Python path")
    for name, version in expected.items():
        if name == "isaacsim" and name in missing and release:
            continue
        if packages.get(name) != version:
            errors.append(f"{name}: expected {version}, found {packages.get(name)}")
    if sys.version_info[:2] != (3, 11):
        errors.append("Expected Python 3.11")
    origins = {}
    for name in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl"):
        spec = importlib.util.find_spec(name)
        origins[name] = spec.origin if spec else None
        if spec is None or not Path(spec.origin).resolve().is_relative_to(ROOT / "IsaacLab"):
            errors.append(f"{name} does not resolve to the preserved source: {origins[name]}")
    # Only CPU modules/kernels. Do not query CUDA devices or import Isaac Sim's application.
    cpu = {}
    try:
        import torch
        import numpy
        import tensordict
        from rsl_rl.runners import OnPolicyRunner
        torch.set_num_threads(1)
        value = (torch.arange(4, dtype=torch.float32) ** 2).sum()
        cpu = {"torch_cpu_sum": value.item(), "tensor_device": str(value.device),
               "numpy_version": numpy.__version__, "rsl_runner_import": OnPolicyRunner.__name__,
               "tensordict_version": tensordict.__version__}
    except Exception as exc:
        errors.append(f"CPU import/kernel check failed: {type(exc).__name__}: {exc}")
    report = {"status": "passed" if not errors else "mismatches", "errors": errors,
              "packages": packages, "metadata_missing": missing,
              "isaac_sim_standalone_version": standalone,
              "python": sys.version, "libc": platform.libc_ver(), "origins": origins,
              "cpu_checks": cpu, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "gpu_used": False, "simulation_started": False, "job_submitted": False}
    report["isolated_asset_tools_on_training_path"] = tool_path_leaked
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "origins"}, indent=2))
    if args.verify and errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
