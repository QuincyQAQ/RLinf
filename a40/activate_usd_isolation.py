#!/usr/bin/env python3
"""Enable the exporter boundary only after both separate CPU environments pass."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    converter = json.loads((ROOT / "validation/usd_converter_check.json").read_text())
    assert converter["status"] == "passed" and converter["pip_check_clean"]
    environment = dict(os.environ)
    environment["PYTHONEXE"] = str(ROOT / "a40/training_python.sh")
    code = """
import importlib.metadata as m, importlib.util, sys
assert m.version('lxml') == '5.4.0'
assert not any('isaacsim.asset.exporter.urdf/pip_prebundle' in p for p in sys.path)
assert not any('.usd_converter' in p or '.asset_tools' in p for p in sys.path)
for name in ('nvidia-srl-usd','nvidia-srl-usd-to-urdf'):
    try:m.version(name)
    except m.PackageNotFoundError:pass
    else:raise RuntimeError('Standalone exporter still visible: '+name)
print('Training/exporter Python paths isolated; training lxml unchanged')
"""
    for args in (["-c", code], ["-m", "pip", "check"],
                 [str(ROOT / "a40/check_container_environment.py"), "--verify", "--report",
                  str(ROOT / "validation/container_verify.json")],
                 [str(ROOT / "a40/audit_container_dependencies.py")]):
        subprocess.run(["bash", "/isaac-sim/python.sh", *args], env=environment,
                       check=True, timeout=120)
    report = {"status": "passed", "finished_unix": time.time(),
              "training_pip_check_clean": True, "converter_pip_check_clean": True,
              "converter_actual_conversion_checked": True,
              "training_lxml": "5.4.0", "converter_lxml": converter["versions"]["lxml"],
              "excluded_training_prebundle": "/isaac-sim/exts/isaacsim.asset.exporter.urdf/pip_prebundle",
              "original_image_modified": False, "simulation_started": False,
              "gpu_used": False, "scope": "CPU process isolation; actual Kit/PhysX validation pending allocation"}
    (ROOT / "validation/usd_isolation_ready.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
