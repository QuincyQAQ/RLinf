#!/usr/bin/env python3
"""Execute Python arguments in the independent USD conversion venv (CPU only).

Example: bash a40/hpc_environment.sh python-cpu a40/usd_converter.py -m pip check
"""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    python = ROOT / ".usd_converter/bin/python"
    if not python.is_file():
        raise SystemExit("Prepare .usd_converter inside the official container first")
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["CUDA_VISIBLE_DEVICES"] = ""
    libs = ROOT / ".usd_converter/lib/python3.11/site-packages/usd_core.libs"
    environment["LD_LIBRARY_PATH"] = str(libs) + os.pathsep + environment.get("LD_LIBRARY_PATH", "")
    os.execve(str(python), [str(python), *sys.argv[1:]], environment)


if __name__ == "__main__":
    main()
