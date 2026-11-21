#!/usr/bin/env python3
"""Check the auxiliary dependency layer in a child process before activating it."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / ".extra_overlay") + os.pathsep + environment.get("PYTHONPATH", "")
    checks = r'''
import json, importlib.metadata as metadata
import numpy as np
from fastapi import FastAPI
from lxml import etree
import quadprog, docstring_parser, jwt
app=FastAPI()
@app.get('/prestart')
def prestart():return {'ok': True}
assert '/prestart' in app.openapi()['paths']
assert etree.fromstring(b'<prestart/>').tag == 'prestart'
assert np.allclose(quadprog.solve_qp(np.eye(2), np.ones(2))[0], np.ones(2))
assert docstring_parser.parse('Prestart CPU check.').short_description == 'Prestart CPU check.'
key='local-cpu-check-key-with-32-characters'
assert jwt.decode(jwt.encode({'ok':True},key,algorithm='HS256'),key,algorithms=['HS256'])['ok']
print(json.dumps({name:metadata.version(name) for name in ['fastapi','starlette','lxml','quadprog','docstring-parser','PyJWT']},indent=2))
'''
    result = subprocess.run([sys.executable, "-c", checks], env=environment,
                            capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    versions = json.loads(result.stdout)
    subprocess.run([sys.executable, str(ROOT / "a40/check_container_environment.py"), "--verify",
                    "--report", str(ROOT / "validation/extra_overlay_core_check.json")],
                   env=environment, check=True, timeout=120)
    subprocess.run([sys.executable, str(ROOT / "a40/audit_container_dependencies.py")],
                   env=environment, check=True, timeout=120)
    report = {"status": "passed", "finished_unix": time.time(), "versions": versions,
              "checks": ["FastAPI OpenAPI generation", "lxml parse", "quadprog CPU QP",
                         "docstring parse", "JWT local roundtrip", "core versions and CPU imports",
                         "no introduced dependency conflicts"],
              "simulation_started": False, "gpu_used": False}
    (ROOT / "validation/extra_overlay_ready.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
