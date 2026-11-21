#!/usr/bin/env python3
"""Wait for this workspace's image pull, then perform CPU-only installation.

Never requests scheduler resources, enables GPU passthrough, or launches a simulator.
The status file records failure instead of silently treating a queued setup as complete.
"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "validation"


def status(phase, **extra):
    data = dict(phase=phase, unix=time.time(), pid=os.getpid(), gpu_used=False,
                slurm_job_submitted=False, simulation_started=False, training_started=False)
    data.update(extra)
    target = VALIDATION / "environment_worker_status.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(target)
    print(json.dumps(data), flush=True)


def main():
    os.chdir(ROOT)
    VALIDATION.mkdir(exist_ok=True)
    with (VALIDATION / "environment_worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            status("waiting_for_official_container")
            deadline = time.monotonic() + 4 * 3600
            receipt = VALIDATION / "container_pull_status.json"
            while True:
                if receipt.exists():
                    result = json.loads(receipt.read_text())
                    if result["exit_code"] != 0:
                        raise RuntimeError("Container pull failed; see container_pull.log")
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("Container pull has not finished after four hours")
                time.sleep(30)
            image = ROOT / "containers/isaac-lab_2.3.2.sif"
            if not image.is_file() or image.stat().st_size == 0:
                raise RuntimeError("Successful pull receipt has no container file")
            status("installing_and_verifying_cpu_environment")
            cpus = ",".join(map(str, sorted(os.sched_getaffinity(0))[:2]))
            environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                               MKL_NUM_THREADS="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
            with (VALIDATION / "environment_install.log").open("a") as output:
                result = subprocess.run(
                    ["nice", "-n", "15", "taskset", "-c", cpus,
                     "bash", "a40/prepare_hpc_environment.sh", "--install"],
                    env=environment, stdout=output, stderr=subprocess.STDOUT, timeout=7200)
            if result.returncode:
                raise RuntimeError("CPU environment setup failed; see environment_install.log")
            completion = json.loads((VALIDATION / "HPC_SETUP_STATUS.json").read_text())
            if completion.get("status") not in {"cpu_environment_prepared", "cpu_environment_prepared_with_known_image_conflicts"}:
                raise RuntimeError("CPU setup completion receipt is invalid")
            status(completion["status"], remaining=completion["remaining"])
        except Exception as exc:
            status("failed", error=f"{type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    main()
