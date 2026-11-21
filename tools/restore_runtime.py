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

"""Restore generated benchmark files and checkpoint times after cloning."""

import hashlib
import json
import os
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    """Return a file's SHA256 digest."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def restore(root: Path) -> int:
    """Restore the recorded overlay, preserving existing files and source repos."""
    metadata = json.loads((root / "provenance.json").read_text())
    for name in metadata["repositories"]:
        if not (root / name / ".git").exists():
            raise FileNotFoundError(
                f"Initialize source submodules first: git submodule update --init --recursive ({name})"
            )
    inventory = json.loads((root / "artifacts/benchmark_runtime.json").read_text())
    archive = root / "artifacts/benchmark_runtime.tar.gz"
    if digest(archive) != inventory["archive_sha256"]:
        raise ValueError("Benchmark runtime archive failed SHA256 verification")
    overlay = root / "benchmark/logs"
    restored = 0
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        names = [member.name for member in members]
        if len(set(names)) != len(names) or set(names) != set(inventory["files"]):
            raise ValueError("Runtime archive members disagree with the inventory")
        # Validate the full archive before writing any file.
        for member in members:
            destination = root / member.name
            if (
                not member.isfile()
                or Path(member.name).is_absolute()
                or ".." in Path(member.name).parts
                or not destination.resolve().is_relative_to(overlay.resolve())
            ):
                raise ValueError(f"Invalid runtime archive path: {member.name}")
            data = bundle.extractfile(member).read()
            expected = inventory["files"][member.name]["sha256"]
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"Runtime member failed SHA256: {member.name}")
            if destination.exists() and digest(destination) != expected:
                raise FileExistsError(
                    f"Refusing to overwrite changed runtime file: {destination}"
                )
        for member in members:
            destination = root / member.name
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                with (
                    bundle.extractfile(member) as source,
                    destination.open("xb") as output,
                ):
                    output.write(source.read())
                destination.chmod(member.mode & 0o777)
                restored += 1
            modified = inventory["files"][member.name]["mtime_ns"]
            os.utime(destination, ns=(modified, modified))
    for entry in metadata["checkpoints"]:
        path = root / entry["path"]
        if digest(path) != entry["sha256"]:
            raise ValueError(f"Checkpoint failed SHA256: {entry['path']}")
        os.utime(path, ns=(entry["mtime_ns"], entry["mtime_ns"]))
    return restored


if __name__ == "__main__":
    print(
        json.dumps(
            {"status": "passed", "restored_files": restore(ROOT), "gpu_allocations": 0}
        )
    )
