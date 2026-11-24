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

"""Fetch the nine policy inputs or one task's baseline policy."""

import argparse
import hashlib
import json
import logging
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parents[2] / "data/go2_abc"


def data_directory() -> Path:
    """Return the data directory used by the ABC command-line tools."""
    return Path(os.environ.get("GO2_ABC_DATA", DEFAULT_DATA))


def digest(path: Path) -> str:
    """Return a file's SHA256 without loading it into memory."""
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            checksum.update(chunk)
    return checksum.hexdigest()


def verify_digest(path: Path, expected: str) -> None:
    """Raise if the file content differs from its manifest entry."""
    if digest(path) != expected:
        raise ValueError(f"SHA256 mismatch: {path}")


def baseline_input(manifest: dict, task: str) -> dict:
    """Find the manifest entry for one task's initial policy."""
    path = f"models/{task}/baseline.pt"
    matches = [item for item in manifest["files"] if item["path"] == path]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one manifest entry for {path}, found {len(matches)}"
        )
    return matches[0]


def prepare(
    destination: Path, source: Path | None = None, task: str | None = None
) -> None:
    """Stage only the manifest's inputs, verifying every content hash."""
    manifest = json.loads((HERE / "manifest.json").read_text())
    items = manifest["files"] if task is None else [baseline_input(manifest, task)]
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / "weights.tar.gz"
        for item in items:
            target = destination / item["path"]
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError(f"Input escapes data directory: {item['path']}")
            if target.exists():
                verify_digest(target, item["sha256"])
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=target.parent) as staging:
                staged = Path(staging) / target.name
                if source is not None:
                    shutil.copyfile(source / item["source"], staged)
                elif "url" in item:
                    urllib.request.urlretrieve(item["url"], staged)
                else:
                    if not archive.exists():
                        urllib.request.urlretrieve(manifest["archive"]["url"], archive)
                        verify_digest(archive, manifest["archive"]["sha256"])
                    with tarfile.open(archive) as bundle:
                        member = bundle.getmember(item["archive_member"])
                        if not member.isfile():
                            raise ValueError(
                                f"Expected a regular weight file: {member.name}"
                            )
                        with (
                            bundle.extractfile(member) as stream,
                            staged.open("wb") as out,
                        ):
                            shutil.copyfileobj(stream, out)
                verify_digest(staged, item["sha256"])
                try:
                    target.hardlink_to(staged)
                except FileExistsError:
                    # Another preparation may have published this input while we copied it.
                    verify_digest(target, item["sha256"])
    logging.getLogger(__name__).info(
        "Verified %s inputs in %s", len(items), destination
    )


def main() -> None:
    """Download inputs or copy them from the frozen experiment workspace."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=data_directory())
    parser.add_argument("--source", type=Path)
    parser.add_argument("--task", choices=("A", "B", "C"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    prepare(args.data, args.source, args.task)


if __name__ == "__main__":
    main()
