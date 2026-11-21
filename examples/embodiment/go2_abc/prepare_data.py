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

"""Fetch the nine policies and their recorded evaluation inputs."""

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


def digest(path: Path) -> str:
    """Return a file's SHA256 without loading it into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(destination: Path, source: Path | None = None) -> None:
    """Stage only the manifest's inputs, verifying every content hash."""
    manifest = json.loads((HERE / "manifest.json").read_text())
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / "weights.tar.gz"
        for item in manifest["files"]:
            target = destination / item["path"]
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError(f"Input escapes data directory: {item['path']}")
            if target.exists():
                if digest(target) != item["sha256"]:
                    raise ValueError(f"SHA256 mismatch: {target}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            staged = Path(temporary) / "input"
            if source is not None:
                shutil.copyfile(source / item["source"], staged)
            elif "url" in item:
                urllib.request.urlretrieve(item["url"], staged)
            else:
                if not archive.exists():
                    urllib.request.urlretrieve(manifest["archive"]["url"], archive)
                    if digest(archive) != manifest["archive"]["sha256"]:
                        raise ValueError("Weight archive SHA256 mismatch")
                with tarfile.open(archive) as bundle:
                    member = bundle.getmember(item["archive_member"])
                    if not member.isfile():
                        raise ValueError(
                            f"Expected a regular weight file: {member.name}"
                        )
                    with bundle.extractfile(member) as stream, staged.open("wb") as out:
                        shutil.copyfileobj(stream, out)
            if digest(staged) != item["sha256"]:
                raise ValueError(f"SHA256 mismatch: {item['path']}")
            shutil.move(staged, target)
    logging.getLogger(__name__).info(
        "Verified %s inputs in %s", len(manifest["files"]), destination
    )


def main() -> None:
    """Download inputs or copy them from the frozen experiment workspace."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path(os.environ.get("GO2_ABC_DATA", DEFAULT_DATA))
    )
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    prepare(args.data, args.source)


if __name__ == "__main__":
    main()
