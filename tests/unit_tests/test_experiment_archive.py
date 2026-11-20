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

"""Offline experiment archive integrity and reproducible report contracts."""

import copy
import importlib.util
import io
import tarfile
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples/embodiment/go2_a40_reference/reproduce.py"
)
SPEC = importlib.util.spec_from_file_location("experiment_archive", SCRIPT)
archive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archive)


def evaluation():
    return {
        "episodes": [
            {
                "episode_success": True,
                "termination_reason": "time_out",
                "success_step_ratio": 0.9,
                "mean_lin_vel_error_xy": 0.1,
                "mean_ang_vel_error_z": 0.2,
                "alive_time_s": 20.0,
            }
        ],
        "summary": {
            "num_episodes": 1,
            "success_rate": 1.0,
            "failure_rate": 0.0,
            "mean_success_step_ratio": 0.9,
            "mean_lin_vel_error": 0.1,
            "mean_ang_vel_error": 0.2,
            "mean_alive_time_s": 20.0,
        },
    }


def test_recorded_episode_summary_is_recomputed():
    data = evaluation()
    assert archive.verify_evaluation(data) == 1
    before = copy.deepcopy(data)
    data["summary"]["success_rate"] = 0.0
    with pytest.raises(ValueError, match="success_rate"):
        archive.verify_evaluation(data)
    assert before["episodes"] == data["episodes"]


def test_episode_count_mismatch_is_rejected():
    data = evaluation()
    data["summary"]["num_episodes"] = 2
    with pytest.raises(ValueError, match="Episode count"):
        archive.verify_evaluation(data)


def test_same_episode_passage_intersection_cannot_be_invented():
    data = evaluation()
    data["episodes"][0].update(
        physical_obstacle_passage=False, joint_tracking_and_passage=True
    )
    data["summary"].update(
        physical_passage_successes=0,
        physical_obstacle_passage_rate=0.0,
        joint_tracking_and_passage_success_rate=1.0,
    )
    with pytest.raises(ValueError, match="intersection"):
        archive.verify_evaluation(data)


@pytest.mark.parametrize("name", ["../escape", "/absolute"])
def test_archive_paths_cannot_escape_destination(tmp_path, name):
    path = tmp_path / "input.tar.gz"
    with tarfile.open(path, "w:gz") as bundle:
        member = tarfile.TarInfo(name)
        member.size = 1
        bundle.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="relative regular file"):
        archive.unpack(path, tmp_path / "out")


def test_archive_restore_does_not_overwrite_existing_work(tmp_path):
    path = tmp_path / "input.tar.gz"
    with tarfile.open(path, "w:gz") as bundle:
        member = tarfile.TarInfo("kept.txt")
        member.size = 1
        bundle.addfile(member, io.BytesIO(b"x"))
    destination = tmp_path / "out"
    destination.mkdir()
    (destination / "kept.txt").write_text("keep")
    with pytest.raises(FileExistsError, match="overwrite"):
        archive.unpack(path, destination)
    assert (destination / "kept.txt").read_text() == "keep"


def test_complete_report_regenerates_all_retained_experiments(tmp_path):
    count = archive.export_csv(SCRIPT.parent, tmp_path / "table.csv")
    assert count == 29
    assert (tmp_path / "table.csv").read_bytes() == (
        SCRIPT.parent / "reports/A40_five_tasks_complete_20261121.csv"
    ).read_bytes()
