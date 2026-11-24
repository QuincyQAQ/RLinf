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

"""Recorded simulator recipes and runtime paths for the ABC example."""

import json
import math
import os
from pathlib import Path

ROOT = Path(os.environ["GO2_ABC_RUNTIME"])


def read_manifest() -> dict:
    """Load the three retained simulator recipes."""
    return json.loads((Path(__file__).parent / "reference.json").read_text())


def normalize(value):
    if isinstance(value, dict):
        return {str(k): normalize(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [normalize(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def nested(data, path):
    for key in path.split("."):
        data = data[key]
    return data


def compare(actual, expected, path=""):
    errors = []
    for key, value in expected.items():
        try:
            found = nested(actual, key)
        except (KeyError, TypeError):
            errors.append(f"Missing {path}{key}")
            continue
        equal = (
            math.isclose(found, value, rel_tol=1e-7, abs_tol=1e-10)
            if type(value) is float and type(found) in (int, float)
            else found == value
        )
        if not equal:
            errors.append(f"{path}{key}: expected {value!r}, actual {found!r}")
    return errors
