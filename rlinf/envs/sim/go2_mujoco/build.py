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

import pathlib
import subprocess
import sysconfig

import mujoco
import pybind11

root = pathlib.Path(__file__).resolve().parent
mj = pathlib.Path(mujoco.__file__).parent
lib = next(mj.glob("libmujoco.so*"))
out = root / ("batched_go2" + sysconfig.get_config_var("EXT_SUFFIX"))
cmd = [
    "g++",
    "-O3",
    "-std=c++17",
    "-shared",
    "-fPIC",
    "-fopenmp",
    str(root / "batched_go2.cpp"),
    "-I" + pybind11.get_include(),
    "-I" + sysconfig.get_path("include"),
    "-I" + str(mj / "include"),
    str(lib),
    "-Wl,-rpath," + str(mj),
    "-o",
    str(out),
]
subprocess.run(cmd, check=True)
