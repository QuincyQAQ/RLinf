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

"""Force-threshold contact timing used by the B2 reference reward."""

from dataclasses import dataclass

from mjlab.sensor import ContactSensor, ContactSensorCfg


class ForceThresholdContactSensor(ContactSensor):
    """Use the reference's 1 N contact criterion for foot air-time tracking."""

    def _extract_sensor_data(self):
        data = super()._extract_sensor_data()
        data.found = (data.force.norm(dim=-1) > 1.0).to(data.found.dtype)
        return data


@dataclass
class ForceThresholdContactSensorCfg(ContactSensorCfg):
    """Build a contact sensor whose timing uses net-force contact detection."""

    def build(self) -> ForceThresholdContactSensor:
        return ForceThresholdContactSensor(self)
