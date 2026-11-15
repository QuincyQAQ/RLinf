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

"""In-memory SDK2 edge fake; no sockets, DDS participants, or robot access."""

import threading
from types import ModuleType, SimpleNamespace
from typing import Any


def lowstate() -> SimpleNamespace:
    """Return deterministic Go2 lowstate data with twelve joints."""
    return SimpleNamespace(
        imu_state=SimpleNamespace(rpy=[0.1, 0.2, 0.3], gyroscope=[1, 2, 3]),
        motor_state=[SimpleNamespace(q=i / 10, dq=i / 20) for i in range(12)],
        power_v=31.52,
        foot_force=[40, 0, 0, 0],
    )


class FakeGo2SDK:
    """Publish state on subscription and record every motion/stop call."""

    def __init__(self) -> None:
        self.publish_initial = True
        self.stop_code = 0
        self.move_code = 0
        self.move_error = False
        self.events: list[tuple[Any, ...]] = []
        self.stopped = threading.Event()
        self.close_entered = threading.Event()
        self.close_release = threading.Event()
        self.close_release.set()
        self.callback: Any = None

    def publish(self) -> None:
        """Deliver one recent state through the real adapter's SDK callback."""
        self.callback(lowstate())

    def modules(self) -> dict[str, ModuleType]:
        """Return importable SDK modules for an isolated test process."""
        fake = self

        class Subscriber:
            def __init__(self, topic: str, dtype: type) -> None:
                fake.events.append(("subscribe", topic))

            def Init(self, callback: Any, queue_len: int) -> None:
                fake.callback = callback
                if fake.publish_initial:
                    fake.publish()

            def Close(self) -> None:
                fake.close_entered.set()
                if not fake.close_release.wait(2):
                    raise TimeoutError("Test did not release subscriber close")
                fake.events.append(("close",))

        class Client:
            def __init__(self) -> None:
                fake.events.append(("client",))

            def SetTimeout(self, timeout: float) -> None:
                fake.events.append(("timeout", timeout))

            def Init(self) -> None:
                fake.events.append(("init",))

            def StopMove(self) -> int:
                fake.events.append(("stop",))
                fake.stopped.set()
                return fake.stop_code

            def Move(self, vx: float, vy: float, vyaw: float) -> int:
                fake.events.append(("move", vx, vy, vyaw))
                if fake.move_error:
                    raise OSError("Simulated DDS send failure")
                return fake.move_code

        def initialize(domain: int, interface: str) -> None:
            fake.events.append(("dds", domain, interface))

        definitions = {
            "unitree_sdk2py.core.channel": {
                "ChannelFactoryInitialize": initialize,
                "ChannelSubscriber": Subscriber,
            },
            "unitree_sdk2py.go2.sport.sport_client": {"SportClient": Client},
            "unitree_sdk2py.idl.unitree_go.msg.dds_": {"LowState_": SimpleNamespace},
        }
        modules = {}
        for name, values in definitions.items():
            module = ModuleType(name)
            module.__dict__.update(values)
            modules[name] = module
        return modules
