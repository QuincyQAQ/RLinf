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

"""In-memory Unitree SDK2 channels and messages for Go2 control tests."""

import copy
import sys
from types import ModuleType, SimpleNamespace


class FakeGo2SDK:
    """Record DDS operations and expose a scripted mode and write result."""

    def __init__(self):
        self.events = []
        self.mode = ""
        self.write_ok = True
        self.callback = None

    @staticmethod
    def command():
        return SimpleNamespace(
            head=[0, 0],
            level_flag=0,
            gpio=0,
            crc=0,
            motor_cmd=[
                SimpleNamespace(mode=0, q=0.0, dq=0.0, kp=0.0, kd=0.0, tau=0.0)
                for _ in range(20)
            ],
        )

    @staticmethod
    def state(tick=1):
        return SimpleNamespace(
            tick=tick,
            crc=42,
            motor_state=[SimpleNamespace(q=0.0, dq=0.0) for _ in range(20)],
            imu_state=SimpleNamespace(
                quaternion=[1.0, 0.0, 0.0, 0.0], gyroscope=[0.0, 0.0, 0.0]
            ),
        )

    def install(self, monkeypatch):
        sdk = self

        class Publisher:
            def __init__(self, topic, message_type):
                sdk.events.append(("publisher", topic))

            def Init(self):
                sdk.events.append(("writer_open",))

            def Write(self, message, timeout=None):
                sdk.events.append(("write", copy.deepcopy(message)))
                return sdk.write_ok

            def Close(self):
                sdk.events.append(("writer_close",))

        class Subscriber:
            def __init__(self, topic, message_type):
                sdk.events.append(("subscriber", topic))

            def Init(self, callback, queue_length):
                sdk.callback = callback

            def Close(self):
                sdk.events.append(("reader_close",))

        class Switcher:
            def SetTimeout(self, timeout):
                pass

            def Init(self):
                pass

            def CheckMode(self):
                sdk.events.append(("check_mode",))
                return 0, {"name": sdk.mode}

        class CRC:
            def Crc(self, message):
                return 42

        modules = {
            "unitree_sdk2py.core.channel": {
                "ChannelFactoryInitialize": lambda domain, interface: sdk.events.append(
                    ("initialize", interface)
                ),
                "ChannelPublisher": Publisher,
                "ChannelSubscriber": Subscriber,
            },
            "unitree_sdk2py.comm.motion_switcher.motion_switcher_client": {
                "MotionSwitcherClient": Switcher
            },
            "unitree_sdk2py.idl.default": {"unitree_go_msg_dds__LowCmd_": self.command},
            "unitree_sdk2py.idl.unitree_go.msg.dds_": {
                "LowCmd_": self.command,
                "LowState_": self.state,
            },
            "unitree_sdk2py.utils.crc": {"CRC": CRC},
        }
        for name, members in modules.items():
            parts = name.split(".")
            for index in range(1, len(parts) + 1):
                package = ".".join(parts[:index])
                if package not in sys.modules:
                    module = ModuleType(package)
                    module.__path__ = []
                    monkeypatch.setitem(sys.modules, package, module)
            module = ModuleType(name)
            module.__dict__.update(members)
            monkeypatch.setitem(sys.modules, name, module)
