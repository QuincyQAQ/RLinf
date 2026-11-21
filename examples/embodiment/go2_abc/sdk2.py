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

"""Unitree SDK2 transport for Go2 low-level state and torque messages."""

import contextlib
import threading
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from .control import JointState, vector


def torque_message(
    message: Any, torques: Sequence[float] | np.ndarray, crc: Any
) -> Any:
    """Fill the official LowCmd torque fields, disable position PD, and sign CRC."""
    effort = vector(torques, 12, "SDK torques")
    if np.abs(effort).max() > 23.5:
        raise ValueError("Expected SDK torques within the recorded 23.5 Nm limit")
    message.head[:] = [0xFE, 0xEF]
    message.level_flag = 0xFF
    message.gpio = 0
    for index, motor in enumerate(message.motor_cmd):
        motor.mode = 0x01
        motor.q, motor.dq = 2.146e9, 16000.0
        motor.kp = motor.kd = 0.0
        motor.tau = float(effort[index]) if index < 12 else 0.0
    message.crc = crc.Crc(message)
    return message


class Go2SDK2:
    """Own one SDK2 session; command publishing requires explicit opt-in.

    Args:
        interface: Network interface connected to Go2.
        enable_motion: Permit LowCmd publishing after the built-in controller
            has already been released by the operator.
    """

    _lock = threading.Lock()
    _active = False
    _interface = None

    def __init__(self, interface: str, *, enable_motion: bool = False) -> None:
        if not interface.strip():
            raise ValueError("Expected an explicit Go2 network interface")
        self.interface, self.enable_motion = interface, enable_motion
        self._resources = contextlib.ExitStack()
        self._state_lock = threading.Lock()
        self._state = None
        self._tick = None
        self._owned = False
        self._published = False
        self._publisher = None

    def connect(self) -> None:
        """Subscribe to LowState and optionally prepare the torque publisher."""
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
            MotionSwitcherClient,
        )
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
            ChannelPublisher,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        with self._lock:
            if Go2SDK2._active or Go2SDK2._interface not in (None, self.interface):
                raise RuntimeError(
                    "Expected exclusive SDK2 ownership on one process-wide interface"
                )
            Go2SDK2._active = self._owned = True
        try:
            ChannelFactoryInitialize(0, self.interface)
            Go2SDK2._interface = self.interface
            self._crc, self._message = CRC(), unitree_go_msg_dds__LowCmd_()
            subscriber = ChannelSubscriber("rt/lowstate", LowState_)
            self._resources.callback(subscriber.Close)
            subscriber.Init(self._receive, 1)
            if self.enable_motion:
                switcher = MotionSwitcherClient()
                switcher.SetTimeout(1.0)
                switcher.Init()
                status, mode = switcher.CheckMode()
                if status != 0 or mode["name"]:
                    raise RuntimeError(
                        "Expected the operator to release the built-in controller before joint control"
                    )
                self._publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
                self._resources.callback(self._publisher.Close)
                self._publisher.Init()
        except BaseException:
            self.close()
            raise

    def _receive(self, message: Any) -> None:
        if message.crc != self._crc.Crc(message):
            return
        with self._state_lock:
            if message.tick == self._tick:
                return
            self._tick = message.tick
            self._state = JointState(
                [motor.q for motor in message.motor_state[:12]],
                [motor.dq for motor in message.motor_state[:12]],
                list(message.imu_state.quaternion),
                list(message.imu_state.gyroscope),
                time.monotonic(),
            )

    def read(self) -> JointState:
        """Return the most recent checksum-valid state without blocking."""
        with self._state_lock:
            if self._state is None:
                raise RuntimeError("No checksum-valid Go2 LowState has arrived")
            return self._state

    def write(self, torques: Sequence[float] | np.ndarray) -> None:
        """Send twelve motor torques through the explicitly enabled publisher."""
        if not self.enable_motion or not self._owned or self._publisher is None:
            raise PermissionError("Go2 SDK2 command publishing is disabled")
        age = time.monotonic() - self.read().timestamp
        if not 0 <= age <= 0.05:
            raise TimeoutError(
                "Go2 joint state exceeded the 50-ms publication deadline"
            )
        self._send(torques)

    def _send(self, torques: Sequence[float] | np.ndarray) -> None:
        message = torque_message(self._message, torques, self._crc)
        self._published = True
        if not self._publisher.Write(message, timeout=0.005):
            raise OSError("Go2 LowCmd write failed")

    def close(self) -> None:
        """Attempt zero torque after publishing, then release this session's channels."""
        if not self._owned:
            return
        try:
            if self._published:
                self._send(np.zeros(12, dtype=np.float32))
        finally:
            try:
                self._resources.close()
            finally:
                self._publisher = None
                self._state = None
                self._tick = None
                self._published = False
                with self._lock:
                    Go2SDK2._active = self._owned = False
