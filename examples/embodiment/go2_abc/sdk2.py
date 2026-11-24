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
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Protocol

import numpy as np

from rlinf.robotics.go2_dds import go2_dds_session

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


@dataclass(frozen=True)
class StandFeedback:
    """One measured supervisor sample on the control host's monotonic clock."""

    timestamp: float
    standing_controller_active: bool
    feet_in_contact: tuple[bool, bool, bool, bool]
    body_stable: bool


class StandingWindow:
    """Require 0.5 s of continuous, fresh support measurements in one phase."""

    def __init__(self, started: float) -> None:
        self.started = started
        self.previous = started
        self.stable_since: float | None = None

    def observe(
        self, feedback: StandFeedback, now: float, *, owner_active: bool
    ) -> bool:
        stamp = feedback.timestamp
        if not (self.started < stamp <= now and now - stamp <= 0.05):
            self.stable_since = None
            return False
        if stamp - self.previous > 0.05:
            self.stable_since = None
        supported = (
            feedback.standing_controller_active == owner_active
            and feedback.feet_in_contact == (True, True, True, True)
            and feedback.body_stable
        )
        if not supported or stamp < self.previous:
            self.stable_since = None
        if stamp <= self.previous:
            return False
        self.previous = stamp
        if supported and self.stable_since is None:
            self.stable_since = stamp
        return self.stable_since is not None and stamp - self.stable_since >= 0.5


class SupervisedControl(Protocol):
    """Exclusive session with an independent motor controller.

    The supervisor owns the sole LowCmd publisher, fences expired sessions and
    takes over on heartbeat loss. All calls must be bounded. State feedback
    must come from measured contact and upright, stationary body estimates.
    """

    def write(self, message: Any, *, timeout: float) -> bool:
        """Send a policy packet while this session still owns control."""
        ...

    def request_stand(self) -> None:
        """Revoke this session's writes and independently stop and support the robot.

        This request is idempotent. Support must persist after client exit.
        """
        ...

    def stand_feedback(self) -> StandFeedback:
        """Return current contact, body stability, and motor ownership during this session."""
        ...


@dataclass(eq=False)
class _ControlLease:
    """One supervisor acquisition and its reusable, serialized LowCmd buffer."""

    supervisor: SupervisedControl
    message: Any
    crc: Any
    packet_lock: threading.Lock = field(default_factory=threading.Lock)


@lru_cache(maxsize=1)
def _mode_checker(client_type: type) -> Any:
    # The pinned SDK starts a reply reader per client and has no teardown API.
    client = client_type()
    client.SetTimeout(1.0)
    client.Init()
    return client


class Go2SDK2:
    """Own one SDK2 session; command publishing requires explicit opt-in.

    Args:
        interface: Network interface connected to Go2.
        enable_motion: Enable policy output through the independent supervisor.
        acquire_control: Obtain an exclusive, watchdog-backed session for this
            interface. Acquisition must roll back on failure.
    """

    def __init__(
        self,
        interface: str,
        *,
        enable_motion: bool = False,
        acquire_control: Callable[[str], SupervisedControl] | None = None,
    ) -> None:
        if not isinstance(enable_motion, bool):
            raise TypeError("enable_motion must be a boolean")
        if not interface.strip():
            raise ValueError("Expected an explicit Go2 network interface")
        if enable_motion and acquire_control is None:
            raise PermissionError("Motion requires an independent standing supervisor")
        self._acquire_control = acquire_control
        self.interface, self.enable_motion = interface, enable_motion
        self._resources = contextlib.ExitStack()
        self._state_lock = threading.Condition()
        self._session_lock = threading.RLock()
        self._loop_active = False
        self._state = None
        self._tick = None
        self._owned = False
        self._session: _ControlLease | None = None
        self._standing_requested = False
        self._policy_writes_enabled = False

    def connect(self) -> None:
        """Subscribe to LowState and optionally prepare the torque publisher."""
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
            MotionSwitcherClient,
        )
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
        from unitree_sdk2py.utils.crc import CRC

        with self._session_lock:
            if self._loop_active:
                raise RuntimeError(
                    "The previous Go2 control loop must exit before reconnect"
                )
            self._resources.enter_context(go2_dds_session(self.interface))
            self._owned = True
            try:
                self._crc = CRC()
                subscriber = ChannelSubscriber("rt/lowstate", LowState_)
                self._resources.callback(subscriber.Close)
                subscriber.Init(self._receive, 1)
                if self.enable_motion is True:
                    if self._acquire_control is None:
                        raise PermissionError(
                            "Motion requires an independent standing supervisor"
                        )
                    switcher = _mode_checker(MotionSwitcherClient)
                    message = unitree_go_msg_dds__LowCmd_()
                    self.wait_for_state()
                    self._session = _ControlLease(
                        self._acquire_control(self.interface), message, self._crc
                    )
                    status, mode = switcher.CheckMode()
                    if status != 0 or mode["name"]:
                        raise RuntimeError(
                            "Expected exclusive low-level control after supervisor acquisition"
                        )
                    self._policy_writes_enabled = True
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
            self._state_lock.notify_all()

    def wait_for_state(self) -> None:
        """Wait up to one second for a fresh, checksum-valid sample."""
        with self._state_lock:
            ready = self._state_lock.wait_for(
                lambda: not self._owned
                or (
                    self._state is not None
                    and 0 <= time.monotonic() - self._state.timestamp < 0.05
                ),
                timeout=1.0,
            )
            if not self._owned:
                raise RuntimeError("Go2 connection is closed")
            if not ready:
                raise TimeoutError(
                    "No fresh checksum-valid Go2 LowState within one second"
                )

    def read(self) -> JointState:
        """Return the most recent checksum-valid state without blocking."""
        with self._state_lock:
            if self._state is None:
                raise RuntimeError("No checksum-valid Go2 LowState has arrived")
            return self._state

    def write(
        self, torques: Sequence[float] | np.ndarray, *, valid_until: float | None = None
    ) -> None:
        """Send twelve motor torques through the explicitly enabled publisher."""
        lease = self._session
        if lease is None or not self._policy_writes_enabled:
            raise PermissionError("Go2 SDK2 command publishing is disabled")
        with lease.packet_lock:
            message = torque_message(lease.message, torques, lease.crc)
            with self._session_lock:
                if self._session is not lease or not self._policy_writes_enabled:
                    raise PermissionError(
                        "Go2 policy control was revoked before publication"
                    )
                state_stamp = self.read().timestamp
                now = time.monotonic()
                if not 0 <= now - state_stamp < 0.05:
                    raise TimeoutError(
                        "Go2 joint state exceeded the 50-ms publication deadline"
                    )
                expires = state_stamp + 0.05
                if valid_until is not None:
                    if not now < valid_until:
                        raise TimeoutError(
                            "Go2 policy command expired before publication"
                        )
                    expires = min(expires, valid_until)
                if not lease.supervisor.write(
                    message, timeout=min(0.005, expires - now)
                ):
                    raise OSError("Go2 LowCmd write failed")

    def _request_stand(self) -> None:
        self._policy_writes_enabled = False
        if self._session is not None and not self._standing_requested:
            self._session.supervisor.request_stand()
            self._standing_requested = True

    def stand_feedback(self) -> StandFeedback:
        """Read bounded, nonblocking support and ownership feedback."""
        with self._session_lock:
            if self._session is None:
                raise RuntimeError("Standing feedback requires an active supervisor")
            return self._session.supervisor.stand_feedback()

    def finish_standing(self) -> None:
        """Transfer control and confirm the supervisor continues supporting the robot."""
        with self._session_lock:
            lease = self._session
            if lease is None:
                raise RuntimeError(
                    "Standing confirmation requires an active supervisor"
                )
            window = StandingWindow(time.monotonic())
            self._request_stand()
        deadline = window.started + 3.0
        while time.monotonic() < deadline:
            with self._session_lock:
                if self._session is not lease:
                    raise RuntimeError(
                        "Go2 session closed during standing confirmation"
                    )
                feedback = lease.supervisor.stand_feedback()
            now = time.monotonic()
            if now >= deadline:
                break
            if window.observe(feedback, now, owner_active=True):
                return
            time.sleep(0.01)
        raise TimeoutError("Standing handover was not confirmed within 3 seconds")

    def close(self) -> None:
        """Relinquish policy control on faults and release local channels."""
        with self._session_lock:
            if not self._owned:
                return
            try:
                self._request_stand()
            finally:
                try:
                    self._resources.close()
                finally:
                    self._session = None
                    self._standing_requested = False
                    with self._state_lock:
                        self._owned = False
                        self._state = None
                        self._tick = None
                        self._state_lock.notify_all()

    @contextlib.contextmanager
    def control_loop(self) -> Iterator[None]:
        """Reserve this connection until its loop has finished fault cleanup."""
        with self._session_lock:
            if not self._owned or self._loop_active:
                raise RuntimeError(
                    "Expected an open Go2 connection with no active loop"
                )
            self._loop_active = True
        try:
            yield
        finally:
            with self._session_lock:
                try:
                    self.close()
                finally:
                    self._loop_active = False
