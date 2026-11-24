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

"""Go2 high-level SDK2 adapter; no joint commands or posture transitions.

Velocity uses m/s, m/s, rad/s, as SportClient.Move does. A watchdog requests
StopMove when commands or state reception expire. It cannot stop a robot
through a broken network or after this process is killed.
"""

from __future__ import annotations

import threading
import time
from contextlib import ExitStack
from typing import Any

import numpy as np

from rlinf.robotics.go2_dds import go2_dds_session
from rlinf.utils.logging import get_logger

from ..base import Action, Features, Observation
from .base import MobileBase

STATE_SHAPES = {
    "rpy": (3,),
    "gyroscope": (3,),
    "joint_position": (12,),
    "joint_velocity": (12,),
    "battery_voltage": (1,),
    "foot_force_raw": (4,),
}


@MobileBase.register("unitree_go2")
class Go2Base(MobileBase):
    """One Go2 high-level controller per process, read-only by default.

    Args:
        interface: Explicit network interface connected to the robot.
        enable_motion: Permit nonzero Move calls after on-site authorization.
        command_timeout: Seconds without a new action before requesting stop.
        state_timeout: Maximum local state reception age in seconds.
        rpc_timeout: Timeout for acknowledged SDK calls, in seconds.

    Nonzero velocities are rejected outside the official example's axis
    magnitudes: 0.3 m/s forward/lateral and 0.5 rad/s yaw. These are adapter
    limits, not manufacturer safety ratings. Foot forces are unvalidated raw
    readings. State age measures reception, not sensor acquisition time.
    """

    SDK = "unitree_sdk2py"

    def __init__(
        self,
        interface: str,
        *,
        enable_motion: bool = False,
        command_timeout: float = 0.5,
        state_timeout: float = 0.5,
        rpc_timeout: float = 1.0,
    ) -> None:
        if not interface.strip():
            raise ValueError("Go2 requires an explicit network interface.")
        for name, value in (
            ("command_timeout", command_timeout),
            ("state_timeout", state_timeout),
            ("rpc_timeout", rpc_timeout),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value}.")
        self.interface = interface
        self.enable_motion = enable_motion
        self.command_timeout = command_timeout
        self.state_timeout = state_timeout
        self.rpc_timeout = rpc_timeout
        self._lock = threading.RLock()
        self._dds = ExitStack()
        self._state_lock = threading.Lock()
        self._state: Observation | None = None
        self._received_at = 0.0
        self._ready = threading.Event()
        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None
        self._subscriber: Any = None
        self._client: Any = None
        self._deadline: float | None = None
        self._retry_stop_at: float | None = None
        self._fault: str | None = None
        self._logger = get_logger()

    @property
    def observation_features(self) -> Features:
        """Return raw SDK state shapes; feet are not contact labels."""
        return {
            name: {"shape": shape, "dtype": "float32"}
            for name, shape in STATE_SHAPES.items()
        }

    @property
    def action_features(self) -> Features:
        """Return body velocity shape, in m/s, m/s, rad/s."""
        return {"velocity": {"shape": (3,), "dtype": "float32"}}

    def _on_state(self, state: Any) -> None:
        try:
            reading = {
                "rpy": state.imu_state.rpy,
                "gyroscope": state.imu_state.gyroscope,
                "joint_position": [m.q for m in state.motor_state[:12]],
                "joint_velocity": [m.dq for m in state.motor_state[:12]],
                "battery_voltage": [state.power_v],
                "foot_force_raw": state.foot_force,
            }
            values = {
                k: np.asarray(v, dtype=np.float32).copy() for k, v in reading.items()
            }
            if any(
                values[k].shape != shape or not np.isfinite(values[k]).all()
                for k, shape in STATE_SHAPES.items()
            ):
                return
        except (AttributeError, TypeError, ValueError):
            return
        with self._state_lock:
            self._state = values
            self._received_at = time.monotonic()
        self._ready.set()

    def _open(self) -> Any:
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.go2.sport.sport_client import SportClient
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

        self._dds.enter_context(go2_dds_session(self.interface))
        try:
            self._ready.clear()
            self._shutdown.clear()
            self._fault = None
            self._retry_stop_at = None
            self._deadline = None
            self._subscriber = ChannelSubscriber("rt/lowstate", LowState_)
            self._subscriber.Init(self._on_state, 1)
            if not self._ready.wait(self.state_timeout):
                raise TimeoutError(
                    "No valid Go2 lowstate received; motion stays disabled."
                )
            if self.enable_motion:
                self._client = SportClient()
                self._client.SetTimeout(self.rpc_timeout)
                self._client.Init()
                self._stop()
                self._thread = threading.Thread(
                    target=self._watchdog, name="go2-command-watchdog", daemon=True
                )
                self._thread.start()
            return self._subscriber
        except BaseException:
            self._cleanup()
            raise

    def _stop(self) -> None:
        self._deadline = None
        if self._client is None:
            return
        try:
            code = self._client.StopMove()
            if code != 0:
                raise RuntimeError(f"StopMove returned {code}")
        except Exception as exc:
            self._fault = (
                f"StopMove unconfirmed ({exc}); use the remote emergency stop."
            )
            self._retry_stop_at = time.monotonic() + 0.2
            raise RuntimeError(self._fault) from exc
        self._retry_stop_at = None

    def _trip(self, reason: str) -> None:
        if self._retry_stop_at is not None:
            return
        self._fault = reason
        self._stop()

    def _watchdog(self) -> None:
        while not self._shutdown.wait(0.02):
            with self._lock:
                now = time.monotonic()
                try:
                    if self._retry_stop_at is not None:
                        if now >= self._retry_stop_at:
                            self._stop()
                        continue
                    if self._deadline is None:
                        continue
                    with self._state_lock:
                        state_age = now - self._received_at
                    if now >= self._deadline or state_age > self.state_timeout:
                        self._trip(
                            "Command or state timed out; reconnect before motion."
                        )
                except Exception:
                    self._logger.exception(
                        "Go2 stop unconfirmed; use remote emergency stop"
                    )

    def reset(self) -> None:
        """Request StopMove; faults remain latched until reconnect."""
        with self._lock:
            self._stop()

    def get_observation(self) -> Observation:
        """Read a copied recent state; stale reception trips the motion latch."""
        if not self.is_connected:
            raise RuntimeError("Connect Go2 before reading observations.")
        with self._lock:
            with self._state_lock:
                age = time.monotonic() - self._received_at
                reading = (
                    None
                    if self._state is None
                    else {k: v.copy() for k, v in self._state.items()}
                )
            if reading is None or age > self.state_timeout:
                self._trip("Go2 state reception expired; reconnect before motion.")
                raise TimeoutError(self._fault)
            return reading

    def send_action(self, action: Action) -> Observation:
        """Send a bounded velocity; returned values are commands, not feedback."""
        if not self.is_connected:
            raise RuntimeError("Connect Go2 before sending actions.")
        with self._lock:
            if self._shutdown.is_set():
                raise RuntimeError(
                    "Go2 connection is closing; no actions are permitted."
                )
            try:
                if set(action) != {"velocity"}:
                    raise ValueError("Go2 action must contain only 'velocity'.")
                velocity = np.asarray(action["velocity"], dtype=np.float32)
                limit = np.asarray([0.3, 0.3, 0.5], dtype=np.float32)
                if (
                    velocity.shape != (3,)
                    or not np.isfinite(velocity).all()
                    or (np.abs(velocity) > limit).any()
                ):
                    raise ValueError(
                        "Expected finite velocity (3,) within [0.3, 0.3, 0.5]."
                    )
                if not np.any(velocity):
                    self._stop()
                    return {"velocity": velocity.copy()}
                if not self.enable_motion:
                    raise PermissionError("Go2 is read-only; enable_motion is false.")
                if self._fault is not None:
                    raise RuntimeError(self._fault)
                if self._deadline is not None and time.monotonic() >= self._deadline:
                    raise TimeoutError("Go2 command expired; reconnect before motion.")
                self.get_observation()
                self._deadline = time.monotonic() + self.command_timeout
                code = self._client.Move(*(float(v) for v in velocity))
                if code != 0:
                    raise RuntimeError(f"Move send returned {code}.")
                return {"velocity": velocity.copy()}
            except BaseException:
                self._trip("Go2 action failed; reconnect before motion.")
                raise

    def _cleanup(self) -> None:
        with self._lock:
            self._shutdown.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        try:
            with self._lock:
                self._stop()
        finally:
            try:
                if self._subscriber is not None:
                    self._subscriber.Close()
            finally:
                self._subscriber = None
                self._client = None
                self._dds.close()

    def _release(self, device: Any) -> None:
        """Stop before closing the state subscriber; propagate stop failures."""
        self._cleanup()
