Go2 High-Level Control
======================

Read Go2 EDU state through the official Unitree SDK2 and exercise a velocity-command environment offline. Hardware access defaults to read-only. Motion through this RLinf adapter has not yet been verified on a physical Go2.

Overview
--------

.. grid:: 2 4 4 4

   .. grid-item-card:: Models

      Manual commands; no trained policy attached.

   .. grid-item-card:: Algorithms

      No training recipe.

   .. grid-item-card:: Tasks

      High-level drive bring-up.

   .. grid-item-card:: Hardware

      Go2 EDU and a Linux controller.

Tasks
~~~~~

.. list-table::
   :header-rows: 1

   * - Task
     - Result
   * - ``Go2DriveEnv-v1``
     - State reception and bounded body velocity commands; zero reward.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Field
     - Contract
   * - Observation
     - RPY, gyroscope, 12 joint positions, 12 joint velocities, battery voltage, and four raw foot-force readings; 35 values total. No camera.
   * - Action
     - ``[vx, vy, vyaw]`` in m/s, m/s, rad/s. Reject nonfinite or out-of-range values. The absolute limits are 0.3 m/s, 0.3 m/s, and 0.5 rad/s, respectively.
   * - Reward and episode
     - Zero reward; truncate after 20 steps by default. This task supplies no success detector.

Safety Before Motion
~~~~~~~~~~~~~~~~~~~~

Verify forward and stop using the `official SportClient example <https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/example/go2/high_level/go2_sport_client.py>`_ first. These adapter limits match that example's axis magnitudes; they are not manufacturer safety ratings. Keep the area clear and have a second operator ready to use the remote emergency stop.

.. warning::

   Software cannot guarantee a stop after network loss, process termination, or a stalled SDK call. A successful Move send is not execution feedback, and a successful StopMove reply does not verify physical rest. Observe the robot directly.

Motion requires explicit ``enable_motion=True`` in the hardware configuration. Read-only connections create no SportClient and send no commands, including on reset and disconnect. Motion-enabled connections request StopMove on connect, reset, close, episode expiry, and errors; they never request posture or gait changes.

While a nonzero command is active, the watchdog requests a stop if no new action arrives for 0.5 seconds or no valid state arrives for 0.5 seconds. An expired command, stale state, or invalid action latches a fault; reset requests stop but does not clear the fault. Reconnect after investigating the cause. These deadlines are software targets, not hard real-time bounds. State freshness measures local reception, not sensor acquisition time. Raw foot forces are not validated contact labels.

Installation
------------

Use a configured RLinf development environment with Gymnasium and NumPy. Follow the `official SDK2 installation instructions <https://github.com/unitreerobotics/unitree_sdk2_python#installation>`_ in the same Python environment before hardware access. The offline example below imports no vendor SDK and creates no DDS participant.

Run It
------

From the RLinf repository root, exercise reset, a zero action, and close without hardware:

.. code-block:: bash

   python - <<'PYTHON'
   import numpy as np
   from rlinf.envs.real.go2 import Go2DriveEnv

   env = Go2DriveEnv({"is_dummy": True})
   try:
       observation, _ = env.reset()
       result = env.step(np.zeros(3, dtype=np.float32))
       print(observation["state"].keys(), result[1:4])
   finally:
       env.close()
   PYTHON

Expect six state fields and ``(0.0, False, False)``. Dummy observations are deterministic zeros, not simulated dynamics.

For an on-site read-only check, set ``interface`` to the verified robot network interface. This code opens only the lowstate subscriber:

.. code-block:: python

   from rlinf.robotics import Go2Robot

   robot = Go2Robot.build(interface="<network-interface>")
   try:
       robot.connect()
       print(robot.get_observation()["base"])
   finally:
       robot.disconnect()

Only one Go2 connection per process can own DDS. SDK2 has a process-global factory and no public teardown API; restart Python to change interfaces. Do not share the process with another SDK2 robot controller. Other processes and remote controllers are not excluded by this adapter.

Visualization and Results
-------------------------

Offline tests use an in-memory SDK fake to check command units, watchdog stops, state expiry, invalid actions, cleanup, and fault latching. They do not certify physical behavior. Camera input, a training reward, and deployment of the existing 12-joint policy remain separate work. See :doc:`Go2 MuJoCo <go2_mujoco>` for the simulation integration.
