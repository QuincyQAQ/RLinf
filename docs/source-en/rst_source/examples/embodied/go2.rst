Go2 High-Level Control
======================

Read Go2 EDU state and check six directions through the Unitree controller. The operator observes the motion and confirms the stop.

Overview
--------

.. grid:: 2 4 4 4

   .. grid-item-card:: Models

      Manual velocity commands.

   .. grid-item-card:: Algorithms

      No training recipe.

   .. grid-item-card:: Tasks

      State reception and six directions.

   .. grid-item-card:: Hardware

      Go2 EDU and a Linux controller.

Tasks
~~~~~

.. list-table::
   :header-rows: 1

   * - Check
     - Result
   * - Read-only
     - State samples without motion commands.
   * - One direction
     - Forward, backward, left, right, turn-left, or turn-right, followed by a stop request.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Field
     - Contract
   * - Observation
     - Orientation, angular velocity, twelve joint positions and velocities, battery voltage, and four raw foot-force readings.
   * - Action
     - Body velocity in m/s, m/s, rad/s, bounded by absolute values of 0.3, 0.3, and 0.5.
   * - Reward
     - Zero. This check does not train a policy.

Installation
------------

Prepare a Python 3.11 RLinf environment using :doc:`Installation </rst_source/start/installation>`. From the repository root, install the optional SDK at the revision used for the hardware checks:

.. code-block:: bash

   bash requirements/install.sh embodied --env go2_sport
   source .venv/bin/activate

Follow the pinned `SDK installation guide <https://github.com/unitreerobotics/unitree_sdk2_python/tree/814556d15970dd2ecf1c9984e845ca02ab07e206>`_ for CycloneDDS prerequisites and robot networking. The controller computer needs no GPU or policy weights.

Run It
------

Replace ``<interface>`` with the network interface attached to the Go2. Preview a command without opening SDK channels:

.. code-block:: bash

   python -m examples.embodiment.go2_sport.run --interface <interface> --direction forward --dry-run

The plan reports ``hardware_access: false``. Check state reception separately:

.. code-block:: bash

   python -m examples.embodiment.go2_sport.run --interface <interface> --readonly

This writes ten state-validation results to ``outputs/go2_sport/``. To inspect a single checksum-verified joint and IMU sample from ``rt/lowstate``, run the read-only subscriber in a fresh process.

.. code-block:: bash

   python -m examples.embodiment.go2_abc.sdk2 --interface <interface>

For motion, keep the original controller active, verify standing posture and clear space, and have the remote ready on site. Close other motion programs before running one direction:

.. code-block:: bash

   python -m examples.embodiment.go2_sport.run --interface <interface> --direction forward

Enter ``GO`` only after the on-site checks. Choose ``forward``, ``backward``, ``left``, ``right``, ``turn_left``, or ``turn_right``. Each invocation checks one direction for one second at 10 Hz; ``--seconds 2`` selects two seconds. ``--output <path>`` selects a fresh report path and refuses to overwrite a file.

.. warning::

   A stop request or a low SDK velocity estimate does not establish physical rest. Observe the robot directly and use the remote if motion continues. Network loss or process failure can defeat software stop requests.

Visualization and Results
-------------------------

Reports separate interface errors, SDK stop estimates, and operator confirmation. The command never marks a physical test as passed automatically. Quantitative distance, turn angle, and stopping-distance acceptance remains incomplete.

For policy training in simulation, see :doc:`go2_abc`.
