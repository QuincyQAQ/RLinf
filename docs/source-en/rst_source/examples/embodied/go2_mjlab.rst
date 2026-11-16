RL with Go2 Mjlab
=================

Train the official Unitree Go2 flat-ground velocity task with native RLinf PPO. CPU workers run Mjlab physics; policy inference and training run on NVIDIA GPUs or Huawei Ascend NPUs. The Ascend recipe uses eight NPUs and 4096 environments.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      Asymmetric Go2 MLP

   .. grid-item-card:: Algorithms
      :text-align: center

      PPO

   .. grid-item-card:: Tasks
      :text-align: center

      Flat-ground velocity tracking

   .. grid-item-card:: Hardware
      :text-align: center

      CPU physics · NVIDIA or :ref:`Huawei Ascend <go2-mjlab-hardware>` learning

Tasks
~~~~~

.. list-table::
   :header-rows: 1

   * - Environment
     - Task / Suite
     - Config / Weights
     - Focus
   * - Go2 Mjlab
     - Unitree-Go2-Flat
     - ``go2_mjlab_ppo_ascend``; random initialization
     - Standing, planar velocity and turning commands

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Field
     - Specification
   * - Observation
     - 47 actor inputs and 74 privileged critic inputs, with observation normalization.
   * - Action
     - 12 Gaussian joint actions. Joint position targets equal the default pose plus 0.25 times the action.
   * - Reward
     - The pinned upstream velocity, gait, stability and contact rewards, at a 0.02-second control period.
   * - Prompt
     - Not used.

The task follows `Unitree RL Mjlab <https://github.com/unitreerobotics/unitree_rl_mjlab>`_ commit ``1425b15f73bd4095f0df53709d7c389c3eb9e790``. The full configuration uses 10001 updates, 24 steps per update, five PPO epochs and four minibatches. It enables clipped squared value loss and analytic-KL adaptive learning rate with target 0.01.

Installation
------------

.. include:: _setup_common.rst

On NVIDIA hosts, install the task and its pinned physics packages:

.. code:: bash

   bash requirements/install.sh embodied --env go2_mjlab --install-rlinf
   source .venv/bin/activate

Run It
------

Run a three-update single-GPU smoke test from the repository root:

.. code:: bash

   export REPO_PATH=$(pwd)
   bash tests/e2e_tests/embodied/run.sh go2_mjlab_ppo

The run writes TensorBoard events and checkpoints under ``/workspace/results/go2_mjlab_ppo``. Use the Ascend instructions below for the verified eight-device configuration.

.. _go2-mjlab-hardware:

Run on Different Hardware Backends
----------------------------------

Huawei Ascend CANN
~~~~~~~~~~~~~~~~~~~~

The verified stack is Linux aarch64 with glibc 2.34, CANN 8.5, PyTorch 2.7.1 and TorchNPU 2.7.1.post2. Physics uses a conda-forge CPU build of Warp 1.12; the PyPI ARM wheel requires glibc 2.35. The system libraries remain unchanged.

Follow the :download:`Ascend runtime recipe <../../../../../examples/embodiment/go2_mjlab/README.md>` to create the isolated runtime from the :download:`conda lock <../../../../../examples/embodiment/go2_mjlab/ascend-conda-explicit.lock>` and :download:`pip lock <../../../../../examples/embodiment/go2_mjlab/ascend-requirements.lock>`. The automatic NVIDIA installer does not create this Ascend runtime.

After installation, select the matching interpreter and CANN setup script, then run the single-NPU smoke test:

.. code:: bash

   export GO2_PYTHON="$PWD/.venv-mjlab-ascend/bin/python"
   export CANN_ENV_FILE=/usr/local/Ascend/ascend-toolkit/set_env.sh
   bash examples/embodiment/go2_ascend/run_native.sh go2_mjlab_ppo_ascend_smoke

Run the 10001-update configuration after validation:

.. code:: bash

   bash examples/embodiment/go2_ascend/run_native.sh go2_mjlab_ppo_ascend

The placement uses NPU devices 8–15, eight actors and 64 CPU environment workers. Each CPU worker runs 64 environments with two Torch threads. Edit the placement in the configuration if device indices differ. CPU physics can dominate runtime; NPU occupancy alone does not measure training throughput.

Visualization and Results
-------------------------

The full run writes TensorBoard events under ``outputs/go2_mjlab/native_ascend/tensorboard`` and resumable checkpoints under ``go2_mjlab/checkpoints`` within that output directory. Checkpoints include model, optimizer, scheduler and normalization statistics. Resume resets physical state and restores the time-based command curriculum from the saved update count.

Single-NPU and eight-NPU short runs validate sampling, PPO updates and saving. These checks do not establish convergence, a success rate or real-robot deployment.

RLinf merges observation moments across actors once per rollout; upstream RSL-RL updates local moments after each environment step. RLinf bootstraps timeouts from terminal observations, while the upstream RSL wrapper uses its stored transition value. The task and listed settings follow upstream, with these execution differences documented rather than a claim of numerical equivalence. See :doc:`../../reference/algorithms/ppo` for native PPO settings.

Cross-Task Benchmark
--------------------

The separate ``rlinf-mjlab-cross-task-v1`` variant provides forward walking, omnidirectional walking, rough terrain, stairs and gap tasks. Each actor and critic receives a 235-dimensional observation, including a 187-ray height scan. Initialize new policies for these tasks; the existing 47/74-observation Flat configuration retains its interface.

Use ``go2_mjlab_benchmark_a1_ascend`` for forward walking, replacing ``a1`` with ``a2``, ``b1``, ``b2`` or ``c2`` for the other tasks. The 16-environment, three-update short configuration is ``go2_mjlab_benchmark_a1_smoke``. Its accelerator validation requires an available device; CPU task checks have passed. Schedule it after the original job releases device 8.

Evaluate each trained policy in a separate process for every task, with at least 1024 episodes per cell. The evaluator and matrix exporter are ``examples/embodiment/go2_mjlab/evaluate_benchmark.py`` and ``examples/embodiment/go2_mjlab/export_benchmark_matrix.py``. The complete commands, thresholds and simulator differences are documented in ``examples/embodiment/go2_mjlab/BENCHMARK.md``. Missing cells remain empty and short tests are excluded by default. This Mjlab variant uses the reference success criteria but retains Mjlab rewards, startup randomization and physics; it is not a numerical reproduction of the reference Isaac Lab results.
