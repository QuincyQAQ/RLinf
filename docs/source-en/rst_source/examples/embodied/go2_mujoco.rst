RL with Go2 MuJoCo
==================

Train a Unitree Go2 velocity policy with CPU MuJoCo simulation and RLinf PPO, then save and resume a short run. This recipe provides a simulation integration; the trained policy requires separate validation before deployment on a robot.

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

      CPU simulation · 1 NVIDIA GPU for learning

The smoke recipe uses 64 parallel environments, 24 control steps per rollout and three training updates. The environment process runs MuJoCo on CPU; policy inference and optimization use the configured accelerator. The current integration supports one environment process and one action per request. Multi-process curriculum synchronization and Ascend execution of this RLinf recipe have not been validated.

Tasks
~~~~~

.. list-table::
   :header-rows: 1

   * - Task
     - Configuration
   * - Go2 planar velocity tracking
     - ``env_type: go2_mujoco``; 20-second episodes; tilt and body-contact termination.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Field
     - Specification
   * - Observation
     - 45 actor inputs and 60 privileged critic inputs, including body motion, command, joint feedback and the previous action.
   * - Action
     - 12 joint targets computed as default pose plus 0.25 times the policy action; PD gains Kp = 25 and Kd = 0.5.
   * - Reward
     - 16 weighted velocity-tracking, stability and motion terms, integrated at a 0.02-second control period.
   * - Prompt
     - Not used.

The actor and critic each use 512/256/128 ELU hidden layers. The task follows Unitree RL Lab commit ``4960b84732b0c2ec593dccbfe963fda1bcd7b1e3`` and the bundled model comes from Unitree MuJoCo commit ``1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d``. Physics uses MuJoCo, so contact and friction differ from the Isaac Lab task. RLinf PPO uses its own value loss and optimizer schedule; this recipe does not claim numerical equivalence with an RSL-RL training run.

Installation
------------

.. include:: _setup_common.rst

Install the CPU simulator and compile its parallel stepping extension from the repository root. A C++17 compiler, OpenMP and Python development headers are required:

.. code:: bash

   bash requirements/install.sh embodied --env go2_mujoco --install-rlinf
   source .venv/bin/activate
   python -m rlinf.envs.sim.go2_mujoco.build

The bundled flat-ground model needs no asset download. The native extension must be rebuilt after changing Python version or CPU architecture.

Run It
------

Start the three-update smoke run from the repository root:

.. code:: bash

   export PYTHONPATH="$PWD"
   export EMBODIED_PATH="$PWD/examples/embodiment"
   export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
   export OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0
   python examples/embodiment/train_embodied_agent.py --config-name go2_mujoco_ppo

Checkpoints appear under ``../rlinf_integration/runs/go2_mujoco_smoke/checkpoints/``. Resume from step three and run two more updates:

.. code:: bash

   python examples/embodiment/train_embodied_agent.py --config-name go2_mujoco_ppo \
       runner.resume_dir="$PWD/../rlinf_integration/runs/go2_mujoco_smoke/checkpoints/global_step_3" \
       runner.max_epochs=5 runner.max_steps=5 \
       runner.logger.experiment_name=go2_mujoco_resume

Resume restores the policy, optimizer and scheduler through RLinf checkpoints. Simulation state is reinitialized. To initialize a new RLinf run from an RSL-RL Go2 checkpoint instead, set ``actor.model.initial_checkpoint=<checkpoint>``; this imports model weights, not the old optimizer or training step. Keep ``num_action_chunks=1`` and disable Torch compilation and CUDA graphs for this recipe.

Visualization and Results
-------------------------

The smoke run was checked on an RTX 3070 Laptop GPU with PyTorch 2.7.0 and MuJoCo 3.3.7. It completed three updates, saved checkpoints, resumed for two further updates, and changed model weights. A separate two-update end-to-end test exercises two-step episode truncation and evaluation. These checks establish the training loop, not convergence or robot readiness. The earlier 50000-update Go2 baseline was trained independently with RSL-RL and is not an RLinf training result.

Run the reset, action-format and policy consistency tests on CPU after compiling the extension:

.. code:: bash

   PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/unit_tests/test_go2_env.py -q

See :doc:`/rst_source/guides/resume` for the general checkpoint workflow.
