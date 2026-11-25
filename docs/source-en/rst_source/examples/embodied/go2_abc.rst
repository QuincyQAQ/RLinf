Go2 Locomotion with PPO
=======================

Train a 12-joint Go2 policy in Isaac Lab with RLinf PPO. Task A tracks body velocity on flat ground, B walks over rough terrain, and C climbs stairs. Each policy uses 48 proprioceptive and command values plus 187 terrain height samples.

**Prerequisites:** Prepare RLinf on an NVIDIA GPU machine using the :doc:`installation guide </rst_source/start/installation>`.

Setup and Run
-------------

.. tab-set::

   .. tab-item:: Isaac Lab

      **1. Install the Go2 runtime**

      Install the Go2 runtime (Isaac Lab 2.3.2 and Isaac Sim 5.1), then install the three task definitions from the pinned `Go2 locomotion benchmark <https://github.com/viktoriageorrg04/bsc-thesis-multi-task-RL-env>`_ in the activated environment:

      .. code-block:: bash

         bash requirements/install.sh embodied --env go2_abc
         source .venv/bin/activate
         uv pip install -r requirements/embodied/envs/go2_abc.txt

      Run ``python -c "import isaacsim"`` in the activated environment to answer the `NVIDIA license prompt <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html#running-isaac-sim>`_. For a batch job, you can set ``OMNI_KIT_ACCEPT_EULA=YES`` in the job environment after accepting the terms.

      **2. Prepare the starting weights**

      From the RLinf repository root, fetch the frozen starting weights for the stairs task. The command verifies their SHA-256 before placing them in the training data directory.

      .. code-block:: bash

         python -m examples.embodiment.go2_abc.prepare_baseline C

      The C recipe reads ``models/C/baseline.pt`` under ``GO2_ABC_DATA`` (default ``data/go2_abc``). Pass ``A`` or ``B`` to fetch the starting weights for that task.

      To use your own weights, set ``actor.model.model_path`` to a checkpoint with a 235-input, 12-action actor and matching critic. Set it to ``null`` to train from scratch.

      **3. Train the policy**

      From the RLinf repository root, continue training the stairs policy.

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh isaaclab_go2_c_ppo

      Choose ``isaaclab_go2_a_ppo`` for flat ground or ``isaaclab_go2_b_ppo`` for rough terrain. By default A and B use 2,048 environments and C uses 1,024. The runner saves actor checkpoints every 50 updates for 1,000 updates. The launch log and run directory are under ``logs/``.

      The Go2 actor and critic train together from the first update and share the learning rate set by ``actor.optim.lr``. The default training seed is 0. Append ``actor.seed=2`` to the launch command to set the actor, rollout, and training environments to seed 2.

      For a local Isaac Sim 5.1 asset mirror, launch the same configuration with Hydra. Replace the asset path with the directory on your machine.

      .. code-block:: bash

         export GO2_ASSET_ROOT=/absolute/path/to/isaac_sim_5.1
         python -m examples.embodiment.train_embodied_agent \
           --config-path "$PWD/examples/embodiment/config" --config-name isaaclab_go2_c_ppo \
           "env.train.init_params.asset_root=$GO2_ASSET_ROOT" "runner.logger.log_path=$PWD/logs/go2-stairs"

Evaluate Your Stairs Checkpoint
-------------------------------

Install the RSL-RL checkpoint loader and fetch the four evaluator source files. The evaluator uses the task definitions installed in step 1. Run this setup from the RLinf repository root.

.. code-block:: bash

   uv pip install "rsl-rl-lib==3.1.2"
   (
     set -e
     runtime=../public-evaluator
     mkdir -p "$runtime/a40" "$runtime/benchmark/scripts"
     wrappers=https://raw.githubusercontent.com/QuincyQAQ/RLinf/b405f67382ce63af2401a2a9ff5b0b3c7f28ccb7/a40
     for module in evaluate_continuation evaluate_dual passage_metrics; do
       curl -fsSL "$wrappers/$module.py" -o "$runtime/a40/$module.py"
     done
     benchmark=https://raw.githubusercontent.com/viktoriageorrg04/bsc-thesis-multi-task-RL-env/743ac04d9543b04b7854f1100bfa0cc69c993106
     curl -fsSL "$benchmark/scripts/evaluate.py" -o "$runtime/benchmark/scripts/evaluate.py"
   )

In a Slurm GPU allocation, set ``RUN_DIR`` to the log directory of your training run. The standard launcher uses a timestamped directory under ``logs/``; the mirror command above uses ``logs/go2-stairs``. Choose a fresh output directory for each checkpoint. Append ``--asset-root "$GO2_ASSET_ROOT"`` if using the mirror.

.. code-block:: bash

   RUN_DIR=logs/go2-stairs
   CHECKPOINT="$RUN_DIR/C_rlinf/checkpoints/global_step_1000/actor/model_state_dict/full_weights.pt"
   python -m examples.embodiment.go2_abc.evaluate_stairs "$CHECKPOINT" "$RUN_DIR/stairs-eval"

The evaluator exports your trained weights to ``stairs-eval/policy.pt`` and runs 256 parallel environments to collect at least 1,024 episodes per seed. The two seed directories contain the episode records. After verifying both reports, the command prints the pooled tracking and simulated stair-passage rates and writes their counts and rates to ``stairs-eval/summary.json``.

For the RSL-RL PPO comparison, place the pinned `train.py <https://github.com/viktoriageorrg04/bsc-thesis-multi-task-RL-env/blob/743ac04d9543b04b7854f1100bfa0cc69c993106/scripts/train.py>`_ beside ``evaluate.py``. Place ``train_continuation_rsl.py`` from the `trainer sources <https://github.com/QuincyQAQ/RLinf/tree/b405f67382ce63af2401a2a9ff5b0b3c7f28ccb7/a40>`_ in ``../public-evaluator/a40/``. Task B also imports ``train_legacy_distribution.py`` and ``b1_numerical_boundary.py`` from the same directory.

Select your task’s ``*_ppo.json`` file from the `training recipes <https://github.com/QuincyQAQ/RLinf/tree/23f16691d4dede5dab949a5cbf7934f6021e7ddf/examples/embodiment/go2_abc/config>`_. Run ``train_continuation_rsl.py`` with the arguments and environment variables in that file. Set the frozen checkpoint and asset paths for your machine and choose training seed 0 or 2 with ``--seed``.

Add this RLinf checkout to ``PYTHONPATH`` and append ``env.rewards.base_height.func=rlinf.envs.sim.isaaclab.tasks.go2:terrain_base_height_l2`` to both training and evaluation commands.

For evaluation, use the same task's ``*_baseline.json`` with ``evaluate_continuation.py``. Set ``--checkpoint`` to the frozen input or the final RSL-RL checkpoint (``model_999.pt`` for A and B, ``model_960.pt`` for C after its 1,000 updates).

Set ``GO2_EVAL_SEED`` to 23456 for the first evaluation and 34567 for the second. Pass a fresh ``--output_dir`` for each.

Simulation Results
------------------

For each task, RSL-RL PPO and RLinf PPO start from the same baseline and train for 1,000 updates with training seeds 0 and 2. The frozen baseline is evaluated once per task and appears in both rows.

Each cell shows the success rate and the number of successful episodes out of all episodes pooled across evaluation seeds 23456 and 34567. For task C, the evaluator scores velocity tracking and uses the simulated trajectory to determine whether Go2 crosses the stairs.

.. list-table::
   :header-rows: 1

   * - Metric
     - Training seed
     - Frozen baseline
     - RSL-RL PPO
     - RLinf PPO (Huber)
   * - A velocity tracking
     - 0
     - 91.47% (1887/2063)
     - 93.33% (1917/2054)
     - 96.78% (1984/2050)
   * - A velocity tracking
     - 2
     - 91.47% (1887/2063)
     - 90.90% (1878/2066)
     - 98.29% (2013/2048)
   * - B velocity tracking
     - 0
     - 99.85% (2045/2048)
     - 99.61% (2040/2048)
     - 99.85% (2045/2048)
   * - B velocity tracking
     - 2
     - 99.85% (2045/2048)
     - 99.76% (2043/2048)
     - 99.80% (2044/2048)
   * - C velocity tracking
     - 0
     - 99.85% (2045/2048)
     - 100.00% (2048/2048)
     - 100.00% (2048/2048)
   * - C velocity tracking
     - 2
     - 99.85% (2045/2048)
     - 99.80% (2044/2048)
     - 99.90% (2046/2048)
   * - C stair passage
     - 0
     - 21.39% (438/2048)
     - 44.82% (918/2048)
     - 56.69% (1161/2048)
   * - C stair passage
     - 2
     - 21.39% (438/2048)
     - 33.50% (686/2048)
     - 62.06% (1271/2048)

Replay on CPU
-------------

Replay one synthetic frame from the control test fixture with your trained checkpoint. The output records the timestamp, twelve joint targets (rad), and twelve motor torques (N·m) in SDK order.

.. code-block:: bash

   REPLAY_DIR=$(mktemp -d "${GO2_ABC_DATA:-data/go2_abc}/replay.XXXXXX")
   python -c 'import json; print(json.dumps(json.load(open("tests/fixtures/go2_control_reference.json"))["frame"]))' > "$REPLAY_DIR/frame.jsonl"
   python -m examples.embodiment.go2_abc.deploy --task C --checkpoint "$CHECKPOINT" \
     --frames "$REPLAY_DIR/frame.jsonl" --output "$REPLAY_DIR/commands.jsonl"

To replay the frozen starting weights from step 2, replace ``--checkpoint "$CHECKPOINT"`` with ``--group baseline``. The optional ``--data`` argument selects the data root for group-based loading.

For your own recording, follow the structure of ``frame.jsonl``. Supply joint and IMU data plus measured body velocity and terrain heights. Record timestamps on one monotonic clock at 5 ms intervals. The policy updates every fourth frame. Express base and ground heights in the same world frame. Sample the yaw-aligned grid in 0.1 m steps, with y from −0.5 to 0.5 m and x from −0.8 to 0.8 m within each row.

For six-direction commands on hardware, see :doc:`go2`.
