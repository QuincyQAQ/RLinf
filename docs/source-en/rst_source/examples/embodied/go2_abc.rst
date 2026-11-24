Go2 Locomotion with PPO
=======================

Train a 12-joint Go2 policy in Isaac Lab with RLinf PPO. Task A tracks body velocity on flat ground, B walks over rough terrain, and C climbs stairs. Each policy uses 48 proprioceptive and command values plus 187 terrain height samples.

Installation
------------

Install the Go2 runtime (Isaac Lab 2.3.2 and Isaac Sim 5.1), then install the three task definitions from the pinned `Go2 locomotion benchmark <https://github.com/viktoriageorrg04/bsc-thesis-multi-task-RL-env>`_ in the activated environment:

.. code-block:: bash

   bash requirements/install.sh embodied --env go2_abc
   source .venv/bin/activate
   uv pip install -r requirements/embodied/envs/go2_abc.txt

Run ``python -c "import isaacsim"`` in the activated environment to answer the `NVIDIA license prompt <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html#running-isaac-sim>`_. For a batch job, you can set ``OMNI_KIT_ACCEPT_EULA=YES`` in the job environment after accepting the terms.

Prepare the Starting Weights
----------------------------

The stairs recipe starts from the C baseline. Unzip the `reference experiment <https://github.com/QuincyQAQ/RLinf/archive/23f16691d4dede5dab949a5cbf7934f6021e7ddf.zip>`_ and move the extracted project directory to ``../go2-reference``. From the RLinf repository root, download and verify the weights.

.. code-block:: bash

   python ../go2-reference/examples/embodiment/go2_abc/prepare_data.py --task C --data "${GO2_ABC_DATA:-$PWD/data/go2_abc}"

The C recipe reads ``models/C/baseline.pt`` under ``GO2_ABC_DATA`` (default ``data/go2_abc``). Use ``--task A`` or ``--task B`` for the other baselines. To use your own weights, set ``actor.model.model_path`` to a checkpoint with a 235-input, 12-action actor and matching critic. Set it to ``null`` to train from scratch.

Run It
------

From the RLinf repository root, continue training the stairs policy.

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh isaaclab_go2_c_ppo

Choose ``isaaclab_go2_a_ppo`` for flat ground or ``isaaclab_go2_b_ppo`` for rough terrain. By default A and B use 2,048 environments and C uses 1,024. The runner saves actor checkpoints every 50 updates for 1,000 updates. The launch log and run directory are under ``logs/``.

The Go2 actor and critic train together from the first update and share the learning rate set by ``actor.optim.lr``.

For a local Isaac Sim 5.1 asset mirror, launch the same configuration with Hydra. Replace the asset path with the directory on your machine.

.. code-block:: bash

   export GO2_ASSET_ROOT=/absolute/path/to/isaac_sim_5.1
   python -m examples.embodiment.train_embodied_agent \
     --config-path "$PWD/examples/embodiment/config" --config-name isaaclab_go2_c_ppo \
     "env.train.init_params.asset_root=$GO2_ASSET_ROOT" "runner.logger.log_path=$PWD/logs/go2-stairs"

Evaluate Your Stairs Checkpoint
-------------------------------

Keep the ``../go2-reference`` directory from the weights step. In the activated environment, install the reference RSL-RL loader and fetch the evaluator and benchmark sources alongside it. Run this setup once from the RLinf repository root.

.. code-block:: bash

   uv pip install "rsl-rl-lib==3.1.2"
   (
     set -e
     mkdir -p ../public-evaluator
     cd ../public-evaluator
     curl -fLO https://raw.githubusercontent.com/QuincyQAQ/RLinf/2bb493a79e4e923cb2dcafd999d7e372a803afb1/artifacts/go2_seed0_20261122/runtime_sources.tar.gz
     echo '827c9fd752c8c4cc8d5c6787a66df2b75199aab774d7a8901523e82e11284751  runtime_sources.tar.gz' | sha256sum -c -
     tar -xzf runtime_sources.tar.gz a40/{evaluate_continuation.py,evaluate_dual.py,passage_metrics.py}
     git clone --filter=blob:none --no-checkout https://github.com/viktoriageorrg04/bsc-thesis-multi-task-RL-env.git benchmark
     git -C benchmark checkout --detach 743ac04d9543b04b7854f1100bfa0cc69c993106
   )

In a Slurm GPU allocation, set ``RUN_DIR`` to the log directory of your training run. The standard launcher uses a timestamped directory under ``logs/``; the mirror command above uses ``logs/go2-stairs``. Choose a fresh output directory for each checkpoint. Append ``--asset-root "$GO2_ASSET_ROOT"`` if using the mirror.

.. code-block:: bash

   RUN_DIR=logs/go2-stairs
   CHECKPOINT="$RUN_DIR/C_rlinf/checkpoints/global_step_1000/actor/model_state_dict/full_weights.pt"
   python examples/embodiment/go2_abc/evaluate_stairs.py "$CHECKPOINT" "$RUN_DIR/stairs-eval"

The evaluator exports your trained weights to ``stairs-eval/policy.pt`` and runs 256 parallel environments to collect at least 1,024 episodes per seed. The two seed directories contain the episode records. After verifying both reports, the command prints the pooled tracking and simulated stair-passage rates and writes their counts and rates to ``stairs-eval/summary.json``.

Reference PPO Results
---------------------

In the `A40 reference experiments <https://github.com/QuincyQAQ/RLinf/blob/23f16691d4dede5dab949a5cbf7934f6021e7ddf/examples/embodiment/go2_abc/README.md#results-and-offline-checks>`_, both PPO methods continued from the same task-specific baseline for 1,000 updates. Each cell gives the success rate and the number of successful episodes out of all episodes pooled across evaluation seeds 23456 and 34567. Both training seeds use the same frozen baseline. Task C measures velocity tracking and crossing the simulated stairs separately.

.. list-table::
   :header-rows: 1

   * - Metric
     - Training seed
     - Baseline
     - RSL-RL PPO
     - RLinf PPO (Huber)
   * - A tracking
     - 0
     - 91.47% (1887/2063)
     - 93.33% (1917/2054)
     - 96.73% (1981/2048)
   * - A tracking
     - 2
     - 91.47% (1887/2063)
     - 90.90% (1878/2066)
     - 98.73% (2022/2048)
   * - B tracking
     - 0
     - 99.85% (2045/2048)
     - 99.61% (2040/2048)
     - 99.85% (2045/2048)
   * - B tracking
     - 2
     - 99.85% (2045/2048)
     - 99.76% (2043/2048)
     - 99.90% (2046/2048)
   * - C tracking
     - 0
     - 99.85% (2045/2048)
     - 100.00% (2048/2048)
     - 100.00% (2048/2048)
   * - C tracking
     - 2
     - 99.85% (2045/2048)
     - 99.80% (2044/2048)
     - 99.80% (2044/2048)
   * - C stair passage
     - 0
     - 21.39% (438/2048)
     - 44.82% (918/2048)
     - 45.17% (925/2048)
   * - C stair passage
     - 2
     - 21.39% (438/2048)
     - 33.50% (686/2048)
     - 41.94% (859/2048)

Replay on CPU
-------------

Run the C baseline on the four synthetic frames from ``go2-reference``. Choose a new output path. Each line contains the timestamp, twelve joint targets (rad), and twelve motor torques (N·m) in SDK order.

.. code-block:: bash

   python -m examples.embodiment.go2_abc.deploy --task C --group baseline \
     --frames ../go2-reference/examples/embodiment/go2_abc/frames.example.jsonl \
     --output commands.jsonl

For a trained RLinf actor or an RSL-RL checkpoint, replace ``--group baseline`` with ``--checkpoint path/to/policy.pt`` and select its task. To load weights by group (``baseline``, ``ppo``, or ``rlinf``), pass ``--data`` to override ``GO2_ABC_DATA``.

To replay your own recording, follow the `sample frame format <https://github.com/QuincyQAQ/RLinf/blob/23f16691d4dede5dab949a5cbf7934f6021e7ddf/examples/embodiment/go2_abc/frames.example.jsonl>`_. Supply joint and IMU data plus measured body velocity and terrain heights. Record timestamps on one monotonic clock at 5 ms intervals. The policy updates every fourth frame. Express base and ground heights in the same world frame. Sample the yaw-aligned grid in 0.1 m steps, with y from −0.5 to 0.5 m and x from −0.8 to 0.8 m within each row.

For six-direction commands on hardware, see :doc:`go2`.
