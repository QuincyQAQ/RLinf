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

If you use a local Isaac asset mirror, set ``env.train.init_params.asset_root`` to its Isaac Sim 5.1 asset directory.

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

To reproduce the published stair-passage rate, use the `reference workflow <https://github.com/QuincyQAQ/RLinf/blob/23f16691d4dede5dab949a5cbf7934f6021e7ddf/examples/embodiment/go2_abc/README.md>`_ in ``go2-reference``. Its ``C_rlinf`` run trains the policy and evaluates it with seeds 23456 and 34567.

For six-direction commands on hardware, see :doc:`go2`.
