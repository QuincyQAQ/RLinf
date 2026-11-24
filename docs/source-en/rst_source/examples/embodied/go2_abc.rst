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

Choose a task and set ``actor.model.model_path`` in its YAML to the corresponding Go2 weights. The continuation recipes expect a 235-input, 12-action actor and a matching critic. For a newly initialized policy leave this path empty. When using a local Isaac asset mirror, set ``env.train.init_params.asset_root`` to its Isaac Sim 5.1 asset directory.

Run It
------

From the RLinf root, start the stairs continuation:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh isaaclab_go2_c_ppo

Choose ``isaaclab_go2_a_ppo`` for flat ground or ``isaaclab_go2_b_ppo`` for rough terrain. By default A and B use 2,048 environments and C uses 1,024. The runner saves actor checkpoints every 50 updates for 1,000 updates. The launch log and run directory are under ``logs/``.

For six-direction commands on hardware, see :doc:`go2`.
