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

In the training environment, run ``python -c "import isaacsim"`` once in an interactive terminal and respond to the NVIDIA license prompt. For a batch job, set ``OMNI_KIT_ACCEPT_EULA=YES`` in the job environment if you accept the terms. See `NVIDIA's Isaac Sim setup instructions <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html#running-isaac-sim>`_.

If you use a local Isaac asset mirror, set ``env.train.init_params.asset_root`` to its Isaac Sim 5.1 asset directory.

Prepare the Starting Weights
----------------------------

The default stairs recipe starts from the C baseline. Download the `reference experiment (ZIP) <https://github.com/QuincyQAQ/RLinf/archive/23f16691d4dede5dab949a5cbf7934f6021e7ddf.zip>`_ and place its extracted project directory next to this checkout as ``go2-reference``. From the root of this RLinf checkout, download and verify the weights.

.. code-block:: bash

   python ../go2-reference/examples/embodiment/go2_abc/prepare_data.py --task C --data "${GO2_ABC_DATA:-$PWD/data/go2_abc}"

The C recipe reads ``models/C/baseline.pt`` under ``GO2_ABC_DATA`` (default ``data/go2_abc``). Use ``--task A`` or ``--task B`` for the other baselines. To use your own weights, set ``actor.model.model_path`` to a checkpoint with a 235-input, 12-action actor and matching critic. Set it to ``null`` to train from scratch.

Run It
------

To reproduce the published stair-passage rate, use the `reference setup and workflow <https://github.com/QuincyQAQ/RLinf/blob/23f16691d4dede5dab949a5cbf7934f6021e7ddf/examples/embodiment/go2_abc/README.md>`_ in ``go2-reference``. The ``C_rlinf`` run trains the policy and evaluates it with seeds 23456 and 34567.

For native RLinf training, start the stairs continuation from the RLinf repository root.

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh isaaclab_go2_c_ppo

Choose ``isaaclab_go2_a_ppo`` for flat ground or ``isaaclab_go2_b_ppo`` for rough terrain. By default A and B use 2,048 environments and C uses 1,024. The runner saves actor checkpoints every 50 updates for 1,000 updates. The launch log and run directory are under ``logs/``.

The Go2 actor and critic share ``actor.optim.lr``. Setting ``actor.optim.value_lr`` separately has no effect. Keep ``actor.optim.critic_warmup_steps`` at its default of zero.

For six-direction commands on hardware, see :doc:`go2`.
