用 PPO 训练 Go2 运动策略
===============================

在 Isaac Lab 中用 RLinf PPO 训练 Go2 的 12 个关节。任务 A 在平地跟踪机体速度指令，B 穿越崎岖地形，C 爬楼梯。每个策略读取 48 个本体状态和指令值，以及 187 个地形高度采样值。

安装
----

先安装 Go2 运行环境（Isaac Lab 2.3.2 和 Isaac Sim 5.1），再在已激活的环境中安装固定版本的 `Go2 运动基准库 <https://github.com/viktoriageorrg04/bsc-thesis-multi-task-RL-env>`_，其中包含三项仿真任务：

.. code-block:: bash

   bash requirements/install.sh embodied --env go2_abc
   source .venv/bin/activate
   uv pip install -r requirements/embodied/envs/go2_abc.txt

选择任务，在对应 YAML 中将 ``actor.model.model_path`` 设为该任务的 Go2 权重路径。续训权重需包含输入 235 维、输出 12 维的 actor 和配套的 critic。如需随机初始化，将此路径留空。使用本地 Isaac 资产镜像时，将 ``env.train.init_params.asset_root`` 设为 Isaac Sim 5.1 的资产目录。

运行
----

从 RLinf 仓库根目录启动楼梯策略续训：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh isaaclab_go2_c_ppo

平地任务使用 ``isaaclab_go2_a_ppo``，越野任务使用 ``isaaclab_go2_b_ppo``。A 和 B 默认使用 2,048 个并行环境，C 使用 1,024 个。训练共更新 1,000 次，每 50 次保存一份 actor 权重。启动日志和运行目录位于 ``logs/`` 下。

真机六方向控制见 :doc:`go2`。
