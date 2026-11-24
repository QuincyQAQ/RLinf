用 PPO 训练 Go2 运动策略
===============================

用 RLinf PPO 在 Isaac Lab 中训练 Go2 的运动 policy，控制 12 个关节。任务 A 在平地跟踪机体速度指令，B 穿越崎岖地形，C 爬楼梯。每个 policy 读取 48 个本体状态和指令值，以及 187 个地形高度采样值。

安装
----

先安装 Go2 运行环境（Isaac Lab 2.3.2 和 Isaac Sim 5.1），再在已激活的环境中安装固定版本的 `Go2 运动基准库 <https://github.com/viktoriageorrg04/bsc-thesis-multi-task-RL-env>`_，其中包含三项仿真任务：

.. code-block:: bash

   bash requirements/install.sh embodied --env go2_abc
   source .venv/bin/activate
   uv pip install -r requirements/embodied/envs/go2_abc.txt

在已激活的环境中运行 ``python -c "import isaacsim"``，按终端提示确认 `NVIDIA 许可条款 <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html#running-isaac-sim>`_。提交批处理作业时，同意条款后可在作业环境中设置 ``OMNI_KIT_ACCEPT_EULA=YES``。

若使用本地 Isaac 资产镜像，将 ``env.train.init_params.asset_root`` 设为 Isaac Sim 5.1 的资产目录。

准备初始权重
------------

楼梯任务从 C 的基线权重续训。将 `参考实验源码压缩包 <https://github.com/QuincyQAQ/RLinf/archive/23f16691d4dede5dab949a5cbf7934f6021e7ddf.zip>`_ 解压后的项目目录放到 ``../go2-reference``，再从当前 RLinf 仓库根目录下载并校验权重。

.. code-block:: bash

   python ../go2-reference/examples/embodiment/go2_abc/prepare_data.py --task C --data "${GO2_ABC_DATA:-$PWD/data/go2_abc}"

C 配置读取 ``GO2_ABC_DATA`` 目录下的 ``models/C/baseline.pt``，默认目录为 ``data/go2_abc``。任务 A 和 B 分别用 ``--task A`` 和 ``--task B``。使用自己的权重时，将 ``actor.model.model_path`` 设为对应文件路径，权重需包含输入 235 维、输出 12 维的 actor 和配套的 critic。从头训练可将该项设为 ``null``。

运行
----

从 RLinf 仓库根目录启动楼梯任务续训。

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh isaaclab_go2_c_ppo

平地任务使用 ``isaaclab_go2_a_ppo``，越野任务使用 ``isaaclab_go2_b_ppo``。A 和 B 默认使用 2,048 个并行环境，C 使用 1,024 个。训练共更新 1,000 次，每 50 次保存一份 actor 权重。启动日志和运行目录位于 ``logs/`` 下。

Go2 的 actor 和 critic 从第一步起一起训练，共用 ``actor.optim.lr`` 设置的学习率。

要复现已公布的楼梯通过率，在 ``go2-reference`` 中按 `参考实验说明 <https://github.com/QuincyQAQ/RLinf/blob/23f16691d4dede5dab949a5cbf7934f6021e7ddf/examples/embodiment/go2_abc/README.zh-CN.md>`_ 运行 ``C_rlinf``。续训结束后，该实验分别用种子 23456 和 34567 评测。

真机六方向控制见 :doc:`go2`。
