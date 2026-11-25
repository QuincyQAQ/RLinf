用 PPO 训练 Go2 运动策略
===============================

用 RLinf PPO 在 Isaac Lab 中训练 Go2 的运动 policy，控制 12 个关节。任务 A 在平地跟踪机体速度指令，B 穿越崎岖地形，C 爬楼梯。每个 policy 读取 48 个本体状态和指令值，以及 187 个地形高度采样值。

**前置条件：** 按 :doc:`安装说明 </rst_source/start/installation>` 在配备 NVIDIA GPU 的机器上准备 RLinf。

安装与运行
----------

.. tab-set::

   .. tab-item:: Isaac Lab

      **1. 安装 Go2 运行环境**

      先安装 Go2 运行环境（Isaac Lab 2.3.2 和 Isaac Sim 5.1），再在已激活的环境中安装固定版本的 `Go2 运动基准库 <https://github.com/viktoriageorrg04/bsc-thesis-multi-task-RL-env>`_，其中包含三项仿真任务：

      .. code-block:: bash

         bash requirements/install.sh embodied --env go2_abc
         source .venv/bin/activate
         uv pip install -r requirements/embodied/envs/go2_abc.txt

      在已激活的环境中运行 ``python -c "import isaacsim"``，按终端提示确认 `NVIDIA 许可条款 <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html#running-isaac-sim>`_。提交批处理作业时，同意条款后可在作业环境中设置 ``OMNI_KIT_ACCEPT_EULA=YES``。

      **2. 准备初始权重**

      楼梯任务从 C 的基线权重续训。将 `参考实验源码压缩包 <https://github.com/QuincyQAQ/RLinf/archive/23f16691d4dede5dab949a5cbf7934f6021e7ddf.zip>`_ 解压后的项目目录放到 ``../go2-reference``，再从当前 RLinf 仓库根目录下载并校验权重。

      .. code-block:: bash

         python ../go2-reference/examples/embodiment/go2_abc/prepare_data.py --task C --data "${GO2_ABC_DATA:-$PWD/data/go2_abc}"

      C 配置读取 ``GO2_ABC_DATA`` 目录下的 ``models/C/baseline.pt``，默认目录为 ``data/go2_abc``。任务 A 和 B 分别用 ``--task A`` 和 ``--task B``。

      使用自己的权重时，将 ``actor.model.model_path`` 设为对应文件路径，权重需包含输入 235 维、输出 12 维的 actor 和配套的 critic。从头训练可将该项设为 ``null``。

      **3. 训练策略**

      从 RLinf 仓库根目录启动楼梯任务续训。

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh isaaclab_go2_c_ppo

      平地任务使用 ``isaaclab_go2_a_ppo``，越野任务使用 ``isaaclab_go2_b_ppo``。A 和 B 默认使用 2,048 个并行环境，C 使用 1,024 个。训练共更新 1,000 次，每 50 次保存一份 actor 权重。启动日志和运行目录位于 ``logs/`` 下。

      Go2 的 actor 和 critic 从第一步起一起训练，共用 ``actor.optim.lr`` 设置的学习率。

      如果使用本地 Isaac Sim 5.1 资产镜像，可直接用 Hydra 启动同一份训练配置。将下面的资产目录换成本机路径。

      .. code-block:: bash

         export GO2_ASSET_ROOT=/absolute/path/to/isaac_sim_5.1
         python -m examples.embodiment.train_embodied_agent \
           --config-path "$PWD/examples/embodiment/config" --config-name isaaclab_go2_c_ppo \
           "env.train.init_params.asset_root=$GO2_ASSET_ROOT" "runner.logger.log_path=$PWD/logs/go2-stairs"

评测训练好的楼梯任务权重
------------------------

保留准备权重时解压的 ``../go2-reference`` 目录。在已激活的环境中安装参考实验使用的 RSL-RL 版本，并下载评测程序及对应版本的基准库源码。从 RLinf 仓库根目录执行一次即可。

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

在 Slurm 分配的 GPU 节点上，将 ``RUN_DIR`` 设为本次训练的日志目录。标准启动器在 ``logs/`` 下创建带时间戳的目录，上面的镜像示例使用 ``logs/go2-stairs``。每次评测指定新的输出目录。使用镜像时，在评测命令末尾加上 ``--asset-root "$GO2_ASSET_ROOT"``。

.. code-block:: bash

   RUN_DIR=logs/go2-stairs
   CHECKPOINT="$RUN_DIR/C_rlinf/checkpoints/global_step_1000/actor/model_state_dict/full_weights.pt"
   python examples/embodiment/go2_abc/evaluate_stairs.py "$CHECKPOINT" "$RUN_DIR/stairs-eval"

评测程序将本次训练的权重导出到 ``stairs-eval/policy.pt``，用 256 个并行环境为每个种子采集至少 1,024 个回合。两个种子目录分别保存逐回合记录。两份报告校验通过后，终端会显示合并后的速度跟踪成功率和仿真楼梯通过率，成功回合数、总回合数和比例写入 ``stairs-eval/summary.json``。

PPO 参考实验结果
----------------

在 `A40 参考实验 <https://github.com/QuincyQAQ/RLinf/blob/23f16691d4dede5dab949a5cbf7934f6021e7ddf/examples/embodiment/go2_abc/README.zh-CN.md#结果与离线检查>`_ 中，两种 PPO 都从各任务的同一份基线权重续训 1,000 次。表中列出成功率，括号内为成功回合数与总回合数，合并了评测种子 23456 和 34567 的全部回合。两组训练种子共用冻结的基线。任务 C 分别统计速度跟踪和仿真楼梯通过率。

.. list-table::
   :header-rows: 1

   * - 指标
     - 训练种子
     - 基线
     - RSL-RL PPO
     - RLinf PPO (Huber)
   * - A 速度跟踪
     - 0
     - 91.47% (1887/2063)
     - 93.33% (1917/2054)
     - 96.73% (1981/2048)
   * - A 速度跟踪
     - 2
     - 91.47% (1887/2063)
     - 90.90% (1878/2066)
     - 98.73% (2022/2048)
   * - B 速度跟踪
     - 0
     - 99.85% (2045/2048)
     - 99.61% (2040/2048)
     - 99.85% (2045/2048)
   * - B 速度跟踪
     - 2
     - 99.85% (2045/2048)
     - 99.76% (2043/2048)
     - 99.90% (2046/2048)
   * - C 速度跟踪
     - 0
     - 99.85% (2045/2048)
     - 100.00% (2048/2048)
     - 100.00% (2048/2048)
   * - C 速度跟踪
     - 2
     - 99.85% (2045/2048)
     - 99.80% (2044/2048)
     - 99.80% (2044/2048)
   * - C 楼梯通过
     - 0
     - 21.39% (438/2048)
     - 44.82% (918/2048)
     - 45.17% (925/2048)
   * - C 楼梯通过
     - 2
     - 21.39% (438/2048)
     - 33.50% (686/2048)
     - 41.94% (859/2048)

在 CPU 上回放
-------------

用 C 的基线权重回放 ``go2-reference`` 中的四帧合成数据。请指定新的输出文件，每行记录时间戳、12 个目标关节角（rad）和 12 个电机力矩（N·m），均按 SDK 关节顺序排列。

.. code-block:: bash

   python -m examples.embodiment.go2_abc.deploy --task C --group baseline \
     --frames ../go2-reference/examples/embodiment/go2_abc/frames.example.jsonl \
     --output commands.jsonl

使用训练好的 RLinf actor 或 RSL-RL 权重时，将 ``--group baseline`` 换成 ``--checkpoint path/to/policy.pt``，并选择对应任务。

按组别加载权重支持 ``baseline``、``ppo`` 和 ``rlinf``，显式指定 ``--data`` 优先于 ``GO2_ABC_DATA``。

回放自己的记录时，按 `样例帧格式 <https://github.com/QuincyQAQ/RLinf/blob/23f16691d4dede5dab949a5cbf7934f6021e7ddf/examples/embodiment/go2_abc/frames.example.jsonl>`_ 提供关节和 IMU 数据、实测机身线速度及地形高度。所有时间戳使用同一单调时钟，每 5 ms 采样一次，policy 每四帧更新一次。机身与地面高度使用同一世界坐标系。高度网格随机身偏航角旋转，间距为 0.1 m，按 y 从 −0.5 到 0.5 m 排列各行，每行按 x 从 −0.8 到 0.8 m 排列。

真机六方向控制见 :doc:`go2`。
