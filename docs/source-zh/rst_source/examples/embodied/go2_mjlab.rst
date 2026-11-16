使用 Go2 Mjlab 训练
==========================

使用原生 RLinf PPO 训练宇树官方 Go2 平地速度跟踪任务。CPU worker 执行 Mjlab 仿真，NVIDIA GPU 或华为昇腾 NPU 执行 policy 推理和训练。昇腾配置使用八个 NPU 和 4096 个环境。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      非对称 Go2 MLP

   .. grid-item-card:: 算法
      :text-align: center

      PPO

   .. grid-item-card:: 任务
      :text-align: center

      平地速度跟踪

   .. grid-item-card:: 硬件
      :text-align: center

      CPU 仿真 · NVIDIA 或 :ref:`华为昇腾 <go2-mjlab-hardware>` 训练

任务
~~~~

.. list-table::
   :header-rows: 1

   * - 环境
     - 任务 / 套件
     - 配置 / 权重
     - 重点
   * - Go2 Mjlab
     - Unitree-Go2-Flat
     - ``go2_mjlab_ppo_ascend``，随机初始化
     - 站立、平面速度和转向指令

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - 字段
     - 说明
   * - 观测
     - Actor 接收 47 维输入，critic 接收 74 维特权输入，启用观测归一化。
   * - 动作
     - 12 维高斯关节动作；目标关节位置等于默认姿态加上动作乘以 0.25。
   * - 奖励
     - 固定版本的上游速度、步态、稳定性和接触奖励，控制周期为 0.02 s。
   * - 提示词
     - 不使用。

任务来自 `Unitree RL Mjlab <https://github.com/unitreerobotics/unitree_rl_mjlab>`_，提交固定为 ``1425b15f73bd4095f0df53709d7c389c3eb9e790``。完整配置训练 10001 次更新，每次采样 24 步，训练五轮 PPO，每轮四个 minibatch。价值损失使用裁剪平方误差，学习率按解析 KL 自适应调整，KL 目标为 0.01。

安装
----

.. include:: _setup_common.rst

在 NVIDIA 主机上安装任务和固定版本的物理引擎依赖：

.. code:: bash

   bash requirements/install.sh embodied --env go2_mjlab --install-rlinf
   source .venv/bin/activate

运行
----

在仓库根目录执行单 GPU 三次更新测试：

.. code:: bash

   export REPO_PATH=$(pwd)
   bash tests/e2e_tests/embodied/run.sh go2_mjlab_ppo

输出的 TensorBoard 事件和 checkpoint 位于 ``/workspace/results/go2_mjlab_ppo``。八设备昇腾配置使用下面的运行步骤。

.. _go2-mjlab-hardware:

在不同硬件后端上运行
--------------------

华为昇腾 CANN
~~~~~~~~~~~~~

验证环境为 Linux aarch64、glibc 2.34、CANN 8.5、PyTorch 2.7.1 和 TorchNPU 2.7.1.post2。仿真使用 conda-forge 的 Warp 1.12 CPU 构建；PyPI 的 ARM wheel 要求 glibc 2.35。系统库保持不变。

按照 :download:`昇腾环境复现说明 <../../../../../examples/embodiment/go2_mjlab/README.zh-CN.md>`，使用 :download:`conda 锁文件 <../../../../../examples/embodiment/go2_mjlab/ascend-conda-explicit.lock>` 和 :download:`pip 锁文件 <../../../../../examples/embodiment/go2_mjlab/ascend-requirements.lock>` 创建独立运行环境。NVIDIA 自动安装命令不负责创建此昇腾环境。

安装完成后，指定匹配的解释器和 CANN 初始化脚本，先跑单 NPU 测试：

.. code:: bash

   export GO2_PYTHON="$PWD/.venv-mjlab-ascend/bin/python"
   export CANN_ENV_FILE=/usr/local/Ascend/ascend-toolkit/set_env.sh
   bash examples/embodiment/go2_ascend/run_native.sh go2_mjlab_ppo_ascend_smoke

验证后启动 10001 次更新的配置：

.. code:: bash

   bash examples/embodiment/go2_ascend/run_native.sh go2_mjlab_ppo_ascend

默认 placement 使用 8–15 号 NPU、八个 actor 和 64 个 CPU 环境 worker。每个 CPU worker 运行 64 个环境，使用两个 Torch 线程。设备编号不同时需修改配置中的 placement。CPU 仿真可能占据主要耗时，不能只用 NPU 占用率判断训练吞吐。

可视化与结果
------------

完整训练的 TensorBoard 事件位于 ``outputs/go2_mjlab/native_ascend/tensorboard``，可恢复 checkpoint 位于该输出目录下的 ``go2_mjlab/checkpoints``。Checkpoint 包括模型、优化器、调度器和归一化统计。恢复会重置物理状态，并按保存的更新次数恢复基于时间的指令课程。

单 NPU 与八 NPU 短测验证了采样、PPO 更新和保存。这些检查不代表已经收敛、取得任务成功率或通过真机部署验证。

RLinf 每批 rollout 汇总一次各 actor 的观测统计，上游 RSL-RL 每个环境步后更新本地统计。RLinf 用终止前观测计算超时回合的 bootstrap 值，上游 RSL 包装器使用保存的 transition value。任务与上述参数对齐上游，但执行方式存在这些差异，不宣称数值等价。原生 PPO 配置见 :doc:`../../reference/algorithms/ppo`。

跨任务评估
----------

独立的 ``rlinf-mjlab-cross-task-v1`` 版本提供平地前进、平地全向、粗糙地形、阶梯和跨沟五个任务。Actor 和 Critic 均使用 235 维观测，其中高度扫描为 187 维。新任务需要重新初始化 policy；原有 47/74 维 Flat 配置保留原接口。

平地前进配置为 ``go2_mjlab_benchmark_a1_ascend``，将 ``a1`` 换成 ``a2``、``b1``、``b2`` 或 ``c2`` 可选择其他任务。短验证配置 ``go2_mjlab_benchmark_a1_smoke`` 使用 16 个环境、3 次更新。CPU 任务检查已通过，加速器验证需要空闲设备，应等原训练释放 8 号卡后再运行。

对每个训练好的 policy，分别启动独立进程评估五项任务，每个单元格至少收集 1024 个回合。评估和矩阵导出入口为 ``examples/embodiment/go2_mjlab/evaluate_benchmark.py`` 和 ``examples/embodiment/go2_mjlab/export_benchmark_matrix.py``，完整命令、门槛和仿真差异见 ``examples/embodiment/go2_mjlab/BENCHMARK.zh-CN.md``。未测单元格保留空值，短测默认不进入正式矩阵。本版沿用参考项目的成功判据，物理、奖励和启动随机化仍使用 Mjlab，不能将结果当作原 Isaac Lab 数值的直接复现。
