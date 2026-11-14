用 RLinf 训练 Go2 MuJoCo policy
===============================

本例用 CPU MuJoCo 仿真和 RLinf PPO 跑通 Go2 的速度跟踪训练，并验证检查点保存和恢复。真机部署需要另做验证。

概述
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      Go2 非对称 MLP

   .. grid-item-card:: 算法
      :text-align: center

      PPO

   .. grid-item-card:: 任务
      :text-align: center

      平地速度跟踪

   .. grid-item-card:: 硬件
      :text-align: center

      CPU 仿真 · 单张 NVIDIA GPU 学习

默认配置使用 64 个环境，每轮采样 24 个控制步，共执行 3 次训练更新。物理仿真运行在 CPU 上，policy 推理和优化使用配置的加速设备。当前支持一个环境进程，每个请求执行一个动作；多进程课程同步和这份 RLinf 配置的昇腾运行尚未验证。

任务
~~~~

.. list-table::
   :header-rows: 1

   * - 任务
     - 配置
   * - Go2 平面速度跟踪
     - ``env_type: go2_mujoco``；每回合 20 秒；倾斜或机身触地时终止。

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - 字段
     - 规格
   * - 观测
     - Actor 为 45 维，Critic 为 60 维特权观测，包含机身运动、指令、关节反馈和上一动作。
   * - 动作
     - 12 个关节目标角，按默认姿态加 policy 动作的 0.25 倍计算；PD 增益为 Kp = 25、Kd = 0.5。
   * - 奖励
     - 16 项加权的速度跟踪、稳定性与运动奖励，按 0.02 秒控制周期积分。
   * - 提示词
     - 不使用。

Actor 和 Critic 均使用 512/256/128 ELU 隐藏层。任务参考 Unitree RL Lab 提交 ``4960b84732b0c2ec593dccbfe963fda1bcd7b1e3``，内置模型来自 Unitree MuJoCo 提交 ``1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d``。MuJoCo 的接触和摩擦与 Isaac Lab 存在差异；RLinf PPO 使用自身的价值损失与优化器调度，因此不能把本例视为 RSL-RL 训练的数值等价复现。

安装
----

.. include:: _setup_common.rst

在仓库根目录安装依赖并编译并行仿真扩展。编译需要支持 C++17 的编译器、OpenMP 和 Python 开发头文件：

.. code:: bash

   bash requirements/install.sh embodied --env go2_mujoco --install-rlinf
   source .venv/bin/activate
   python -m rlinf.envs.sim.go2_mujoco.build

平地模型已内置，无需另行下载资源。更换 Python 版本或 CPU 架构后，需要重新编译扩展。

运行
----

从仓库根目录启动三次更新的最小训练：

.. code:: bash

   export PYTHONPATH="$PWD"
   export EMBODIED_PATH="$PWD/examples/embodiment"
   export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
   export OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0
   python examples/embodiment/train_embodied_agent.py --config-name go2_mujoco_ppo

检查点保存在 ``../rlinf_integration/runs/go2_mujoco_smoke/checkpoints/``。从第 3 步恢复，再执行两次更新：

.. code:: bash

   python examples/embodiment/train_embodied_agent.py --config-name go2_mujoco_ppo \
       runner.resume_dir="$PWD/../rlinf_integration/runs/go2_mujoco_smoke/checkpoints/global_step_3" \
       runner.max_epochs=5 runner.max_steps=5 \
       runner.logger.experiment_name=go2_mujoco_resume

RLinf 检查点恢复 policy、优化器和调度器；仿真状态会重新初始化。若要用已有 RSL-RL Go2 权重初始化新任务，设置 ``actor.model.initial_checkpoint=<checkpoint>``。这只导入模型参数，不恢复旧优化器或训练步数。本例应保持 ``num_action_chunks=1``，关闭 Torch 编译和 CUDA graphs。

可视化与结果
------------

已在 RTX 3070 Laptop GPU、PyTorch 2.7.0 和 MuJoCo 3.3.7 上完成三次更新并保存检查点，随后恢复并执行两次更新，确认权重发生变化。另一个执行两次更新的端到端测试覆盖两步回合超时和评估。这些结果证明训练链路可用，不代表 policy 收敛或真机可用；此前完成的 50000 次更新的基线由独立 RSL-RL 脚本训练，不属于 RLinf 训练成果。

编译扩展后，可在 CPU 上检查重置、动作格式和 policy 一致性：

.. code:: bash

   PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/unit_tests/test_go2_env.py -q

通用检查点流程见 :doc:`/rst_source/guides/resume`。
