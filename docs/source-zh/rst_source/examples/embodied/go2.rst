Go2 高层控制
============

通过宇树官方 SDK2 读取 Go2 EDU 状态，并离线检查速度控制环境。真机连接默认只读。RLinf 适配层的运动控制尚未通过 Go2 实机验收。

概览
----

.. grid:: 2 4 4 4

   .. grid-item-card:: 模型

      手动命令，尚未接入训练好的 policy。

   .. grid-item-card:: 算法

      尚无训练流程。

   .. grid-item-card:: 任务

      高层移动接口验证。

   .. grid-item-card:: 硬件

      Go2 EDU 和 Linux 控制电脑。

任务
~~~~

.. list-table::
   :header-rows: 1

   * - 任务
     - 结果
   * - ``Go2DriveEnv-v1``
     - 读取本体状态、发送有界机身速度命令，奖励为零。

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - 字段
     - 约定
   * - 观测
     - RPY、陀螺仪、12 个关节位置、12 个关节速度、电池电压和四个原始足部力读数，共 35 维，不含相机。
   * - 动作
     - ``[vx, vy, vyaw]`` 的单位依次为 m/s、m/s、rad/s，拒绝非有限值和越界值，三个轴的绝对值上限依次为 0.3、0.3、0.5。
   * - 奖励与回合
     - 奖励为零，默认 20 步截断回合，尚无任务成功判据。

运动前检查
~~~~~~~~~~

先用 `官方 SportClient 示例 <https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/example/go2/high_level/go2_sport_client.py>`_ 验证前进和停止。适配层的轴向速度上限取自该示例，不代表厂商认定的安全速度。场地保持空旷，并由另一位操作员持遥控器准备急停。

.. warning::

   断网、进程终止或 SDK 调用阻塞后，软件无法保证机器人停止。Move 发送成功不代表机器人已执行动作，StopMove 返回成功也不能证明机器人已静止，必须现场观察。

只有在硬件配置中显式设置 ``enable_motion=True`` 后，才允许运动。只读连接不创建 SportClient，也不发送任何命令，reset 和 disconnect 同样不发送命令。启用运动后，连接、重置、关闭、回合结束和异常均请求 StopMove，不自动切换姿态或步态。

非零命令生效期间，连续 0.5 秒未收到新动作，或连续 0.5 秒未收到有效状态，看门狗会请求停止。动作过期、状态过期或动作无效后，运动保持锁定；reset 请求停止，但不会解除锁定。查明原因后重新连接。这些超时是软件目标，不是硬实时保证。状态新鲜度按本地接收时间计算，不代表传感器采样时间。原始足部力读数尚未验证，不能当作接触标签。

安装
----

使用已配置的 RLinf 开发环境，其中需要 Gymnasium 和 NumPy。真机读取前，在同一 Python 环境中按照 `SDK2 官方安装说明 <https://github.com/unitreerobotics/unitree_sdk2_python#installation>`_ 安装依赖。下面的离线示例不导入厂商 SDK，也不创建 DDS participant。

运行
----

在 RLinf 仓库根目录运行以下命令，离线检查重置、零动作和关闭：

.. code-block:: bash

   python - <<'PYTHON'
   import numpy as np
   from rlinf.envs.real.go2 import Go2DriveEnv

   env = Go2DriveEnv({"is_dummy": True})
   try:
       observation, _ = env.reset()
       result = env.step(np.zeros(3, dtype=np.float32))
       print(observation["state"].keys(), result[1:4])
   finally:
       env.close()
   PYTHON

预期输出六个状态字段和 ``(0.0, False, False)``。离线观测是确定的零值，不包含动力学仿真。

现场只读检查需要将 ``interface`` 设置为已验证的机器人网卡名称。下面的代码只打开 lowstate 订阅：

.. code-block:: python

   from rlinf.robotics import Go2Robot

   robot = Go2Robot.build(interface="<network-interface>")
   try:
       robot.connect()
       print(robot.get_observation()["base"])
   finally:
       robot.disconnect()

每个进程只允许一个 Go2 连接占用 DDS。SDK2 的工厂是进程全局对象，且没有公开的销毁接口；切换网卡需要重启 Python。不要在同一进程中运行其他 SDK2 机器人控制器。适配层不能排除其他进程或遥控器同时控制机器人。

可视化与结果
------------

离线测试使用内存中的 SDK 替身，覆盖命令单位、看门狗停止、状态过期、无效动作、资源清理和故障锁定，不代表实机行为已经验证。相机、训练奖励以及已有 12 关节 policy 的部署仍需分别实现。仿真接入见 :doc:`Go2 MuJoCo <go2_mujoco>`。
