# CineBotRL 纠偏 GIK 教师请求（2026-07-14）

## 结论

当前不允许继续 PPO，也不应继续用名义轨迹做普通 BC。`split_reference_v2`
的离线 holdout RMSE 已从 `0.02932` 降到 `0.01588`，但 120 步闭环位置
p95 恶化到 `0.60182 m`。诊断表明，第 5 步开始策略就进入名义教师从未覆盖的
状态分布，`95.8%` 的 rollout 状态超过教师 future-attitude error 的 p99。

下一步是让 GIK/WBC 对 **Isaac 中策略实际到达的状态** 重新求解下一步纠偏动作，
而不是按相同 waypoint 抄回名义动作。

## 固定合同

- 目标位姿：physical `cam_link`。
- 姿态转换：`R_world_cam = R_world_DFR * Rz(+pi/2)`。
- 四元数顺序：`wxyz`。
- RL 动作合同：`split_base_arm_attitude_v1`，共 9 维。
- 学习标签：`[0,1,2,6,7,8]`，即机械臂前三关节和底盘 `vx/vy/wz`。
- 保留通道：`[3,4,5]`，不得写入策略标签。
- 三个 DJI 物理云台关节可参与 GIK 的完整 physical-camera 可行性求解和诊断，
  但不能成为 RL action label。运行时 Option-B DLS attitude adapter 控制它们。
- 虚拟关节 `ee1_rot_z/y/x` 不得作为物理状态或标签。

## Isaac 请求数据

`evaluate_stage_rollout_gate.py` 新增：

```text
--output_corrective_teacher_request_npz <path>
```

它在每次 `env.step()` 之前保存：

- 策略 observation、policy action、实际 applied action；
- base 世界坐标系位置、四元数、线速度和角速度；
- 6 个物理机械臂/云台关节位置和速度；
- 当前 physical `cam_link` 世界位姿和 twist；
- 目标 physical `cam_link` 位姿；
- 对应的 semantic DFR 目标姿态；
- 未来 `0.5 s` 的 10 个 20 Hz physical `cam_link`/semantic DFR 目标；
- episode、waypoint、progress、剩余时间和 first-episode validity。

该 NPZ 的 schema 为 `corrective_teacher_request_v1`。它只是请求状态，不含教师
标签，不能直接拼入 BC 数据。

使用独立导出器验证并生成 MATLAB 可直接读取的 CSV：

```bash
python scripts/imitation/export_corrective_teacher_request.py \
  artifacts/split_teacher_corrective_request_20260714/rejected_v2_episode1_120.npz \
  --output_dir artifacts/split_teacher_corrective_request_20260714/gik_request
```

输出：

- `corrective_teacher_request_samples.csv`
- `corrective_teacher_request_manifest.json`

导出器会拒绝非有限值、错误维度、错误 action ownership、非单位四元数，以及
不满足 Option-B 转换的目标姿态。

## GIK 求解任务

对 CSV 每一行，从保存的真实状态开始做短时域纠偏求解。推荐先做单步或
`0.25-0.50 s` receding-horizon solve：

1. 使用当前 base pose/twist、前三个 arm joint 和三个 physical gimbal joint 作为
   初始状态。
2. 将当前目标和 `target_horizon_t01...t10` 作为 `0.5 s` physical `cam_link`
   receding-horizon 目标，不能只追当前 waypoint 造成一个周期以上的固定滞后。
3. 完整求解可使用 physical gimbal DOF，以验证姿态可达性和耦合影响。
4. 只将下一控制周期的前三个 arm target 和 body-frame base `vx/vy/wz` 映射到
   `teacher_action_0/1/2/6/7/8`。
5. physical gimbal 解只保存为诊断字段，不写 `teacher_action_3/4/5`。
6. 不允许先超限再 clip。违反关节、速度、加速度、碰撞或物理相机误差门限的行
   必须标记 rejected。

教师返回 CSV 至少应包含：

```text
sample_id
source_episode_index
rollout_step
teacher_action_0
teacher_action_1
teacher_action_2
teacher_action_6
teacher_action_7
teacher_action_8
solver_success
physical_cam_position_residual_m
physical_cam_orientation_residual_deg
joint_limit_margin_min_rad
```

建议额外返回 solver exit reason、碰撞余量、速度/加速度峰值、完整 6 关节预测值
以及 physical gimbal 诊断值，便于判断 runtime DLS 和离线完整解是否一致。

## 数据门限（已按物理可达性修订）

首轮只处理已失败策略的 120 个状态，不做全量导出。每一行必须满足：

- source identity 唯一且可追溯；
- 若初始 position error `<= 0.05 m`，0.5 s 终点 residual 必须 `<= 0.06 m`；
- 否则 0.5 s 终点必须达到 `<= 0.05 m`，或至少改善
  `min(0.02 m, 5% * initial_error)`；
- orientation 终点不得恶化超过 `2 deg`，且低误差状态目标门限为 `4 deg`；
- arm/base action 在 `proto2_safe_v1` envelope 内；
- 无关节限位、速度、加速度、碰撞或非有限值违规；
- 不使用虚拟关节或 physical DJI joint 作为策略标签。

原先要求所有大偏差状态在一个 0.5 s runtime-bounded horizon 内达到 `0.05 m`，与
底盘加速度、机械臂单周期 slew 和相机初始误差不相容。修订后的门限只判断教师
是否给出物理可执行且正确方向的恢复标签；最终 tracking accuracy 仍由 Isaac 闭环
rollout gate 判断，不能用离线 horizon improvement 代替。

若 120 行中有效率太低，不应放宽门限掩盖不可行性；应调整短时域、底盘自由度或
目标时序，再重新求解。

## 返回后的唯一训练实验

1. 只接收通过门限的纠偏行。
2. 保留原 62 条 accepted nominal teacher，按 source episode 做 grouped split。
3. 给纠偏行有限权重，不允许简单重复单个 episode 主导训练。
4. 从 v2 BC checkpoint warm-start，只做一次有上限的 BC 更新。
5. 至少在 5 条不同轨迹上做 120 步闭环 gate。

相对于当前 84D primary BC baseline，必须同时满足：

- position p95 `< 0.26433 m`；
- orientation mean `< 4.362 deg`；
- 不得只改善 mean 而恶化 p95；
- 不得只改善 episode 1。

失败即停止，不继续 DAgger 轮次，不运行 PPO。通过后才扩展到完整 horizon 和全部
accepted no-obstacle trajectories。障碍物 curriculum 必须等 no-obstacle recovery
通过后再启动。

## 当前边界

截至 2026-07-14，纠偏教师已完成：

- MATLAB/Isaac physical `cam_link` FK parity：position max
  `9.33e-7 m`，orientation max `9.11e-5 deg`；
- runtime-aligned smoke：`3/3`、`12/12` 通过；
- 全部请求状态：`120/120` 通过，每行完成 10 个 50 ms bounded step；
- 已显式验证底盘 `vx/vy`（含 lateral `vy`）、`wz`、底盘/偏航加速度、机械臂
  `0.015 rad/step` slew、云台 rate/acceleration、SRDF collision 和 action envelope；
- corrective BC 数据已按 `sample_id` 与原始 98D observation 连接，保留通道
  `3/4/5` 仍为 zero/masked；每行权重 `0.5`，总有效权重 `60`；
- 与 62 条 accepted nominal teacher 合并后为 `21,137` 行、63 个 disjoint source
  group，其中 nominal 为 `21,017` 行，corrective 为 120 行。

当前仍未授权 PPO。下一步只允许从 accepted62 reference-v2 checkpoint 做一次低学习率、
有限 epoch 的 BC warm-start，然后执行 5 条轨迹、每条 120 步的 Isaac 闭环 gate。
