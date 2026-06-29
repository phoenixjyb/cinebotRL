# DJI RS4/RS5 云台 Action Contract 二次审计与迁移建议

日期：2026-06-30

审计对象：

- `.98` CineBotRL 仓库：`/mnt/g/wSpace/cinebotRL`
- 分支与提交：`win-recomoPro1 @ 488f12cac57b2aff164ac0afeecc28c4afe03e67`
- 机器人描述：`assets_own/recomoProto2-1190_moveit.urdf`
- 仿真环境：`src/rl_platform/tasks/mobile_mm/`
- 部署侧参考：`/Users/yanbo/Projects/recomo-app-monorepo/src/pnc/gimbal_composition/cine_gimbal_control/`
- 部署侧消息：`/Users/yanbo/Projects/recomo-app-monorepo/src/common/recomo_msgs/msg/`

## 结论摘要

当前答案不是“完全确定可以直接改”，而是：

1. **当前 CineBotRL 9D contract 在仿真内部是自洽的。**
   当前 policy action 是：

   ```text
   [6 个 URDF arm/gimbal joint position targets, base_vx, base_vy, base_wz]
   ```

   其中 `joint3_gimbal_yaw / joint2_gimbal_roll / joint1_gimbal_pitch` 确实在 URDF 的 `cam_link` 主 FK 链路上，不是无效关节。

2. **当前 9D contract 不能直接证明等价于真实 DJI RS4/RS5 部署接口。**
   部署侧代码显示，真实 RS4 路径默认使用 `rs4_control_mode=attitude` 和 `rs4_use_velocity_control=true`，发布 `RoninRs4Control::CMD_VELOCITY` 到 `/gimbal/cmd/tracking`，字段是 `velocity_deg_s`，而不是按 URDF joint 名字发布 `joint3_gimbal_yaw / joint2_gimbal_roll / joint1_gimbal_pitch` 的绝对位置目标。

3. **最稳妥的下一步不是继续 full 9D arm/gimbal BC，而是新建一个 RS4-aware experimental contract。**
   推荐优先尝试：

   ```text
   action_dim = 9
   [arm_yaw, arm_pitch, arm_elbow,
    rs4_yaw_rate, rs4_pitch_rate, rs4_roll_rate,
    base_vx, base_vy, base_wz]
   ```

   但 roll 是否纳入 v1 要谨慎：部署侧当前 yaw/pitch velocity loop 是 2-DoF，composition preset 里 roll 被记录但注释明确说 mixed-mode static roll while velocity-controlling yaw/pitch 尚未确认。

4. **`ee1_rot_z/y/x` 不应该被直接改成 policy action。**
   它们是 `arm_link_3` 下的 sibling virtual branch，终点是 `ee1_tool`，不在当前 `cam_link` 主链路中。除非环境 reward / observation / camera body 改成使用 `ee1_tool`，否则驱动 `ee1_rot_*` 不等价于驱动相机。

## 审计证据

### 1. 当前 CineBotRL action slicing

`src/rl_platform/tasks/mobile_mm/joint_names.py` 当前定义：

```text
ARM_JOINT_NAMES =
  joint6_arm_yaw
  joint5_arm_pitch
  joint4_elbow_pitch
  joint3_gimbal_yaw
  joint2_gimbal_roll
  joint1_gimbal_pitch

EE_LINK_NAME = cam_link
EE_VIRTUAL_JOINT_NAMES = ee1_rot_z, ee1_rot_y, ee1_rot_x

POLICY_ARM_ACTION_SLICE = slice(0, 6)
POLICY_BASE_VX_ACTION_INDEX = 6
POLICY_BASE_VY_ACTION_INDEX = 7
POLICY_BASE_WZ_ACTION_INDEX = 8
POLICY_ACTION_DIM = NUM_ARM_JOINTS + 3
```

`env.py` 当前行为：

- `num_actions = POLICY_ACTION_DIM`
- action space 注释为 `[arm_joint_targets (6), base_vel_x, base_vel_y, base_angular_vel]`
- `_pre_physics_step()` 要求 action shape 为 9
- 前 6 维经过 `_scale_actions_to_joint_limits()` 和 `_filter_arm_position_targets()`
- 然后 `robot.set_joint_position_target(..., joint_ids=ARM_JOINT_NAMES)`
- `ee1_rot_z/y/x` 被 `_lock_passive_joints()` 锁住
- base 三维按 body-frame velocity 解释，写入 root velocity

这说明当前 CineBotRL 没有把 action 输出转换成 RS4 attitude command，也没有 RS4-specific adapter。

### 2. URDF frame chain

URDF 主相机链路：

```text
base_link
  -> arm_base_joint fixed -> arm_base_link
  -> joint6_arm_yaw       -> arm_link_1
  -> joint5_arm_pitch     -> arm_link_2
  -> joint4_elbow_pitch   -> arm_link_3
  -> joint3_gimbal_yaw    -> arm_link_4
  -> joint2_gimbal_roll   -> arm_link_5
  -> joint1_gimbal_pitch  -> arm_link_6
  -> camera_optical_center fixed -> cam_link
```

虚拟 EE 分支：

```text
arm_link_3
  -> ee1_rot_z -> ee1_rotz_link
  -> ee1_rot_y -> ee1_roty_link
  -> ee1_rot_x -> ee1_rotx_link
  -> ee1_tool_mount_fixed -> ee1_tool
```

关键含义：

- `joint3_gimbal_yaw / joint2_gimbal_roll / joint1_gimbal_pitch` 在仿真中确实能改变 `cam_link`。
- `ee1_rot_z/y/x` 不在 `cam_link` 链路中；它们不能直接代表当前环境里的 camera attitude action。
- USD 文件是 binary crate，无法用简单 grep 审文本结构；本次以 URDF 和环境 joint-name verification 为主要证据。

### 3. 部署侧 RS4 command surface

消息定义 `RoninRs4Control.msg`：

```text
CMD_POSITION = 1
CMD_VELOCITY = 2
geometry_msgs/Vector3 position_deg
geometry_msgs/Vector3 velocity_deg_s
```

消息定义 `RoninRs4Status.msg`：

```text
VALID_ATTITUDE = 1
VALID_JOINT = 2
geometry_msgs/Vector3 attitude_deg
geometry_msgs/Vector3 joint_deg
```

`gimbal_tracking_node` 当前默认参数：

```text
gimbal_driver: "rs4"
rs4_control_mode: "attitude"
topics.rs4_status: "/ronin_rs4_driver/status/state"
topics.rs4_command: "/gimbal/cmd/tracking"
rs4_axis_map_from_gimbal: [2, 0, 1]
rs4_use_velocity_control: true
```

源代码行为：

- 如果 `rs4_control_mode=attitude`，状态回调读取 `msg->attitude_deg`，否则读取 `msg->joint_deg`。
- 默认 RS4 velocity 路径发布 `CMD_VELOCITY`。
- velocity vector 先按 local gimbal order 组织为：

  ```text
  gimbal_vel = [roll, pitch, yaw]
             = [0.0, last_calc_vel_pitch_deg_s, last_calc_vel_yaw_deg_s]
  ```

- 再通过 `rs4_axis_map_from_gimbal=[2,0,1]` 映射到 RS4 axis order。
- 当前 tracking node 的 Ruckig smoother 是 2-DoF：`[yaw, pitch]`。

因此，部署侧“当前验证过的默认路径”更像：

```text
RS4 attitude frame / velocity command surface
```

而不是：

```text
URDF joint3_gimbal_yaw / joint2_gimbal_roll / joint1_gimbal_pitch absolute position targets
```

### 4. 部署侧 Realman arm surface

`arm_height_controller.cpp` 明确写着：

```text
Robot: recomoProto1-190
Arm: 3-DOF (joint6_yaw, joint5_pitch, joint4_elbow)
```

运行时订阅：

```text
/realman_arm_driver/joint_states
current_arm_[0] = yaw
current_arm_[1] = pitch
current_arm_[2] = elbow
```

发布：

```text
/arm_cmd/pnc
cmd.name = ["base_yaw", "base_pitch", "elbow"]
cmd.position[0..2]
cmd.velocity[0..2]
```

所以真实部署 arm command surface 至少在这个 Smart Follow/height-control 路径下是 3-DOF arm，而不是 CineBotRL 当前的 6 个 arm/gimbal position target。

## 回答交接文档中的问题

### Q1. 当前 CineBotRL 的 action contract 是否和真实机器人控制接口一致？

**结论：不完全一致；base 部分可信，arm/gimbal 部分不应直接视为部署一致。**

确认项：

- `base_vx/base_vy/base_wz` 作为 body-frame velocity command 是合理的，且之前 base-only BC 验证结果很好。
- 当前 6-joint arm/gimbal contract 对 Isaac 仿真是可执行的。

不一致 / 不确定项：

- 真实 Realman arm 路径显示 3-DOF arm command：yaw/pitch/elbow。
- 真实 RS4 路径显示 attitude/joint telemetry 都存在，但默认控制模式是 attitude + velocity。
- 当前 CineBotRL 没有 RS4 command abstraction，也没有将 policy 输出映射到 `RoninRs4Control` 的逻辑。

建议：

- 不要继续把当前 full 9D BC 叫做“可部署 9D BC”。
- 保留当前环境作为 `sim_6joint_gimbal` baseline。
- 新建 RS4-aware 实验环境或显式 contract 名称，避免污染已有 checkpoint。

### Q2. `joint3_gimbal_yaw / joint2_gimbal_roll / joint1_gimbal_pitch` 到底是什么？

**它们是 URDF/USD 仿真中的主相机链路 revolute joints。**

它们不是“无效虚拟关节”，因为 `cam_link` 在它们之后：

```text
joint3_gimbal_yaw -> joint2_gimbal_roll -> joint1_gimbal_pitch -> arm_link_6 -> cam_link
```

但它们是否等于真实 DJI RS4/RS5 部署 command，当前证据不足。

更准确的表述：

- 在 Isaac/URDF 中：它们是可控的 sim gimbal/wrist joints。
- 在部署代码中：真实 RS4 使用 `RoninRs4Control` 的 attitude/joint vector command，不使用这些 URDF joint names。
- 如果要复用它们，需要写一个明确 adapter：

  ```text
  policy gimbal action
    -> sim: joint3/joint2/joint1 target or velocity
    -> deploy: RoninRs4Control position_deg / velocity_deg_s
  ```

  并验证 axis order、sign、offset、frame、unit、rate limit。

### Q3. `ee1_rot_z / ee1_rot_y / ee1_rot_x` 是否应该成为 attitude action？

**当前不应该直接成为 action。**

原因：

- 它们是 `arm_link_3` 下的 sibling branch。
- 当前 `EE_LINK_NAME` 是 `cam_link`。
- 当前 `cam_link` FK 主链不包含 `ee1_rot_*`。
- 当前审计文档 `GIK_ARM_GIMBAL_IMITATION_AUDIT_CN.md` 也确认：`ee1_rot_z/y/x` 不在 `cam_link` FK 链路中。

如果要让它们成为 attitude action，需要先做模型层变更：

1. 明确 `ee1_tool` 是否才是真正的 camera attitude frame。
2. 改 reward / observation / FK target link，从 `cam_link` 切到新 camera frame，或把 `ee1_rot_*` 结构移到 `cam_link` 主链上。
3. 重新导出 USD 并做 Isaac smoke。
4. 验证和 RS4 axis order `[yaw, roll, pitch]` / local gimbal order `[roll, pitch, yaw]` 的映射。

在完成这些之前，`ee1_rot_*` 更适合作为 MoveIt/GIK virtual frame 线索，而不是直接 policy action。

### Q4. camera attitude 应该用什么表示？

推荐分层：

#### v1 推荐：RS4 rate/residual command

```text
[yaw_rate, pitch_rate, optional_roll_rate]
```

单位与归一化建议：

```text
policy [-1, 1]
-> deg/s or rad/s
-> clamp by RS4-safe max velocity
-> apply acceleration/jerk limits
```

理由：

- 部署默认就是 `CMD_VELOCITY`。
- 已有 Ruckig 2-DoF yaw/pitch velocity smoother。
- yaw wrap 风险小于 absolute yaw。
- 和视觉 tracking / residual correction 的闭环逻辑一致。

#### v2 可选：absolute attitude target

```text
[yaw_abs, pitch_abs, roll_abs]
```

适用条件：

- 明确 RS4 attitude frame 是 world pointing、body-relative 还是 gimbal-local。
- 有 wrap-safe yaw error 表示，例如 `sin/cos(yaw_error)` 或 unwrapped yaw。
- 明确 roll 的控制模式不会和 yaw/pitch velocity mode 冲突。

#### 暂不推荐：quaternion target

不建议第一版直接用 quaternion action。原因：

- 部署消息目前暴露的是 vector position/velocity deg，不是 quaternion。
- policy 输出 quaternion 需要 normalization 和 antipodal handling。
- 对 RS4 三轴约束、axis order、work mode 的可解释性更差。

### Q5. GIK imitation 数据应该如何重新解释？

当前导出数据里的 base label 可以继续使用：

```text
base pose [x, y, yaw]
-> finite difference
-> body-frame [vx, vy, wz]
-> normalized base action
```

arm/gimbal 需要重新导出，不要继续把 `qTraj` 后 6 维直接当作可部署 action。

推荐新导出 schema：

```text
actions:
  0 arm_yaw
  1 arm_pitch
  2 arm_elbow
  3 rs4_yaw_rate_or_residual
  4 rs4_pitch_rate_or_residual
  5 rs4_roll_rate_or_residual
  6 base_vx
  7 base_vy
  8 base_wz

action_valid_mask:
  per-channel validity

extra fields:
  current_camera_attitude
  target_camera_attitude
  camera_attitude_error
  actual_ee_quat_wxyz
  target_quat_wxyz
  source_frame
  attitude_frame_convention
```

需要回答的导出细节：

- `target_quat_wxyz` 是 target camera attitude 还是 target EE pose?
- `actual_ee_quat_wxyz` 对应 `cam_link`、`ee1_tool`，还是 MATLAB/GIK 的另一个 frame?
- attitude residual 是否要相对 base frame、world frame，还是 RS4 attitude frame?
- `roll` 是可控目标，还是只允许 roll-window / DJI tracker handoff 之类高层模式？

## 推荐迁移方案

### Phase 0：冻结旧结论边界

- 旧 base-only BC 保留。
- 旧 full 9D arm/gimbal BC 不继续作为部署路径。
- 当前 `RecomoProto2TrackEE-v0` 不直接破坏。

### Phase 1：新增 contract 文档和命名

建议明确两个 contract：

```text
sim_6joint_gimbal_v1:
  [6 URDF arm/gimbal joint targets, base_vx, base_vy, base_wz]

rs4_attitude_rate_v1:
  [3 Realman arm targets, 3 RS4 attitude/rate commands, base_vx, base_vy, base_wz]
```

### Phase 2：先做 sim adapter，不碰真实硬件

在 Isaac 中让 `rs4_attitude_rate_v1` 可以驱动相机：

选项 A：

- policy 输出 3 arm + 3 gimbal rates。
- 仿真 adapter 把 gimbal rates 积分成 `joint3/joint2/joint1` target。
- reward 评估 `cam_link` attitude tracking。

选项 B：

- 修改模型，使 attitude action 作用于新的 camera attitude frame。
- 代价更大，但语义更干净。

### Phase 3：重建 GIK dataset

- base label 沿用现有 finite-difference body velocity 逻辑。
- arm label 只取 3 个真实 arm DOF。
- gimbal label 从 camera attitude error / target-current residual 生成。
- 记录 frame convention 和 axis order。

### Phase 4：短 BC + replay 验证

必须先过：

- action mask shape 正确。
- observation replay 中 camera attitude error 下降。
- Isaac short rollout 无 NaN、无 joint-limit saturation、无 self-collision。
- 和旧 base-only BC 对比，base 行为不退化。

### Phase 5：再考虑硬件 adapter

硬件 adapter 需要单独验证：

```text
policy rs4_yaw/pitch/roll command
-> axis/sign/offset
-> RoninRs4Control CMD_VELOCITY or CMD_POSITION
-> /gimbal/cmd/tracking or dedicated RL topic
-> mux priority
-> RS4 telemetry attitude_deg/joint_deg closed-loop response
```

## action_dim 建议

### 推荐：先保持 action_dim = 9，但改语义

```text
0 arm_yaw
1 arm_pitch
2 arm_elbow
3 rs4_yaw_rate_or_residual
4 rs4_pitch_rate_or_residual
5 rs4_roll_rate_or_residual
6 base_vx
7 base_vy
8 base_wz
```

优点：

- 不增加 policy 输出维度。
- 更贴近当前真实 arm 3-DOF + RS4 3-axis abstraction。
- 旧训练框架和 BC loader 的 shape 改动较小。

风险：

- 旧 9D checkpoint 不能语义兼容，只能部分迁移 base/head 或重新训练。
- roll 的部署控制尚未充分验证，v1 可以先 mask 掉 roll 或让 roll_rate=0。

### 暂不推荐：action_dim = 12

```text
[6 URDF joints, 3 RS4 attitude commands, base_vx, base_vy, base_wz]
```

不推荐原因：

- 同时控制 URDF gimbal joints 和 RS4 attitude 会语义重复。
- 部署侧真实 arm 证据更支持 3-DOF arm + RS4，而不是 6 arm + extra RS4。
- 数据和 reward 更容易混淆：相机姿态到底由后三个 URDF joints 产生，还是由 RS4 attitude command 产生？

只有在证明真实机器人确实有 6 个独立 Realman/arm joints 加一个额外 DJI gimbal 时，才考虑 12D。

## 需要硬件或团队确认的问题

1. RS4 driver 的 `position_deg` / `velocity_deg_s` 三个字段的物理 axis order 是否稳定为 `[yaw, roll, pitch]`？
2. `attitude_deg` 是 world pointing、body-relative，还是 DJI SDK 定义的 gimbal attitude？
3. `joint_deg` 是否可作为部署控制目标，还是只作为 telemetry？
4. 当前 RS4 是否支持 yaw/pitch velocity 控制同时施加 static roll？
5. 真实机器人 arm 是否永远只有 `base_yaw/base_pitch/elbow` 三个 Realman command，还是某些模式还有额外 wrist/gimbal motor command？
6. GIK/MATLAB 中的 `target_quat_wxyz` / `actual_ee_quat_wxyz` 对应哪个 frame？

## 最终建议

我现在对下面判断有较高信心：

- 当前 CineBotRL policy 确实在控制 6 个 URDF joint target。
- `joint3_gimbal_yaw / joint2_gimbal_roll / joint1_gimbal_pitch` 在 sim 中影响 `cam_link`。
- `ee1_rot_z/y/x` 不在当前 `cam_link` 主链上，不应直接作为 action。
- 部署侧默认 RS4 路径是 attitude mode + velocity command，和当前 CineBotRL 6-joint target contract 不同。
- full 9D BC 不应继续按旧语义推进。

我对下面判断只给中等信心，需要硬件/driver 确认：

- 最终部署 policy 是否应该输出 roll。
- RS4 action 最好是 absolute attitude 还是 velocity/residual。
- `joint_deg` 是否有可用的低层 joint control 模式。

因此建议：

1. 保留 base-only BC 作为可复用成果。
2. 新增 `rs4_attitude_rate_v1` contract。
3. 重导 GIK imitation dataset。
4. 先在 Isaac 中做 adapter 和 short rollout。
5. 真实硬件只在 adapter 过 smoke 后再接入。
