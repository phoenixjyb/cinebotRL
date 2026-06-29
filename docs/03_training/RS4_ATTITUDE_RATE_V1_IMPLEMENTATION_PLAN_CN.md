# rs4_attitude_rate_v1 实施计划

日期：2026-06-30  
适用仓库：`cinebotRL`  
前置设计文档：`docs/03_training/DJI_GIMBAL_ACTION_CONTRACT_PROPOSAL_CN.md`

## 目标

新增一个面向真实 DJI RS4/RS5 部署语义的实验性 action contract：

```text
rs4_attitude_rate_v1

action_dim = 9

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

这个 contract 的目标不是马上替换当前训练主线，而是建立一条可验证路径，让仿真、GIK imitation 数据、以及真实 DJI gimbal command surface 对齐。

## 非目标

当前阶段不要做：

- 不要把旧 `sim_6joint_gimbal_v1` checkpoint 直接当成可部署 RS4 policy。
- 不要继续做旧语义 full 9D BC。
- 不要直接把 `ee1_rot_z/y/x` 改成 policy action 后训练。
- 不要直接连接真实 RS4 硬件做闭环，除非 sim adapter 已通过 smoke。
- 不要删除或破坏现有 `RecomoProto2TrackEE-v0`；先新增实验 contract。

## 当前保留项

继续保留：

```text
sim_6joint_gimbal_v1:
  [6 URDF arm/gimbal joint position targets, base_vx, base_vy, base_wz]
```

它是当前 Isaac 仿真内部自洽的 baseline。

继续保留并复用：

```text
base-only BC
logs/bc/gik_strict_base_only_20260629/bc_policy.zip
```

原因：base label 已验证干净，和 RS4 action contract 问题解耦。

## Phase 1：代码中显式命名 contract

目标：让代码和日志不再只说 “9D action”，而是明确 action 语义。

建议新增：

```text
src/rl_platform/tasks/mobile_mm/action_contracts.py
```

建议内容：

```text
SIM_6JOINT_GIMBAL_V1
RS4_ATTITUDE_RATE_V1
```

每个 contract 至少定义：

- `name`
- `action_dim`
- `arm_slice`
- `gimbal_or_attitude_slice`
- `base_slice`
- action channel names
- normalized range
- physical unit / scaling

验收：

- 现有训练默认仍使用 `SIM_6JOINT_GIMBAL_V1`。
- 日志打印当前 contract 名称。
- 现有 smoke/test 不退化。

## Phase 2：新增 RS4-aware adapter，不改真实硬件

目标：policy 输出 RS4 attitude/rate 语义，但 Isaac 中仍能驱动相机姿态。

建议新增：

```text
src/rl_platform/tasks/mobile_mm/rs4_adapter.py
```

v1 推荐先做 rate adapter：

```text
policy [-1, 1]
-> yaw/pitch/roll rate command
-> clamp by safe max deg/s or rad/s
-> apply accel/rate smoothing
-> integrate to simulated gimbal target
-> map to current sim controllable joints or a camera-attitude proxy
```

注意：

- roll 在部署侧尚未完全确认，v1 可以支持 channel 但默认 mask 或置零。
- yaw/pitch 是优先路径，因为部署侧已有 2-DoF velocity loop。
- adapter 要显式记录 axis order、sign、unit、frame。

验收：

- 单步 adapter 单元测试通过。
- 输入 zero rate 时 camera attitude 不漂移。
- 输入 yaw rate 时 camera yaw 按预期方向变化。
- 输入 pitch rate 时 camera pitch 按预期方向变化。
- roll 被禁用时 roll channel 不影响 sim 状态。

## Phase 3：新增实验环境或 config flag

目标：不破坏当前 `RecomoProto2TrackEE-v0` 的情况下跑新 contract。

两个可选方案：

方案 A：新增任务名：

```text
RecomoProto2TrackEE-RS4-v0
```

方案 B：保留任务名，但新增显式参数：

```text
--action_contract rs4_attitude_rate_v1
```

推荐方案 A 或 A+B，因为 checkpoint 和日志更不容易混淆。

需要改动：

- `env.py` action parsing。
- observation composition 是否加入 current camera attitude / attitude error。
- reward 是否改为 camera attitude tracking，而不是 gimbal motor joint tracking。
- diagnostics 增加 `rs4_yaw_rate`, `rs4_pitch_rate`, `rs4_roll_rate`, `camera_attitude_error`。

验收：

- 旧任务仍能启动。
- 新任务 action shape 为 9。
- 新任务日志明确打印 `rs4_attitude_rate_v1`。
- 新任务无 NaN、无 action shape mismatch。

## Phase 4：重建 imitation dataset

目标：不要再把 `qTraj` 后 6 维直接当作可部署 action。

建议新增导出 schema：

```text
schema = cinebotrl_gik_rs4_attitude_rate_demo_v1
```

新 action：

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

必需字段：

```text
actions
action_valid_mask
q_current
q_next
current_camera_attitude
target_camera_attitude
camera_attitude_error
actual_ee_quat_wxyz
target_quat_wxyz
attitude_frame_convention
rs4_axis_order
source_frame
dt
```

需要先确认：

- `target_quat_wxyz` 是哪个 frame。
- `actual_ee_quat_wxyz` 是哪个 frame。
- attitude residual 相对 world、base、还是 gimbal-local。
- roll 是否有效。

验收：

- base labels 仍然全部有效。
- arm 3-DoF labels 在 Realman/URDF safe range 内。
- RS4 yaw/pitch labels 在 velocity/rate limit 内。
- roll 若未确认，应 mask 掉或固定为 0。

## Phase 5：短 BC，不接长 PPO

目标：验证新数据和新 contract 是否自洽。

建议先跑：

```text
strict small dataset
2-5 epochs smoke BC
deterministic prediction error check
```

指标：

- action mask 正确。
- base MSE 不退化。
- yaw/pitch rate MSE 有明显下降。
- arm 3-DoF MSE 有明显下降。
- roll 若 masked，不参与 loss。

不通过则回到 Phase 4，不要进 PPO。

## Phase 6：Isaac short rollout

目标：验证 BC policy 在仿真闭环里不是只会离线拟合。

最小 gate：

- 10-20 个短 episode。
- 无 NaN。
- 无 joint-limit saturation。
- 无 self-collision 明显增加。
- camera attitude error 有下降趋势。
- base tracking 不比 base-only BC 明显退化。

只有通过后，才考虑 PPO continuation。

## Phase 7：硬件 adapter 设计，不直接上线

真实部署路径需要单独设计：

```text
policy rs4 yaw/pitch/roll rate
-> axis/sign/offset adapter
-> RoninRs4Control CMD_VELOCITY
-> /gimbal/cmd/tracking or RL-specific topic
-> mux / safety arbitration
-> RS4 status feedback
```

必须确认：

- RS4 axis order。
- `attitude_deg` 是 world、body-relative、还是 DJI SDK frame。
- yaw/pitch/roll 正方向。
- deg/s 最大安全速度。
- roll 是否可和 yaw/pitch velocity control 同时使用。
- fail-safe：policy 输出异常时是否置零速度。

## 建议的首批代码变更顺序

1. 只加 contract definitions 和日志，不改变默认行为。
2. 加 adapter 的纯函数测试。
3. 加新 task/config flag，但默认仍旧 contract。
4. 加新 dataset builder，不覆盖旧 dataset。
5. 加 short BC smoke。
6. 再决定是否接 PPO。

## 回滚策略

任何阶段失败时：

- 保留 `sim_6joint_gimbal_v1`。
- 保留 base-only BC。
- 不删除旧数据。
- 新实验输出全部放在独立目录：

```text
data/gik_rs4_attitude_demos/
logs/bc/rs4_attitude_rate_v1_*/
logs/sb3/recomoproto2trackee_rs4_v0/
```

## 最小验收清单

进入 PPO 前必须满足：

```text
[ ] action contract 名称明确
[ ] 新旧 9D 语义不会混淆
[ ] RS4 adapter axis/sign/unit 有测试
[ ] roll 策略明确：enabled or masked
[ ] dataset schema 明确记录 frame convention
[ ] BC smoke loss 正常下降
[ ] Isaac short rollout camera attitude error 改善
[ ] base behavior 不退化
[ ] 无 NaN / 无明显 collision / 无 joint-limit saturation
```
