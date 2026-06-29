# GIK 轨迹用于手臂/云台模仿学习的问题分析与修复建议

日期：2026-06-29  
适用仓库：`cinebotRL`，分支 `win-recomoPro1`  
相关脚本：`scripts/imitation/audit_gik_arm_labels.py`

## 结论先行

当前 **可以继续使用底盘三维动作** 做模仿学习：

- `base_vx`
- `base_vy`
- `base_wz`

但当前 **不建议直接启动手臂/云台的 6 维动作模仿学习**。原因不是 GIK 数据完全没用，而是这些标签和 CineBotRL 当前强化学习环境的动作契约不完全一致。直接做 9 维 BC/IL 会有较高风险：策略可能学到大量超出当前安全动作包络、超过环境动作滤波速度、或与 `cam_link` 末端链路不一致的手臂目标。

最稳妥路线是：

1. 继续使用底盘 BC 作为当前有效增益。
2. 对手臂/云台先做标签修复，而不是直接训练。
3. 优先让 GIK 生成端就使用和 RL 一致的安全包络、限速和末端 frame。
4. 只有当审计报告显示通过率足够高后，再启用手臂/云台 imitation。

## 当前动作契约是什么

当前 Proto2 RL policy 的动作维度是 9：

```text
[arm_0, arm_1, arm_2, arm_3, arm_4, arm_5, base_vx, base_vy, base_wz]
```

其中前 6 维是实际由 RL 控制的手臂/腕部关节：

```text
joint6_arm_yaw
joint5_arm_pitch
joint4_elbow_pitch
joint3_gimbal_yaw
joint2_gimbal_roll
joint1_gimbal_pitch
```

这里名字里带 `gimbal` 的后 3 个关节是 **实际受控的腕部/云台关节**，可以属于 RL action。

但 URDF 里还有另一组 MoveIt 风格的虚拟末端云台关节：

```text
ee1_rot_z
ee1_rot_y
ee1_rot_x
```

这三个是为了 MoveIt/规划 frame 处理方便存在的虚拟关节。当前 RL 环境会把它们锁在 0，并且不把它们作为 policy action。任何 imitation 数据如果实际上依赖这三个虚拟关节来达到末端姿态，就不能直接拿来训练当前 RL policy。

## 目前审计了什么

审计脚本：

```bash
/mnt/g/isaaclab_venv/Scripts/python.exe scripts/imitation/audit_gik_arm_labels.py \
  --demo-dir data/gik_ik_demos \
  --manifest manifest_strict.json \
  --output data/gik_ik_demos/arm_label_audit_strict.json

/mnt/g/isaaclab_venv/Scripts/python.exe scripts/imitation/audit_gik_arm_labels.py \
  --demo-dir data/gik_ik_demos \
  --manifest manifest.json \
  --output data/gik_ik_demos/arm_label_audit_full.json
```

审计内容包括：

- action 是否仍在 `[-1, 1]`。
- `q_next` 的 6 个手臂目标是否在 URDF 硬限制内。
- `q_next` 是否在 RL 当前安全动作包络内。
- 相邻样本之间的手臂目标跳变是否超过环境动作滤波器允许值。
- 用 URDF 做轻量 FK，检查 6 个实际受控关节能否复现导出的实际 `cam_link` 位姿。
- 确认虚拟关节 `ee1_rot_z/y/x` 不在当前 RL `cam_link` FK 链路中。

## 关键数据

### strict 数据集

`manifest_strict.json`：

- 31 条轨迹
- 10,123 个样本

逐关节 RL-safe envelope 通过率：

```text
joint6_arm_yaw       0.898
joint5_arm_pitch     0.370
joint4_elbow_pitch   0.898
joint3_gimbal_yaw    0.053
joint2_gimbal_roll   0.944
joint1_gimbal_pitch  0.604
```

逐关节 URDF 硬限制通过率：

```text
1.000 0.999 1.000 1.000 1.000 0.958
```

逐关节 slew 通过率：

```text
0.974 0.921 0.926 0.910 0.896 0.879
```

轨迹级别通过情况：

```text
action in range                 31 / 31
all targets in URDF limits      18 / 31
all targets in RL safe envelope  0 / 31
all steps within slew limit      0 / 31
FK position p95 OK              10 / 31
FK orientation p95 OK           13 / 31
all gates pass                   0 / 31
```

FK 误差：

```text
mean position error        0.0747 m
worst file p95 position    1.8063 m
mean orientation error     0.0469 rad
worst file p95 orientation 1.2404 rad
```

### full 数据集

`manifest.json`：

- 79 条轨迹
- 24,224 个样本

逐关节 RL-safe envelope 通过率：

```text
0.837 0.410 0.700 0.100 0.748 0.614
```

逐关节 URDF 硬限制通过率：

```text
1.000 0.988 1.000 1.000 0.997 0.956
```

逐关节 slew 通过率：

```text
0.937 0.882 0.875 0.838 0.800 0.853
```

轨迹级别通过情况：

```text
action in range                 79 / 79
all targets in URDF limits      47 / 79
all targets in RL safe envelope  0 / 79
all steps within slew limit      0 / 79
FK position p95 OK              26 / 79
FK orientation p95 OK           39 / 79
all gates pass                   0 / 79
```

FK 误差：

```text
mean position error        0.0847 m
worst file p95 position    1.8704 m
mean orientation error     0.0515 rad
worst file p95 orientation 1.5569 rad
```

## 为什么会发生这个问题

### 1. GIK 的可行空间大于当前 RL 的安全动作空间

URDF 硬限制比较宽：

```text
lower: [-3.142, -1.570, -2.356, -3.142, -3.200, -3.200]
upper: [ 3.142,  1.570,  2.356,  3.142,  1.570,  1.570]
```

但当前 RL 环境为了训练稳定性，把手臂 action 映射到一个更窄的安全包络：

```text
lower: [-1.000, 0.550, -2.000, -1.000, -0.800, -0.800]
upper: [ 1.000, 1.450, -0.400,  1.000,  0.800,  0.800]
```

所以一个 GIK 关节值可能满足 URDF，但是不满足当前 RL action envelope。审计数据显示，这正是主问题：URDF 通过率接近 99%，但 RL-safe envelope 通过率只有 56.8% 到 62.8%，轨迹级别没有任何一条完整通过。

### 2. 当前 GIK 轨迹不是按 RL action filter 生成的

RL 环境里手臂目标不是直接瞬间跳到 action 对应位置，而是有目标滤波：

```text
max_arm_target_delta = max_joint_acceleration * control_dt^2
                     = 6.0 * 0.05^2
                     = 0.015 rad / control step
```

GIK 日志里的相邻目标可能比这个跳得更大。这样会造成两个问题：

- 如果用这些标签做 BC，policy 学到的目标变化速度会比环境实际执行得快。
- 训练时 policy 输出和环境真实执行之间会出现系统性偏差，导致 imitation loss 看起来小，但 rollout 不一定正确。

注意：这里还有一个时间采样细节。如果 GIK 样本间隔是 0.1s，而 RL control tick 是 0.05s，那么严格说两个 GIK 样本之间也许应该允许两个 control step 的变化量，即约 `0.03 rad`。但即使用这个更宽松解释，当前数据仍然没有通过安全包络问题。因此，时间对齐需要后续修正，但不是当前唯一 blocker。

### 3. `gimbal` 名字有歧义，容易把两类关节混在一起

当前 policy 的后 3 个手臂关节名字里有 `gimbal`：

```text
joint3_gimbal_yaw
joint2_gimbal_roll
joint1_gimbal_pitch
```

这些是实际 RL 控制的腕部关节。

但 MoveIt URDF 还包含虚拟末端云台：

```text
ee1_rot_z
ee1_rot_y
ee1_rot_x
```

这些不是 RL action。如果 GIK 或 MoveIt 侧某些姿态调整隐含依赖虚拟关节，而 RL 侧锁住它们，那么同一条末端轨迹在两个系统里就不是同一个控制问题。

当前审计确认：到 `cam_link` 的 FK 链路不包含 `ee1_rot_z/y/x`，这是好的。但这也意味着 GIK 数据必须真正由 6 个 RL 受控关节实现 `cam_link` 位姿，不能靠虚拟关节补偿。

### 4. FK 平均误差不大，但 outlier 太大

strict 数据集 FK 平均位置误差约 `7.5 cm`，平均姿态误差约 `0.047 rad`。平均值看起来还能接受。

但是 worst-file p95 位置误差达到 `1.8 m` 级别，姿态 p95 也超过 `1.2 rad`。这说明至少一部分轨迹存在明显 frame、链路、样本对齐、或者不可达/跳变问题。

这种情况下不能只看平均指标。如果直接训练，policy 会被 outlier 污染，尤其是在 PPO 后续 fine-tune 时可能表现为：

- EE tracking 退化。
- 关节抖动。
- self-collision 增加。
- critic 学到不稳定 value。
- 底盘学到的可达性改善被手臂错误动作抵消。

## 三种修复方向

### 方向 A：放宽或重设 RL arm envelope

做法：把 RL 环境里的 `arm_safe_home` 和 `arm_action_radius` 调整得更接近 GIK 数据分布。

优点：

- 可以最大程度保留现有 GIK 数据。
- 不需要重新生成所有 MATLAB/GIK 轨迹。
- 如果真实机器人确实需要这些姿态范围，RL envelope 太窄本身就是问题。

风险：

- 当前窄包络是为了早期 PPO 稳定和避免乱撞。直接放宽可能重新引入 self-collision、关节极限附近动作和训练爆炸。
- 放宽 envelope 后，已有 PPO checkpoint 的行为分布会变化，不能假设兼容。
- 需要重新做 Isaac rollout 安全验证，而不是只看离线数据。

建议：

- 不要一次性放到 URDF 全范围。
- 先用审计报告统计 GIK 数据的 1%、5%、95%、99% 分位数。
- 针对失败最严重的 `joint5_arm_pitch`、`joint3_gimbal_yaw`、`joint1_gimbal_pitch` 分别设计 envelope。
- 每次只扩大一个小范围，跑短 Isaac smoke test。
- 如果 self-collision 或 PPO 稳定性退化，立即回退。

适合情况：

- 团队确认 GIK 动作范围是物理上合理且真实任务必须的。
- 当前 RL 安全包络被认为过于保守。

### 方向 B：对现有 GIK 标签做 retime / slew-limit / remap

做法：保留现有 GIK 轨迹，但在导出 imitation dataset 时对手臂标签做处理：

- 按 RL-safe envelope clip 或 project。
- 对相邻目标做时间重采样或低通滤波。
- 按 env 的 target filter 反推可执行目标序列。
- 重新计算对应的 EE pose / error / observation。

优点：

- 比重新跑 GIK 成本低。
- 可以快速产生一个“可被当前 RL 环境执行”的 arm label 版本。
- 适合做短期实验，验证 arm BC 是否有潜力。

风险：

- 简单 clip 会改变末端位姿。clip 后的关节不再对应原始 target EE pose。
- 如果只改 action，不改 observation/target，训练数据会变成自相矛盾：观测说目标在 A，action 却只能做到 B。
- 过度滤波会造成轨迹滞后，EE tracking 可能系统性落后。

建议：

- 不要只做盲目 clip。
- 每次 remap 后必须重新跑 FK，重新计算 EE error。
- 生成新的 `manifest_arm_retimed.json`，不要覆盖原始数据。
- 保留每个样本的 `label_source = raw/remapped/clipped/filtered` 诊断字段。
- 先只对通过 FK gate 的轨迹做 retime，不要全量硬修。

适合情况：

- 需要快速验证 arm imitation 的可行性。
- 团队暂时不想重跑 MATLAB/GIK。
- 可以接受它只是中间实验数据，不是最终数据源。

### 方向 C：从 GIK 生成端重新生成数据，并使用 RL 同款约束

做法：在 MATLAB/GIK 侧直接使用 CineBotRL 当前约束：

- 同样的 6 个 RL 控制关节。
- 同样的 joint order。
- 同样的 RL-safe envelope。
- 同样的 per-step slew / velocity / acceleration 限制。
- 同样的末端 frame：`cam_link`。
- 明确锁住 `ee1_rot_z/y/x` 虚拟关节。
- 输出时按 RL control timestep 对齐。

优点：

- 数据语义最干净。
- imitation label 和 RL 环境真正一致。
- 后续可以自然扩展到 full 9D BC 或 DAgger 类流程。

风险：

- 需要改 GIK 生成端，重新跑数据。
- 如果 RL envelope 太窄，GIK 可能大量失败，这会暴露出“当前 RL 动作空间无法完成任务”的事实。
- 需要团队对 frame、URDF、USD、MoveIt 约束做一次对齐确认。

建议：

- 这是中长期最正确的修复方向。
- 生成前先做一个 5 到 10 条轨迹的小批量验证。
- 每条轨迹必须输出质量指标：
  - 是否成功。
  - 最大/均值位置误差。
  - 最大/均值姿态误差。
  - joint envelope margin。
  - per-step slew margin。
  - obstacle clearance。
  - 是否使用了虚拟 EE joint。
- 小批量通过后再跑 full batch。

适合情况：

- 团队要把 GIK imitation 作为长期训练数据来源。
- 希望后续不仅学底盘，也学手臂/云台。
- 希望减少后续 RL fine-tune 的不确定性。

## 推荐执行路线

### 第一步：保持当前训练策略

继续使用底盘 BC：

```text
obs -> base_vx/base_vy/base_wz
```

不要把当前 arm/gimbal 标签直接放进 BC loss。

原因：底盘标签已经验证干净，而手臂标签当前没有任何轨迹通过完整 arm gate。

### 第二步：补充 arm 数据分布报告

在现有 `audit_gik_arm_labels.py` 基础上增加：

- 每个关节的 min/max/percentile。
- 超 envelope 的方向和幅度。
- 超 slew 的连续片段位置。
- FK outlier 对应的文件名和样本 index。

目标是回答：

```text
到底是 envelope 太窄，还是 GIK 数据真的有错？
```

### 第三步：做小规模修复实验

建议并行做两个小实验：

实验 1：轻微放宽 envelope。

- 只针对失败最严重的关节。
- 每次只扩大 10% 到 20%。
- 跑 Isaac 短 rollout，看 self-collision、joint limit、tracking 是否恶化。

实验 2：retime/slew-limit 一小批 FK 干净的轨迹。

- 只选 FK p95 位置和姿态都好的轨迹。
- 重新导出 `obs_dataset_arm_retimed_trial.npz`。
- 只做 masked arm BC smoke，不接入长 PPO。

### 第四步：决定最终路线

如果轻微放宽 envelope 后安全性仍然好：

- 可以考虑将 RL envelope 正式更新。
- 然后重新生成 imitation dataset。

如果 retime 后 FK 和 tracking 保持干净：

- 可以把 retimed 数据作为过渡方案。

如果两者都不稳定：

- 应该回到 GIK 生成端，使用 RL 同款约束重新生成数据。

## 当前不建议做的事情

不建议直接做：

```text
full 9D BC = arm6 + base3
```

也不建议简单把 arm label clip 到 `[-1, 1]` 后训练。当前 action 本来已经在 `[-1, 1]`，真正的问题是 normalized action 对应的物理目标不在 RL-safe envelope，以及时序跳变和 FK outlier。

不建议把 MoveIt 的虚拟 EE joints 纳入 RL action，除非团队明确决定修改 USD/env/action contract。否则训练数据、仿真、真实机器人接口会继续不一致。

## 验收标准

只有满足以下条件后，才建议打开 arm/gimbal imitation：

```text
all targets in RL safe envelope:   接近 100%
all steps within slew limit:       接近 100%，或使用 time-aware slew 后接近 100%
FK p95 position error:             < 5 cm，且无大 outlier
FK p95 orientation error:          < 0.2 rad，且无大 outlier
virtual EE joints:                 不参与 RL action
short Isaac rollout:               无 self-collision、无 joint-limit saturation
PPO continuation smoke:            critic/clip/KL 稳定
```

## 相关文件

```text
scripts/imitation/audit_gik_arm_labels.py
data/gik_ik_demos/arm_label_audit_strict.json
data/gik_ik_demos/arm_label_audit_full.json
data/gik_ik_demos/obs_dataset_strict_base_only.npz
logs/bc/gik_strict_base_only_20260629/bc_policy.zip
```

## 最终建议

短期：继续 chassis-only imitation，并把它作为 PPO 后续训练的底盘先验。

中期：先做 arm 数据分布和 outlier 定位，再做小范围 envelope/retime 实验。

长期：让 GIK 生成端使用 CineBotRL 完全一致的 action contract、joint envelope、slew limit 和 `cam_link` frame。这样生成出来的数据才能作为真正可靠的 full-body imitation 数据源。
