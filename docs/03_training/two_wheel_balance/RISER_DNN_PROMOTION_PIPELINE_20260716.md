# 两轮升降云台机器人 DNN 策略晋级流程

> **2026-07-17 上游隔离：** 原 `corrected_all79_stage`/v4 plans 继承了截断、重采样的 GIK 源日志。`20260716_residual_all79_phase_v3_clean` 已整体隔离，不能用于 BC/PPO 或期望轨迹晋级。本流程仅在质量合格的 79-case `exact_source_v1` package 到达后恢复。详见 `RISER_UPSTREAM_TRAJECTORY_QUARANTINE_20260717.md`。

## 目标与边界

目标是在已验证的平衡 LQR 之上训练高层残差 DNN，使无机械臂的两轮升降云台机器人完成全部 79 条修正轨迹。

优先级固定为：

1. 自平衡与有限状态安全；
2. 升降柱与云台安全；
3. 全时长轨迹完成；
4. 相机位置与姿态跟踪精度；
5. 控制代价。

DNN 只输出 `delta-vx`、`delta-wz` 和升降目标增量。轮端力矩仍由冻结的 LQR 控制；DJI 云台姿态适配仍是确定性模块。物理云台电机关节角、旧版隔离 NPZ 和源 GIK 动作都不能作为策略标签。PPO 在本流程中保持未授权。

早期 26 维瞬时观测合同已在正式采集前废弃。它只描述“现在发生了什么”，无法区分当前状态相同、但随后转弯、升降或改变相机姿态的轨迹，因此不具备完整轨迹跟踪所需的前视信息。任何 `executed_residual_v1`、`residual_merged_v1` 或 `residual_policy_v1` 文件都只能作为历史审计，不能升级、拼接或静默转换为当前训练数据。

最终归一化动作范围为 `0.30 m/s`、`0.40 rad/s`、`0.10 m`。该范围来自全部诊断轨迹的原始教师命令审计、`1.10x` 余量和 `0.05` 量化，不得从旧 NPZ 的裁剪动作反推。

## 固定运行合同

- 机器人总质量暂定 `28 kg`；
- 轮距 `0.620 m`；
- 轮径 `0.2032 m`（8 英寸）；
- 相机高度范围 `0.6--1.8 m`；
- 升降速度上限 `1.0 m/s`；
- 机械臂自由度为 `0`；
- 相机观测使用物理 `cam_link` FK；
- 语义 DFR 到物理相机：`R_world_cam = R_world_DFR * Rz(+pi/2)`；
- 跟踪配置：`riser_phase_consistent_v2`；
- 相位合同：`derivatives_scaled_by_progress_v1`；
- 默认规划偏航上限为 `0.25 rad/s`；case 15 单独为 `0.20 rad/s`，case 45 单独为 `0.325 rad/s`，case 78 单独为 `0.35 rad/s`。所有值都低于不变的 `0.40 rad/s` 公共上限，且各自的云台代理速率仍必须通过 `24 deg/s` gate。

## DNN v2 观测与动作合同

策略输入固定为 65 维：原 26 维已执行物理状态前缀，加上 `0.25 s`、`0.50 s`、`1.00 s` 三个前视点，每个前视点 13 维。前视时钟使用重定时后的 `execution_time_s`，查询超过终点时固定在最后一个不可变 source anchor，不使用 source clock 冒充执行时钟。

每个前视点包含：

- 机体坐标系中的未来 base 目标 `x/y/yaw` 误差；
- 机体坐标系中的未来物理 `cam_link` 目标 `x/y/z` 误差；
- 未来物理相机姿态误差向量；
- 未来升降目标误差；
- 按当前 progress scale 缩放的未来 `vx/wz/riser_velocity` 前馈。

相机目标先按 `R_world_cam = R_world_DFR * Rz(+pi/2)` 从语义 DFR 转为物理 `cam_link`，再与物理 FK 比较。策略动作仍严格为 3 维，不增加机械臂动作、轮端力矩动作或 DJI 物理电机关节动作。数据、模型和 gate 必须分别声明 `executed_residual_v2`、`residual_merged_v2`、`residual_policy_v2` 以及 `executed_state_with_execution_time_lookahead_v2`；任一声明缺失或不一致即 fail closed。

网络结构固定为 `state_shared_lookahead_fusion_v1`，而不是把 65 维输入直接送入一个无结构的平坦 MLP：

- 26 维已执行状态由 `128,128` 状态编码器处理；
- 三个 13 维前视点使用同一个权重共享的 `64,64` 编码器，确保相同物理量在不同前视时域具有一致特征提取规则；
- 三个按时间顺序排列的前视 embedding 与状态 embedding 拼接，再经 `256,128` 融合网络输出 3 维 `tanh` 动作；
- 所有 encoder 使用正交初始化、LayerNorm 和 SiLU；观测归一化只从 train case 计算；
- checkpoint、离线报告、holdout 和全 79 gate 都必须声明并核验该结构，禁止用旧平坦 MLP 冒充当前策略。

该结构保留前视时域顺序，同时共享每个时域的局部编码参数，目的是在只有 79 条轨迹时减少不必要的时域专用拟合。它不是训练成功的证明；仍须通过完整 BC 和动态 gate。

## Gate A：79 条确定性稠密采集

`20260716_residual_all79_phase_v2` 和隔离后的 `phase_v3_clean` 都只能作为控制器诊断。正式采集必须使用新的 `20260717_all79_playback_exact_source_v1` 和空的 `20260717_residual_all79_exact_source_lookahead_v2`，不得复用旧 gate/NPZ。

启动 Gate A 前必须设置 `RISER_EXACT_SOURCE_MANIFEST_WSL`。该 manifest 必须声明 `exact_source_v1`，包含连续 1--79 case，并对每个 case 证明 `N` 源姿态、`N` 源时间戳、`N` retarget waypoint states、`N-1` transitions、顺序几何保持、初始化分离、完整性通过、独立质量 gate 通过和 `valid_for_training=true`。完整性 canary 的 `valid_for_training=false`，因此不能启动采集。

运行：

```bash
bash scripts/two_wheel_balance/run_riser_all79_dataset_gate.sh
```

每条轨迹单独启动 Isaac 进程，成功后才保留 JSON gate 与 NPZ。启动时要求 tracked worktree 无改动，并生成不可回填的 `exact_source_admission.json` 和 `admission.json`，锁定上游 manifest、Git commit、规划 manifest SHA-256、case 1--79、跟踪配置和相位合同。运行可恢复，但 admission 与当前状态必须完全一致，且只接受以下合同完全一致的已有结果：

- `riser_phase_consistent_v2`；
- `derivatives_scaled_by_progress_v1`；
- 动态 gate 全部通过；
- NPZ 与 JSON 同时存在。

合并前必须满足：上游 exact-source/quality admission 通过、`79/79`、无终止、标签无裁剪、有限数值、教师命令重建误差不超过 `2e-6`、按完整 case 分割且无轨迹泄漏。每个样本必须在相同执行时刻从不可变 plan 在线构造三个前视点；不得从旧 NPZ 猜测或回填未来目标。最终 summary 记录上游 manifest/admission SHA-256、运行 commit、capture admission SHA-256、规划 manifest SHA-256、观测合同、前视时域和所有源 NPZ SHA-256。BC 入口会再次验证整条 provenance 链。

任一 case 失败即停止。禁止用已通过的部分数据提前训练。

## Gate B：离线 BC

运行：

```bash
bash scripts/two_wheel_balance/run_riser_residual_bc_gate.sh
```

BC 指行为克隆。训练数据仅来自 Gate A 通过后的 Isaac 执行状态与确定性控制器残差。归一化统计只从 train case 计算，validation 与 holdout 均按完整 case 隔离。validation 用于早停和离线晋级；holdout 不参与模型选择，只在 Gate C 首次打开。

训练损失按每条轨迹的样本数反向加权，使每个 train case 的总权重相同；validation MSE 也先在每个 case 内计算，再跨 case 平均。禁止让 50 秒长轨迹仅因行数更多而压过 10 秒轨迹。

训练入口要求 tracked worktree 无改动，并把完整 Git SHA-1 同时写入 checkpoint 和报告。Gate C、Gate D 必须在同一 commit 上评估该策略；策略 SHA-256、代码 commit 或 case 集任一变化都禁止恢复旧 rollout。

每个动作通道在 validation 上必须比零动作 MSE 至少改善 `5%`。如果某个通道的零动作信号为零，则不能把“同样为零”误报为改善。Gate B 不计算 holdout 指标。离线 gate 失败时只写审计报告，不生成 checkpoint 或 TorchScript。

## Gate C：case-disjoint 动态 holdout

运行：

```bash
bash scripts/two_wheel_balance/run_riser_residual_holdout_gate.sh
```

只评估保留至此的 8 个 holdout case。每个 case 分别运行：

- 零策略动作基线（不是完整确定性升降控制器）；
- 学习残差策略；
- Gate A 已保存的确定性教师结果用于对照。

晋级条件：

- 学习策略每个 case 的硬安全与完成 gate 全部通过；
- 学习策略平均位置 p95 比零策略动作至少改善 `5%`；
- 超过一半 case 的位置 p95 优于零策略动作；
- 相对确定性教师，位置、姿态、pitch、升降和云台误差各项不得回退超过 `5%`；
- 残差动作保持在 `[-1, 1]`。

输出按策略 SHA-256 与 case 集合锁定并可恢复，禁止混用不同 checkpoint。

## Gate D：学习策略全 79 条验证

运行：

```bash
bash scripts/two_wheel_balance/run_riser_residual_all79_policy_gate.sh
```

只有 Gate C 通过才允许启动。学习策略必须完成 `79/79`，每条轨迹硬 gate 通过，并在上述所有安全和跟踪指标上保持教师 `5%` 回退预算。该 gate 仍不授权 PPO。

## Gate E：渲染与最终晋级

Gate D 通过后，从简单、case 15、固定路径、联合自适应、低位、高位和长时长轨迹中选择代表 case 录制 RTX offscreen MP4。检查机器人结构、轮地接触、升降柱、云台姿态、目标轨迹和异常摆动。

只有以下证据同时存在，DNN 才能成为新 baseline：

- Gate A 数据集 summary；
- Gate B 离线报告与哈希匹配的 TorchScript；
- Gate C case-disjoint summary；
- Gate D 全 79 summary；
- Gate E 可视化视频和对应 JSON gate；
- 代码 commit 与远端分支已推送。

任何 gate 失败时都回到最高优先级根因，不放宽阈值，不盲目延长训练，不自动转向 PPO。
