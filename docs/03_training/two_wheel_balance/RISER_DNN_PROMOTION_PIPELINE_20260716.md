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

> **2026-07-22 模型规划器残差合同更新：** 上述
> `[0.30, 0.40, 0.10]` 只属于以 phase feed-forward 为基准的历史
> `residual_merged_v2/v3` 路径。当前 case-30 纠偏教师已经改为在完整模型规划器之上
> 输出小残差，其固定尺度为 `[0.05 m/s, 0.05 rad/s, 0.02 m]`，命令合同为
> `model_based_planner_plus_bounded_policy_residual_v1`。两套数据不得拼接、重标记或
> 共用同一个 merged schema。
>
> 新的 `model_based_corrective_case_dataset_v1` 转换只接受通过独立 finalizer 的
> `corrective_teacher_capture_v2`。训练标签必须来自 safety supervisor 之后的
> `effective_corrective_normalized_actions`；supervisor 之前的 requested action 只保留
> 作审计，不得作为训练标签。由于采集时 previous-action 观测记录的是 requested
> action，转换器必须按 effective action 重新构建递推历史：首行归零，其后第 `i`
> 行等于第 `i-1` 行 effective label。单 case 输出只允许进入后续 merge 评审，保持
> `valid_for_training=false`、BC/PPO 未授权，不能直接送入现有 BC trainer。
>
> **2026-07-23 时序投影补充：** case-30 的 `11,410` 个 transition 中，
> requested teacher intent 在物理 slew 上 `0/0/0` 违规，但 effective
> post-supervisor label 有 `30/49/8` 个通道 transition 超限；全部 `87`
> 个超限 transition 都与相同通道的 command clipping 相邻。这些是安全投影导致的
> effective label 跳变，不能误判成 teacher chatter，也不能简单把 effective label
> 的逐帧差分作为 policy slew gate。后续 model-based BC 必须把网络输出定义为
> requested bounded residual，先经过可微的
> `model_based_residual_safety_projection_v1`，再与 effective label 计算
> pointwise loss；requested output 的 slew regularization/gate 独立执行。
> requested action 仍不得作为 pointwise training target，最终 runtime safety
> supervisor 仍保持最高控制权限。
>
> **2026-07-23 CPU 实现状态：** 已实现
> `model_based_projected_effective_action_bc_loss_v1`。网络输出仍表示
> requested normalized residual；损失内部先经过不变的 safety projection，再与
> effective target 比较。独立的
> `requested_physical_residual_slew_hinge_v1` 只约束同一 case 内相邻 requested
> 输出，不用 effective label 的投影跳变判定网络 chatter。case-30 的真实
> `11,411` 行审计证明 projected pointwise loss 约为 `2.86e-13`，而错误的
> requested-vs-effective 直接 MSE 为 `0.00538`；requested slew 违规为
> `0/0/0`，effective label 的 `30/49/8` 个违规仍全部来自 supervisor
> clipping。该结果只证明损失合同正确，不代表多 case corpus 已满足、不授权 BC，
> 也未生成 checkpoint 或启动训练。现有 review-only corpus 仍被 BC 入口拒绝。
>
> **2026-07-23 训练 schema 晋级合同：** 已实现
> `cinebotrl_two_wheel_riser_model_based_corrective_training_v1` 的晋级器，但
> 当前没有创建该数据集。晋级必须输入至少 `4` 个 train case 和 `2` 个 validation
> case 的 review corpus，保留 holdout `[3,5,13,19,24]` 完全未打开，并由独立
> admission 精确绑定 corpus、commit、loss module、loss audit、promotion module
> 和 promotion CLI 的 SHA。晋级只派生同 case previous-row index、正
> `delta_time_s`、transition mask 和每 case 总权重为 `1` 的 sample weight；
> 不修改 effective label，也不把 requested audit action 变成 target。仓库中的
> admission template 缺少 corpus SHA/commit，且
> `training_schema_promotion_approved=false`，因此不能执行。即使未来晋级成功，
> `bc_authorized=false`、`ppo_authorized=false`、`learned_rollout_authorized=false`
> 和 `training_started=false` 仍保持不变；BC 入口在单独集成和授权前继续拒绝
> 新 schema。
>
> **2026-07-23 BC 入口预检：** BC 入口现在能够完整加载并验证上述训练
> schema，但只在显式 `--preflight-only` 下运行 projection-aware loss 的
> CPU 诊断。预检使用 `requested_actions_audit` 检查 supervisor projection
> 和时序 slew 合同，不把它当成训练 target；训练 target 仍是 effective
> post-supervisor label。由于历史 supervisor 还可能包含时序限制，
> audit request 不要求逐点重建 effective label，重建误差作为独立诊断报告。
> 预检不创建 output directory、checkpoint 或 TorchScript。相同数据若不带
> `--preflight-only`，入口会在 optimizer 创建前因缺少独立、hash-bound BC
> 授权而拒绝；legacy 数据也不能借用这个预检开关。
>
> **2026-07-23 projection-aware optimizer kernel：** 已实现但尚未接入可执行
> trainer。kernel 为每个当前 observation 显式取同 case、同 split 的前一
> observation，分别计算当前和前一 requested action。minibatch 不各自重新
> 归一化为不同目标；它先按全 split 的 case-balanced pointwise weight 和有效
> transition weight 缩放每个 batch 的梯度，再在完整一轮后只做一次 optimizer
> step，因此不同 batch 切分保持同一目标。validation 必须先通过 frozen safety
> projection，再对 effective label 计算逐 case 平衡 MSE，并独立报告 zero-request
> baseline、clipping、requested-action 范围和 slew。CPU synthetic test 证明不同
> batch 切分的一步参数更新一致，并能确定性学习一个可达的三通道 teacher；无效
> 的跨 case/split predecessor 会被拒绝。该 kernel 目前没有 CLI 路由，不创建
> policy，且不改变 `bc_authorized=false`。
>
> **2026-07-23 BC execution admission/report 合同：** 已定义严格的
> `cinebotrl_two_wheel_riser_model_based_corrective_bc_execution_admission_v1`
> 和 execution report v1。admission 必须绑定 promoted dataset、promotion
> commit、execution commit、trainer、adapter、loss、policy、training-dataset
> module 和 admission module 的 SHA，并固定 optimizer、网络尺寸、seed、CUDA
> device、case split 和 unopened holdout。仓库模板故意保留 dataset/commit
> 为 `null`、split 为空、`bc_execution_approved=false`，不能通过验证。成功
> report 必须绑定 admission 和 dataset，包含连续 epoch history、train/validation
> projected metrics、zero-request improvement、requested-action margin、真实
> checkpoint/TorchScript SHA，且 holdout、PPO、learned rollout 保持关闭。
> 缺 artifact、伪造 code/dataset hash、stale commit、改超参数、打开 holdout
> 或虚假 success 均 fail closed。trainer 现已接入该 admission 和 reviewed
> optimizer kernel：只有 clean `HEAD==upstream`、dataset/code/config/hash 全匹配
> 且 admission 显式批准时才创建 optimizer。synthetic end-to-end test 可以生成并
> 重新验证 checkpoint/TorchScript；仓库模板和当前真实数据状态仍无法进入该路径，
> 因此没有授权或启动真实 BC。

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

每个动作通道在 validation 上必须比零动作 MSE 至少改善 `5%`，且
teacher-forced 与启用时的递归 validation 预测在每个通道上都必须严格小于
归一化动作幅值 `0.95`，保留执行安全余量。如果某个通道的零动作信号为零，
则不能把“同样为零”误报为改善。Gate B 不计算 holdout 指标。离线 gate
失败时只写审计报告，不生成 checkpoint 或 TorchScript。

对 `model_based_corrective_merged_v1` 的后续可训练 schema，Gate B 还必须
使用 safety-projected effective-action loss，并报告 requested policy output 的
物理 slew 指标。clipped transition 可参与投影后的 pointwise effective loss，
但不得直接用于判定 requested policy output 的 teacher-slew 一致性。

Gate B 通过只产生 `offline_policy_candidate_ready=true` 的候选
checkpoint，不授权任何 Isaac rollout。训练器必须始终写入
`learned_rollout_authorized=false`、`dynamic_holdout_authorized=false` 和
`separate_dynamic_authorization_required=true`。Gate C 需要另行评审并签发
一次性、绑定策略 SHA-256、代码 commit、case 集和新 namespace 的运行授权；
不得把 validation 指标通过直接解释为动态运行授权。

## Gate C：case-disjoint 动态 validation 与 holdout

当前 `model_based_planner + [0.05,0.05,0.02]` 策略必须先运行 validation：

```bash
bash scripts/two_wheel_balance/run_model_based_learned_split_policy_gate.sh \
  validation_canary --preflight
```

另行签发一次性运行 admission 后才可把 `--preflight` 改为 `--execute`。
validation 只允许 BC 报告中的完整 validation case，不能包含 train 或保留
holdout case。只有 validation 动态 gate 通过、模型选择正式结束并把该报告
逐字节绑定到新的 holdout admission 后，才能单独预检 holdout：

```bash
bash scripts/two_wheel_balance/run_model_based_learned_split_policy_gate.sh \
  holdout --preflight
```

holdout 固定为 `[3,5,13,19,24]`，不得参与训练、早停、超参数选择或
validation 阈值调整。每个 case 分别运行：

- 完整 model-based planner 加零学习残差的基线；
- 学习残差策略；
- 同一个 model-based 零残差结果作为教师与 null-action 对照。

晋级条件：

- 学习策略每个 case 的硬安全与完成 gate 全部通过；
- 学习策略平均位置 p95 比零策略动作至少改善 `5%`；
- 超过一半 case 的位置 p95 优于零策略动作；
- 相对确定性教师，位置、姿态、pitch、升降和云台误差各项不得回退超过 `5%`；
- 残差动作保持在 `[-1, 1]`。

输出按策略 SHA-256 与 case 集合锁定并可恢复，禁止混用不同 checkpoint。
admission、CPU preflight、exact-source、plan、LQR、USD、drive profile、
代码 commit 与每条原始 rollout 都必须哈希绑定。validation 与 holdout 的
summary 分别独立报告；validation 通过不自动授权 holdout。

旧的 `run_riser_residual_holdout_gate.sh` 属于历史
`phase_feedforward + [0.3,0.4,0.1]` 合同，不得用于当前策略。

## Gate D：学习策略全 79 条验证

运行：

```bash
bash scripts/two_wheel_balance/run_model_based_learned_all79_policy_gate.sh
```

只有 Gate C 通过才允许启动。学习策略必须完成 `79/79`，每条轨迹硬 gate 通过，并在上述所有安全和跟踪指标上保持教师 `5%` 回退预算。该 gate 仍不授权 PPO。

旧的 `run_riser_residual_all79_policy_gate.sh` 属于历史
`phase_feedforward + [0.3,0.4,0.1]` 合同，不得用于当前
`model_based_planner + [0.05,0.05,0.02]` 策略。新入口在任何 Isaac 启动前
必须验证独立的 all-79 admission、BC/validation/holdout 报告、策略、
exact-source manifest、79 个 plan 文件、LQR、当前 400 W 仿真资产与代码
哈希。运行采用新 namespace，并支持只恢复 admission/preflight
逐字节一致且已有 case JSON 重新验证通过的中断任务。最终 summary
必须哈希 admission、preflight、plan manifest 以及每条 teacher/learned
rollout JSON。

## Gate E：渲染与最终晋级

Gate D 通过后，固定录制代表 case `[1,15,31,50,73,79]`，覆盖简单、
case-15 限偏航、固定路径、联合自适应、低位、高升降运动和长时长轨迹。
当前入口为：

```bash
bash scripts/two_wheel_balance/run_model_based_learned_render_gate.sh \
  --preflight
```

另行签发 hash-bound render admission 后才允许 `--execute`。入口必须绑定
Gate-D all-79 报告、策略、exact-source、plan、LQR、USD、drive profile、
代码 commit 和新的 namespace，并独占 GPU。录制使用 D3D12 offscreen RTX、
完整 model-based planner、`[0.05,0.05,0.02]` 学习残差和与 Gate D
完全相同的控制参数。

运行结束只生成 rollout gate、MP4 与 `ffprobe` media manifest，不会自动
宣称视觉通过。人工必须基于实际六个视频填写独立 visual-review JSON，
逐项确认机器人结构完整、升降运动可见、camera/gimbal 可见、轮地接触可信、
无脱落 link、无异常振荡。然后运行：

```bash
python3 scripts/two_wheel_balance/finalize_model_based_learned_render.py \
  --media-manifest <media_manifest.json> \
  --visual-review <completed_visual_review.json> \
  --output <learned_render_audit.json>
```

finalizer 和最终 goal auditor 会重新验证 admission、preflight、每条 rollout、
每个 MP4 哈希与媒体参数、reviewer 和 review 时间。模板中的 false 值不能
作为通过证据。

只有以下证据同时存在，DNN 才能成为新 baseline：

- Gate A 数据集 summary；
- Gate B 离线报告与哈希匹配的 TorchScript；
- Gate C case-disjoint summary；
- Gate D 全 79 summary；
- Gate E 可视化视频和对应 JSON gate；
- 代码 commit 与远端分支已推送。

任何 gate 失败时都回到最高优先级根因，不放宽阈值，不盲目延长训练，不自动转向 PPO。
