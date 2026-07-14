# Corrective Teacher / BC Gate 结果（2026-07-14）

## 结论

本轮完成了纠偏教师生成、accepted-only packaging、一次 bounded BC warm-start 和
5 轨迹 Isaac 闭环对照。候选策略的位置跟踪明显改善，但没有通过预先固定的 gate，
因此 **候选 checkpoint 不升级为 baseline，不运行 PPO，也不启动第二轮 BC**。

当前 baseline 仍为：

```text
artifacts/split_teacher_bc_smoke_20260714/policy_grouped_accepted62_reference_v2.zip
```

本轮 rejected diagnostic candidate：

```text
artifacts/corrective_teacher_bc_20260714/policy_grouped_accepted62_plus_corrective120_w05_e3_lr1e5.zip
```

## 已完成的数据链路

1. Isaac 从失败策略的 episode 1 保存 120 个 policy-visited state，observation 为
   `split_reference_v2` 98D，并附带 0.5 s physical `cam_link` 目标 horizon。
2. MATLAB/Isaac physical `cam_link` FK parity 通过：position max `9.33e-7 m`，
   orientation max `9.11e-5 deg`。
3. runtime-aligned differential IK 对 120/120 状态完成 10 x 50 ms bounded solve；
   `vx/vy/wz`、加速度、arm slew、gimbal rate/acceleration、SRDF collision 全部通过。
4. corrective label 按 `sample_id` 连接原始 observation，只学习 action
   `[0,1,2,6,7,8]`；保留的 attitude-adapter action `[3,4,5]` 为 zero/masked。
5. corrective 120 行权重为 `0.5`，有效权重 60；与 nominal accepted62 的 21,017
   行合并后共 21,137 行、63 个 disjoint source group。
6. pytest 在 macOS、`.98` WSL 和 Windows Isaac Python 均通过；当前 focused suite
   为 8/8。

相关代码 commit：

```text
codex/split-teacher-attitude-v1: 605924f
win-recomoPro1:                 8acc4f2
```

## 唯一一次 BC 更新

固定配置：

- warm-start：accepted62 reference-v2 baseline；
- grouped policy，不改变网络结构；
- source-grouped split：45 train / 9 validation / 9 holdout；
- corrective source group 62 只进入 train；
- action mask enabled；
- dataset sample weight enabled；
- 3 epochs，learning rate `1e-5`，无 PPO。

训练中 validation 最优值出现在 epoch 1，保存前已恢复 best state。

离线结果：

| 数据 | Baseline RMSE | Candidate RMSE | 变化 |
|---|---:|---:|---:|
| corrective 120 | 0.25823 | 0.25169 | -2.5% |
| nominal accepted62 | 0.01034 | 0.01001 | -3.2% |

离线 nominal 总 RMSE 没有退化，但 base action row 的误差略有增加，因此不能用
离线 MSE 直接批准候选。

## 5 x 120 Isaac Gate

同一个 seed、同一批 sequential trajectory（episode 1-5）、同一 physical-camera
contract 下的结果：

| 指标 | Baseline | Candidate | 变化 |
|---|---:|---:|---:|
| position mean (m) | 0.25215 | 0.21005 | -16.7% |
| position p95 (m) | 0.57688 | 0.47750 | -17.2% |
| position max (m) | 0.72291 | 0.53431 | -26.1% |
| orientation mean (deg) | 4.37223 | 4.43988 | +1.5% |
| orientation p95 (deg) | 8.20717 | 8.22993 | +0.3% |
| reward mean | 24.5635 | 31.2648 | +27.3% |

候选失败原因：

- position p95 仍高于固定 gate `0.26433 m`；
- orientation mean 高于固定 gate `4.362 deg`，且相对 baseline 有退化；
- episode 4/5 原本是 easy case，candidate 的 position mean/p95 反而变差；
- episode 1 虽然 position mean 改善 16.2%，orientation mean 从 `4.68 deg`
  恶化到 `6.17 deg`。

因此不能宣称本轮策略已通过，也不能靠增加 epoch 或直接 PPO 掩盖问题。

## 教师质量诊断

第一版教师虽然 runtime-feasible，但属于接近 bang-bang 的恢复标签：

- first-step arm slew saturation：91/120；
- first-step chassis acceleration saturation：103/120；
- first-step gimbal acceleration saturation：74/120。

仅把 response horizon 从 0.5 s 改为 1.0 s 会明显削弱 position/orientation recovery，
已拒绝。velocity-change penalty `2.0` 会导致 orientation regression，已拒绝。

当前最佳离线教师候选为：

```text
DifferentialVelocityChangePenalty = 0.5
DifferentialOrientationWeight = 1.0
DifferentialResponseHorizonS = 0.5
```

与第一版教师比较：

| 指标 | 第一版 | 最佳平滑候选 |
|---|---:|---:|
| mean position improvement (m) | 0.24335 | 0.24084 |
| mean terminal orientation residual (deg) | 3.9895 | 3.3383 |
| arm saturation rows | 91 | 56 |
| chassis saturation rows | 103 | 95 |
| gimbal saturation rows | 74 | 11 |
| terminal orientation > 4 deg rows | 35 | 33 |

该候选 120/120 通过 runtime/collision gate，但尚未用于 BC。chassis saturation 仍然
过高，且数据仍只覆盖 episode 1，所以当前状态仍为 `training_allowed=false`。

## 下一步

下一轮必须先做结构修复，而不是直接训练：

1. 在 GIK objective 中增加针对 chassis acceleration 的独立平滑代价，不要继续用
   一个全 DOF scalar penalty；保留 orientation weight `1.0`。
2. 将 teacher feasibility 与 teacher learnability 分开：对 first-step acceleration/slew
   接近饱和的行降低权重或拒绝，不能因为“可执行”就认为“适合 BC”。
3. 从 hard/easy 分层轨迹采集 policy-visited state，至少覆盖 episode 1-5；不能再让
   单一 episode 代表整个 corrective distribution。
4. BC 更新应冻结 grouped shared encoder，先只更新 arm/base action heads；避免少量
   corrective 行改变 nominal/easy-case 的共享表示。
5. 新教师和 head-only BC 必须重新执行同一个 5 x 120 gate；未同时满足 position
   p95 和 orientation mean 门限前，继续禁止 PPO 与 obstacle curriculum。

## 证据路径

```text
artifacts/corrective_teacher_bc_20260714/accepted62_plus_corrective120_weight05.npz
artifacts/corrective_teacher_bc_20260714/offline_baseline_on_corrective.json
artifacts/corrective_teacher_bc_20260714/offline_candidate_on_corrective.json
artifacts/corrective_teacher_bc_20260714/offline_baseline_on_nominal62.json
artifacts/corrective_teacher_bc_20260714/offline_candidate_on_nominal62.json
artifacts/corrective_teacher_bc_20260714/isaac5x120_baseline.json
artifacts/corrective_teacher_bc_20260714/isaac5x120_candidate.json
```

GIK 最佳平滑教师证据：

```text
/Users/yanbo/Projects/gikWBC9DOF/artifacts/rl_corrective_teacher_response_20260714/all120_velocity_penalty0p5_ori1p0/summary.json
/Users/yanbo/Projects/gikWBC9DOF/artifacts/rl_corrective_teacher_response_20260714/all120_velocity_penalty0p5_ori1p0/corrective_teacher_response_smoke.csv
```
