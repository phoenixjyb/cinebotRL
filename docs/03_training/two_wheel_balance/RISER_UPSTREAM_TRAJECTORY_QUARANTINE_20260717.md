# 两轮升降机器人上游轨迹完整性隔离说明

## 结论

`20260716_residual_all79_phase_v3_clean` 虽然是单一 commit 下重新执行的 Isaac 采集，但其目标轨迹仍来自已被证明截断/重采样的 `no_obstacle_episode_*_split_teacher_v2.npz` 链路。干净的仿真重放不能修复错误的上游目标，因此该语料整体不得用于期望轨迹 BC、PPO、holdout 或策略晋级。

远端目录已于 2026-07-17 重命名为：

```text
/mnt/g/wSpace/cinebotRL-two-wheel-riser/artifacts/two_wheel_riser/
20260716_residual_all79_phase_v3_clean_QUARANTINED_UPSTREAM_TRUNCATED_SOURCE_20260717
```

暂停时 case 65 已完成，case 66 尚未启动。隔离目录保留 `65` 个 JSON gate 和 `65` 个执行态 NPZ。它们仅可用于以下控制器诊断：

- 冻结 LQR、自平衡和 phase governor；
- 物理 `cam_link` 观测、语义 DJI 姿态适配；
- 残差动作范围与教师命令重建；
- Isaac 动态安全、饱和与数值稳定 gate。

禁止将这些标签描述为完整的期望轨迹课程，也禁止把 `65/65` 动态通过解释为源轨迹跟踪通过。

## 上游不一致证据

相对于权威源 JSON：

| episode | 旧两轮 stage | 权威源 | 旧路径长度 | 权威路径长度 |
|---|---:|---:|---:|---:|
| 1 | 253 poses / 25.123689 s | 256 / 4.634756 s | 1.092 m | 2.452 m |
| 4 | 224 poses / 22.291202 s | 723 / 14.042191 s | 2.007 m | 3.849 m |
| 7 | 174 poses / 17.247189 s | 663 / 12.940941 s | 2.260 m | 3.808 m |

## 替代合同

唯一允许进入新残差课程的上游合同为 `exact_source_v1`：

1. 每个 case 保留权威源的 `N` 个姿态和 `N` 个时间戳；
2. retarget 后仍有 `N` 个按原顺序对应的 waypoint states；
3. 动作/transition 数严格为 `N-1`；
4. 初始化/回零段与源轨迹显式分离，不伪装成源标签；
5. 轨迹完整性 gate 与质量/安全 gate 分开；
6. 每个 case 以及 package 顶层都必须显式 `valid_for_training=true`；
7. 79 个 case 全部通过后，才允许重新生成 Isaac residual curriculum。

当前 canary 包位于：

```text
/mnt/g/wSpace/cinebotRL/data/
gikWBC9DOF_exact_source_teacher_integrity_canaries_20260716
```

episode 1/4/7 的计数、时间戳和几何 transport 均通过 `exact_source_v1` 完整性检查，但这些 canary 使用 free-GIK seed，只用于 loader/transport 验证，明确 `valid_for_training=false`。

## 当前停止规则

- 不启动 BC；
- 不启动或恢复 PPO；
- 不从 v4 plans 或隔离 NPZ 恢复采集；
- 等待质量合格、`valid_for_training=true` 的 79-case `exact_source_v1` teacher package；
- 新 package 到达后先运行 `validate_riser_exact_source_manifest.py`，再 retarget、执行独立质量/安全 gate，最后从空目录重新采集 residual curriculum。
