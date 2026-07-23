# 升降柱台架原始日志与自动归约合同

日期：2026-07-23  
状态：CPU-only 工具就绪；尚无实测日志，不构成采购、实机迁移或训练许可

## 目的

`RISER_BENCH_MEASUREMENT_TEMPLATE_20260723.json` 是最终工程审核输入，但其中
RMS 电流、末段温升斜率和最差停车距离不能依赖人工抄写。

新工具 `scripts/two_wheel_balance/reduce_riser_bench_log.py` 从校准后的 CSV
时序日志自动生成以下数字字段：

- 连续运行时长和按时间积分的占空比；
- 命令速度、稳态最低实测速度；
- 相电流 RMS/峰值、直流输入电流 RMS；
- 直流母线最高电压；
- 环境温度、电机外壳/驱动器最高温度；
- 最后 300 s 的电机与驱动器温升斜率；
- 每次急停的触发初速和停车距离；
- 十次急停中的最小初速、最差停车距离、电流和母线峰值；
- fault、跳齿和位置丢失状态。

输出只允许合并到测量模板的 `continuous_duty` 和 `emergency_stop` 数字区。
仪器校准、供应商书面批准、抱闸、防坠、限位、端挡和安全断能仍须人工审查并绑定
独立证据 SHA-256。

原始数字归约本身不代表电机身份。合并时必须再验证目标模板中的
`candidate.drive_profile`：400 W 工程样机和 750 W 生产设计候选不得交叉使用
厂家批准、力学摘要或台架结论。

## CSV 合同

模板文件：
`docs/03_training/two_wheel_balance/RISER_BENCH_RAW_LOG_TEMPLATE_20260723.csv`

列顺序固定，不能增删或重排：

| 列 | 语义 |
|---|---|
| `time_s` | 全日志严格递增时间 |
| `phase` | 仅 `continuous` 或 `emergency_stop` |
| `trial_id` | 连续段为 0；急停试验为正整数 |
| `active_command` | 当前处于连续运行命令段，0/1 |
| `steady_state` | 已排除加减速过渡、可用于最低速度统计，0/1 |
| `stop_trigger_event` | 每个急停 trial 恰好一行置 1 |
| `commanded_velocity_mps` | 升降速度命令 |
| `measured_velocity_mps` | 独立位置测量求得的实测速度 |
| `position_m` | 独立标定后的 carriage 位置 |
| `phase_current_a` | 相电流绝对值 |
| `dc_input_current_a` | 驱动直流输入电流绝对值 |
| `dc_bus_voltage_v` | 驱动直流母线电压 |
| `ambient_temperature_c` | 环境温度 |
| `motor_housing_temperature_c` | 电机外壳温度 |
| `drive_temperature_c` | 驱动器温度 |
| `fault_active` | 任意驱动/系统 fault，0/1 |
| `tooth_jump_detected` | 编码器交叉校验发现跳齿，0/1 |
| `position_loss_detected` | 急停时位置丢失，0/1 |

布尔列只允许 `0/1`；所有数字必须有限；相电流和直流输入电流采用绝对值且不得为
负。`continuous` 必须是一个连续日志块，并包含至少三个
`active_command=1 && steady_state=1` 样本。

每个 `emergency_stop` trial 必须：

- 使用唯一正整数 `trial_id`；
- 恰好一行 `stop_trigger_event=1`；
- 触发后继续记录，直到 `abs(measured_velocity_mps) <= 0.02`；
- 保留触发点和首次停止点的位置，工具据此计算停车距离；
- 保留 fault、母线、电流和位置丢失状态。

## 执行

```bash
python3 scripts/two_wheel_balance/reduce_riser_bench_log.py \
  --input /path/to/calibrated_riser_bench.csv \
  --candidate-profile leadshine_750w_production_candidate_v1 \
  --output /path/to/riser_bench_reduction.json
```

工具拒绝覆盖已有输出。输出记录原始 CSV 的 SHA-256，并始终保持：

- `ready_for_production_design_review=false`；
- `valid_for_production_procurement=false`；
- `valid_for_hardware_transfer=false`；
- `valid_for_training=false`；
- runtime、GPU、BC 和 PPO 未授权。

指定 `--candidate-profile` 时输出 v2 candidate-bound schema，可进入同一候选的
证据组装。未指定时保留历史 v1 数字归约行为，但
`valid_for_candidate_bound_bench_merge=false`，不得用于 400 W/750 W
候选绑定的最终台架组装。

只有将归约结果与完整校准记录、供应商证据和机械/功能安全试验合并后，才能再运行
`audit_riser_bench_measurements.py`。即使最终台架 gate 全部通过，也只进入生产设计
评审，不自动批准采购或实机部署。
