# 750 W 升降柱台架证据组装合同

日期：2026-07-23  
状态：CPU-only 工具就绪；尚无真实供应商回复或台架数据

## 目的

本合同把三类互相独立的输入合并为一个 750 W 台架测量记录：

1. 人工填写并审核的校准、机构配置、抱闸、防坠、限位、端挡和安全断能记录；
2. 由原始 CSV 自动归约、且绑定 750 W candidate profile 的连续运行/急停数字；
3. 通过独立审核、且明确绑定 750 W 电机/驱动/减速/导程的供应商书面回复。

组装器不会推断缺失字段，也不会把 400 W、旧 v1 数字归约或通用供应商回复升级
为 750 W 证据。

## 固定候选身份

所有输入必须同时匹配：

| 字段 | 固定值 |
|---|---|
| profile | `leadshine_750w_production_candidate_v1` |
| motor | `ELVM8075V48EH-M17-HD` |
| drive | `ELD2-CAN7020B` |
| reduction | `3.0` |
| linear lead | `0.07 m/rev` |

厂家快照必须是
`cinebotrl_two_wheel_riser_production_candidate_vendor_snapshot_v1`，力学摘要必须是
`cinebotrl_two_wheel_riser_hardware_production_candidate_v1`。

## 输入 1：人工记录

以 `RISER_750W_BENCH_MEASUREMENT_TEMPLATE_20260723.json` 为起点。人工只填写：

- `test_id`；
- 四类校准记录 SHA-256 和有效性；
- 完整运动质量、摩擦、配重、机械行程和软件高度范围；
- 抱闸静态保持、防坠测试；
- 上下硬限位、上下吸能端挡和安全断能；
- 安全测试视频 SHA-256。

以下自动字段必须保持模板中的 `null/false`，否则组装器拒绝输入：

- `continuous_duty`；
- `emergency_stop`；
- `supplier_evidence`；
- `evidence.raw_log_sha256`；
- `evidence.supplier_approval_package_sha256`。

## 输入 2：候选绑定的原始日志归约

新台架采集必须显式指定 profile：

```bash
python3 scripts/two_wheel_balance/reduce_riser_bench_log.py \
  --input /path/to/calibrated_riser_bench.csv \
  --candidate-profile leadshine_750w_production_candidate_v1 \
  --output /path/to/riser_bench_reduction_v2.json
```

组装器只接受：

- schema `cinebotrl_two_wheel_riser_bench_log_reduction_v2`；
- `candidate_profile=leadshine_750w_production_candidate_v1`；
- `passed=true`；
- `valid_for_candidate_bound_bench_merge=true`；
- 原始 CSV SHA-256 存在；
- runtime、GPU、训练、BC、PPO、采购和硬件迁移全部为 false。

没有 profile 的历史 v1 归约仍可用于数字检查，但不能进入候选绑定组装。

## 输入 3：750 W 供应商审核

供应商审核必须：

- 通过 `cinebotrl_two_wheel_riser_supplier_response_audit_v1`；
- 通用和 400 W merge flag 均为 false；
- `valid_for_750w_bench_supplier_evidence_merge=true`；
- `required_candidate` 与上述五项身份完全相同；
- 四项供应商批准均为 true；
- 供应商批准包 SHA-256 存在，并与供应商审核输入中封存的回复 SHA-256 完全相同；
- 所有 runtime、训练和采购权限保持关闭。

## 组装和最终 gate

```bash
python3 scripts/two_wheel_balance/assemble_riser_750w_bench_evidence.py \
  --manual-measurements /path/to/manual_750w_measurements.json \
  --numeric-reduction /path/to/riser_bench_reduction_v2.json \
  --supplier-audit /path/to/supplier_response_audit.json \
  --output-measurements /path/to/assembled_750w_measurements.json \
  --output-audit /path/to/assembled_750w_audit.json
```

组装器拒绝覆盖输出。结构合同不通过时不生成结果；结构合同通过但最终台架 gate
失败时，保留 fail-closed 测量和审核文件并返回非零退出码。只有完整台架 gate
通过时，`ready_for_production_design_review=true`。

即使最终 gate 通过，以下状态仍固定为 false：

- `valid_for_production_procurement`；
- `valid_for_hardware_transfer`；
- `simulation_motor_model_updated`；
- runtime/GPU；
- `valid_for_training`、BC、PPO。

生产采购、仿真 plant 切换和实机迁移仍需独立评审。
