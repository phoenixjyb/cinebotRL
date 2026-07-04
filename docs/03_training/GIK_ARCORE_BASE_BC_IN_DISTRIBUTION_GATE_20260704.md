# GIK/ARCore Base-BC In-Distribution Gate - 2026-07-04

## Purpose

This gate checks whether the accepted GIK/ARCore base-only BC policy actually imitates its own teacher dataset before using it as a PPO warm-start.

The answer is yes. The policy fits the accepted teacher base labels in-distribution. The failed recovery warm-start is therefore a distribution-transfer problem, not a basic BC fitting failure.

## Command

```bash
PYTHONUTF8=1 /mnt/g/isaaclab_venv/Scripts/python.exe -X utf8 \
  scripts/reinforcement_learning/bc/evaluate_bc_dataset.py \
  --policy logs/bc/gik_accepted130_base_smoke_20260704/bc_policy.zip \
  --demo_file data/gik_offline_teacher_obs/obs_dataset_offline_base_only.npz \
  --use_action_mask \
  --action_indices 6,7,8 \
  --device cpu \
  --output_json evaluation_results/bc/gik_accepted130_base_in_distribution_20260704.json
```

## Inputs

- Policy: `logs/bc/gik_accepted130_base_smoke_20260704/bc_policy.zip`
- Dataset: `data/gik_offline_teacher_obs/obs_dataset_offline_base_only.npz`
- Samples: `4,557`
- Observation/action shape: `85 / 9`
- Evaluated action rows: `6,7,8` (`base_vx`, `base_vy`, `base_wz`)
- Effective label mask mean: `[0, 0, 0, 0, 0, 0, 0.9908, 0.9888, 1.0000]`
- Selected labels: `13,578`

## Result

| Metric | Value |
| --- | ---: |
| Masked base MSE | `0.00193070` |
| Masked base RMSE | `0.04393972` |
| Masked base MAE | `0.02457343` |
| Max absolute error | `0.70619857` |

Per base action:

| Action index | Meaning | Count | RMSE | MAE | Max abs | Bias |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 6 | `base_vx` | 4,515 | `0.05091583` | `0.02834063` | `0.70619857` | `-0.00222254` |
| 7 | `base_vy` | 4,506 | `0.04847480` | `0.02853925` | `0.44322780` | `0.00121963` |
| 8 | `base_wz` | 4,557 | `0.02933697` | `0.01691953` | `0.42844003` | `-0.00146063` |

## Interpretation

The BC policy reproduces the accepted GIK/ARCore base labels at roughly the same level as its training/validation smoke result.

This means the poor raw `stage1_recovery` PPO warm-start result should not be treated as "BC failed." It should be treated as "the accepted GIK/ARCore teacher distribution does not directly match the recovery trajectory/reset distribution."

## Decision

Keep the BC policy and dataset as useful learning material.

Do not use this result to justify longer training from the rejected recovery warm-start. A valid next PPO/IL gate needs either:

1. an in-distribution GIK/ARCore-style evaluation stage, or
2. teacher labels generated for the actual `stage1_recovery` distribution.

## Tool Added

Added `scripts/reinforcement_learning/bc/evaluate_bc_dataset.py`.

The script:

- loads a saved SB3-compatible BC policy,
- loads any `.npz` dataset with `observations`, `actions`, and optional `action_valid_mask`,
- applies optional action-index filtering, such as `6,7,8` for base-only,
- reports masked aggregate and per-action MSE/RMSE/MAE/max/bias,
- optionally writes JSON evidence under `evaluation_results/`.

It is intentionally Isaac-free so it can run as a fast pre-PPO gate.
