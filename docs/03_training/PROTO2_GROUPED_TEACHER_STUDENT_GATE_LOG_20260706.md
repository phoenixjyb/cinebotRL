# Proto2 Grouped Teacher/Student Gate Log - 2026-07-06

## Purpose

First bounded HOVER-style bridge test for CineBotRL:

- grouped actor heads for `arm`, `gimbal`, and `base`
- GIK base-only teacher labels as behavioural-cloning warm start
- capped PPO gate, not open-ended training
- recovery evaluator as the promotion gate

## Teacher Dataset Check

Source:

`data/gik_stage1_recovery_gate20_basehome0_iter300_20260705/obs_dataset_base_only.npz`

Result:

- samples: `2,826`
- source observation dim: `85`
- action dim: `9`
- valid mask mean: `[0,0,0,0,0,0,1,1,1]`
- base-only labels are valid; full masked arm/gimbal labels remain diagnostic only.

The 85D dataset appends progress, while the PPO training environment currently exposes 84D policy observations. The first BC policy trained on 85D achieved low validation error, but could not warm-start PPO:

```text
[WARN] Failed to load BC pretrained policy: size mismatch ... [256, 85] vs [256, 84]
```

## Corrected BC Gate

Derived dataset:

`data/gik_stage1_recovery_gate20_basehome0_iter300_20260705/obs_dataset_base_only_obs84.npz`

Transformation:

- dropped the final progress column
- preserved actions and `action_valid_mask`
- set `observation_dim=84`

Command output:

- output policy: `logs/bc/gik_stage1_gate20_grouped_baseonly_obs84_20260706/bc_policy.zip`
- best validation MSE: `0.000082`
- architecture: grouped shared `[256,256]`, per-head hidden `128`

## Verified PPO Warm Start

Run:

`logs/sb3/recomoproto2trackee_v0/stage1_gate20_grouped_basebc_obs84_64k_20260706`

Gate:

- `64` envs
- `65,536` timesteps
- `stage1_recovery`
- `max_trajectories=20`
- `min_trajectory_duration=5.0`
- copied only base rows `[6,7,8]` from BC

Warm-start evidence:

```text
[OK] BC policy feature weights loaded
[OK] BC policy grouped shared encoder and selected action rows [6, 7, 8]; zeroed non-selected rows loaded
[OK] kept PPO log_std_init; set non-selected log_std=-2.0
```

Final training metrics:

- `approx_kl=0.010740501`
- `explained_variance=0.601`
- `value_loss=0.0677`
- `std=0.135`
- exit code: `0`

## Evaluator Repair

`scripts/reinforcement_learning/sb3/evaluate_recovery_candidate.py` now infers the checkpoint observation dim and drops one trailing observation column only when the evaluator env returns exactly `expected_dim + 1`.

Observed adapter evidence:

```text
[obs-adapter] Dropping final observation column for checkpoint compatibility: 85 -> 84
```

This fixes evaluation of older/current 84D PPO checkpoints against recorded-trajectory eval configs that append progress.

## Evaluation Result

Output:

`evaluation_results/recovery_candidate/stage1_gate20_grouped_basebc_obs84_64k_20260706/recovery_eval_raw-policy_20260706_234006.json`

Raw-policy metrics:

- `ee_pos_error_mean_m.mean=1.7677`
- `ee_pos_error_p95_m.mean=1.9410`
- `ee_ori_error_mean_deg.mean=166.6523`
- `unreachable_zone_pct.mean=91.2037`
- `workspace_hard_exceed_pct.mean=15.6481`
- `obstacle_unsafe_pct.mean=0.0`
- `obstacle_collision_pct.mean=0.0`

Comparison candidate:

`evaluation_results/recovery_candidate/fortykg_planar_smoke_20260706_2148/recovery_eval_raw-policy_20260706_214759.json`

- `ee_pos_error_mean_m.mean=1.1442`
- `ee_pos_error_p95_m.mean=1.3883`
- `ee_ori_error_mean_deg.mean=132.1662`
- `unreachable_zone_pct.mean=47.4211`
- `workspace_hard_exceed_pct.mean=3.7318`
- `obstacle_unsafe_pct.mean=0.0`
- `obstacle_collision_pct.mean=0.0`

## Decision

Do not promote `stage1_gate20_grouped_basebc_obs84_64k_20260706`.

The infrastructure works, but base-only BC warm-start by itself regressed recovery tracking versus the current yaw-assist candidate. The likely issue is that the student receives only base labels while arm/gimbal remain RL-only and undertrained in a short 64k gate. This is useful as a negative gate, not as a better policy.

## Next Recommendation

Keep the grouped architecture and evaluator adapter. Do not run a longer version of this same base-only gate blindly.

Next policy work should target one of:

- train a stronger teacher/student phase with validated arm/gimbal labels or replay-safe filtered labels
- add an auxiliary base imitation loss during PPO instead of only copying BC weights at initialization
- evaluate grouped architecture starting from the current best yaw-assist/recovery checkpoint if compatible, rather than starting from base-only BC

## Follow-up: Active Base Aux Loss Gate

Code update:

`scripts/reinforcement_learning/sb3/train.py`

The existing `MaskedActionAuxCallback` previously assumed a flat SB3 `action_net`, so it did not support the grouped actor where `action_net` is `Identity()` and the real actor heads live under `GroupedActionMlpExtractor.action_heads`.

The callback now supports both paths:

- flat policy: preserve previous behavior and train selected `action_net` rows
- grouped policy: train only grouped action heads touched by the selected rows

Smoke evidence:

```text
[BaseAssistAux] enabled
  action rows: [6, 7, 8]
  grouped heads: ['base']
[BaseAssistAux] rollout=1 loss=0.001829
```

Bounded run:

`logs/sb3/recomoproto2trackee_v0/stage1_gate20_grouped_basebc_aux_obs84_64k_20260706`

Gate:

- `64` envs
- `65,536` timesteps
- `stage1_recovery`
- `max_trajectories=20`
- `min_trajectory_duration=5.0`
- warm-start from corrected 84D base-only BC
- active aux dataset: `obs_dataset_base_only_obs84.npz`
- aux rows: `[6,7,8]`
- aux gradient steps per rollout: `8`
- aux batch size: `512`
- aux lr: `5e-5`

Final training metrics:

- `base_assist_aux_loss=7.5e-05`
- `approx_kl=0.008889617`
- `explained_variance=0.554`
- `value_loss=0.0642`
- `std=0.135`
- exit code: `0`

Evaluation output:

`evaluation_results/recovery_candidate/stage1_gate20_grouped_basebc_aux_obs84_64k_20260706/recovery_eval_raw-policy_20260706_235028.json`

Raw-policy metrics:

- `ee_pos_error_mean_m.mean=1.8187`
- `ee_pos_error_p95_m.mean=1.9484`
- `ee_ori_error_mean_deg.mean=164.9784`
- `unreachable_zone_pct.mean=92.6562`
- `workspace_hard_exceed_pct.mean=6.0938`
- `obstacle_unsafe_pct.mean=0.0`
- `obstacle_collision_pct.mean=0.0`

Comparison:

- grouped base-BC without aux: `unreachable_zone_pct.mean=91.2037`, `ee_pos_error_mean_m.mean=1.7677`
- current yaw-assist candidate: `unreachable_zone_pct.mean=47.4211`, `ee_pos_error_mean_m.mean=1.1442`

Decision:

Do not promote `stage1_gate20_grouped_basebc_aux_obs84_64k_20260706`.

The grouped auxiliary-loss infrastructure works, but active base-only imitation still does not improve the policy. It slightly improves workspace hard-exceed versus the no-aux grouped run, but reachability and EE tracking remain much worse than the current yaw-assist candidate.

Updated lesson:

Base-only teacher labels are not sufficient for this recovery policy, even when applied continuously during PPO. The next useful policy update should either bring in validated arm/gimbal teacher labels or reuse the stronger yaw-assist policy as the starting point while changing one thing at a time.

## 2026-07-07 Attempt: Yaw-Assist Baseline + Base Aux

Planned next gate:

- resume from `logs/sb3/recomoproto2trackee_v0/stage1_recovery_yaw_assist_gate_20260702_1508/final_model.zip`
- keep the existing flat yaw-assist policy
- add only active base auxiliary imitation using `obs_dataset_base_only.npz`
- run a tiny resume smoke before any 64k gate

Precheck:

- yaw-assist checkpoint exists
- yaw-assist `vec_normalize.pkl` exists
- checkpoint policy: `ActorCriticPolicy`
- checkpoint extractor: `MlpExtractor`
- checkpoint observation dim: `85`
- action dim: `9`
- matching 85D aux dataset: `obs_dataset_base_only.npz`

Blocked before policy validation:

```text
RuntimeError: No CUDA GPUs are available
```

Windows/WSL GPU status:

```text
NVIDIA RTX PRO 4000 Blackwell
Status: Error
Problem: CM_PROB_FAILED_POST_START
ConfigManagerErrorCode: 43
```

`nvidia-smi` also fails from both WSL and Windows paths:

```text
Failed to initialize NVML: N/A
NVIDIA-SMI has failed because you do not have sufficient permissions.
```

Decision:

Do not interpret this as a training or policy failure. The next gate is blocked by the host GPU/driver state. Resume the yaw-assist + base-aux smoke only after Windows reports the NVIDIA display adapter as `OK` and WSL/Isaac can see CUDA again.

Follow-up recovery attempts from WSL:

- Cleared stale Isaac/Windows-Python processes from previous evaluation commands.
- Rechecked `/usr/lib/wsl/lib/nvidia-smi`; NVML still failed.
- Tried `pnputil /restart-device` for the exact NVIDIA PCI instance; blocked with `Access denied`.
- Tried restarting `NVDisplay.ContainerLocalSystem`; blocked because the shell is not elevated.

Privilege state:

- Windows token is medium integrity.
- `BUILTIN\Administrators` appears as deny-only, so this WSL-launched PowerShell cannot perform device-manager or service-control recovery.

Required host action:

Run one of the following from an elevated Windows desktop/session, then recheck WSL CUDA:

```powershell
pnputil /restart-device "PCI\VEN_10DE&DEV_2C34&SUBSYS_20521028&REV_A1\287078D8CC2DB04800"
nvidia-smi
```

If the device remains Code 43, reboot Windows or clean-reinstall the NVIDIA driver. After recovery, WSL should satisfy:

```bash
/usr/lib/wsl/lib/nvidia-smi
```

Only then rerun the yaw-assist + base-aux smoke.
