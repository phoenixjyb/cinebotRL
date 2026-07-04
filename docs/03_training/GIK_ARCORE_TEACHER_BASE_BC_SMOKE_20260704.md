# GIK/ARCore Teacher Base-BC Smoke

Date: 2026-07-04

Host:

```text
ssh -p 2222 yanbo@192.168.100.98
repo: /mnt/g/wSpace/cinebotRL
branch: win-recomoPro1
python: /mnt/g/isaaclab_venv/Scripts/python.exe -X utf8
```

## Purpose

Use the accepted GIK/ARCore teacher materials as learning data without treating
them as validated full-body 9D imitation labels. The safe scope for this smoke is
base-only behavior cloning over `[base_vx, base_vy, base_wz]`.

## Dataset Boundary

Input dataset:

```text
data/gik_offline_teacher_obs/obs_dataset_offline_base_only.npz
```

Source manifest:

```text
data/gik_offline_teachers_20260701_142322/accepted_npz/manifest.json
```

Manifest summary:

```text
accepted teacher items: 130
  no_obstacle: 79 accepted
  one_obstacle: 51 accepted
candidate teacher items:
  two_obstacles_case_a: 27 candidate
  two_obstacles_case_b: 27 candidate
```

Dataset check:

```text
observations: (4557, 85)
actions: (4557, 9)
action_valid_mask: (4557, 9)
finite observations/actions: true
base label validity: [0.9908, 0.9888, 1.0000]
arm/gimbal labels: masked out
max abs action: 1.0
```

Important boundary:

```text
Do not use this result as proof that full arm/gimbal BC is ready.
The GIK arm/gimbal labels still collide with the current conservative RL action
envelope. Failed/candidate two-obstacle cases remain curriculum/eval material,
not positive BC labels.
```

## BC Smoke

Command:

```bash
PYTHONUTF8=1 /mnt/g/isaaclab_venv/Scripts/python.exe -X utf8 \
  scripts/reinforcement_learning/bc/pretrain_bc.py \
  --demo_file data/gik_offline_teacher_obs/obs_dataset_offline_base_only.npz \
  --output_path logs/bc/gik_accepted130_base_smoke_20260704/bc_policy \
  --obs_dim 85 \
  --act_dim 9 \
  --epochs 5 \
  --batch_size 256 \
  --lr 0.0003 \
  --use_action_mask \
  --device auto
```

Result:

```text
total transitions: 4,557
train / val split: 4,102 / 455
architecture: 85 -> 256 -> 256 -> 128 -> 9
best validation MSE: 0.001932
saved policy: logs/bc/gik_accepted130_base_smoke_20260704/bc_policy.zip
```

Validation load/predict smoke:

```text
loaded policy ok
prediction shape: (32, 9)
finite predictions: true
base action mean: [0.161130, 0.016765, 0.007291]
base action maxabs: [0.333943, 0.048794, 0.027883]
```

## Interpretation

The accepted GIK/ARCore materials are useful for base-motion learning. The
current base-only BC path is compatible with the 85D observation contract and
produces a learnable supervised signal.

This is not a promotion candidate by itself. It is a teacher-data smoke that
should feed the next bounded policy gate.

## Next Gate

Before any broad PPO run, run one small policy-initialization experiment:

```text
candidate: initialize or distill base head from
  logs/bc/gik_accepted130_base_smoke_20260704/bc_policy.zip

eval gate:
  64-episode raw recovery smoke
  obstacles enabled
  no runtime base assist
  compare against current Round 4 / Round 9 recovery baselines
```

Promotion criteria:

```text
unreachable mean must improve
p95/max must not regress
obstacle unsafe/collision must remain 0
EE tracking/orientation must be reported separately, not hidden by base metrics
```
