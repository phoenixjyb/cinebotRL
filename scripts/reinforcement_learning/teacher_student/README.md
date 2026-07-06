# HOVER-Style Teacher/Student Path

This directory documents the first CineBotRL teacher/student route. The code
currently reuses the existing masked BC dataset format and the new grouped SB3
policy. It is not yet the full RSL-RL privileged-teacher pipeline.

## Current Student Policy Smoke

Train a grouped-head student from a masked dataset:

```bash
PYTHONUTF8=1 /mnt/g/isaaclab_venv/Scripts/python.exe -X utf8 \
  scripts/reinforcement_learning/bc/pretrain_bc.py \
  --demo_file data/or/path/to/obs_dataset_base_only.npz \
  --obs_dim 85 \
  --act_dim 9 \
  --use_action_mask \
  --policy_arch grouped \
  --output_path logs/bc/grouped_base_student/bc_policy
```

Warm-start PPO from that student:

```bash
PYTHONUTF8=1 NO_PROXY='*' no_proxy='*' /mnt/g/isaaclab_venv/Scripts/python.exe -X utf8 \
  scripts/reinforcement_learning/sb3/train.py \
  --headless \
  --policy_arch grouped \
  --pretrained_policy logs/bc/grouped_base_student/bc_policy.zip \
  --pretrained_action_indices 6,7,8 \
  --pretrained_unselected_log_std -2.0
```

## Contract

- Output action order stays unchanged: arm indices `0..2`, gimbal/RS4 indices
  `3..5`, base indices `6..8`.
- Base-only or masked teacher labels should use `action_valid_mask` so invalid
  arm/gimbal channels do not train the student.
- This path is intended as the low-risk bridge before a full RSL-RL
  privileged-teacher plus deployable-student implementation.

## Next Upgrade

Port the same action groups and masks to an RSL-RL teacher/student runner:

- teacher observation: privileged sim state, trajectory progress, obstacle state
- student observation: deployable state, command/mode mask, action history
- training: DAgger-style student rollout mixed with teacher labels
- evaluation: per-group base/arm/gimbal metrics plus EE tracking and obstacle safety
