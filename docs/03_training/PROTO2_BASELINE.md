# RecomoProto2-1190 RL Baseline

## Active Baseline

- Active asset: `assets_own/recomoProto2-1190_moveit/recomoProto2-1190_moveit.usd`
- Primary task alias: `RecomoProto2TrackEE-v0`
- Compatibility task: `MobileMMTrackEE-v0`
- Legacy compatibility alias: `RecomoProto1TrackEE-v0`

The Proto2 USD preserves the MoveIt-style virtual gimbal joints. They are not
part of the RL policy action space.

## V1 Policy Contract

The first Proto2 baseline intentionally keeps the existing 8-action SB3 policy
shape:

```text
[arm_j1, arm_j2, arm_j3, arm_j4, arm_j5, arm_j6, base_vx, base_wz]
```

Locked/passive joints:

```text
base_joint_vy, ee1_rot_z, ee1_rot_y, ee1_rot_x
```

Do not switch to a 9-action full-PPR policy without also updating the reward,
observation history, launchers, checkpoint compatibility notes, and evaluation
scripts.

## Regenerate USD

Run from the project root on the Windows/WSL host:

```powershell
G:\isaaclab_venv\Scripts\python.exe G:\wSpace\cinebotRL\scripts\convert_urdf_to_usd.py `
  --urdf assets_own\recomoProto2-1190_moveit.urdf `
  --usd assets_own\recomoProto2-1190_moveit\recomoProto2-1190_moveit.usd `
  --headless
```

## Validation Gates

Smoke test:

```powershell
G:\isaaclab_venv\Scripts\python.exe G:\wSpace\cinebotRL\scripts\test_mobile_mm_env.py `
  --task RecomoProto2TrackEE-v0 `
  --headless `
  --num_envs 1 `
  --steps 1
```

Multi-env smoke:

```powershell
G:\isaaclab_venv\Scripts\python.exe G:\wSpace\cinebotRL\scripts\test_mobile_mm_env.py `
  --task RecomoProto2TrackEE-v0 `
  --headless `
  --num_envs 2 `
  --steps 10
```

Short PPO gate:

```powershell
.\scripts\launch_training_windows.ps1 `
  -Task RecomoProto2TrackEE-v0 `
  -Headless `
  -NumEnvs 2 `
  -TotalTimesteps 2048
```

Start a longer run only after the smoke tests and short PPO gate pass from a
fresh checkout of `win-recomoPro1`.

