# RecomoProto2-1190 RL Baseline

## Active Baseline

- Active asset: `assets_own/recomoProto2-1190_moveit/recomoProto2-1190_moveit.usd`
- Primary task alias: `RecomoProto2TrackEE-v0`
- Compatibility task: `MobileMMTrackEE-v0`
- Legacy compatibility alias: `RecomoProto1TrackEE-v0`

The Proto2 USD preserves the MoveIt-style virtual gimbal joints. They are not
part of the RL policy action space.

## V3 Policy Contract

The active Proto2 baseline uses a 9-action SB3 policy shape:

```text
[arm_j1, arm_j2, arm_j3, arm_j4, arm_j5, arm_j6, base_vx, base_vy, base_wz]
```

Locked/passive joints:

```text
ee1_rot_z, ee1_rot_y, ee1_rot_x
```

`base_vx` and `base_vy` are body-frame root linear velocities, and `base_wz`
is root yaw rate. The USD PPR base joints are still zeroed every control/readout
step, so policies do not accumulate base joint offsets.

Checkpoint compatibility: prior 8D Proto2 policies are not shape-compatible
with this v3 action contract and must not be resumed directly.

## Regenerate USD

Run from the project root on the Windows/WSL host:

```powershell
G:\isaaclab_venv\Scripts\python.exe G:\wSpace\cinebotRL\scripts\convert_urdf_to_usd.py `
  --urdf assets_own\recomoProto2-1190_moveit.urdf `
  --usd assets_own\recomoProto2-1190_moveit\recomoProto2-1190_moveit.usd `
  --headless
```


## Proto2 v2 Policy Safety Profile

Before starting another long training run, use the v2 safety defaults now baked
into the task and SB3 launcher:

- `log_std_init=-2.0` (`std ~= 0.14`) instead of the old `-1.0` startup noise.
- Default `learning_rate=1e-4` and `ent_coef=0.001` for lower-variance PPO updates.
- Arm position targets are slew-limited per control step before being sent to Isaac.
- Arm reset positions are written directly to sim state, then used to seed the command filter.
- Initial joint reset noise is reduced to `0.03 rad`.
- Self-collision termination/reward contact is masked for the first `12` post-reset control steps.
- Hard self-collision now uses filtered base-arm/EE contact only, not broad arm-ground net forces.
- Base PPR joints are locked to zero every control/readout step; 9D base policy actions drive body-frame root velocity only.
- Arm actions are mapped to a conservative safe-home envelope for v2 stability, not the full joint range.
- Arm joint state is projected back into finite safe limits before dones/observations/rewards.

The previous 512-env probing run reached `524,288` steps but auto-paused on
negative explained variance. Treat v2 as the next stability gate, not as a final
performance policy.

## Obstacle Avoidance Mode

The static-disc obstacle task is opt-in so the plain 9D baseline stays reproducible.
It spawns a kinematic ground cylinder in each replicated environment and feeds one
additional observation dimension: signed base-footprint clearance normalized by
the configured safety radius. The reward path uses the raw signed clearance via
`obstacle_distance_reward`.

Example short gate:

```powershell
G:\isaaclab_venv\Scripts\python.exe G:\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py `
  --task RecomoProto2TrackEE-v0 `
  --headless `
  --num_envs 8 `
  --total_timesteps 2048 `
  --n_steps 64 `
  --batch_size 128 `
  --enable_obstacles
```

Default disc: local env-frame `xy=(0.0, 0.5)`, radius `0.18m`, height `0.08m`.
Override with `--obstacle_x`, `--obstacle_y`, and `--obstacle_radius`.

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

