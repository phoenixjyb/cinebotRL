# Two-wheel riser Gate 0--2 results

Date: 2026-07-15
Worktree: `/mnt/g/wSpace/cinebotRL-two-wheel-riser`
Branch: `codex/two-wheel-riser-rl`

> **Superseded asset notice:** these measurements predate the fixed gimbal
> bracket that restores the removed arm's accepted-corpus nominal orientation.
> They remain valid evidence for the riser actuator/controller implementation,
> but are not final-asset acceptance evidence. Gate 0, static heights, and all
> dynamic runs must be repeated on the regenerated USD with unchanged limits.

## Scope

This milestone proves the isolated riser asset and scripted vertical/balance
baseline. It does not claim 79-trajectory tracking or a trained DNN policy.
Obstacle avoidance remains excluded from this round.

## Gate 0: asset and frame contract

Status: **PROVISIONAL PASS; FINAL-ASSET RERUN REQUIRED**

- 14 rigid bodies and 13 joints imported into Isaac;
- exactly six movable joints: two wheels, one riser, three physical gimbal
  joints;
- all three arm joints absent;
- total imported mass: 28.0000007 kg;
- wheel track: 0.620 m; wheel diameter: 0.2032 m;
- USD riser range: 0.0--1.20000005 m;
- USD riser anchor: 0.57353055 m;
- URDF camera optical-center range: exactly 0.6--1.8 m;
- importer drives disabled; Isaac actuator config owns the drives;
- `ee1_tool` is collocated with `cam_link` and satisfies
  `R_world_cam = R_world_DFR * Rz(+pi/2)`.

An initial conversion with global `--mesh-scale 0.001` was rejected because it
also shrank the prismatic range and origin by 1000. The corrected URDF carries
`0.001` only on CAD meshes and is converted with `--mesh-scale 1.0`.

Evidence: `evidence_20260715_riser/gate0_asset_audit.json`.

## Gate 1: static balance at three heights

Status: **PROVISIONAL PASS; FINAL-ASSET RERUN REQUIRED**, 2000 steps (10 s) per height.

| Camera height | Pitch p95 | Pitch max | Height error p95 | Wheel saturation | Riser saturation |
|---:|---:|---:|---:|---:|---:|
| 0.6 m | 1.145 deg | 1.210 deg | 3.2 mm | 0% | 0% |
| 0.9 m | 1.224 deg | 1.275 deg | 13.7 mm | 0% | 0% |
| 1.8 m | 1.645 deg | 1.798 deg | 14.2 mm | 0% | 0% |

There was no termination, non-finite state, or actuator saturation.

Evidence: `evidence_20260715_riser/gate1_static_heights.json`.

## Gate 2: jerk-limited up/down motion

Status: **PROVISIONAL PASS; FINAL-ASSET RERUN REQUIRED** after one
controller-interface correction.

The first run completed safely but failed the unchanged 30 mm tracking gate at
higher speed because the implicit drive received only a position target; its
damping opposed the commanded velocity. The fix supplies position and velocity
feedforward from the analytic quintic reference. No gain or gate threshold was
relaxed.

| Requested speed | Measured peak | Pitch p95 | Pitch max | Height error p95 | Overshoot | Saturation |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 m/s | 0.111 m/s | 1.466 deg | 1.808 deg | 14.8 mm | 0 mm | 0% |
| 0.25 m/s | 0.260 m/s | 1.206 deg | 1.692 deg | 14.3 mm | 0 mm | 0% |
| 0.50 m/s | 0.509 m/s | 1.174 deg | 1.685 deg | 14.0 mm | 0 mm | 0% |
| 1.00 m/s | 0.998 m/s | 1.166 deg | 1.638 deg | 13.5 mm | 0 mm | 0% |

All four environments completed a full 0.0--1.2--0.0 m round trip. The
reference limits are 1.0 m/s, 2.0 m/s^2, and 8.0 m/s^3. There was no fall,
termination, non-finite state, wheel saturation, riser saturation, or travel
overshoot.

Evidence: `evidence_20260715_riser/gate2_dynamic.json`.

## Remaining boundary

- Add riser-specific physical camera FK and semantic DFR attitude IK without
  reintroducing arm-joint assumptions.
- Retarget and feasibility-audit corrected accepted teacher trajectories using
  base xy/yaw plus riser z.
- Build synthetic xyz tracking and then accepted-corpus/full-79 dynamic gates.
- Only after those scripted gates pass, define and train the residual DNN
  action/observation contract.
- Replace provisional mass, COM, inertia, drive force, and delay values with
  measured hardware data before sim-to-real claims.
