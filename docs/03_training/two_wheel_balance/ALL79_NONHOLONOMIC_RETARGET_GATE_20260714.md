# Two-Wheel All-79 Nonholonomic Retarget Gate (2026-07-14)

## Decision

The corrected all-79 position references pass the deterministic kinematic
retarget gate for the 28 kg, 620 mm wheel-track, 8-inch-wheel whole-body model.
This authorizes the next bounded step: scripted Isaac playback with the balance
controller active. It does not authorize PPO or teacher ingestion.

## Inputs and contract

- Full references: `trajectoryToLearn/stage_gik_no_obstacle79_nominal`
- Corrected sparse contract: `gik_no_obstacle_monorepo_ee1_moveit_urdf_v3_20260713`
- Position target: world-frame `ee1_tool`
- Base action: differential-drive `v, wz`; lateral body action is always zero
- Arm action: three physical arm-joint deltas
- Gimbal: excluded from this gate
- Attitude: excluded from this gate; case 34 remains quarantined by the attitude audit

The source acquisition prefix was generated for a different initial robot pose.
For every case it is replaced by a quintic Cartesian transition from the Stage 0
home pose to the audited first semantic sample. Semantic samples are not shifted.
If the generated acquisition exceeds a bound, only that prefix may be stretched;
semantic inter-sample timing remains unchanged.

## Results

- Cases passed: 79/79
- Full source duration: 1394.645 s
- Cases requiring acquisition retiming: 6 and 9
- Retiming used: 1.25x for the generated acquisition prefix only
- Maximum whole-path position error: 0.07674 m (case 23)
- Maximum semantic-path position error: 0.06013 m (case 57)
- Maximum linear velocity: 0.38981 m/s (bound 0.4 m/s)
- Maximum yaw rate: 0.4 rad/s (bound 0.4 rad/s)
- Maximum arm rate: 0.5 rad/s (bound 0.5 rad/s)
- Training started: false

Evidence:

- `evidence_20260714_all79_reference_contract/summary.json`
- `evidence_20260714_all79_reference_contract/cases.csv`
- `evidence_20260714_all79_nonholonomic_retarget/summary.json`
- `evidence_20260714_all79_nonholonomic_retarget/cases.csv`

## Proven boundary

This gate proves that all 79 position paths can be represented by bounded
unicycle-plus-arm commands under the URDF position kinematics and the stated
error gates. It also proves that direct imitation of the old holonomic teacher
actions is invalid: 62 cases contain source lateral body velocity above 0.02 m/s.

This gate does not prove dynamic tracking in Isaac, simultaneous self-balance,
wheel torque feasibility, collision clearance, obstacle avoidance, camera
attitude tracking, DJI physical-gimbal adaptation, or learned-policy quality.

## Next gate

1. Replay cases 1, 20, 28, 50, and 79 in Isaac with the balance controller active.
2. Stop on excessive pitch, wheel/arm saturation, contact faults, or divergence.
3. Expand deterministic playback to all 79 only after the representative gate passes.
4. Add Option-B physical `cam_link` attitude tracking after the position/dynamics gate.
5. Keep PPO paused until these scripted gates pass and observations/actions are frozen.
