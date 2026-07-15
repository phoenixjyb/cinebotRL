# Two-Wheel All-79 Position-Only Dynamic Gate (2026-07-15)

## Decision

The frozen Stage-0 controller passes all 79 no-obstacle references dynamically
with the 28 kg, 620 mm wheel-track, 8-inch-wheel model. This is a scripted
position-tracking baseline, not a learned policy and not a full camera-pose
gate. PPO remains blocked.

## Result

- Cases: `79/79`
- Completed reference time: `1401.424 s`
- Worst pitch: `11.652 deg` in case 77, limit `12 deg`
- Worst arm servo error: `5.608 deg` in case 6, limit `10 deg`
- Worst arm effort: `27.480 Nm` in case 34, hard limit `30 Nm`
- Worst tool p95: `0.1476 m` in case 57, limit `0.15 m`
- Worst tool maximum: `0.1970 m` in case 57, limit `0.25 m`
- Worst wheel-action saturation ratio: `0.000683` in case 34, limit `0.20`
- Worst arm-effort saturation ratio: `0.0`, limit `0.20`
- Terminations: `0`

The generated acquisition prefix is retimed only for cases 6, 7, 9, and 62.
Case 6 was selected by the kinematic bound search; cases 7, 9, and 62 were
promoted after isolated dynamic recovery. Semantic inter-sample timing and all
dynamic thresholds remain unchanged.

## Proven Boundary

The controller tracks world position of semantic `ee1_tool` using the
differential-drive chassis and three physical arm joints. The complete physical
camera/gimbal subtree contributes mass, COM, inertia, and gravity load, but the
three physical gimbal joints and four virtual MoveIt frame joints are fixed in
this Stage-0 asset. Camera attitude, physical `cam_link` orientation, and DJI
attitude adaptation are not part of this gate.

The next whole-body profile must retain this evidence unchanged and add a
separate attitude contract:

```text
position target: semantic ee1_tool world position
attitude target: semantic DFR quaternion or optical axis
sim adapter:     R_world_cam_des = R_world_DFR * Rz(+pi/2)
hardware:        DJI firmware receives attitude, not motor-joint labels
```

## Evidence

The unabridged result remains at
`evaluation_results/two_wheel_all79_playback/all79_promoted_defaults_v2` on
`.98`. Its `summary.json` SHA-256 is
`1bbe5b8170bb1443c7047eafd1e9b62d73dc1f5bf5275b2f34b17e4c950570db`.

The case-63 visual smoke records the same Stage-0 contract with a blue reference
path and red current target. Its metadata explicitly reports
`camera_attitude_tracking_enabled=false` and
`physical_gimbal_control_enabled=false`.
