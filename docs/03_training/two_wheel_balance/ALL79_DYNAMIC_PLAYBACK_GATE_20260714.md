# Two-Wheel All-79 Dynamic Playback Gate (2026-07-14)

## Decision

The first representative dynamic playback does not pass. PPO and all-79 batch
playback remain blocked. The robot stays balanced, but open-loop arm and chassis
references do not maintain physical `ee1_tool` tracking in Isaac.

## Case 1 result

- Full horizon completed: 25.124 s / 5,026 simulation steps
- Termination: none
- Peak pitch: 3.833 deg (pass, limit 12 deg)
- Wheel action saturation: 0.0 (pass, limit 0.20 ratio)
- Peak arm tracking error: 21.899 deg (fail, limit 10 deg)
- Tool position p95 error: 0.528 m (fail, limit 0.15 m)
- Tool position maximum error: 0.623 m (fail, limit 0.25 m)
- Chassis `vx` RMSE: 0.113 m/s
- Chassis `wz` RMSE: 0.025 rad/s
- Training started: false

Evidence: `evidence_20260714_all79_dynamic_playback/case1_smoke.json`

## Diagnosis

The kinematic retarget assumes that the base follows its planned pose exactly and
therefore emits an open-loop arm-joint path. In dynamics, the balance controller
moves the base to reject arm/COM disturbances, while arm gravity and actuator
limits introduce arm lag. Replaying the planned arm joints cannot correct either
error, so the physical tool diverges even though the robot remains upright.

This is not evidence that PPO should replace the controller. It is evidence that
the scripted teacher/control stack is missing closed-loop task-space feedback.

## Required next controller

1. Measure actual base world pose and physical `ee1_tool` pose from Isaac.
2. Add bounded base-pose feedback around the retargeted `v, wz` feed-forward path.
3. Use conventional damped-least-squares IK on the three physical arm joints to
   reduce world-frame tool-position error from the measured state.
4. Preserve the balance LQR as the wheel-effort inner loop.
5. Enforce arm position/rate/effort, wheel action, pitch, contact, and tracking
   stop rules at every step.
6. Re-run case 1 before cases 20, 28, 50, and 79.
7. Add camera attitude and the DJI gimbal adapter only after position dynamics pass.

RL can later learn bounded residuals or arbitration on top of this controller.
It should not be asked to compensate for a missing IK/pose-feedback layer.
