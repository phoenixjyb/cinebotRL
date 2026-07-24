# Two-wheel riser end-goal completion audit v4

This CPU-only checkpoint preserves the full 6-of-10 goal assessment and adds
an explicit learned control-ownership contract to the future validation,
holdout, all-79, and rendered-rollout gates.

For every model-based learned rollout, the evidence must now prove:

- learned outputs are only normalized residual velocity, yaw-rate, and riser
  target commands;
- the frozen cascaded LQR owns wheel effort;
- the deterministic semantic attitude adapter owns gimbal attitude;
- deterministic supervisors own riser hard limits and runtime safety gates;
- learned actions never directly command wheel effort or physical gimbal
  joints.

The macOS and `.98` reports are byte-identical. No conversion, runtime,
capture, BC, PPO, or training operation was started.

