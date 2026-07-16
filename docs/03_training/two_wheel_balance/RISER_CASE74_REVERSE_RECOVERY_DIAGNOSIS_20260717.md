# Case 74 reverse-recovery and proxy-yaw diagnosis

## Scope

This is a CPU-only diagnosis of the sealed final Gate C reject in
`20260717_gate_c_case74_77_v3_exclusive_timing_resealed`. It does not rerun
Isaac, change the controller, create residual labels, or authorize training.

Authoritative inputs:

- Gate C case JSON SHA-256:
  `9bec49cf68d37d100b800e6505f5d0e5b6df2d1af30cd5f4e89bbe10d7794eb4`
- Final status SHA-256:
  `b6bbd2dc25783ddff8364bafea1a23b06555d7f2dfe089095dfad29304cde4ee`
- Gate B portfolio manifest SHA-256:
  `851a7b2751cd397ba35daf57d1a8c6971fb14ed0186683af48d3c6109090570a`
- CPU reverse-recovery audit SHA-256:
  `0a8e611a567640451103a80b05de955601718f5c63aa19d0e21fe9ab475df60a`
  (`CPU_REVERSE_RECOVERY_AUDIT.json` in the final case-74 namespace)

## Result

Case 74 contains valid planned reverse motion. The 1 Hz sealed trace shows
bounded reverse feed-forward and corrective chassis commands before the first
proxy-yaw branch fault. At `58 s`, the semantic yaw target and PhysX state are
orientation-equivalent but differ by almost `720 deg` under raw subtraction.
Only after that event does the chassis enter full-scale bidirectional recovery
and then fail the base-position and contact gates.

The hash-bound audit counted `54` reverse-motion samples before the fault.
None saturated the `0.4 m/s` outer-loop limit; maximum absolute command was
`0.233591 m/s`, maximum base error was `0.144051 m`, and maximum camera
position error was `0.182615 m`. After the fault, `7/11` sampled commands
saturated, the command changed direction, base error reached `1.361215 m`, and
camera position error reached `1.615056 m` before the final contact.

This ordering supports the following classification:

1. The continuous-yaw branch mismatch is the primary observed precursor.
2. The later `+0.4/-0.4 m/s` recovery is a downstream response to the injected
   physical disturbance and rapidly changing position error.
3. The old trace does not justify changing reverse-tracking gains, clipping
   commands, adding hysteresis, or relaxing any gate before the corrected-yaw
   case-74 canary is executed.

The trace is sampled at 1 Hz, so it is suitable for event ordering but not for
controller identification or stability-margin estimation. A corrected dynamic
rerun remains necessary to prove causality. If corrected case 74 still exhibits
recovery oscillation, the next CPU/runtime audit must capture full-rate values
for along/cross/yaw error, unsaturated and saturated `vx/wz`, actual `vx/wz`,
LQR action, pitch bias, and governor state before proposing a bounded recovery
law change.

## Stop rules

- Keep riser GPU/Isaac work stopped until explicitly authorized.
- Do not launch case 77 or the accepted-71 batch.
- Do not start residual capture, BC, PPO, or any learned-action path.
- Preserve source anchors, both clocks, dynamic gates, and frozen action scales.
- Do not describe this CPU diagnosis as dynamic validation.
