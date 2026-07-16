# Corrected all-79 riser recovery evidence

This directory records the deterministic recovery milestone for the arm-free
two-wheel riser robot. It does not authorize PPO or claim a learned policy.

## Admission contract

- The source package is the corrected all-79 v3 semantic teacher export.
- Source actions and physical DJI gimbal-joint labels are not used.
- The unexported MATLAB initialization attitude sample is not used as a label.
- The commanded gimbal target is semantic world-DFR attitude. Isaac reward and
  observation continue to use physical `cam_link` FK through Option B.
- A home-to-first-target acquisition segment is regenerated from the riser
  robot's physical home pose rather than copied from the removed arm robot.

## Recovery history

The first direct full-stage attempt passed `0/79`: every case violated the
new robot's acquisition-rate contract. Regenerating the acquisition segment
from the riser home pose raised this to `72/79`. The seven remaining families
were repaired without changing any public gate or actuator limit:

- Whole-trajectory timing: case 18 `1.5x`, case 79 `1.5x`.
- Acquisition-only timing: case 23 `6.5x`, case 24 `4.5x`, case 41 `1.3x`,
  case 72 `1.5x`.
- Geometry: case 50 applies its low-height shift before acquisition generation,
  avoiding the previous double shift.

The final stage contains `79` accepted and `0` rejected cases. Source duration
is `1394.644931 s`; executable duration is `1435.965017 s`.

## Accepted gates

- Pure full-duration audit: `79/79` passed. Strategy mix is 30 fixed-path and
  49 joint-adaptive plans. Worst position p95 is `0.149015 m`, worst position
  maximum is `0.152459 m`, and worst proxy rate is `0.413218 rad/s`.
- Isaac repaired-family replay: cases 18, 23, 24, 41, 50, 72, and 79 passed
  `7/7`, totalling 39,430 simulation steps with no termination or attitude-IK
  failure.
- Dynamic worst values: position p95 `0.131495 m`, position maximum
  `0.144621 m`, attitude p95 `0.182725 deg`, attitude maximum `0.225993 deg`,
  pitch maximum `5.968319 deg`, riser-servo p95 `0.011064 m`, and internal
  proxy rate `62.243368 deg/s`.
- Action and riser saturation ratios were zero. The worst proxy saturation
  ratio was the accepted `0.000762` in case 72.

## Files

- `stage_summary.json`: strict source admission, provenance, and retiming.
- `pure_all79_summary.json`: aggregate pure reference gate.
- `pure_all79_cases.csv`: per-case pure reference metrics.
- `repaired7_playback_manifest.json`: hashes for self-contained replay plans.
- `repaired7_isaac.json`: complete Isaac results for repaired families.

The next gate is a versioned residual observation/action dataset contract with
case-level train/validation/holdout separation. PPO remains blocked until that
dataset passes leakage, finite-value, dimensional, and deterministic-baseline
regression checks.
