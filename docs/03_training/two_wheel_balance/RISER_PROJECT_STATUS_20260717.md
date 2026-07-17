# Two-wheel riser project status - 2026-07-17

This is the current authoritative status for the two-wheel riser branch. It
separates implemented engineering, proven evidence, failed evidence, and work
that has not started. The older gate plan remains the design contract, but this
document is the operational handoff.

## 1. Current snapshot

| Item | Current state |
|---|---|
| Execution host | `yanbo@192.168.100.98:2222` |
| Worktree | `/mnt/g/wSpace/cinebotRL-two-wheel-riser` |
| Branch | `codex/two-wheel-riser-rl` |
| Latest pushed implementation | `789be2d8927cf0a4c133fee543da7bf15fa942d3` |
| CPU regression suite | 246 tests passed |
| GPU/Isaac ownership | Idle; no playback owner |
| Exact-source integrity | 79/79 |
| Kinematic Gate-B admission | 71/79 |
| Final dynamic qualification under the latest contract | 0 cases |
| Residual dataset | Not created |
| BC, PPO, or other policy training | Not started |
| Immediate block | Explicit authorization for the bounded case-74 recovery-v4 canary |

The project has a usable robot asset, scripted control stack, exact-source
trajectory pipeline, guarded evaluation infrastructure, and a defined residual
network. It does not yet have a trained or dynamically qualified learned policy.

## 2. Engineering completed

### Robot and camera contract

- Created the arm-free `recomoProto2_two_wheel_riser` form with two wheels, one
  vertical riser, the three-axis physical gimbal, and the camera tool. The old
  arm joints are absent from this asset.
- Corrected the plant assumptions to 28 kg total mass, 620 mm wheel track, and
  203.2 mm (8 inch) wheel diameter.
- Defined 0.6 to 1.8 m physical camera height from a 1.2 m riser with a
  provisional 1.0 m/s speed limit.
- Implemented camera-frame Option B: the semantic DFR attitude is converted by
  `R_world_cam = R_world_DFR * Rz(+pi/2)`, while observations and rewards use
  physical `cam_link` forward kinematics.
- Kept DJI gimbal motor angles internal to the deterministic attitude adapter.
  They are not teacher labels or learned actions.

### Scripted control and safety

- Built the balance-first controller stack: wheel LQR, path-progress control,
  jerk-limited riser control, gravity compensation, semantic gimbal adaptation,
  and balance/saturation governors.
- Added feed-forward and bounded reverse-recovery behavior without replacing
  the frozen wheel LQR.
- Added continuous-yaw handling and isolated the remaining case-74 failure from
  the earlier proxy-yaw wrapping defect.
- Added guarded GPU ownership, fail-closed evaluation, independent dynamic and
  residual-label outcomes, source/execution clock separation, evidence hashing,
  and explicit no-dataset/no-training markers.
- Added a provisional riser thermal admission monitor using actual applied
  force. The current assumptions are 292.397 N continuous force, 877.191 N peak
  force, and a provisional 30 s thermal time constant. This is an admission
  monitor, not an active force derater.

### Exact-source trajectory pipeline

- Quarantined the old teacher NPZ lineage affected by source truncation,
  resampling, pose-transpose, and physical-gimbal-index errors.
- Implemented the `exact_source_v1` contract with immutable source poses and
  timestamps, ordered anchor preservation, explicit initialization separation,
  and separate execution timing.
- Verified exact-source integrity for 79/79 episodes.
- Produced a sealed Gate-B portfolio with 71 kinematically admitted episodes.
  The eight rejects are `[6, 13, 18, 21, 22, 27, 55, 64]`. Episode 27 remains
  an honest vertical-workspace reject.
- Preserved planning admission separately from dynamic, dataset, and training
  admission. No Gate-B artifact is marked valid for training.

### Residual policy design

- Defined a 65-dimensional observation: 26 executed-state features plus three
  13-dimensional execution-clock lookahead samples at 0.25, 0.50, and 1.00 s.
- Defined a three-dimensional residual action over the scripted baseline:
  base linear-velocity delta, base yaw-rate delta, and riser-target increment.
  Wheel torque remains the output of the frozen LQR; the policy does not command
  wheel effort directly.
- Implemented `state_shared_lookahead_fusion_v1`: a 128-128 state encoder, a
  shared 64-64 horizon encoder, a 256-128 fusion trunk, and a three-output tanh
  head.
- Fixed checkpoint/report schema handling and quarantined incompatible v1
  datasets and checkpoints.

### Mechanism assumptions

- Documented the provisional vertical-axis recommendation: 400 W, 48 V servo
  with holding brake, approximately 3:1 reduction, and 70 mm travel per output
  revolution, subject to measured payload, inertia, duty cycle, and stopping
  requirements.
- Kept mass, COM, yaw/pitch inertia, friction, wheel torque, command delay, and
  riser thermal parameters explicitly provisional for later hardware update.

## 3. Evidence and current failure

The sealed exact-source portfolio is:

`artifacts/two_wheel_riser/20260717_exact_source_all79_portfolio_v4_threshold71`

- manifest SHA-256:
  `851a7b2751cd397ba35daf57d1a8c6971fb14ed0186683af48d3c6109090570a`
- summary SHA-256:
  `688b5bc23d801705c3132c511f009e1deb3d2af0a16a2a3ae33467764272db83`
- result: 79/79 integrity, 71/79 kinematic admission, no dynamic or training
  admission.

The latest completed dynamic evidence is the corrected-yaw case-74 canary:

`artifacts/two_wheel_riser/20260717_gate_c_case74_continuous_yaw_fix_v4_exclusive`

- case JSON SHA-256:
  `f5686d491cc3dff069a58f54fb974e718057993261a1924976e907b737fff65d`
- summary SHA-256:
  `3deb477ab7ee45cca9aafaac801b05ce4935523460d32f3cfd9e6b94cb37535f`
- source duration: 11.373883 s
- retimed execution duration: 188.546638 s
- completed phase: 159.901892 s
- position p95/max: 1.065665/1.094696 m
- attitude max: 0.458524 deg
- pitch max: 7.645328 deg
- proxy servo error max: 0.500490 deg
- result: dynamic quality failed because reverse-recovery/path progress became
  unstable; no residual labels or training data were produced.

The primary failure window is approximately wall time 89 to 166 s, corresponding
to plan time 52.1 to 59.8 s. The current recovery-v4 change is a structural
candidate for this failure, but it has not been run in Isaac and must not be
described as a fix.

## 4. Work not achieved

- The recovery-v4 case-74 GPU canary has not run.
- Episode 77 has not run under the latest dynamic contract.
- The accepted 71-case portfolio has not passed dynamic Gate C.
- No v2 residual dataset has been captured.
- No BC model, holdout evaluation, learned all-79 evaluation, or PPO run exists
  for this corrected riser pipeline.
- No learned policy has demonstrated improved trajectory tracking.
- The thermal model is not calibrated and does not actively derate the riser.
- Physical COM, inertias, friction, actuator strength, and control delay remain
  simulation assumptions rather than measured values.
- Obstacle avoidance is intentionally deferred until balance and unobstructed
  exact-source tracking pass their gates.

Historical cases 1 and 52 produced provisional metric passes before the final
evidence-contract corrections. They remain useful diagnostics, but they are not
counted as final dynamic qualifications under the latest contract.

## 5. Required continuation sequence

No GPU work starts without explicit case-74 authorization. The guarded command
is:

```bash
RISER_CASE74_GPU_AUTHORIZATION=AUTHORIZED_CASE74_RECOVERY_V4 \
  bash scripts/two_wheel_balance/run_riser_case74_recovery_canary.sh
```

Then proceed only in this order:

1. Run case 74 alone in the fresh recovery-v4 namespace and audit all safety,
   tracking, thermal, clock, ownership, hash, and no-dataset fields.
2. Stop on any dynamic reject. Do not tune by relaxing thresholds.
3. If case 74 passes, run the bounded case-77 canary under the same contract.
4. If both pass, dynamically qualify the sealed 71-case portfolio with the
   existing fail-fast rules.
5. Only after dynamic qualification, capture a new residual dataset from the
   exact-source lineage and recompute the raw residual envelope.
6. Train BC only after dataset schema, integrity, holdout split, and action
   envelope admission pass.
7. Evaluate BC on held-out and all-79 references before considering PPO.
8. Keep PPO closed until the scripted baseline, dataset, BC, and holdout gates
   are all green.

## 6. Correct status language

Safe claims:

- "Exact-source integrity is 79/79 and kinematic admission is 71/79."
- "The controller, evaluation gates, and residual network are implemented."
- "Case 74 still fails dynamic tracking; recovery-v4 awaits a bounded canary."
- "No corrected residual dataset or trained policy exists yet."

Incorrect claims:

- "The robot can track all 79 trajectories."
- "The 71 accepted plans have passed Isaac dynamics."
- "BC or PPO has trained a usable riser policy."
- "Recovery-v4 has fixed case 74."
