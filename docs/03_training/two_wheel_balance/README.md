# Two-Wheel Balance DirectRLEnv Bring-Up

Date: 2026-07-14

Branch: `codex/two-wheel-balance-rl`

Task: `RecomoTwoWheelBalance-v0`

## Scope and status

| Capability | Status | Evidence |
| --- | --- | --- |
| Plain URDF and generated USD | Passed | Asset audit |
| Floating articulation and effort drives | Passed | Asset audit |
| Passive fall and forbidden body contact | Passed | Gate 0 logs/artifacts |
| Common/yaw effort direction | Passed after 8-inch wheel-axis correction | 2026-07-12 8-inch evidence |
| 10D observation / 2D action contract | Passed | 22 pure contract tests |
| Deterministic `32 x 2048` smoke | Passed | Two byte-equivalent metric runs |
| Corrected-plant scripted PD controllability | Passed | Mean upright duration 113 -> 409 steps |
| Fresh 28 kg inner-LQR recovery | Passed | 6/6 signed `+/-2/5/8 deg` starts recovered |
| Upright-start LQR push recovery | Passed | 32/32 recovered from up to 6 N s at 0.5 m application height |
| Cascaded low-speed `vx/wz` tracking | Passed | 32/32 survived signed commands and reversals |
| Corrected 28 kg combined tracking plus push | Passed | 36/36 passed; exact seeded repeat |
| Provisional guessed-plant signed safety gate | Passed for balance | 112/112 survive and recover balance; 91/112 strict tracking |
| USD visual-reference composition | Passed after empty-visual repair | Asset audit and warning-free stage open |
| Robust fixed-gain tracking closure | Rejected; defaults unchanged | Best sweep result missed safety/95% acceptance |
| Deterministic plant-uncertainty smoke | Failed | 16/16 survived, but only 7/16 met all tracking limits |
| URDF-to-PhysX mass contract | Passed | 28.000 kg authored and 27.999998 kg resolved |
| PPO learning signal at 65,536 steps | Failed; stop rule active | PPO gate metrics |
| Learned product policy | Not achieved | PPO stop rule remains active |
| Obstacles, arm, sim-to-real | Not started | After deterministic robustness |

Do not resume or extend PPO from the failed checkpoint. The corrected 28 kg nominal controller is the scripted baseline. Combined command-plus-disturbance robustness has passed; plant uncertainty remains a deterministic controller/system-identification task, not a request for more PPO timesteps.

### 2026-07-12 geometry correction

The confirmed wheel diameter is 8 inches (`0.2032 m`), not 6 inches. The source URDF and generated USD now use:

```text
left wheel center:   [0, +0.310, 0.1016] m
right wheel center:  [0, -0.310, 0.1016] m
wheel radius:        0.1016 m
wheel track:         0.620 m
wheel axes:          [0, +1, 0]
```

The geometry audit also exposed that the previous `[0,-1,0]` wheel axes contradicted the claimed `+X` forward direction. Both axes are now `+Y`, and the original differential mixer is restored. Dynamic tests prove positive common action gives `+vx` and positive yaw action gives `+wz`.

All PPO and PD evidence in `evidence_20260711/` and `evidence_20260712/` predates this physical-model correction. It remains historical diagnostic evidence and must not be used as the baseline for the corrected model. No PPO has been run on the corrected 8-inch plant.

### 2026-07-14 mass and outer-loop correction

The current robot is approximately `28 kg`, not `40 kg`. The model now authors exactly `28.0 kg`: `24.996 kg` on the aggregate `base_link`, `1.5 kg` on each wheel, and explicit `1 g` inertials on each of the four sensor/reference frames. The frame inertials prevent PhysX from silently assigning its `1 kg` fallback. Static authored mass and live resolved mass now agree within floating-point tolerance.

The prior accepted gates actually ran on an unintended `30 kg` plant and are retained only as historical evidence. The old inner gain survived but failed `0/6` corrected signed-pitch recovery cases with `24.67%` saturation. A fresh 28 kg identification selected scale `0.6`; it recovers `6/6` signed `+/-2/5/8 deg` starts with a worst time of `0.825 s` and zero saturation. The provisional defaults use `vx_ki=0.05`, `wz_ki=0.10`, conditional anti-windup, and latched two-phase startup equilibrium-pitch estimation. Nominal tracking-plus-push remains 36/36; the guessed-plant signed gate is complete for balance safety but only 91/112 for strict tracking.

See `LQR_28KG_MODEL_AND_OUTER_LOOP_GATE_20260714.md` for the mass allocation, corrected metrics, uncertainty boundary, and stop rule.
See `PROVISIONAL_PLANT_PRIOR_AND_ROBUST_CONTROL_20260714.md` and `PLANT_PRIOR_PROVISIONAL_V1.json` for guessed parameters and their replacement procedure.
See `ROBUST_TRACKING_CLOSURE_DIAGNOSIS_20260714.md` for the repaired USD composition, rejected gain sweep, and next structural controller boundary.

### Historical pre-correction diagnosis and residual gate

Before the wheel-axis correction, the failed direct PPO policy appeared to learn the wrong stabilizing sign:

- `corr(a_common, pitch) = +0.941`
- `corr(a_common, pitch_rate) = +0.998`
- meaningful action sign agreement with the proven PD controller: `0%`
- direct PPO common action at `+10 deg`: `+0.058`, while the PD action is `-0.175`

Those results were generated against the old `[0,-1,0]` wheel convention. The opt-in residual mode has been updated for the corrected plant and now composes:

```text
pd_common = clip(+pitch + 0.2 * pitch_rate, -0.5, 0.5)
applied_common = clip(pd_common + 0.15 * policy_residual_common)
applied_yaw = 0.15 * policy_residual_yaw
```

Zero residual exactly reproduces the 364-step PD baseline. The bounded 8,192-step residual PPO gate preserved the prior and remained much safer than direct PPO, but it did not learn an improvement:

| Evaluation | Mean episode length | Pitch p95 | Fall rate |
| --- | ---: | ---: | ---: |
| Frozen direct random baseline | 125.16 | 13.87 deg | 1.0 |
| Residual random baseline | 362.48 | 12.74 deg | 1.0 |
| Zero-residual PD prior | 364.00 | 12.83 deg | 1.0 |
| Learned residual after 8,192 steps | 353.00 | 12.94 deg | 1.0 |

The table above is retained only for provenance. See the supersession warning in `LEARNING_SIGNAL_DIAGNOSIS_20260712.md`.

## Runtime contract

Physics runs at 1,000 Hz with decimation 5, giving a 200 Hz policy rate.

Actor observation order:

```text
pitch, pitch_rate,
mean_wheel_position, mean_wheel_velocity, wheel_velocity_difference,
yaw_rate, vx_ref, wz_ref,
previous_a_common, previous_a_yaw
```

The actor receives no world-frame linear-velocity truth. Simulation truth `vx` is used only for reward and metrics.

Action order:

```text
a_common, a_yaw
```

The live direction audit showed that positive `a_yaw` must produce positive body `+Z` yaw. With both positive wheel velocities driving `+X`, the correct mixer is:

```text
tau_left  = clamp(a_common - a_yaw) * tau_limit
tau_right = clamp(a_common + a_yaw) * tau_limit
```

This matches the original handoff formula after correcting both wheel joint axes to `+Y`. On the 8-inch plant, a `+0.1` yaw action produces approximately `+0.296 rad/s`.

## Validation commands

Run from `/mnt/g/wSpace/cinebotRL-two-wheel-balance` on `.98`.

```bash
# Pure contract tests without pytest.
/mnt/g/isaaclab_venv/Scripts/python.exe -X utf8 -c \
  "import sys; sys.path[:0]=[r'G:\\wSpace\\cinebotRL-two-wheel-balance',r'G:\\wSpace\\cinebotRL-two-wheel-balance\\src']; from tests import test_two_wheel_balance_contract as t; tests=[getattr(t,n) for n in dir(t) if n.startswith('test_')]; [f() for f in tests]; print(f'{len(tests)} contract tests passed')"

# Regenerate the two-wheel USD. The effort-drive flag is mandatory.
/mnt/g/isaaclab_venv/Scripts/python.exe -u -X utf8 \
  G:\\wSpace\\cinebotRL-two-wheel-balance\\scripts\\convert_urdf_to_usd.py \
  --urdf assets_own/recomoProto2_two_wheel_balance/recomoProto2_two_wheel_balance.urdf \
  --usd assets_own/recomoProto2_two_wheel_balance/recomoProto2_two_wheel_balance.usd \
  --mesh-scale 1.0 --default-drive-type none --headless

# Asset audit.
/mnt/g/isaaclab_venv/Scripts/python.exe -X utf8 \
  G:\\wSpace\\cinebotRL-two-wheel-balance\\scripts\\two_wheel_balance\\smoke_asset.py \
  --urdf G:\\wSpace\\cinebotRL-two-wheel-balance\\assets_own\\recomoProto2_two_wheel_balance\\recomoProto2_two_wheel_balance.urdf \
  --usd G:\\wSpace\\cinebotRL-two-wheel-balance\\assets_own\\recomoProto2_two_wheel_balance\\recomoProto2_two_wheel_balance.usd \
  --output G:\\wSpace\\cinebotRL-two-wheel-balance\\artifacts\\two_wheel_balance\\gate0\\asset_audit.json \
  --headless

# Required vector smoke. Run twice with the same seed.
/mnt/g/isaaclab_venv/Scripts/python.exe -u -X utf8 \
  G:\\wSpace\\cinebotRL-two-wheel-balance\\scripts\\two_wheel_balance\\smoke_env.py \
  --num-envs 32 --steps 2048 --zero-steps 256 \
  --action-mode zero_then_random --seed 20260712 \
  --output G:\\wSpace\\cinebotRL-two-wheel-balance\\artifacts\\two_wheel_balance\\gate1\\smoke.json \
  --headless

# Scripted controllability gate.
/mnt/g/isaaclab_venv/Scripts/python.exe -u -X utf8 \
  G:\\wSpace\\cinebotRL-two-wheel-balance\\scripts\\two_wheel_balance\\smoke_env.py \
  --num-envs 1 --steps 1000 --zero-steps 0 \
  --action-mode pd --reset-pitch-deg 2.0 --pd-kp 1.0 --pd-kd 0.2 \
  --output G:\\wSpace\\cinebotRL-two-wheel-balance\\artifacts\\two_wheel_balance\\gate2\\pd.json \
  --headless

# Nominal LQR identification and recovery gate. This does not train a policy.
/mnt/g/isaaclab_venv/Scripts/python.exe -u -X utf8 \
  G:\\wSpace\\cinebotRL-two-wheel-balance\\scripts\\two_wheel_balance\\tune_lqr.py \
  --num-envs 32 --horizon-steps 2000 --control-interval-steps 4 \
  --gain-scales 0.4,0.5,0.6 \
  --output-dir G:\\wSpace\\cinebotRL-two-wheel-balance\\artifacts\\two_wheel_balance\\28kg_lqr_tune \
  --headless

# Deterministic signed initial-pitch recovery gate. This does not train a policy.
/mnt/g/isaaclab_venv/Scripts/python.exe -u -X utf8 \
  G:\\wSpace\\cinebotRL-two-wheel-balance\\scripts\\two_wheel_balance\\evaluate_lqr_push.py \
  --gains G:\\wSpace\\cinebotRL-two-wheel-balance\\docs\\03_training\\two_wheel_balance\\evidence_20260714_28kg\\lqr_gains.json \
  --num-envs 6 --initial-pitch-deg=-8,-5,-2,2,5,8 --push-forces-n=0 \
  --output G:\\wSpace\\cinebotRL-two-wheel-balance\\artifacts\\two_wheel_balance\\28kg_inner_recovery\\gate.json \
  --headless

# Cascaded signed vx/wz command and reversal gate. This does not train a policy.
/mnt/g/isaaclab_venv/Scripts/python.exe -u -X utf8 \
  G:\\wSpace\\cinebotRL-two-wheel-balance\\scripts\\two_wheel_balance\\evaluate_lqr_tracking.py \
  --gains G:\\wSpace\\cinebotRL-two-wheel-balance\\docs\\03_training\\two_wheel_balance\\evidence_20260714_28kg\\lqr_gains.json \
  --output G:\\wSpace\\cinebotRL-two-wheel-balance\\artifacts\\two_wheel_balance\\lqr_tracking\\tracking_gate.json \
  --headless

# Combined signed vx/wz tracking plus 2/4/6 N s push gate. No policy is trained.
/mnt/g/isaaclab_venv/Scripts/python.exe -u -X utf8 \
  G:\\wSpace\\cinebotRL-two-wheel-balance\\scripts\\two_wheel_balance\\evaluate_lqr_tracking_push.py \
  --gains G:\\wSpace\\cinebotRL-two-wheel-balance\\docs\\03_training\\two_wheel_balance\\evidence_20260714_28kg\\lqr_gains.json \
  --num-envs 36 --push-forces-n=-60,-40,-20,20,40,60 \
  --output G:\\wSpace\\cinebotRL-two-wheel-balance\\artifacts\\two_wheel_balance\\lqr_tracking_push\\combined_gate.json \
  --headless
```

The bounded PPO command is retained in `scripts/two_wheel_balance/train_short_ppo.py` for reproducibility, but must not be rerun until the failed-gate diagnosis is addressed.
Use `scripts/two_wheel_balance/evaluate_policy.py` to reevaluate an existing checkpoint without starting training.

## Evidence summary

- Corrected asset audit: 0.620 m track, 0.2032 m diameter, `+Y` wheel axes, seven explicit inertials totaling `28.0000003 kg`, and zero wheel-drive stiffness/damping.
- Runtime mass audit: PhysX resolves the same seven bodies to `27.9999981 kg`; no fallback mass remains.
- Current Gate 1 run 1 and run 2: byte-identical metrics, 295 accounted body-contact resets, zero non-finite values.
- Corrected passive baseline from 2 degrees: 113 policy steps mean upright duration.
- Corrected PD sanity: 409 policy steps mean upright duration and two contacts per 1,000 steps.
- Fresh 28 kg LQR scale `0.6`: nominal tuner pitch p95 `1.690 deg`, pitch max `7.991 deg`, and zero saturation.
- Corrected signed-pitch recovery: 6/6 `+/-2/5/8 deg` starts recovered; worst recovery was `0.825 s` and saturation was zero. The rejected old gain recovered 0/6 with `24.67%` saturation.
- Original corrected 28 kg combined gate: 36/36 scenarios passed with `vx_ki=wz_ki=0.05`; this remains model/inner-LQR provenance and is superseded as the outer-loop default.
- The corrected combined-gate repeat was exactly equal on all aggregate and per-scenario metrics. The rendered `-6 N s` case survived without termination while tracking `vx=+0.2 m/s`, `wz=-0.4 rad/s`.
- Provisional controller nominal gate: 36/36 passed with an exact repeat; peak pitch was `10.91 deg`, `vx/wz` RMSE was `0.0819/0.0520`, and saturation was `0.213%`.
- Full signed guessed-plant gate at `+/-2 N s`: 112/112 survived and recovered balance, 91/112 met strict tracking, peak pitch was `11.47 deg`, and saturation was zero.
- Broad 6 N s stress diagnostic: 16/16 survived and recovered balance while 11/16 met strict tracking; peak pitch was `10.28 deg` and saturation was zero.
- The default-disabled path-progress governor activated in 12/112 bias-reinforcing provisional cases and improved strict admitted tracking from 91/112 to 99/112 while preserving 112/112 balance recovery and a 0.759 progress floor. It missed the 106/112 promotion target and is rejected; the governor-off 36-case regression remained bit-for-bit equal.
- Structural diagnosis disproved hidden wheel mixing as the failure cause: baseline wheel saturation and common/yaw authority loss were zero.
- The explicit `structural_robust_v1` profile passed 111/112 strict provisional cases with 112/112 balance recovery, 9.95 degree peak pitch, zero wheel-mixer saturation, and a 0.752 progress floor.
- The same profile passed 36/36 nominal cases with no governor activation and improved post `vx` RMSE from 0.0819 to 0.0678 m/s. It remains opt-in; conservative defaults are unchanged.
- The broad diagnostic stress boundary remains 11/16 strict, though eventual tracking recovery improved from 12/16 to 15/16. This is not a hardware-readiness claim.
- The Stage 0 whole-body asset now exposes only two wheel and three physical arm DOFs; DJI gimbal and MoveIt virtual joints are fixed while `cam_link` remains the physical observation frame.
- Static `home_v0` balance passed 1x2,000 and 16x2,000-step gates. The 16-environment gate had 3.260 degree peak pitch, 0.236 degree peak arm error, and no reset or non-finite state.
- No PPO checkpoint is valid for the corrected 8-inch/`+Y`-axis model.

The rendered file under `evidence_20260711/` shows the obsolete 6-inch/pre-axis-fix asset and is retained only for provenance.

## Next gate

Before another optimizer run or hardware claim:

1. Measure or tightly bound aggregate COM, yaw inertia, available wheel torque, tire friction, and control delay.
2. Replace broad assumed uncertainty ranges with credible hardware ranges.
3. Add a low-amplitude no-obstacle `cam_link` tracking gate using conventional three-joint arm IK without weakening the balance safety layer.
4. Replace guessed plant values with measurements before tightening tracking claims; add slopes, sensor noise, and command-rate limits only after reacceptance.
5. Keep the repaired empty-visual USD composition check in every regenerated-asset audit.
6. Pass no-obstacle whole-body tracking before obstacle avoidance, and only then evaluate whether a bounded residual policy adds value over the frozen controller.

See `WHOLE_BODY_STAGE0_GATE_20260714.md`, `STRUCTURAL_TRACKING_DIAGNOSIS_20260714.md`, and `evidence_20260714_structural_tracking/summary.json` for the accepted boundaries. Earlier LQR documents remain provenance. The gains remain simulation-only because COM, inertia, friction, actuator torque, and delay are guessed rather than measured.
