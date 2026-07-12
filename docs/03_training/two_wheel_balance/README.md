# Two-Wheel Balance DirectRLEnv Bring-Up

Date: 2026-07-11

Branch: `codex/two-wheel-balance-rl`

Task: `RecomoTwoWheelBalance-v0`

## Scope and status

| Capability | Status | Evidence |
| --- | --- | --- |
| Plain URDF and generated USD | Passed | Asset audit |
| Floating articulation and effort drives | Passed | Asset audit |
| Passive fall and forbidden body contact | Passed | Gate 0 logs/artifacts |
| Common/yaw effort direction | Passed after yaw-sign correction | Gate 0 logs/artifacts |
| 10D observation / 2D action contract | Passed | Four pure contract tests |
| Deterministic `32 x 2048` smoke | Passed | Two byte-equivalent metric runs |
| Scripted PD controllability | Passed | Mean upright duration 125 -> 364 steps |
| PPO learning signal at 65,536 steps | Failed; stop rule active | PPO gate metrics |
| Product stand policy | Not achieved | Blocked by failed Gate 3 |
| Velocity tracking, obstacles, arm, sim-to-real | Not started | Out of scope |

Do not resume or extend PPO from the failed checkpoint. The next task is an environment/reward/normalization diagnosis, not more timesteps.

### 2026-07-12 diagnosis and residual gate

The diagnosis is complete. The failed direct PPO policy learned the wrong stabilizing sign:

- `corr(a_common, pitch) = +0.941`
- `corr(a_common, pitch_rate) = +0.998`
- meaningful action sign agreement with the proven PD controller: `0%`
- direct PPO common action at `+10 deg`: `+0.058`, while the PD action is `-0.175`

Observation magnitudes were moderate during the failed rollout; no channel-scale explosion was found. A new opt-in `pd_residual` mode therefore preserves the direct default while composing:

```text
pd_common = clip(-pitch - 0.2 * pitch_rate, -0.5, 0.5)
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

`prior_preserved=true` and `direct_baseline_improved=true`, but `learning_signal=false`; the gate remains failed and PPO remains stopped. See `LEARNING_SIGNAL_DIAGNOSIS_20260712.md` and `evidence_20260712/`.

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
tau_left  = clamp(a_common + a_yaw) * tau_limit
tau_right = clamp(a_common - a_yaw) * tau_limit
```

This intentionally corrects the sign in the original handoff formula. The old formula produced `-0.264 rad/s` for positive yaw action; the corrected formula produces `+0.264 rad/s`.

## Validation commands

Run from `/mnt/g/wSpace/cinebotRL-two-wheel-balance` on `.98`.

```bash
# Pure contract tests without pytest.
/mnt/g/isaaclab_venv/Scripts/python.exe -X utf8 -c \
  "import sys; sys.path[:0]=[r'G:\\wSpace\\cinebotRL-two-wheel-balance',r'G:\\wSpace\\cinebotRL-two-wheel-balance\\src']; from tests import test_two_wheel_balance_contract as t; tests=[getattr(t,n) for n in dir(t) if n.startswith('test_')]; [f() for f in tests]; print(f'{len(tests)} contract tests passed')"

# Asset audit.
/mnt/g/isaaclab_venv/Scripts/python.exe -X utf8 \
  G:\\wSpace\\cinebotRL-two-wheel-balance\\scripts\\two_wheel_balance\\smoke_asset.py \
  --usd G:\\wSpace\\cinebotRL-two-wheel-balance\\assets_own\\recomoProto2_two_wheel_balance\\recomoProto2_two_wheel_balance.usd \
  --output G:\\wSpace\\cinebotRL-two-wheel-balance\\artifacts\\two_wheel_balance\\gate0\\asset_audit.json \
  --headless

# Required vector smoke. Run twice with the same seed.
/mnt/g/isaaclab_venv/Scripts/python.exe -u -X utf8 \
  G:\\wSpace\\cinebotRL-two-wheel-balance\\scripts\\two_wheel_balance\\smoke_env.py \
  --num-envs 32 --steps 2048 --zero-steps 256 \
  --action-mode zero_then_random --seed 20260711 \
  --output G:\\wSpace\\cinebotRL-two-wheel-balance\\artifacts\\two_wheel_balance\\gate1\\smoke.json \
  --headless

# Scripted controllability gate.
/mnt/g/isaaclab_venv/Scripts/python.exe -u -X utf8 \
  G:\\wSpace\\cinebotRL-two-wheel-balance\\scripts\\two_wheel_balance\\smoke_env.py \
  --num-envs 1 --steps 1000 --zero-steps 0 \
  --action-mode pd --reset-pitch-deg 2.0 --pd-kp -1.0 --pd-kd -0.2 \
  --output G:\\wSpace\\cinebotRL-two-wheel-balance\\artifacts\\two_wheel_balance\\gate2\\pd.json \
  --headless
```

The bounded PPO command is retained in `scripts/two_wheel_balance/train_short_ppo.py` for reproducibility, but must not be rerun until the failed-gate diagnosis is addressed.
Use `scripts/two_wheel_balance/evaluate_policy.py` to reevaluate an existing checkpoint without starting training.

## Evidence summary

- Asset audit: all 11 checks passed; 26.0 kg; zero wheel-drive stiffness/damping.
- Gate 1 run 1 and run 2: exact matching metrics, 267 accounted body-contact resets, zero non-finite values.
- Passive baseline from 2 degrees: approximately 125 policy steps mean upright duration.
- PD sanity: 364 policy steps mean upright duration and two contacts per 1,000 steps.
- PPO deterministic evaluation: 90 steps versus random baseline 125.16; fall rate remained 1.0; pitch p95 worsened from 13.87 to 14.41 degrees.
- PPO critic explained variance became strongly negative and reached approximately -1.07 during the bounded run.

The rendered PD evidence is `evidence_20260711/two-wheel-pd-sanity-step-0.mp4` (H.264, 1280x720, 399 frames, 7.98 seconds).

## Next diagnosis

Before another optimizer run:

1. Audit observation scaling, especially unbounded wheel position versus small-radian attitude channels.
2. Plot per-term reward around recovery and contact; verify the policy cannot improve reward by accelerating a fall.
3. Compare a normalized-observation PPO smoke against a residual policy around the proven PD controller.
4. Require critic explained variance and deterministic episode length to improve in a smaller 8,192-step gate before allowing another 65,536-step run.
5. Replace provisional mass/COM/torque values when measured hardware data is available.

The July 12 evidence narrows the next optimizer experiment further: use more than one optimizer update inside the small gate (the current `n_steps=256` with 32 environments yields only one update at 8,192 timesteps), and add a directly testable pitch-energy/progress signal or supervised PD initialization. Do not increase total timesteps first.
