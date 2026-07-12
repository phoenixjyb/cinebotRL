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
| Common/yaw effort direction | Passed after 8-inch wheel-axis correction | 2026-07-12 8-inch evidence |
| 10D observation / 2D action contract | Passed | Six pure contract tests |
| Deterministic `32 x 2048` smoke | Passed | Two byte-equivalent metric runs |
| Scripted PD controllability | Passed | Mean upright duration 125 -> 364 steps |
| PPO learning signal at 65,536 steps | Failed; stop rule active | PPO gate metrics |
| Product stand policy | Not achieved | Blocked by failed Gate 3 |
| Velocity tracking, obstacles, arm, sim-to-real | Not started | Out of scope |

Do not resume or extend PPO from the failed checkpoint. The next task is an environment/reward/normalization diagnosis, not more timesteps.

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
```

The bounded PPO command is retained in `scripts/two_wheel_balance/train_short_ppo.py` for reproducibility, but must not be rerun until the failed-gate diagnosis is addressed.
Use `scripts/two_wheel_balance/evaluate_policy.py` to reevaluate an existing checkpoint without starting training.

## Evidence summary

- Current asset audit: 0.620 m track, 0.2032 m diameter, `+Y` wheel axes, 26.0 kg, and zero wheel-drive stiffness/damping.
- Current Gate 1 run 1 and run 2: byte-identical metrics, 295 accounted body-contact resets, zero non-finite values.
- Corrected passive baseline from 2 degrees: 113 policy steps mean upright duration.
- Corrected PD sanity: 409 policy steps mean upright duration and two contacts per 1,000 steps.
- No PPO checkpoint is valid for the corrected 8-inch/`+Y`-axis model.

The rendered file under `evidence_20260711/` shows the obsolete 6-inch/pre-axis-fix asset and is retained only for provenance.

## Next diagnosis

Before another optimizer run:

1. Audit observation scaling, especially unbounded wheel position versus small-radian attitude channels.
2. Plot per-term reward around recovery and contact; verify the policy cannot improve reward by accelerating a fall.
3. Compare a normalized-observation PPO smoke against a residual policy around the proven PD controller.
4. Require critic explained variance and deterministic episode length to improve in a smaller 8,192-step gate before allowing another 65,536-step run.
5. Replace provisional mass/COM/torque values when measured hardware data is available.

The July 12 evidence narrows the next optimizer experiment further: use more than one optimizer update inside the small gate (the current `n_steps=256` with 32 environments yields only one update at 8,192 timesteps), and add a directly testable pitch-energy/progress signal or supervised PD initialization. Do not increase total timesteps first.
