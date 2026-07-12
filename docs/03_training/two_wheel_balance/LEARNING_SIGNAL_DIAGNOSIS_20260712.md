# Two-Wheel PPO Learning-Signal Diagnosis

Date: 2026-07-12

## Conclusion

The original 65,536-step direct PPO run did not merely learn too slowly. It learned common-wheel action with the opposite sign to the stabilizing controller. The plant, action direction, contact termination, deterministic vectorization, and scripted controllability gates remain healthy.

No long PPO run is authorized from either failed checkpoint.

## Measured failure

All controllers were replayed for 1,000 steps from the same fixed `+2 deg` pitch reset.

| Controller | Contacts | Mean completed episode | Mean common action | PD sign agreement |
| --- | ---: | ---: | ---: | ---: |
| Zero action | 8 | 115.0 | 0.000 | 0% |
| Random action | 7 | 126.9 | 0.0005 | 50.7% |
| Scripted PD | 2 | 364.0 | -0.0844 | 100% |
| Failed direct PPO | 11 | 90.0 | +0.0917 | 0% |

The failed PPO action correlates positively with both destabilizing state channels:

```text
corr(a_common, pitch)      = +0.9407
corr(a_common, pitch_rate) = +0.9980
```

The scripted PD correlations are approximately `-0.996` and `-0.987` respectively.

The canonical zero-other-channel sweep confirms the error independently of rollout history:

```text
pitch          -10 deg      0 deg      +10 deg
PPO action     +0.0451      +0.0517    +0.0582
PD action      +0.1745       0.0000    -0.1745
```

## Reward evidence

The policy is not exploiting a higher measured return. Mean per-step upright reward is `0.631` for failed PPO versus `0.842` for PD, while PPO also has worse pitch-rate and wheel-speed penalties. The failed policy receives lower overall reward and shorter episodes.

This points to poor credit assignment/optimization rather than a reward loophole that makes falling optimal.

## Observation evidence

No catastrophic observation-scale mismatch was measured during the failed PPO rollout:

```text
pitch abs p95                 0.252 rad
pitch_rate abs p95            0.890 rad/s
mean_wheel_position abs p95   0.812 rad
mean_wheel_velocity abs p95   2.476 rad/s
```

Normalization may still help critic conditioning, but it is not sufficient as the sole explanation for the wrong action sign.

## Residual correction

An opt-in `pd_residual` mode was added. Direct mode remains the default and passes its regression smoke.

```text
pd_common = clip(-pitch - 0.2 * pitch_rate, -0.5, 0.5)
applied_common = clip(pd_common + 0.15 * residual_common)
applied_yaw = 0.15 * residual_yaw
```

Zero policy residual reproduces the proven PD baseline exactly at 364 mean steps. Six pure contract tests cover the mixer and residual composition.

## Bounded A/B result

One 8,192-step residual PPO gate was run with 32 environments and 128 deterministic evaluation episodes.

```text
random residual baseline:  362.48 mean steps, 12.74 deg pitch p95
zero-residual PD prior:     364.00 mean steps, 12.83 deg pitch p95
learned residual:           353.00 mean steps, 12.94 deg pitch p95
```

The residual architecture preserves safety and strongly outperforms the old direct baseline, but the learned residual slightly degrades the controller. Therefore:

```text
finite_training_metrics = true
prior_preserved         = true
direct_baseline_improved = true
learning_signal         = false
passed                  = false
```

## Next bounded experiment

Do not increase the total timestep budget. The next experiment should change update density and learning signal together:

1. Reduce `n_steps` from 256 to 64 for four optimizer updates within an 8,192-step gate.
2. Add a bounded pitch-energy/progress reward with a pure sign/regression test.
3. Alternatively initialize the residual policy from synthetic PD labels and then verify PPO does not erase the stabilizing sign.
4. Require deterministic residual performance to exceed the zero-residual PD prior, not merely the old direct baseline.
5. Stop again at 8,192 if mean episode length and pitch p95 do not both improve.
