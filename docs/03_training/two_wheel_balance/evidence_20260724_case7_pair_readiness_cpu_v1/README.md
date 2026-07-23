# Case-7 paired-canary readiness evidence

This package audits selected training-tranche case 7 without launching Isaac
or opening any capture or training path.

## Result

- The imported exact-source plan and zero-residual dynamic gate match the
  tranche-selection SHA-256 identities.
- The plan preserves 663 source anchors and 662 transitions.
- Source duration is 12.940941 s; retimed execution duration is 18.1173174 s.
- Camera height remains within 0.600000-1.605452 m.
- Maximum feedforward rates remain below the frozen 0.4 m/s base-linear,
  0.4 rad/s base-yaw, 1.0 m/s riser, and 0.418879 rad/s proxy limits.
- The zero-residual gate passes at 0.130904/0.142948 m position p95/max and
  6.169692 degrees peak pitch.
- Four low-motion windows are available, with the longest lasting 3.431994 s.
- Camera lever-arm correction saturation is 0.919479, so case 7 still needs a
  case-specific corrective profile. Reusing case-23, case-6, or case-2
  profiles is forbidden.

This evidence does not authorize runtime, GPU use, label capture, dataset
conversion or merge, BC, PPO, or training.
