# Two-wheel riser fixed-gimbal-mount audit

Date: 2026-07-15

## Decision

The first riser URDF is rejected as the final tracking model. It removed the
three arm joints but also discarded their orientation contribution, attaching
the source gimbal chain directly to the riser with an identity base mount.
That model can balance and move the riser, but it does not represent a suitable
fixed bracket for the corrected camera-pose corpus.

The replacement bracket uses the median physical arm configuration from 1,289
samples across the corrected accepted 62-case teacher set:

```text
joint6_arm_yaw     -0.31901315 rad
joint5_arm_pitch    0.55165980 rad
joint4_elbow_pitch -0.79816264 rad
```

The arm DOFs remain absent. Their median orientation is encoded once as a
fixed mechanical bracket, not restored as state, action, or policy output.

## Numerical reason

For corrected case 1 at the initial path-aligned differential-drive heading:

| Mount | Physical cam attitude error | Normalized gimbal-limit margin |
|---|---:|---:|
| Rejected identity mount | 20.264 deg, no IK convergence | 0.024 |
| Accepted-corpus mean arm orientation | 0.0030 deg | 0.169 |
| Selected median-arm fixed bracket | 0.0068 deg | 0.172 |

The identity result is a structural workspace failure. PPO, reward tuning, or
larger wheel torques cannot correct it. The median-arm bracket was selected
because it is feasible and corresponds to a concrete nominal physical arm
pose, unlike an unconstrained average rotation.

Across the 1,289 source samples, arm orientation differs from the rotation mean
by 14.13 degrees median, 81.86 degrees p95, and 91.90 degrees maximum. A fixed
bracket therefore does not prove all trajectories feasible; the complete
corrected-corpus audit remains mandatory.

## Geometry contract

The bracket rotation is composed into `joint3_gimbal_yaw`. Its translation is
recomputed so the zero-gimbal physical `cam_link` remains centered over the
riser and retains a 0.026469434 m offset from the riser carriage origin.

```text
joint3 origin xyz = [ 0.12829271, -0.07024936, -0.00584255 ] m
joint3 origin rpy = [ 0.0000489776, -1.34982244, -0.319053503 ] rad
```

The prismatic joint remains 0.0--1.2 m. The physical optical-center range is
still exactly 0.6--1.8 m; 1.9 m is not allowed. The Option-B frame contract is
unchanged:

```text
R_world_cam = R_world_DFR * Rz(+pi/2)
```

Physical gimbal joints remain internal simulator adapter state and are not
learned actions or hardware command labels.

## Current validation boundary

Local pure-kinematic tests pass for joint structure, mass, 0.6/1.8 m camera
height, Option-B frame alignment, robust gimbal IK, corrected case-1 attitude,
and strict corrected-reference loading. Remote `.98` USD conversion and Isaac
validation are pending because the host became unreachable during this audit.

Required restart sequence:

1. sync the builder, generated URDF, tests, and audit to the isolated worktree;
2. regenerate USD with mesh scale 1.0;
3. rerun Gate 0 import and all 0.6/0.9/1.8 m static tests;
4. rerun the 0.1/0.25/0.5/1.0 m/s dynamic riser matrix unchanged;
5. rerun corrected-reference workspace and nonholonomic gates;
6. keep PPO disabled until those final-asset gates pass.
