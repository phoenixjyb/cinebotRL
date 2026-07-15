# Two-Wheel Complete Upper-Body Attitude Gate (2026-07-15)

## Decision

The two-wheel robot must retain the complete current `gikWBC9DOF` upper body.
`ee1_tool` is the semantic camera target, including world position and semantic
DFR attitude. The virtual `ee1_rot_z/y/x` joints are MoveIt handling coordinates;
they are not physical DJI motor commands and are not learned actions.

Simulation uses this split contract:

- Policy/teacher target: semantic DFR world quaternion or optical axis.
- Frame conversion: `R_world_cam = R_world_DFR * Rz(+pi/2)`.
- Observation and attitude reward: physical `cam_link` FK.
- Internal simulator adapter: bounded IK to the three physical gimbal joints.
- Hardware deployment: send attitude to DJI firmware; do not export simulated
  physical gimbal angles as the hardware command.

## Asset

The upper-body source is byte-identical to
`recomoProto2-1190_moveit_source.urdf` in `gikWBC9DOF`:

`sha256 aa463a14d84cc5718335f91de7091a49674ec66f8de016cb69d8190f7d98db77`

The split-attitude asset keeps the arm, physical gimbal, `cam_link`, `ee_tool`,
`ee1_tool`, virtual attitude-frame links, meshes, inertials, and camera offset.
The physical gimbal joints are revolute; virtual frame joints are fixed and do
not appear as articulation DOFs. Total robot mass remains 28 kg, wheel spacing
is 620 mm, and wheel diameter is 8 inches.

The source CAD omits reflected motor inertia. A provisional `0.01 kg m^2`
gimbal armature is required to prevent an unrealistic startup velocity impulse.
The 10 Nm effort and 0.5 rad/s velocity assumptions were not increased. This
armature remains a provisional plant prior to replace with measured values.

## Gates

The 1,000-step home hold passed:

- Peak base pitch: 2.929 degrees.
- Peak arm error: 0.170 degrees.
- Peak gimbal error: 2.973 degrees.
- Camera attitude drift: 1.926 degrees.
- No termination or non-finite state.

The 1,000-step semantic DFR motion gate used a +/-5 degree, 0.25 Hz command:

- All attitude IK steps converged; maximum IK residual was 0.100 degrees.
- Physical `cam_link` attitude error was 4.092 degrees p95 and 4.230 degrees max.
- Peak base pitch was 2.982 degrees.
- Gimbal target slew saturation ratio was 0.019.
- Policy action dimension remained two wheel-effort channels.

Evidence:

- `evidence_20260715_complete_upper_body_attitude/home_smoke_1000.json`
- `evidence_20260715_complete_upper_body_attitude/semantic_dfr_5deg.json`

## Data Boundary And Stop Rule

Retarget schema v3 can carry
`target_attitude_world_dfr_quat_wxyz` and generates a smooth acquisition SLERP
from the physical home camera converted to semantic DFR space. It does not add
physical gimbal angles to teacher actions. Schema v3 is explicitly enabled with
`--attitude-urdf`; omitting it preserves the existing Stage-0 schema v2 command.

Do not regenerate the full all-79 attitude set from
`stage_gik_no_obstacle79_nominal`: its metadata points to the older July 1 NPZ
lineage that was quarantined for camera-pose and gimbal-index defects. Preserve
the completed Stage-0 position-only all-79 evidence unchanged.

PPO remains blocked. The next gate is a corrected GIK semantic-attitude export,
followed by one accepted-case full-pose playback with physical `cam_link`
position/orientation metrics. Only then should the accepted corrected cases be
expanded; all-79 attitude training must not start from quarantined labels.
