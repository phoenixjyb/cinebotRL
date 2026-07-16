# Two-wheel riser robot: goal, contract, and gates

Date: 2026-07-15
Branch: `codex/two-wheel-riser-rl`
Worktree: `/mnt/g/wSpace/cinebotRL-two-wheel-riser`

## 1. Goal

Build and validate a new self-balancing two-wheel camera robot that removes the
three arm joints and replaces them with one motorized vertical riser. The film
camera remains on the physical three-axis gimbal. The final system must track
the corrected 79 semantic camera trajectories while remaining balanced.

Obstacle avoidance is explicitly outside this round. It must not be added to
the reward, action space, curriculum, or acceptance claim.

The control priority is:

1. balance and finite-state safety;
2. riser travel, speed, braking, and gimbal limit safety;
3. semantic camera trajectory tracking.

### Recursive improvement objective

The optimization loop is lexicographic, not a weighted average:

1. retain every previously passed balance and finite-state safety gate;
2. increase full-duration corrected-reference pass count;
3. reduce holdout position p95, then position maximum;
4. reduce holdout attitude p95, then attitude maximum;
5. reduce actuator saturation and control effort only after the tracking gates
   above are non-regressing.

Every round changes one structural factor, reruns the same deterministic smoke
set, and records an accept or reject decision. A candidate is accepted only if
it fixes the targeted failure without regressing any higher-priority gate.
Longer training is forbidden as a response to a model, frame, reference, or
feasibility failure.

## 2. Robot contract

The new asset is `recomoProto2_two_wheel_riser`; the existing arm-based assets
remain unchanged.

Physical movable joints:

- `left_wheel_joint`, `right_wheel_joint`: wheel effort control;
- `riser_joint`: vertical prismatic motion;
- `joint3_gimbal_yaw`, `joint2_gimbal_roll`, `joint1_gimbal_pitch`: internal
  attitude adapter controlled from a semantic camera-attitude command.

Forbidden joints:

- `joint6_arm_yaw`;
- `joint5_arm_pitch`;
- `joint4_elbow_pitch`.

Semantic interface:

- position target and observation: physical camera optical center at
  `cam_link`;
- attitude command: world-frame semantic DFR attitude at `ee1_tool`;
- physical attitude observation/reward: `cam_link` after applying
  `R_world_cam = R_world_DFR * Rz(+pi/2)`;
- physical DJI gimbal joint angles are internal adapter states, never teacher
  labels or direct learned actions.
- the fixed riser-to-gimbal bracket reproduces the accepted corrected-corpus
  median orientation of the removed arm; an identity mount is forbidden
  because it makes path-aligned camera attitudes unreachable.

`ee1_tool` is collocated with `cam_link` and fixed at `R_cam_DFR = Rz(-pi/2)`.
This preserves the corrected Option-B camera-frame contract without retaining
the old virtual attitude joints.

## 3. Provisional plant assumptions

These values are simulation priors, not sim-to-real evidence:

| Quantity | Provisional value |
|---|---:|
| Total mass | 28.0 kg |
| Wheel track | 0.620 m |
| Wheel diameter | 0.2032 m (8 in) |
| Wheel effort limit | 20 N m per wheel |
| Camera optical-center range | 0.600 to 1.800 m |
| Initial camera height | 0.900 m |
| Riser stroke | 1.200 m |
| Riser speed limit | 1.000 m/s |
| Riser force limit | 300 N |
| Provisional moving mass | 4.342 kg |
| Riser acceleration target | <= 2.0 m/s^2 initially |
| Riser jerk target | <= 8.0 m/s^3 initially |

The 28 kg total is retained so the established chassis balance model remains a
useful starting point. Removing the arm and adding the riser redistributes mass,
so COM and pitch inertia must later be replaced with CAD or measured values.

## 4. Controller architecture

The scripted baseline is mandatory before DNN training:

1. inner wheel LQR stabilizes pitch and wheel motion;
2. differential-drive outer loop tracks horizontal camera/path progress;
3. a jerk-limited riser reference plus position/velocity loop tracks camera
   height and includes gravity load compensation;
4. the existing semantic DFR adapter solves the three physical gimbal joints;
5. a safety governor slows horizontal and riser progress as pitch margin,
   actuator headroom, or travel margin decreases.

The fixed-path candidate caps chassis yaw at `0.25 rad/s`. The complementary
joint-adaptive candidate may use up to `0.4 rad/s`, but it is eligible only when
all raw continuous proxy targets remain below the `24 deg/s` filming slew
limit. The adaptive optimizer uses a `0.995` internal rate margin rather than
relaxing that public limit. The accepted 62-case portfolio satisfies the gate
with a worst raw proxy rate of `0.417392 rad/s`. These are reference-planning
results, not a claim that the balance plant has passed the same motion in
Isaac.

The first learned controller is a residual policy over the scripted baseline,
not unrestricted PPO from scratch. Its initial residual action contract is:

- common wheel effort correction;
- differential wheel effort correction;
- riser reference/force correction.

The semantic gimbal adapter remains deterministic. A learned gimbal residual is
not permitted until the corrected teacher and holdout attitude gates pass.

## 5. Stage gates and stop rules

### Gate 0: asset integrity

- one floating root and exactly the six movable joints listed above;
- no arm joints or planar virtual base joints;
- total mass 28.0 kg within 1e-6 kg;
- camera optical-center height is 0.600 m at lower travel and 1.800 m at upper
  travel in the URDF root frame;
- fixed gimbal mount contract is
  `accepted62_rs4_semantic_body_basis_yaw025_v1`;
- `joint1_gimbal_pitch`, the legacy-named Ronin-yaw proxy, is continuous in
  simulation; the semantic adapter still wraps and bounds the hardware command
  to `[-pi, pi]`;
- raw proxy position-target deltas, without cyclic correction, must satisfy the
  same `24 deg/s` rate gate;
- riser limit is 0.0 to 1.2 m and speed limit is 1.0 m/s;
- semantic and physical camera frames satisfy the Option-B transform.

The CAD STL files are millimetres and carry an explicit `0.001` scale in the
generated URDF. Convert this asset with `--mesh-scale 1.0`; using the converter
default would incorrectly shrink prismatic limits and joint origins.

### Gate 1: static balance at three heights

Run at camera heights 0.6, 0.9, and 1.8 m for at least 10 s each:

- no fall, termination, NaN, or joint-limit violation;
- maximum absolute pitch <= 12 deg and pitch p95 <= 6 deg;
- no wheel or riser command remains saturated for more than 20% of samples.

### Gate 2: riser dynamics

Execute bounded up/down profiles at 0.1, 0.25, 0.5, then 1.0 m/s:

- each lower-speed stage must pass before the next starts;
- no travel overshoot beyond 10 mm;
- camera-height p95 error <= 30 mm after acceleration transients;
- the Gate-1 balance and saturation limits still hold;
- any non-finite state, fall, hard-limit contact, or persistent saturation stops
  the run immediately.

### Gate 3: synthetic whole-body tracking

Track straight, lateral-by-yaw, vertical, and combined xyz camera paths before
using teacher data:

- position p95 <= 0.15 m and maximum <= 0.25 m;
- attitude p95 <= 5 deg and maximum <= 10 deg;
- full commanded duration completes without safety termination.

### Gate 4: corrected teacher subset

- use only corrected Option-B exports and accepted teacher rows;
- old NPZ exports affected by pose-transpose or physical-gimbal-index bugs stay
  quarantined;
- run the accepted 62-case corpus first;
- rejected cases are not silently promoted to teacher labels.

### Gate 5: all 79 trajectories

Repair/regenerate the remaining teacher references, then require 79/79 full
duration completion under the Gate-3 error and safety bounds. Per-case reports
must preserve failures rather than averaging them away.

### Gate 6: DNN residual policy

Training is allowed only after the scripted controller produces a stable,
non-saturated learning signal. The learned policy must:

- match all scripted safety gates;
- improve median and p95 position error on training cases;
- not regress corrected holdout cases by more than 5%;
- pass mass, COM, friction, motor-strength, command-delay, and sensor-noise
  randomization envelopes;
- survive a deterministic replay and rendered rollout audit.

PPO remains blocked whenever the asset, static-height, riser-dynamics, or
teacher-contract gate is red. A longer run is never a remedy for a structural
gate failure.

## 6. Hardware boundary

The 1 m/s vertical axis is a safety-critical machine subsystem. The simulation
uses a 4.342 kg moving mass and 300 N force envelope, but final motor and
mechanism selection requires measured moving payload, duty cycle, desired
acceleration/stopping distance, supply voltage, and retained brake load. A
normally-closed brake, redundant end limits, independent overspeed detection,
mechanical end stops, and a controlled emergency-stop deceleration are required
before hardware operation.
