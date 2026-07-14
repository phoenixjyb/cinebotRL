# Two-wheel whole-body Stage 0 gate (2026-07-14)

## Scope

This gate integrates the latest frozen `recomoProto2-1190_moveit` arm chain with
the accepted 620 mm-track, 8-inch-wheel chassis. It validates only static
`home_v0` arm hold plus chassis balance. It does not start PPO, track an end
effector trajectory, command the DJI gimbal, or introduce obstacles.

The generated asset contract is:

- total modeled mass: 28 kg;
- active DOFs: two wheel joints and three physical arm joints;
- arm `home_v0`: `[0, pi/2, 3pi/4]` rad;
- physical camera observation frame: `cam_link`;
- DJI physical gimbal joints: fixed at zero for Stage 0;
- MoveIt semantic/virtual joints: fixed at zero and never exposed as policy actions.

The source arm URDF is vendored with SHA256
`aa463a14d84cc5718335f91de7091a49674ec66f8de016cb69d8190f7d98db77`.
The generator redistributes mass from `base_link` so grafting the physical arm
does not double-count the 28 kg aggregate assumption.

## Rejected model

The first imported asset exposed the three physical gimbal axes and four
virtual joints as simulated DOFs with position holds. That model failed the
static gate:

- 10 Nm gimbal limit: 61.72 degree peak pitch-axis error;
- 20 Nm diagnostic limit: 61.33 degree peak pitch-axis error;
- all three gimbal axes approached their 0.5 rad/s limits after the initial
  chassis-settling impulse;
- independent URDF potential-energy differentiation predicted only 0.04,
  0.06, and 0.21 Nm static gravity torque at the three gimbal axes.

Increasing torque therefore did not address the real issue. The real DJI
contract accepts camera attitude targets and resolves motor angles internally;
the virtual MoveIt joints are semantic frames, not physical DOFs. Keeping those
seven joints dynamically actuated would teach against a controller that does
not exist on hardware.

## Accepted evidence

The regenerated URDF/USD asset audit passes:

- one articulation root, 20 rigid bodies, and 19 named joints;
- runtime mass `27.9999994 kg`;
- all bodies have explicit positive inertia;
- 620 mm wheel track, 203.2 mm wheel diameter, and positive-Y wheel axes;
- physical `cam_link` present;
- all three DJI gimbal joints and four virtual joints are fixed.

The deterministic `home_v0` gates use the frozen 28 kg LQR gains and the
opt-in `structural_robust_v1` outer-loop profile:

| Gate | Peak pitch | Peak arm error | Final arm error | Result |
| --- | ---: | ---: | ---: | --- |
| 1 env x 2,000 steps | 3.136 deg | 0.215 deg | 0.015 deg | pass |
| 16 envs x 2,000 steps | 3.260 deg | 0.236 deg | 0.027 deg | pass |

Both gates completed without fall, forbidden contact, wheel overspeed,
timeout, or non-finite state. Machine-readable runtime evidence is under
`artifacts/two_wheel_balance/whole_body_stage0/` on the validated `.98` host.

## Boundary and next gate

Stage 0 is closed for static arm hold in simulation. This is not a dynamic DJI
gimbal validation or a hardware-readiness claim. COM, inertia, friction, wheel
torque, and control delay remain provisional.

The next gate is a no-obstacle, low-amplitude whole-body tracking smoke:

1. observe the physical `cam_link` pose;
2. produce slow three-joint arm references with conventional IK;
3. retain balance/fault handling as the highest-priority controller;
4. reject any reference that violates balance or arm limits;
5. keep the gimbal fixed until an explicit attitude-to-physical-gimbal
   controller is implemented and independently validated.

Obstacle avoidance and PPO remain blocked until that deterministic tracking
gate passes.
