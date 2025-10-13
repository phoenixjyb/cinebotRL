# Robot Specification - Mobile Base + Left Arm

## Document Context
- Source assets: `assets_own/mobile_arm_whole_body`
- URDF analysed: `assets_own/mobile_arm_whole_body/urdf/arm_on_car_center_rotZ_neg90.urdf`
- ROS package metadata: `assets_own/mobile_arm_whole_body/package.xml` (`mobile_arm_whole_body` v0.1.0)
- Mesh library: `assets_own/mobile_arm_whole_body/meshes` (STL, millimetres, Z-up in CAD)

## Chassis (Base)
- Primary link: `chassis_center_link` (placeholder link, no inertial/visual/collision defined in current URDF).
- Physical chassis mesh available as `meshes/cr_no_V.stl`, but it is **not** referenced yet-add it once collision geometry is ready.
- Status: **TBD** - measure total mass, inertia tensor, wheelbase (track width, wheel radius), controller limits, and diff-drive PID gains.
- Measured mass: 30 kg (base only, payload excluded).
- Wheelbase / track width: 0.35 m wheelbase, 0.60 m track width (hub-to-hub).
- Velocity limits: linear capped at 1.5 m/s; angular limit TBD; wheel rotational speed measurement pending.
- Mount interface (`arm_mount_joint`):
  - Parent -> child: `chassis_center_link` -> `left_arm_base_link`.
  - Transform (world units metres, radians): translation `(0.150, -0.0675, 1.050)`; rotation `(roll=0, pitch=0, yaw=-1.5708)` i.e. yaw -90 deg about Z.
  - Interpretation: arm base is 150 mm forward, 67.5 mm to robot-right, and 1.05 m above chassis origin with a -90 deg yaw to face forward.

## Arm (Left Arm, 6 DOF)
- Kinematic chain: `chassis_center_link` -> `left_arm_base_link` -> `left_arm_link1` -> `left_arm_link2` -> `left_arm_link3` -> `left_arm_link4` -> `left_arm_link5` -> `left_arm_link6` -> `left_gripper_link`.
- Joint limits extracted from URDF (radians / rad*s^-1 / N*m):

| Joint             | Type     | Lower | Upper | Max Velocity | Max Effort |
|-------------------|----------|-------|-------|--------------|------------|
| left_arm_joint1   | Revolute | -2.8798 |  2.8798 | 1.6 | 40 |
| left_arm_joint2   | Revolute | 0.0    |  3.2289 | 1.6 | 40 |
| left_arm_joint3   | Revolute | -3.3161 | 0.0   | 4.0 | 27 |
| left_arm_joint4   | Revolute | -2.8798 | 2.8798 | 4.0 | 7  |
| left_arm_joint5   | Revolute | -1.6581 | 1.6581 | 4.0 | 7  |
| left_arm_joint6   | Revolute | -2.8798 | 2.8798 | 4.0 | 7  |

- Link inertial parameters (kg / m / kg*m^2):

| Link                | Mass | CoM (x y z) [m]             | Inertia (ixx, iyy, izz, ixy, ixz, iyz) |
|---------------------|------|-----------------------------|-----------------------------------------|
| left_arm_base_link  | 1.658 | -0.0005634 0.038934 0.0000032 | (0.0010597, 0.0011787, 0.0010647, 1.9821E-05, -1.6752E-07, -1.9146E-07) |
| left_arm_link1      | 1.164 | 0.000015 0.105259 -0.001954 | (0.001125, 0.001084, 0.001158, 0, 0, -2.3E-05) |
| left_arm_link2      | 1.300 | -0.23622 0.016352 -0.000133 | (0.00060638, 0.0075936, 0.0075712, 0.00041817, 0.00014956, -8.0916E-06) |
| left_arm_link3      | 0.818 | 0.045114 0.054616 -0.000456 | (0.00060107, 0.0013959, 0.0015027, -0.00022467, -7.1194E-06, -9.7503E-06) |
| left_arm_link4      | 0.698 | 0.24285 0.0023784 0.0000013 | (8.45E-05, 0.00010174, 9.7044E-05, -8.2627E-07, -2.2607E-09, 5.3612E-09) |
| left_arm_link5      | 0.417 | 0.054309 0.0041811 0.0000041 | (8.3999E-05, 9.8498E-05, 0.00011333, 1.6234E-05, 7.4127E-08, -1.3811E-08) |
| left_arm_link6      | 0.037 | 0.028138 0.00000012 0.00000005 | (3.5662E-06, 2.0238E-06, 2.0238E-06, 6.6514E-12, 2.9628E-12, -4.1666E-12) |
| left_gripper_link   | 0.604 | -0.031107 -0.00000014 -0.00000014 | (0.00017588, 9.8637E-05, 0.00016512, 4.1789E-10, -5.3493E-10, -8.1856E-08) |

(Scientific notation copied verbatim from URDF; convert to decimal form as needed for downstream tools.)

## End-Effector / Camera Rig
- Terminal link: `left_gripper_link`.
- Mount transform from wrist (`left_arm_joint6` frame): translation `(0.1039, 0.0, 0.0)` m; rotation `(0, 0, 0)` rad.
- Inertial properties: mass 0.604 kg; inertia tensor as above.
- Calibration status: **TBD** - add camera extrinsics, payload CG shift when instrumented, and wiring diagram.

## Electrical / Control
- MoveIt configurations: `assets_own/mobile_arm_whole_body/config/*.yaml` (kinematics, controllers, planning pipelines).
- Launch entry point: `assets_own/mobile_arm_whole_body/launch/whole_body_demo.launch.py`.
- Current controllers: FakeController (planning-only). Replace with arm + diff-drive controllers when integrating hardware.
- Base <-> arm coordination: demo assumes diff-drive commands sourced from `world_joint` trajectories (see `scripts/world_joint_to_cmd_vel.py`).

## Reference Assets
- Meshes: `meshes/*.STL` (arm links, chassis shell).
- URDF: `urdf/arm_on_car_center_rotZ_neg90.urdf`.
- SRDF & MoveIt: `srdf/arm_on_car_center_whole_body.srdf`, plus configs under `config/`.
- Supporting docs: this spec (`assets/raw/robot_spec.md`) and template (`assets/raw/robot_spec_template.md`).

## Open TODOs
- Populate chassis inertia tensor and wheel rotational / angular limits once measured.
- Add electrical architecture (power rails, motor drivers, safety interlocks) and network topology.
- Record URDF home pose vs. desired trajectory frame offsets once alignment study is complete (`docs/tracking/ee_frame_alignment.md`).
- Document sensor payload (camera/gimbal model, calibration, data links) when selected.
