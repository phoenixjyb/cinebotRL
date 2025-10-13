# Robot Specification Template

Provide measured or manufacturer data; leave `TBD` if unknown. Duplicate sections when the robot has multiple arms, end-effectors, or interchangeable bases.

## Document Context
- Source asset package:
- URDF or USD analysed:
- Supporting ROS packages (name, version):
- Mesh library root:
- Notes on coordinate conventions:

## Chassis (Base)
- Model name / revision:
- Mass [kg]:
- Footprint (length x width) [m]:
- Wheelbase / track width [m]:
- Drive topology (diff-drive, mecanum, ackermann, etc.):
- Max linear velocity [m/s]:
- Max angular velocity [rad/s]:
- Motor controller / interface notes:
- Inertial parameters (Ixx, Iyy, Izz, Ixy, Ixz, Iyz) [kg*m^2]:
- Sensor mounts (frame, pose relative to chassis_center_link):
- Mount interface to arm (parent -> child link):
  - Transform (translation xyz [m], rotation rpy [rad]):
  - Interpretation / alignment notes:
- Chassis collision mesh path(s):

## Arm (example: 6 DOF)
- Model name / revision:
- Kinematic chain (list links in order):
- Home joint configuration [rad]:
- Joint limits (radians / rad*s^-1 / N*m):

| Joint name | Type | Lower | Upper | Max velocity | Max effort |
|------------|------|-------|-------|--------------|------------|
| TBD        | TBD  | TBD   | TBD   | TBD          | TBD        |

- Inertial parameters per link (mass [kg], COM [m], inertia tensor [kg*m^2]):

| Link name | Mass | CoM xyz | Inertia (ixx, iyy, izz, ixy, ixz, iyz) |
|-----------|------|---------|-----------------------------------------|
| TBD       | TBD  | TBD     | TBD                                     |

- Gear ratios / backdrive info:
- Joint actuators and control interface:

## End-Effector / Camera Rig
- Tooling description (model, payload list):
- Mass [kg]:
- Mount transform from wrist frame (translation [m], rotation [rad]):
- Calibration data (intrinsics, extrinsics):
- Additional wiring / payload details:

## Electrical / Control
- Bus voltages and current limits:
- Embedded controllers (MCU/SBC/IPC) and operating systems:
- Communication protocols (CAN, EtherCAT, ROS 2 topics, DDS domains):
- Safety interlocks (E-stop, watchdogs, FSoE, etc.):
- Networking notes (Windows <-> WSL bridges, VLANs, QoS):

## Reference Assets
- CAD file path(s):
- URDF/SDF/USD source path(s):
- MoveIt or Isaac Lab configuration packages:
- Datasheets / manuals (links or filenames):

## Open Questions / TODO
- [ ] Pending measurements:
- [ ] Vendor follow-ups:
- [ ] Simulation approximation notes:
