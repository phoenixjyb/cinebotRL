"""Central joint name definitions for the mobile manipulator.

Maps between the cinebotRL internal convention and the latest PNC URDF
(recomoProto2-1190_moveit.urdf) joint names.

PNC URDF joint order (actuated only):
  0: base_joint_vx     (prismatic)  — base X translation
  1: base_joint_vy     (prismatic)  — base Y translation
  2: base_joint_wz     (continuous) — base yaw rotation
  3: joint6_arm_yaw    (revolute)   — arm J1 (shoulder yaw)
  4: joint5_arm_pitch  (revolute)   — arm J2 (shoulder pitch)
  5: joint4_elbow_pitch(revolute)   — arm J3 (elbow)
  6: joint3_gimbal_yaw (revolute)   — arm J4 (wrist yaw)
  7: joint2_gimbal_roll(revolute)   — arm J5 (wrist roll)
  8: joint1_gimbal_pitch(revolute)  — arm J6 (wrist pitch)
  9-11: ee1_rot_z/y/x  (revolute)  - virtual EE gimbal (excluded from RL)
"""

# --- Base joints (PPR: prismatic-prismatic-revolute) ---
BASE_JOINT_VX = "base_joint_vx"
BASE_JOINT_VY = "base_joint_vy"
BASE_JOINT_WZ = "base_joint_wz"
BASE_JOINT_NAMES = [BASE_JOINT_VX, BASE_JOINT_VY, BASE_JOINT_WZ]

# --- Arm joints (6-DOF, ordered from base to tip) ---
ARM_JOINT_1 = "joint6_arm_yaw"
ARM_JOINT_2 = "joint5_arm_pitch"
ARM_JOINT_3 = "joint4_elbow_pitch"
ARM_JOINT_4 = "joint3_gimbal_yaw"
ARM_JOINT_5 = "joint2_gimbal_roll"
ARM_JOINT_6 = "joint1_gimbal_pitch"
ARM_JOINT_NAMES = [ARM_JOINT_1, ARM_JOINT_2, ARM_JOINT_3,
                   ARM_JOINT_4, ARM_JOINT_5, ARM_JOINT_6]

# Regex for Isaac Lab actuator config (matches joint6_, joint5_, ... joint1_)
ARM_JOINT_NAMES_EXPR = ["joint[1-6]_.*"]

# --- EE link ---
EE_LINK_NAME = "cam_link"

# --- Virtual EE gimbal joints (kept for MoveIt-style frames; excluded from RL action/observation) ---
EE_VIRTUAL_JOINT_NAMES = ["ee1_rot_z", "ee1_rot_y", "ee1_rot_x"]

# --- Counts ---
NUM_BASE_JOINTS = len(BASE_JOINT_NAMES)
NUM_ARM_JOINTS = len(ARM_JOINT_NAMES)
NUM_ACTUATED_JOINTS = NUM_BASE_JOINTS + NUM_ARM_JOINTS  # 9 (excludes EE virtual)
