import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple


def _repo_root() -> Path:
    # linux_env_dev/pybullet_envs/<this_file> -> repo root
    return Path(__file__).resolve().parents[2]


def _default_gik_root() -> str:
    return os.environ.get("GIK_WBC9DOF_ROOT", "/home/converge/data/yanbo/gikWBC9DOF")


@dataclass(frozen=True)
class RobotSpec:
    name: str
    default_urdf_path: str
    package_rewrites: Dict[str, str] = field(default_factory=dict)
    base_joints_xyz_yaw: Tuple[str, str, str] = ("", "", "")
    arm_joints: Tuple[str, ...] = ()
    stabilized_joints: Tuple[str, ...] = ()
    base_link_name: str = ""
    ee_link_name: str = ""


def get_robot_spec(robot: str) -> RobotSpec:
    robot = (robot or "").strip().lower()
    if robot == "":
        robot = "mobile_mm"

    repo_root = _repo_root()
    if robot == "mobile_mm":
        return RobotSpec(
            name="mobile_mm",
            default_urdf_path=str(repo_root / "assets_own" / "mobile_manipulator_little_xy_link.urdf"),
            package_rewrites={},
            base_joints_xyz_yaw=("joint_x", "joint_y", "joint_theta"),
            arm_joints=(
                "left_arm_joint1",
                "left_arm_joint2",
                "left_arm_joint3",
                "left_arm_joint4",
                "left_arm_joint5",
                "left_arm_joint6",
            ),
            stabilized_joints=(),
            base_link_name="abstract_chassis_link",
            ee_link_name="left_gripper_link",
        )

    if robot == "recomo":
        gik_root = Path(_default_gik_root())
        mesh_dir = gik_root / "meshes" / "recomoDemo1"
        return RobotSpec(
            name="recomo",
            default_urdf_path=str(gik_root / "models" / "recomoDemo1" / "recomoDemo1.urdf"),
            package_rewrites={
                "package://recomoDemo1/meshes/": str(mesh_dir) + "/",
            },
            base_joints_xyz_yaw=("base_joint_vx", "base_joint_vy", "base_joint_wz"),
            arm_joints=("joint6_arm_yaw", "joint5_arm_pitch"),
            stabilized_joints=("joint3_gimbal_yaw", "joint2_gimbal_pitch", "joint1_gimbal_roll"),
            base_link_name="base_link",
            ee_link_name="ee_tool",
        )

    raise ValueError(f"Unknown robot='{robot}'. Expected one of: mobile_mm, recomo")

