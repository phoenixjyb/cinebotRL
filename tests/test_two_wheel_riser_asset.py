import json
import math
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = PROJECT_ROOT / "assets_own/recomoProto2_two_wheel_riser"
URDF = ASSET_DIR / "recomoProto2_two_wheel_riser.urdf"
AUDIT = ASSET_DIR / "build_audit.json"
REMOVED_ARM_JOINTS = {
    "joint6_arm_yaw",
    "joint5_arm_pitch",
    "joint4_elbow_pitch",
}
GIMBAL_JOINTS = {
    "joint3_gimbal_yaw",
    "joint2_gimbal_roll",
    "joint1_gimbal_pitch",
}


def _config_float_constant(name: str) -> float:
    import ast

    source = (
        PROJECT_ROOT / "src/rl_platform/robots/two_wheel_balance/config.py"
    ).read_text(encoding="utf-8")
    module = ast.parse(source)
    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in statement.targets):
            return float(ast.literal_eval(statement.value))
    raise AssertionError(f"missing configuration constant: {name}")


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _transform(origin: ET.Element) -> np.ndarray:
    value = np.eye(4)
    value[:3, :3] = _rpy_matrix(
        np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
    )
    value[:3, 3] = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
    return value


def _zero_fk(root: ET.Element, target: str, riser_q: float) -> np.ndarray:
    by_child = {
        joint.find("child").attrib["link"]: joint for joint in root.findall("joint")
    }
    chain = []
    link = target
    while link != "base_link":
        joint = by_child[link]
        chain.append(joint)
        link = joint.find("parent").attrib["link"]
    value = np.eye(4)
    for joint in reversed(chain):
        value = value @ _transform(joint.find("origin"))
        if joint.attrib["name"] == "riser_joint":
            translation = np.eye(4)
            translation[2, 3] = riser_q
            value = value @ translation
    return value


def test_riser_asset_has_exact_physical_dof_contract() -> None:
    root = ET.parse(URDF).getroot()
    joints = {item.attrib["name"]: item for item in root.findall("joint")}
    movable = {
        name for name, joint in joints.items() if joint.attrib["type"] != "fixed"
    }
    assert movable == {
        "left_wheel_joint",
        "right_wheel_joint",
        "riser_joint",
        *GIMBAL_JOINTS,
    }
    assert not REMOVED_ARM_JOINTS & joints.keys()
    assert not {"base_joint_vx", "base_joint_vy", "base_joint_wz"} & joints.keys()

    riser_limit = joints["riser_joint"].find("limit")
    assert joints["riser_joint"].attrib["type"] == "prismatic"
    assert float(riser_limit.attrib["lower"]) == 0.0
    assert float(riser_limit.attrib["upper"]) == 1.2
    assert float(riser_limit.attrib["velocity"]) == 1.0
    assert float(riser_limit.attrib["effort"]) == 300.0
    yaw_proxy = joints["joint1_gimbal_pitch"]
    assert yaw_proxy.attrib["type"] == "continuous"
    assert "lower" not in yaw_proxy.find("limit").attrib
    assert "upper" not in yaw_proxy.find("limit").attrib


def test_riser_mass_and_camera_height_contract() -> None:
    root = ET.parse(URDF).getroot()
    links = {item.attrib["name"]: item for item in root.findall("link")}
    total_mass = sum(
        float(link.find("./inertial/mass").attrib["value"])
        for link in links.values()
    )
    assert abs(total_mass - 28.0) < 1e-6
    assert {"cam_link", "ee_tool", "ee1_tool", "riser_carriage_link"} <= links.keys()

    lower = _zero_fk(root, "cam_link", 0.0)
    upper = _zero_fk(root, "cam_link", 1.2)
    np.testing.assert_allclose(lower[:2, 3], np.zeros(2), atol=1e-8)
    assert abs(lower[2, 3] - 0.6) < 1e-8
    assert abs(upper[2, 3] - 1.8) < 1e-8


def test_semantic_dfr_frame_is_collocated_and_option_b_aligned() -> None:
    root = ET.parse(URDF).getroot()
    cam = _zero_fk(root, "cam_link", 0.3)
    dfr = _zero_fk(root, "ee1_tool", 0.3)
    np.testing.assert_allclose(cam[:3, 3], dfr[:3, 3], atol=1e-12)
    expected_cam_from_dfr = _rpy_matrix(np.array([0.0, 0.0, math.pi / 2.0]))
    np.testing.assert_allclose(cam[:3, :3], dfr[:3, :3] @ expected_cam_from_dfr, atol=1e-10)


def test_riser_build_audit_preserves_learning_boundary() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert audit["passed"] is True
    assert audit["removed_arm_joints"] == [
        "joint6_arm_yaw",
        "joint5_arm_pitch",
        "joint4_elbow_pitch",
    ]
    assert audit["learned_physical_gimbal_joint_action"] is False
    assert audit["position_target_and_observation_link"] == "cam_link"
    assert audit["semantic_attitude_target_link"] == "ee1_tool"
    assert audit["sim_gimbal_joint_semantics"] == (
        "rs4_attitude_proxy_not_motor_shaft_angles"
    )
    assert audit["sim_gimbal_joint_command_mapping"] == {
        "joint3_gimbal_yaw": "ronin_pitch_from_rot_z",
        "joint2_gimbal_roll": "ronin_roll_from_rot_y",
        "joint1_gimbal_pitch": "ronin_yaw_from_rot_x",
    }
    assert audit["continuous_yaw_proxy_joint"] == "joint1_gimbal_pitch"
    assert audit["hardware_yaw_command_envelope_rad"] == [-math.pi, math.pi]
    assert audit["riser"]["camera_height_range_m"] == [0.6, 1.8]
    assert audit["riser"]["speed_limit_mps"] == 1.0
    mount = audit["fixed_gimbal_mount"]
    assert mount["contract"] == "accepted62_rs4_semantic_body_basis_yaw025_v1"
    assert mount["source_case_count"] == 62
    assert mount["source_sample_count"] == 21017
    assert mount["maximum_reference_base_yaw_rate_rad_s"] == 0.25


def test_rs4_proxy_servo_keeps_recovery_headroom_above_filming_rate() -> None:
    command_rate = _config_float_constant("RS4_PROXY_COMMAND_RATE_LIMIT_RAD_S")
    servo_rate = _config_float_constant("RS4_PROXY_SERVO_VELOCITY_LIMIT_RAD_S")
    stiffness = _config_float_constant("RS4_PROXY_SERVO_STIFFNESS_NM_PER_RAD")
    damping = _config_float_constant("RS4_PROXY_SERVO_DAMPING_NMS_PER_RAD")
    config_source = (
        PROJECT_ROOT / "src/rl_platform/robots/two_wheel_balance/config.py"
    ).read_text(encoding="utf-8")

    assert math.isclose(command_rate, math.radians(24.0), abs_tol=1e-12)
    assert math.isclose(servo_rate, math.radians(360.0), abs_tol=1e-12)
    assert servo_rate >= 4.0 * command_rate
    assert stiffness == 400.0
    assert damping == 8.0
    assert "velocity_limit_sim=RS4_PROXY_SERVO_VELOCITY_LIMIT_RAD_S" in config_source
    assert "damping=RS4_PROXY_SERVO_DAMPING_NMS_PER_RAD" in config_source


def test_riser_meshes_resolve_locally() -> None:
    root = ET.parse(URDF).getroot()
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib["filename"]
        assert not filename.startswith("package://")
        assert mesh.attrib["scale"] == "0.001 0.001 0.001"
        assert (URDF.parent / filename).resolve().is_file(), filename
