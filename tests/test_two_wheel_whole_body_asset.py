from pathlib import Path
from xml.etree import ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[1]
URDF = (
    PROJECT_ROOT
    / "assets_own/recomoProto2_two_wheel_whole_body"
    / "recomoProto2_two_wheel_whole_body.urdf"
)


def test_generated_whole_body_urdf_contract() -> None:
    root = ET.parse(URDF).getroot()
    links = {item.attrib["name"]: item for item in root.findall("link")}
    joints = {item.attrib["name"]: item for item in root.findall("joint")}
    total_mass = sum(
        float(link.find("./inertial/mass").attrib["value"])
        for link in links.values()
    )
    assert abs(total_mass - 28.0) < 1e-6
    assert {"left_wheel_joint", "right_wheel_joint"} <= joints.keys()
    assert {
        "joint6_arm_yaw",
        "joint5_arm_pitch",
        "joint4_elbow_pitch",
        "joint3_gimbal_yaw",
        "joint2_gimbal_roll",
        "joint1_gimbal_pitch",
    } <= joints.keys()
    assert {"ee1_level_pitch", "ee1_rot_z", "ee1_rot_y", "ee1_rot_x"} <= joints.keys()
    for name in (
        "joint3_gimbal_yaw",
        "joint2_gimbal_roll",
        "joint1_gimbal_pitch",
        "ee1_level_pitch",
        "ee1_rot_z",
        "ee1_rot_y",
        "ee1_rot_x",
    ):
        assert joints[name].attrib["type"] == "fixed"
    assert "cam_link" in links
    assert not {"base_joint_vx", "base_joint_vy", "base_joint_wz"} & joints.keys()
    pitch_limit = joints["joint5_arm_pitch"].find("limit")
    assert abs(float(pitch_limit.attrib["upper"]) - 1.5707963268) < 1e-10


def test_whole_body_uses_physical_camera_and_valid_local_meshes() -> None:
    root = ET.parse(URDF).getroot()
    camera_joint = root.find("./joint[@name='camera_optical_center']")
    assert camera_joint is not None
    assert camera_joint.find("child").attrib["link"] == "cam_link"
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib["filename"]
        assert not filename.startswith("package://")
        assert (URDF.parent / filename).resolve().is_file(), filename
