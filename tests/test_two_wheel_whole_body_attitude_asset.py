import json
from pathlib import Path
from xml.etree import ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = PROJECT_ROOT / "assets_own/recomoProto2_two_wheel_whole_body_attitude"
URDF = ASSET_DIR / "recomoProto2_two_wheel_whole_body_attitude.urdf"
AUDIT = ASSET_DIR / "build_audit.json"


def test_split_attitude_asset_contract() -> None:
    root = ET.parse(URDF).getroot()
    links = {item.attrib["name"]: item for item in root.findall("link")}
    joints = {item.attrib["name"]: item for item in root.findall("joint")}
    total_mass = sum(
        float(link.find("./inertial/mass").attrib["value"])
        for link in links.values()
    )
    assert abs(total_mass - 28.0) < 1e-6
    for name in (
        "joint3_gimbal_yaw",
        "joint2_gimbal_roll",
        "joint1_gimbal_pitch",
    ):
        assert joints[name].attrib["type"] == "revolute"
        assert float(joints[name].find("limit").attrib["effort"]) == 10.0
    for name in ("ee1_level_pitch", "ee1_rot_z", "ee1_rot_y", "ee1_rot_x"):
        assert joints[name].attrib["type"] == "fixed"
    assert {"cam_link", "ee_tool", "ee1_tool"} <= links.keys()


def test_split_attitude_build_audit_preserves_action_boundary() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert audit["profile"] == "split_attitude"
    assert audit["physical_gimbal_joint_mode"] == "internal_sim_attitude_adapter"
    assert audit["learned_physical_gimbal_joint_action"] is False
    assert audit["position_target_link"] == "ee1_tool"
    assert audit["physical_camera_observation_link"] == "cam_link"
    assert audit["validation"]["checks"]["physical_gimbal_matches_profile"] is True
