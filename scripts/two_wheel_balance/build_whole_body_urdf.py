#!/usr/bin/env python3
"""Build the 28 kg two-wheel whole-body URDF from frozen source assets."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = (
    PROJECT_ROOT
    / "assets_own/recomoProto2_two_wheel_balance/recomoProto2_two_wheel_balance.urdf"
)
DEFAULT_ARM = (
    PROJECT_ROOT
    / "assets_own/sources/recomoProto2-1190_moveit_aa463a.urdf"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "assets_own/recomoProto2_two_wheel_whole_body"
    / "recomoProto2_two_wheel_whole_body.urdf"
)
DEFAULT_ATTITUDE_OUTPUT = (
    PROJECT_ROOT
    / "assets_own/recomoProto2_two_wheel_whole_body_attitude"
    / "recomoProto2_two_wheel_whole_body_attitude.urdf"
)
EXPECTED_ARM_SHA256 = "aa463a14d84cc5718335f91de7091a49674ec66f8de016cb69d8190f7d98db77"
TARGET_TOTAL_MASS_KG = 28.0
POSITION_ONLY_PROFILE = "position_only"
SPLIT_ATTITUDE_PROFILE = "split_attitude"

PHYSICAL_ARM_JOINTS = (
    "joint6_arm_yaw",
    "joint5_arm_pitch",
    "joint4_elbow_pitch",
    "joint3_gimbal_yaw",
    "joint2_gimbal_roll",
    "joint1_gimbal_pitch",
)
PHYSICAL_GIMBAL_JOINTS = PHYSICAL_ARM_JOINTS[3:]
VIRTUAL_FRAME_JOINTS = (
    "ee1_level_pitch",
    "ee1_rot_z",
    "ee1_rot_y",
    "ee1_rot_x",
)
ARM_LINKS = (
    "arm_base_link",
    "arm_link_1",
    "arm_link_2",
    "arm_link_3",
    "arm_link_4",
    "arm_link_5",
    "arm_link_6",
    "cam_link",
    "ee_tool",
    "ee1_tool",
    "ee1_level_pitch_link",
    "ee1_rotz_link",
    "ee1_roty_link",
    "ee1_rotx_link",
)
ARM_JOINTS = (
    "arm_base_joint",
    *PHYSICAL_ARM_JOINTS,
    "camera_optical_center",
    "ee_tool_fixed",
    *VIRTUAL_FRAME_JOINTS,
    "ee1_tool_mount_fixed",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _named(root: ET.Element, tag: str, name: str) -> ET.Element:
    element = root.find(f"./{tag}[@name='{name}']")
    if element is None:
        raise RuntimeError(f"missing {tag} {name}")
    return element


def _link_mass(link: ET.Element) -> float:
    mass = link.find("./inertial/mass")
    if mass is None:
        raise RuntimeError(f"link {link.attrib['name']} has no explicit mass")
    return float(mass.attrib["value"])


def _remove_empty_geometry(link: ET.Element) -> None:
    for tag in ("visual", "collision"):
        for element in list(link.findall(tag)):
            if element.find("geometry") is None:
                link.remove(element)


def _rewrite_mesh_paths(element: ET.Element) -> None:
    prefix = "package://camera_robot/meshes/"
    for mesh in element.findall(".//mesh"):
        filename = mesh.attrib.get("filename", "")
        if filename.startswith(prefix):
            mesh.attrib["filename"] = f"../meshes/{filename.removeprefix(prefix)}"


def _lock_joint(joint: ET.Element) -> None:
    joint.attrib["type"] = "fixed"
    for tag in ("axis", "limit", "dynamics", "safety_controller", "calibration", "mimic"):
        element = joint.find(tag)
        if element is not None:
            joint.remove(element)


def _set_base_mass(root: ET.Element, target_mass: float) -> tuple[float, float]:
    base_link = _named(root, "link", "base_link")
    base_mass_element = base_link.find("./inertial/mass")
    inertia = base_link.find("./inertial/inertia")
    if base_mass_element is None or inertia is None:
        raise RuntimeError("base_link needs explicit mass and inertia")
    old_mass = float(base_mass_element.attrib["value"])
    scale = target_mass / old_mass
    base_mass_element.attrib["value"] = f"{target_mass:.9f}"
    for key in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
        inertia.attrib[key] = f"{float(inertia.attrib[key]) * scale:.9f}"
    return old_mass, scale


def _validate_tree(root: ET.Element, profile: str) -> dict[str, object]:
    links = root.findall("link")
    joints = root.findall("joint")
    link_names = [item.attrib["name"] for item in links]
    joint_names = [item.attrib["name"] for item in joints]
    children = []
    for joint in joints:
        child = joint.find("child")
        if child is None:
            raise RuntimeError(f"joint {joint.attrib['name']} has no child")
        children.append(child.attrib["link"])
    roots = sorted(set(link_names) - set(children))
    total_mass = sum(_link_mass(link) for link in links)
    checks = {
        "unique_links": len(link_names) == len(set(link_names)),
        "unique_joints": len(joint_names) == len(set(joint_names)),
        "single_base_link_root": roots == ["base_link"],
        "total_mass_28kg": abs(total_mass - TARGET_TOTAL_MASS_KG) < 1e-6,
        "physical_arm_joints_present": all(name in joint_names for name in PHYSICAL_ARM_JOINTS),
        "virtual_frame_joints_present": all(name in joint_names for name in VIRTUAL_FRAME_JOINTS),
        "virtual_frame_joints_fixed": all(
            _named(root, "joint", name).attrib["type"] == "fixed"
            for name in VIRTUAL_FRAME_JOINTS
        ),
        "physical_gimbal_matches_profile": all(
            _named(root, "joint", name).attrib["type"]
            == ("fixed" if profile == POSITION_ONLY_PROFILE else "revolute")
            for name in PHYSICAL_GIMBAL_JOINTS
        ),
        "physical_cam_link_present": "cam_link" in link_names,
        "semantic_ee1_tool_present": "ee1_tool" in link_names,
        "no_planar_base_joints": all(
            name not in joint_names for name in ("base_joint_vx", "base_joint_vy", "base_joint_wz")
        ),
        "two_wheel_joints": all(
            name in joint_names for name in ("left_wheel_joint", "right_wheel_joint")
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"whole-body URDF contract failed: {checks}")
    return {
        "link_count": len(links),
        "joint_count": len(joints),
        "root_links": roots,
        "total_mass_kg": round(total_mass, 9),
        "checks": checks,
    }


def build(
    base_path: Path,
    arm_path: Path,
    output_path: Path,
    profile: str = POSITION_ONLY_PROFILE,
) -> dict[str, object]:
    if profile not in (POSITION_ONLY_PROFILE, SPLIT_ATTITUDE_PROFILE):
        raise ValueError(f"unsupported profile: {profile}")
    arm_sha256 = _sha256(arm_path)
    if arm_sha256 != EXPECTED_ARM_SHA256:
        raise RuntimeError(
            f"arm source hash changed: expected {EXPECTED_ARM_SHA256}, got {arm_sha256}"
        )

    base_tree = ET.parse(base_path)
    base_root = base_tree.getroot()
    arm_root = ET.parse(arm_path).getroot()
    base_root.attrib["name"] = "recomoProto2-two-wheel-whole-body"

    base_root.remove(_named(base_root, "joint", "arm_mount_joint"))
    base_root.remove(_named(base_root, "link", "arm_mount_link"))

    copied_links = []
    for name in ARM_LINKS:
        link = deepcopy(_named(arm_root, "link", name))
        _remove_empty_geometry(link)
        _rewrite_mesh_paths(link)
        base_root.append(link)
        copied_links.append(link)

    copied_joints = []
    for name in ARM_JOINTS:
        joint = deepcopy(_named(arm_root, "joint", name))
        if name == "arm_base_joint":
            parent = joint.find("parent")
            origin = joint.find("origin")
            if parent is None or origin is None:
                raise RuntimeError("arm_base_joint is incomplete")
            parent.attrib["link"] = "base_link"
            origin.attrib["xyz"] = "0 0 0.880"
            origin.attrib["rpy"] = "0 0 0"
        elif name == "joint5_arm_pitch":
            limit = joint.find("limit")
            if limit is None:
                raise RuntimeError("joint5_arm_pitch has no limit")
            # Deployed home_v0 and software limits use an exact +90 degree bound.
            limit.attrib["upper"] = "1.5707963268"
        if name in VIRTUAL_FRAME_JOINTS or (
            profile == POSITION_ONLY_PROFILE and name in PHYSICAL_GIMBAL_JOINTS
        ):
            _lock_joint(joint)
        base_root.append(joint)
        copied_joints.append(joint)

    retained_non_base_mass = sum(
        _link_mass(link)
        for link in base_root.findall("link")
        if link.attrib["name"] != "base_link"
    )
    target_base_mass = TARGET_TOTAL_MASS_KG - retained_non_base_mass
    if target_base_mass <= 0.0:
        raise RuntimeError(f"invalid redistributed base mass: {target_base_mass}")
    old_base_mass, inertia_scale = _set_base_mass(base_root, target_base_mass)

    ET.indent(base_tree, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    xml_bytes = ET.tostring(base_root, encoding="utf-8", xml_declaration=True)
    output_path.write_bytes(xml_bytes.replace(b"\r\n", b"\n"))
    validation = _validate_tree(base_root, profile)
    result = {
        "schema": "recomo_two_wheel_whole_body_urdf_build_v3",
        "profile": profile,
        "base_source": _audit_path(base_path),
        "arm_source": _audit_path(arm_path),
        "arm_source_sha256": arm_sha256,
        "output": _audit_path(output_path),
        "target_total_mass_kg": TARGET_TOTAL_MASS_KG,
        "redistributed_base_mass_kg": round(target_base_mass, 9),
        "previous_aggregate_base_mass_kg": round(old_base_mass, 9),
        "base_inertia_scale": round(inertia_scale, 14),
        "arm_mount_xyz_m": [0.0, 0.0, 0.880],
        "home_v0_rad": {
            "joint6_arm_yaw": 0.0,
            "joint5_arm_pitch": 1.5707963268,
            "joint4_elbow_pitch": 2.3561944902,
        },
        "physical_gimbal_home_rad": {name: 0.0 for name in PHYSICAL_GIMBAL_JOINTS},
        "physical_gimbal_joint_mode": (
            "fixed"
            if profile == POSITION_ONLY_PROFILE
            else "internal_sim_attitude_adapter"
        ),
        "learned_physical_gimbal_joint_action": False,
        "virtual_frame_joints_locked_rad": {name: 0.0 for name in VIRTUAL_FRAME_JOINTS},
        "control_boundary": (
            "wheel effort plus three physical arm joints; semantic DFR attitude is "
            "a separate command and physical gimbal joints are never learned labels"
        ),
        "position_target_link": "ee1_tool",
        "physical_camera_observation_link": "cam_link",
        "validation": validation,
        "passed": True,
    }
    audit_path = output_path.with_name("build_audit.json")
    audit_path.write_bytes((json.dumps(result, indent=2) + "\n").encode("utf-8"))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--arm", type=Path, default=DEFAULT_ARM)
    parser.add_argument(
        "--profile",
        choices=(POSITION_ONLY_PROFILE, SPLIT_ATTITUDE_PROFILE),
        default=POSITION_ONLY_PROFILE,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or (
        DEFAULT_ATTITUDE_OUTPUT
        if args.profile == SPLIT_ATTITUDE_PROFILE
        else DEFAULT_OUTPUT
    )
    result = build(
        args.base.resolve(), args.arm.resolve(), output.resolve(), args.profile
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
