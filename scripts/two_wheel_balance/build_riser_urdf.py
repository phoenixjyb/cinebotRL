#!/usr/bin/env python3
"""Build the isolated 28 kg two-wheel riser-camera URDF."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = (
    PROJECT_ROOT
    / "assets_own/recomoProto2_two_wheel_balance/recomoProto2_two_wheel_balance.urdf"
)
DEFAULT_GIMBAL = (
    PROJECT_ROOT / "assets_own/sources/recomoProto2-1190_moveit_aa463a.urdf"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "assets_own/recomoProto2_two_wheel_riser/recomoProto2_two_wheel_riser.urdf"
)

EXPECTED_GIMBAL_SHA256 = (
    "aa463a14d84cc5718335f91de7091a49674ec66f8de016cb69d8190f7d98db77"
)
TARGET_TOTAL_MASS_KG = 28.0
RISER_LOWER_M = 0.0
RISER_UPPER_M = 1.2
RISER_HOME_M = 0.3
RISER_SPEED_MPS = 1.0
RISER_FORCE_N = 300.0
CAMERA_MIN_HEIGHT_M = 0.6
CAMERA_MAX_HEIGHT_M = 1.8
CAMERA_ZERO_OFFSET_Z_M = 0.02646943433666392
RISER_JOINT_ORIGIN_Z_M = CAMERA_MIN_HEIGHT_M - CAMERA_ZERO_OFFSET_Z_M
# Fixed chassis-to-semantic-DFR zero-command basis fitted from all 21,017
# samples in the corrected accepted 62-case stage with base yaw capped at
# 0.25 rad/s. URDF rpy uses Rz(yaw) * Ry(pitch) * Rx(roll).
RS4_BODY_BASIS_RPY_RAD = (-1.75035342, 1.40878912, -2.99569687)
RS4_BODY_BASIS_QUAT_XYZW = (
    0.37126688383868894,
    0.6136484680880162,
    -0.4508086827044413,
    0.5313830917297113,
)
RS4_HARD_RATE_RAD_S = 2.0 * math.pi
RS4_PROXY_JOINT_LIMITS_RAD = {
    # Proxy order implements Rz(pitch) * Ry(roll) * Rx(yaw).
    "joint3_gimbal_yaw": (math.radians(-112.0), math.radians(214.0)),
    "joint2_gimbal_roll": (math.radians(-95.0), math.radians(240.0)),
    "joint1_gimbal_pitch": (-math.pi, math.pi),
}
RS4_CONTINUOUS_PROXY_JOINT = "joint1_gimbal_pitch"
RS4_PROXY_JOINT_COMMAND_MAPPING = {
    "joint3_gimbal_yaw": "ronin_pitch_from_rot_z",
    "joint2_gimbal_roll": "ronin_roll_from_rot_y",
    "joint1_gimbal_pitch": "ronin_yaw_from_rot_x",
}
RISER_GUIDE_MASS_KG = 1.5
RISER_CARRIAGE_MASS_KG = 1.5

ARM_JOINTS = (
    "joint6_arm_yaw",
    "joint5_arm_pitch",
    "joint4_elbow_pitch",
)
PHYSICAL_GIMBAL_JOINTS = (
    "joint3_gimbal_yaw",
    "joint2_gimbal_roll",
    "joint1_gimbal_pitch",
)
GIMBAL_LINKS = (
    "arm_link_4",
    "arm_link_5",
    "arm_link_6",
    "cam_link",
    "ee_tool",
)
GIMBAL_JOINTS = (
    *PHYSICAL_GIMBAL_JOINTS,
    "camera_optical_center",
    "ee_tool_fixed",
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
            # Source CAD meshes are millimetres, while all URDF kinematics are
            # metres. Keep this scale local to mesh geometry so the importer
            # cannot shrink prismatic travel and joint origins.
            mesh.attrib["scale"] = "0.001 0.001 0.001"


def _rpy_matrix(rpy: tuple[float, float, float] | np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    return np.array(
        [
            [
                math.cos(yaw) * math.cos(pitch),
                math.cos(yaw) * math.sin(pitch) * math.sin(roll)
                - math.sin(yaw) * math.cos(roll),
                math.cos(yaw) * math.sin(pitch) * math.cos(roll)
                + math.sin(yaw) * math.sin(roll),
            ],
            [
                math.sin(yaw) * math.cos(pitch),
                math.sin(yaw) * math.sin(pitch) * math.sin(roll)
                + math.cos(yaw) * math.cos(roll),
                math.sin(yaw) * math.sin(pitch) * math.cos(roll)
                - math.cos(yaw) * math.sin(roll),
            ],
            [
                -math.sin(pitch),
                math.cos(pitch) * math.sin(roll),
                math.cos(pitch) * math.cos(roll),
            ],
        ]
    )


def _matrix_rpy(rotation: np.ndarray) -> tuple[float, float, float]:
    pitch = math.atan2(
        -float(rotation[2, 0]),
        math.hypot(float(rotation[0, 0]), float(rotation[1, 0])),
    )
    roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
    yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    return roll, pitch, yaw


def _origin_rpy(joint: ET.Element) -> np.ndarray:
    return _rpy_matrix(
        np.fromstring(joint.find("origin").attrib.get("rpy", "0 0 0"), sep=" ")
    )


def _origin_xyz(joint: ET.Element) -> np.ndarray:
    return np.fromstring(joint.find("origin").attrib.get("xyz", "0 0 0"), sep=" ")


def _format_vector(values: np.ndarray | tuple[float, ...]) -> str:
    return " ".join(f"{float(value):.12f}" for value in values)


def _configure_rs4_proxy_chain(joints: dict[str, ET.Element]) -> np.ndarray:
    """Make legacy-named sim joints implement the deployed attitude mapping."""

    first = joints["joint3_gimbal_yaw"]
    second = joints["joint2_gimbal_roll"]
    third = joints["joint1_gimbal_pitch"]
    camera = joints["camera_optical_center"]
    r2 = _origin_rpy(second)
    r3 = _origin_rpy(third)
    r23 = r2 @ r3
    rz_cam = _rpy_matrix((0.0, 0.0, math.pi / 2.0))

    first.find("origin").attrib["rpy"] = _format_vector(RS4_BODY_BASIS_RPY_RAD)
    first.find("axis").attrib["xyz"] = "0 0 1"
    second.find("axis").attrib["xyz"] = _format_vector(r2.T @ np.array([0.0, 1.0, 0.0]))
    third.find("axis").attrib["xyz"] = _format_vector(r23.T @ np.array([1.0, 0.0, 0.0]))
    camera.find("origin").attrib["rpy"] = _format_vector(_matrix_rpy(r23.T @ rz_cam))

    for name in PHYSICAL_GIMBAL_JOINTS:
        lower, upper = RS4_PROXY_JOINT_LIMITS_RAD[name]
        limit = joints[name].find("limit")
        if name == RS4_CONTINUOUS_PROXY_JOINT:
            # The DJI yaw command wraps at +/-pi. A continuous simulation
            # coordinate lets the position servo follow the nearest equivalent
            # target instead of attempting a nearly 2*pi reversal.
            joints[name].attrib["type"] = "continuous"
            limit.attrib.pop("lower", None)
            limit.attrib.pop("upper", None)
        else:
            limit.attrib.update(lower=f"{lower:.12f}", upper=f"{upper:.12f}")
        limit.attrib["velocity"] = f"{RS4_HARD_RATE_RAD_S:.12f}"

    # Center the zero-command optical center on the riser axis. Joint
    # translations remain from CAD; only command-frame rotations are adapted.
    p_down = (
        _origin_xyz(second)
        + r2 @ _origin_xyz(third)
        + r23 @ _origin_xyz(camera)
    )
    basis = _rpy_matrix(RS4_BODY_BASIS_RPY_RAD)
    mount_xyz = np.array([0.0, 0.0, CAMERA_ZERO_OFFSET_Z_M]) - basis @ p_down
    first.find("origin").attrib["xyz"] = _format_vector(mount_xyz)
    return mount_xyz


def _inertial(mass_kg: float, size_xyz: tuple[float, float, float]) -> ET.Element:
    x, y, z = size_xyz
    inertial = ET.Element("inertial")
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": f"{mass_kg:.9f}"})
    scale = mass_kg / 12.0
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": f"{scale * (y * y + z * z):.9f}",
            "ixy": "0",
            "ixz": "0",
            "iyy": f"{scale * (x * x + z * z):.9f}",
            "iyz": "0",
            "izz": f"{scale * (x * x + y * y):.9f}",
        },
    )
    return inertial


def _primitive_link(
    name: str,
    mass_kg: float,
    size_xyz: tuple[float, float, float],
    material: str,
) -> ET.Element:
    link = ET.Element("link", {"name": name})
    link.append(_inertial(mass_kg, size_xyz))
    for tag in ("visual", "collision"):
        geometry_owner = ET.SubElement(link, tag)
        geometry = ET.SubElement(geometry_owner, "geometry")
        ET.SubElement(geometry, "box", {"size": " ".join(map(str, size_xyz))})
        if tag == "visual":
            ET.SubElement(geometry_owner, "material", {"name": material})
    return link


def _semantic_tool_link() -> ET.Element:
    link = ET.Element("link", {"name": "ee1_tool"})
    link.append(_inertial(0.001, (0.01, 0.01, 0.01)))
    visual = ET.SubElement(link, "visual")
    geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(geometry, "sphere", {"radius": "0.015"})
    material = ET.SubElement(visual, "material", {"name": "semantic_green"})
    ET.SubElement(material, "color", {"rgba": "0.1 0.9 0.2 1"})
    return link


def _set_base_mass(root: ET.Element, target_mass: float) -> tuple[float, float]:
    base_link = _named(root, "link", "base_link")
    mass = base_link.find("./inertial/mass")
    inertia = base_link.find("./inertial/inertia")
    if mass is None or inertia is None:
        raise RuntimeError("base_link needs explicit mass and inertia")
    old_mass = float(mass.attrib["value"])
    scale = target_mass / old_mass
    mass.attrib["value"] = f"{target_mass:.9f}"
    for key in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
        inertia.attrib[key] = f"{float(inertia.attrib[key]) * scale:.9f}"
    return old_mass, scale


def _validate_tree(root: ET.Element) -> dict[str, object]:
    links = root.findall("link")
    joints = root.findall("joint")
    link_names = [item.attrib["name"] for item in links]
    joint_names = [item.attrib["name"] for item in joints]
    children = [joint.find("child").attrib["link"] for joint in joints]
    roots = sorted(set(link_names) - set(children))
    movable = tuple(
        joint.attrib["name"]
        for joint in joints
        if joint.attrib["type"] not in ("fixed", "continuous")
    )
    continuous = tuple(
        joint.attrib["name"] for joint in joints if joint.attrib["type"] == "continuous"
    )
    total_mass = sum(_link_mass(link) for link in links)
    riser = _named(root, "joint", "riser_joint")
    riser_limit = riser.find("limit")
    checks = {
        "unique_links": len(link_names) == len(set(link_names)),
        "unique_joints": len(joint_names) == len(set(joint_names)),
        "single_base_link_root": roots == ["base_link"],
        "total_mass_28kg": abs(total_mass - TARGET_TOTAL_MASS_KG) < 1e-6,
        "continuous_joint_contract": continuous
        == ("left_wheel_joint", "right_wheel_joint", RS4_CONTINUOUS_PROXY_JOINT),
        "riser_and_gimbal_joints_present": movable
        == ("riser_joint", *PHYSICAL_GIMBAL_JOINTS[:2]),
        "arm_joints_absent": not set(ARM_JOINTS) & set(joint_names),
        "no_planar_base_joints": not {
            "base_joint_vx",
            "base_joint_vy",
            "base_joint_wz",
        }
        & set(joint_names),
        "physical_and_semantic_camera_present": {
            "cam_link",
            "ee_tool",
            "ee1_tool",
        }
        <= set(link_names),
        "riser_prismatic": riser.attrib["type"] == "prismatic",
        "riser_range_1p2m": riser_limit is not None
        and float(riser_limit.attrib["lower"]) == RISER_LOWER_M
        and float(riser_limit.attrib["upper"]) == RISER_UPPER_M,
        "riser_speed_1mps": riser_limit is not None
        and float(riser_limit.attrib["velocity"]) == RISER_SPEED_MPS,
    }
    if not all(checks.values()):
        raise RuntimeError(f"riser URDF contract failed: {checks}")
    return {
        "link_count": len(links),
        "joint_count": len(joints),
        "root_links": roots,
        "movable_joints": [*continuous, *movable],
        "total_mass_kg": round(total_mass, 9),
        "checks": checks,
    }


def build(base_path: Path, gimbal_path: Path, output_path: Path) -> dict[str, object]:
    gimbal_sha256 = _sha256(gimbal_path)
    if gimbal_sha256 != EXPECTED_GIMBAL_SHA256:
        raise RuntimeError(
            "gimbal source hash changed: expected "
            f"{EXPECTED_GIMBAL_SHA256}, got {gimbal_sha256}"
        )

    base_tree = ET.parse(base_path)
    root = base_tree.getroot()
    gimbal_root = ET.parse(gimbal_path).getroot()
    root.attrib["name"] = "recomoProto2-two-wheel-riser"

    root.remove(_named(root, "joint", "arm_mount_joint"))
    root.remove(_named(root, "link", "arm_mount_link"))

    root.append(
        _primitive_link(
            "riser_guide_link", RISER_GUIDE_MASS_KG, (0.12, 0.12, 0.8), "tower_gray"
        )
    )
    guide_joint = ET.Element("joint", {"name": "riser_guide_joint", "type": "fixed"})
    ET.SubElement(guide_joint, "origin", {"xyz": "0 0 0.9", "rpy": "0 0 0"})
    ET.SubElement(guide_joint, "parent", {"link": "base_link"})
    ET.SubElement(guide_joint, "child", {"link": "riser_guide_link"})
    root.append(guide_joint)

    root.append(
        _primitive_link(
            "riser_carriage_link",
            RISER_CARRIAGE_MASS_KG,
            (0.24, 0.24, 0.12),
            "chassis_dark",
        )
    )
    riser_joint = ET.Element("joint", {"name": "riser_joint", "type": "prismatic"})
    ET.SubElement(
        riser_joint,
        "origin",
        {"xyz": f"0 0 {RISER_JOINT_ORIGIN_Z_M:.9f}", "rpy": "0 0 0"},
    )
    ET.SubElement(riser_joint, "parent", {"link": "base_link"})
    ET.SubElement(riser_joint, "child", {"link": "riser_carriage_link"})
    ET.SubElement(riser_joint, "axis", {"xyz": "0 0 1"})
    ET.SubElement(
        riser_joint,
        "limit",
        {
            "lower": f"{RISER_LOWER_M:.1f}",
            "upper": f"{RISER_UPPER_M:.1f}",
            "effort": f"{RISER_FORCE_N:.1f}",
            "velocity": f"{RISER_SPEED_MPS:.1f}",
        },
    )
    ET.SubElement(riser_joint, "dynamics", {"damping": "2.0", "friction": "1.0"})
    root.append(riser_joint)

    for name in GIMBAL_LINKS:
        link = deepcopy(_named(gimbal_root, "link", name))
        _remove_empty_geometry(link)
        _rewrite_mesh_paths(link)
        root.append(link)

    copied_joints = {
        name: deepcopy(_named(gimbal_root, "joint", name)) for name in GIMBAL_JOINTS
    }
    mount_xyz = _configure_rs4_proxy_chain(copied_joints)
    copied_joints["joint3_gimbal_yaw"].find("parent").attrib[
        "link"
    ] = "riser_carriage_link"
    for name in GIMBAL_JOINTS:
        root.append(copied_joints[name])

    root.append(_semantic_tool_link())
    semantic_joint = ET.Element(
        "joint", {"name": "semantic_dfr_from_cam_fixed", "type": "fixed"}
    )
    ET.SubElement(
        semantic_joint,
        "origin",
        {"xyz": "0 0 0", "rpy": f"0 0 {-math.pi / 2.0:.12f}"},
    )
    ET.SubElement(semantic_joint, "parent", {"link": "cam_link"})
    ET.SubElement(semantic_joint, "child", {"link": "ee1_tool"})
    root.append(semantic_joint)

    retained_non_base_mass = sum(
        _link_mass(link) for link in root.findall("link") if link.attrib["name"] != "base_link"
    )
    target_base_mass = TARGET_TOTAL_MASS_KG - retained_non_base_mass
    if target_base_mass <= 0.0:
        raise RuntimeError(f"invalid redistributed base mass: {target_base_mass}")
    old_base_mass, inertia_scale = _set_base_mass(root, target_base_mass)

    ET.indent(base_tree, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    output_path.write_bytes(xml_bytes.replace(b"\r\n", b"\n"))
    validation = _validate_tree(root)

    moving_mass = RISER_CARRIAGE_MASS_KG + sum(
        _link_mass(_named(root, "link", name)) for name in GIMBAL_LINKS
    ) + _link_mass(_named(root, "link", "ee1_tool"))
    result = {
        "schema": "recomo_two_wheel_riser_urdf_build_v1",
        "base_source": _audit_path(base_path),
        "gimbal_source": _audit_path(gimbal_path),
        "gimbal_source_sha256": gimbal_sha256,
        "output": _audit_path(output_path),
        "target_total_mass_kg": TARGET_TOTAL_MASS_KG,
        "redistributed_base_mass_kg": round(target_base_mass, 9),
        "previous_aggregate_base_mass_kg": round(old_base_mass, 9),
        "base_inertia_scale": round(inertia_scale, 14),
        "provisional_moving_mass_kg": round(moving_mass, 9),
        "fixed_gimbal_mount": {
            "contract": "accepted62_rs4_semantic_body_basis_yaw025_v1",
            "source_case_count": 62,
            "source_sample_count": 21017,
            "maximum_reference_base_yaw_rate_rad_s": 0.25,
            "semantic_body_basis_quat_xyzw": list(RS4_BODY_BASIS_QUAT_XYZW),
            "joint3_origin_xyz_m": mount_xyz.tolist(),
            "joint3_origin_rpy_rad": list(RS4_BODY_BASIS_RPY_RAD),
        },
        "riser": {
            "joint": "riser_joint",
            "stroke_m": RISER_UPPER_M - RISER_LOWER_M,
            "home_m": RISER_HOME_M,
            "speed_limit_mps": RISER_SPEED_MPS,
            "force_limit_n": RISER_FORCE_N,
            "camera_height_range_m": [CAMERA_MIN_HEIGHT_M, CAMERA_MAX_HEIGHT_M],
        },
        "removed_arm_joints": list(ARM_JOINTS),
        "physical_gimbal_joints": list(PHYSICAL_GIMBAL_JOINTS),
        "sim_gimbal_joint_semantics": "rs4_attitude_proxy_not_motor_shaft_angles",
        "sim_gimbal_joint_command_mapping": RS4_PROXY_JOINT_COMMAND_MAPPING,
        "continuous_yaw_proxy_joint": RS4_CONTINUOUS_PROXY_JOINT,
        "continuous_yaw_proxy_reason": "avoid_wrapped_position_target_long_way_rotation",
        "hardware_yaw_command_envelope_rad": [-math.pi, math.pi],
        "learned_physical_gimbal_joint_action": False,
        "position_target_and_observation_link": "cam_link",
        "semantic_attitude_target_link": "ee1_tool",
        "semantic_frame_contract": "R_world_cam = R_world_DFR * Rz(+pi/2)",
        "usd_conversion": {
            "mesh_scale_is_explicit_in_urdf": True,
            "converter_mesh_scale_argument": 1.0,
        },
        "validation": validation,
        "passed": True,
    }
    output_path.with_name("build_audit.json").write_bytes(
        (json.dumps(result, indent=2) + "\n").encode("utf-8")
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--gimbal", type=Path, default=DEFAULT_GIMBAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(args.base.resolve(), args.gimbal.resolve(), args.output.resolve())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
