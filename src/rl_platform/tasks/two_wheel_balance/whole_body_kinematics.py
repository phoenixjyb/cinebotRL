"""Pure NumPy position kinematics for bounded two-wheel retargeting audits."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np


ARM_JOINTS = ("joint6_arm_yaw", "joint5_arm_pitch", "joint4_elbow_pitch")


def _rpy_matrix(values: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = values
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _transform(xyz: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    value = np.eye(4)
    value[:3, :3] = rotation
    value[:3, 3] = xyz
    return value


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    cross = 1.0 - c
    return np.array(
        [
            [c + x * x * cross, x * y * cross - z * s, x * z * cross + y * s],
            [y * x * cross + z * s, c + y * y * cross, y * z * cross - x * s],
            [z * x * cross - y * s, z * y * cross + x * s, c + z * z * cross],
        ]
    )


@dataclass(frozen=True)
class ChainJoint:
    name: str
    joint_type: str
    origin: np.ndarray
    axis: np.ndarray
    child_link: str
    child_mass_kg: float
    child_com: np.ndarray


@dataclass(frozen=True)
class TreeJoint:
    name: str
    joint_type: str
    child_link: str
    origin: np.ndarray
    axis: np.ndarray
    child_mass_kg: float
    child_com: np.ndarray


class UrdfPositionKinematics:
    """Evaluate one URDF link position from planar base plus three arm joints."""

    def __init__(self, urdf: Path, target_link: str = "ee1_tool") -> None:
        root = ET.parse(urdf).getroot()
        inertials = {}
        for link_element in root.findall("link"):
            inertial = link_element.find("inertial")
            if inertial is None:
                inertials[link_element.attrib["name"]] = (0.0, np.eye(4))
                continue
            mass = float(inertial.find("mass").attrib["value"])
            origin = inertial.find("origin")
            xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
            rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
            inertials[link_element.attrib["name"]] = (
                mass,
                _transform(xyz, _rpy_matrix(rpy)),
            )
        by_child = {}
        tree_by_parent: dict[str, list[TreeJoint]] = {}
        for joint in root.findall("joint"):
            child = joint.find("child").attrib["link"]
            if child in by_child:
                raise ValueError(f"multiple parent joints for {child}")
            by_child[child] = joint
            parent = joint.find("parent").attrib["link"]
            origin = joint.find("origin")
            xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
            rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
            axis_element = joint.find("axis")
            axis = (
                np.fromstring(axis_element.attrib["xyz"], sep=" ")
                if axis_element is not None
                else np.array([0.0, 0.0, 1.0])
            )
            tree_by_parent.setdefault(parent, []).append(
                TreeJoint(
                    name=joint.attrib["name"],
                    joint_type=joint.attrib["type"],
                    child_link=child,
                    origin=_transform(xyz, _rpy_matrix(rpy)),
                    axis=axis,
                    child_mass_kg=inertials[child][0],
                    child_com=inertials[child][1],
                )
            )
        self._tree_by_parent = {
            parent: tuple(joints) for parent, joints in tree_by_parent.items()
        }

        reverse_chain = []
        link = target_link
        while link != "base_link":
            if link not in by_child:
                raise ValueError(f"cannot trace {target_link} to base_link from {link}")
            joint = by_child[link]
            origin = joint.find("origin")
            xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
            rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
            axis_element = joint.find("axis")
            axis = (
                np.fromstring(axis_element.attrib["xyz"], sep=" ")
                if axis_element is not None
                else np.array([0.0, 0.0, 1.0])
            )
            reverse_chain.append(
                ChainJoint(
                    name=joint.attrib["name"],
                    joint_type=joint.attrib["type"],
                    origin=_transform(xyz, _rpy_matrix(rpy)),
                    axis=axis,
                    child_link=link,
                    child_mass_kg=inertials[link][0],
                    child_com=inertials[link][1],
                )
            )
            link = joint.find("parent").attrib["link"]
        self.chain = tuple(reversed(reverse_chain))
        movable = tuple(joint.name for joint in self.chain if joint.joint_type != "fixed")
        if movable != ARM_JOINTS:
            raise ValueError(f"unexpected movable chain to {target_link}: {movable}")
        self.target_link = target_link

        joint_limits = {}
        for joint in root.findall("joint"):
            if joint.attrib["name"] not in ARM_JOINTS:
                continue
            limit = joint.find("limit")
            joint_limits[joint.attrib["name"]] = (
                float(limit.attrib["lower"]),
                float(limit.attrib["upper"]),
            )
        self.arm_lower = np.array([joint_limits[name][0] for name in ARM_JOINTS])
        self.arm_upper = np.array([joint_limits[name][1] for name in ARM_JOINTS])

    def transform(self, base_arm_q: np.ndarray) -> np.ndarray:
        base_arm_q = np.asarray(base_arm_q, dtype=np.float64)
        if base_arm_q.shape != (6,) or not np.isfinite(base_arm_q).all():
            raise ValueError(f"expected finite base/arm q shape (6,), got {base_arm_q.shape}")
        x, y, yaw = base_arm_q[:3]
        value = _transform(np.array([x, y, 0.0]), _rpy_matrix(np.array([0.0, 0.0, yaw])))
        arm = dict(zip(ARM_JOINTS, base_arm_q[3:], strict=True))
        for joint in self.chain:
            value = value @ joint.origin
            if joint.joint_type != "fixed":
                value = value @ _transform(
                    np.zeros(3), _axis_rotation(joint.axis, arm[joint.name])
                )
        return value

    def position(self, base_arm_q: np.ndarray) -> np.ndarray:
        return self.transform(base_arm_q)[:3, 3]

    def gravitational_effort_nm(
        self,
        base_arm_q: np.ndarray,
        gravity_mps2: float = 9.81,
        epsilon: float = 1e-5,
    ) -> np.ndarray:
        base_arm_q = np.asarray(base_arm_q, dtype=np.float64)
        if base_arm_q.shape != (6,) or gravity_mps2 <= 0.0 or epsilon <= 0.0:
            raise ValueError("invalid state, gravity, or finite-difference epsilon")

        def potential_energy(state: np.ndarray) -> float:
            x, y, yaw = state[:3]
            value = _transform(
                np.array([x, y, 0.0]),
                _rpy_matrix(np.array([0.0, 0.0, yaw])),
            )
            arm = dict(zip(ARM_JOINTS, state[3:], strict=True))

            def subtree(link: str, parent_transform: np.ndarray) -> float:
                energy = 0.0
                for joint in self._tree_by_parent.get(link, ()):
                    child_transform = parent_transform @ joint.origin
                    if joint.joint_type != "fixed":
                        angle = arm.get(joint.name, 0.0)
                        child_transform = child_transform @ _transform(
                            np.zeros(3), _axis_rotation(joint.axis, angle)
                        )
                    if joint.child_mass_kg > 0.0:
                        com_position = (child_transform @ joint.child_com)[:3, 3]
                        energy += (
                            joint.child_mass_kg * gravity_mps2 * com_position[2]
                        )
                    energy += subtree(joint.child_link, child_transform)
                return energy

            return subtree("base_link", value)

        effort = np.empty(3, dtype=np.float64)
        for index in range(3):
            plus = base_arm_q.copy()
            minus = base_arm_q.copy()
            plus[index + 3] += epsilon
            minus[index + 3] -= epsilon
            effort[index] = (
                potential_energy(plus) - potential_energy(minus)
            ) / (2.0 * epsilon)
        return effort


def integrate_unicycle(base_q: np.ndarray, velocity: float, yaw_rate: float, dt: float) -> np.ndarray:
    base_q = np.asarray(base_q, dtype=np.float64)
    if base_q.shape != (3,) or dt <= 0.0:
        raise ValueError("invalid base state or timestep")
    x, y, yaw = base_q
    delta_yaw = yaw_rate * dt
    if abs(yaw_rate) < 1e-9:
        x += velocity * dt * math.cos(yaw)
        y += velocity * dt * math.sin(yaw)
    else:
        x += velocity / yaw_rate * (math.sin(yaw + delta_yaw) - math.sin(yaw))
        y -= velocity / yaw_rate * (math.cos(yaw + delta_yaw) - math.cos(yaw))
    return np.array([x, y, yaw + delta_yaw])
