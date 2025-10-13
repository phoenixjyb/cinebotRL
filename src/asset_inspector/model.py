"""Data structures and math helpers for URDF asset inspection."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, sin
from typing import Dict, Iterable, List, Optional, Tuple


Vector3 = Tuple[float, float, float]
Matrix4 = Tuple[Tuple[float, float, float, float],
                Tuple[float, float, float, float],
                Tuple[float, float, float, float],
                Tuple[float, float, float, float]]


@dataclass(frozen=True)
class Origin:
    translation: Vector3 = (0.0, 0.0, 0.0)
    rpy: Vector3 = (0.0, 0.0, 0.0)

    def as_matrix(self) -> Matrix4:
        return compose_transform(self.translation, self.rpy)


@dataclass(frozen=True)
class Inertial:
    origin: Origin
    mass: float
    inertia: Tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class Mesh:
    filename: str
    scale: Optional[Vector3] = None


@dataclass(frozen=True)
class Visual:
    origin: Origin
    mesh: Optional[Mesh] = None
    material_rgba: Optional[Tuple[float, float, float, float]] = None


@dataclass(frozen=True)
class Collision:
    origin: Origin
    mesh: Optional[Mesh] = None


@dataclass(frozen=True)
class URDFLink:
    name: str
    inertial: Optional[Inertial] = None
    visuals: List[Visual] = field(default_factory=list)
    collisions: List[Collision] = field(default_factory=list)


@dataclass(frozen=True)
class JointLimit:
    lower: Optional[float] = None
    upper: Optional[float] = None
    effort: Optional[float] = None
    velocity: Optional[float] = None


@dataclass(frozen=True)
class URDFJoint:
    name: str
    type: str
    parent: str
    child: str
    origin: Origin
    axis: Vector3
    limit: Optional[JointLimit] = None


@dataclass(frozen=True)
class URDFModel:
    name: str
    links: Dict[str, URDFLink]
    joints: Dict[str, URDFJoint]

    def children_for(self, parent_link: str) -> List[URDFJoint]:
        return [joint for joint in self.joints.values() if joint.parent == parent_link]

    def find_root_link(self) -> Optional[str]:
        child_links = {joint.child for joint in self.joints.values()}
        candidates = [name for name in self.links if name not in child_links]
        if not candidates:
            return None
        # Prefer chassis/base naming if ambiguous.
        for preferred in ("base_link", "chassis", "chassis_link", "chassis_center_link"):
            for name in candidates:
                if preferred in name:
                    return name
        return candidates[0]


IDENTITY_MATRIX: Matrix4 = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def compose_transform(translation: Vector3, rpy: Vector3) -> Matrix4:
    roll, pitch, yaw = rpy
    cx, cy, cz = cos(roll), cos(pitch), cos(yaw)
    sx, sy, sz = sin(roll), sin(pitch), sin(yaw)

    rot = (
        (cy * cz, cz * sx * sy - cx * sz, sx * sz + cx * cz * sy),
        (cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx),
        (-sy, cy * sx, cx * cy),
    )

    x, y, z = translation
    return (
        (rot[0][0], rot[0][1], rot[0][2], x),
        (rot[1][0], rot[1][1], rot[1][2], y),
        (rot[2][0], rot[2][1], rot[2][2], z),
        (0.0, 0.0, 0.0, 1.0),
    )


def matmul(a: Matrix4, b: Matrix4) -> Matrix4:
    def dot(row: Iterable[float], col: Iterable[float]) -> float:
        return sum(i * j for i, j in zip(row, col))

    cols = list(zip(*b))
    data = []
    for row in a:
        data.append(tuple(dot(row, col) for col in cols))
    return tuple(data)  # type: ignore[return-value]


def axis_angle_transform(axis: Vector3, angle: float) -> Matrix4:
    x, y, z = axis
    norm = (x * x + y * y + z * z) ** 0.5
    if norm == 0.0:
        return IDENTITY_MATRIX
    x /= norm
    y /= norm
    z /= norm
    c = cos(angle)
    s = sin(angle)
    C = 1.0 - c
    rot = (
        (c + x * x * C, x * y * C - z * s, x * z * C + y * s),
        (y * x * C + z * s, c + y * y * C, y * z * C - x * s),
        (z * x * C - y * s, z * y * C + x * s, c + z * z * C),
    )
    return (
        (rot[0][0], rot[0][1], rot[0][2], 0.0),
        (rot[1][0], rot[1][1], rot[1][2], 0.0),
        (rot[2][0], rot[2][1], rot[2][2], 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def prismatic_transform(axis: Vector3, displacement: float) -> Matrix4:
    x, y, z = axis
    norm = (x * x + y * y + z * z) ** 0.5
    if norm == 0.0:
        return IDENTITY_MATRIX
    scale = displacement / norm
    return (
        (1.0, 0.0, 0.0, x * scale),
        (0.0, 1.0, 0.0, y * scale),
        (0.0, 0.0, 1.0, z * scale),
        (0.0, 0.0, 0.0, 1.0),
    )


def compute_joint_transform(joint: URDFJoint, position: float = 0.0) -> Matrix4:
    """Compose origin transform with motion transform at the given joint position."""

    base = joint.origin.as_matrix()
    if joint.type in {"fixed", "unknown"}:
        return base
    if joint.type in {"revolute", "continuous"}:
        return matmul(base, axis_angle_transform(joint.axis, position))
    if joint.type == "prismatic":
        return matmul(base, prismatic_transform(joint.axis, position))
    if joint.type == "planar":
        # Planar joints are rare; treat as translation in joint.axis XY plane with position as scalar.
        return matmul(base, prismatic_transform(joint.axis, position))
    return base


def pretty_matrix(matrix: Matrix4) -> str:
    return "[" + "; ".join("({:.4f},{:.4f},{:.4f},{:.4f})".format(*row) for row in matrix) + "]"

