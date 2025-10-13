"""URDF parser tailored for cinebotRL asset inspection."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from .model import (
    Collision,
    Inertial,
    JointLimit,
    Mesh,
    Origin,
    URDFJoint,
    URDFLink,
    URDFModel,
    Visual,
)


def _parse_float_tuple(text: Optional[str], *, length: int = 3) -> Tuple[float, ...]:
    if not text:
        return tuple(0.0 for _ in range(length))
    parts = [float(value) for value in text.split()]
    if len(parts) != length:
        raise ValueError(f"Expected {length} components, got {len(parts)} for '{text}'")
    return tuple(parts)


def _parse_origin(element: Optional[ET.Element]) -> Origin:
    if element is None:
        return Origin()
    xyz = _parse_float_tuple(element.get("xyz"))
    rpy = _parse_float_tuple(element.get("rpy"))
    return Origin(translation=xyz, rpy=rpy)


def _parse_inertial(element: Optional[ET.Element]) -> Optional[Inertial]:
    if element is None:
        return None
    origin = _parse_origin(element.find("origin"))
    mass_elem = element.find("mass")
    inertia_elem = element.find("inertia")
    if mass_elem is None or inertia_elem is None:
        return None
    mass = float(mass_elem.get("value", "0"))
    inertia = (
        float(inertia_elem.get("ixx", "0")),
        float(inertia_elem.get("iyy", "0")),
        float(inertia_elem.get("izz", "0")),
        float(inertia_elem.get("ixy", "0")),
        float(inertia_elem.get("ixz", "0")),
        float(inertia_elem.get("iyz", "0")),
    )
    return Inertial(origin=origin, mass=mass, inertia=inertia)


def _parse_mesh(element: Optional[ET.Element]) -> Optional[Mesh]:
    if element is None:
        return None
    filename = element.get("filename")
    if not filename:
        return None
    scale = element.get("scale")
    scale_tuple = _parse_float_tuple(scale) if scale else None
    return Mesh(filename=filename, scale=scale_tuple)  # type: ignore[arg-type]


def _parse_material_rgba(element: Optional[ET.Element]) -> Optional[Tuple[float, float, float, float]]:
    if element is None:
        return None
    color_elem = element.find("color")
    if color_elem is None:
        return None
    rgba = _parse_float_tuple(color_elem.get("rgba"), length=4)
    return rgba  # type: ignore[return-value]


def _parse_visuals(parent: ET.Element) -> Tuple[Visual, ...]:
    visuals = []
    for visual_elem in parent.findall("visual"):
        origin = _parse_origin(visual_elem.find("origin"))
        geometry_elem = visual_elem.find("geometry")
        mesh = _parse_mesh(geometry_elem.find("mesh") if geometry_elem is not None else None)
        material = _parse_material_rgba(visual_elem.find("material"))
        visuals.append(Visual(origin=origin, mesh=mesh, material_rgba=material))
    return tuple(visuals)


def _parse_collisions(parent: ET.Element) -> Tuple[Collision, ...]:
    collisions = []
    for col_elem in parent.findall("collision"):
        origin = _parse_origin(col_elem.find("origin"))
        geometry_elem = col_elem.find("geometry")
        mesh = _parse_mesh(geometry_elem.find("mesh") if geometry_elem is not None else None)
        collisions.append(Collision(origin=origin, mesh=mesh))
    return tuple(collisions)


def _default_axis(joint_type: str) -> Tuple[float, float, float]:
    if joint_type == "revolute":
        return (0.0, 0.0, 1.0)
    return (1.0, 0.0, 0.0)


def _parse_joint_limit(element: Optional[ET.Element]) -> Optional[JointLimit]:
    if element is None:
        return None
    def _get(attr: str) -> Optional[float]:
        value = element.get(attr)
        return float(value) if value is not None else None
    return JointLimit(
        lower=_get("lower"),
        upper=_get("upper"),
        effort=_get("effort"),
        velocity=_get("velocity"),
    )


def parse_urdf(path: Path) -> URDFModel:
    tree = ET.parse(path)
    robot_elem = tree.getroot()
    if robot_elem.tag != "robot":
        raise ValueError(f"URDF root element must be <robot>, got <{robot_elem.tag}>")

    name = robot_elem.get("name", path.stem)

    links: Dict[str, URDFLink] = {}
    for link_elem in robot_elem.findall("link"):
        link_name = link_elem.get("name")
        if not link_name:
            raise ValueError("Encountered <link> without a name attribute")
        inertial = _parse_inertial(link_elem.find("inertial"))
        visuals = list(_parse_visuals(link_elem))
        collisions = list(_parse_collisions(link_elem))
        links[link_name] = URDFLink(
            name=link_name,
            inertial=inertial,
            visuals=visuals,
            collisions=collisions,
        )

    joints: Dict[str, URDFJoint] = {}
    for joint_elem in robot_elem.findall("joint"):
        joint_name = joint_elem.get("name")
        if not joint_name:
            raise ValueError("Encountered <joint> without a name attribute")
        joint_type = joint_elem.get("type", "fixed")
        parent_elem = joint_elem.find("parent")
        child_elem = joint_elem.find("child")
        if parent_elem is None or child_elem is None:
            raise ValueError(f"Joint '{joint_name}' missing parent or child definition")
        parent = parent_elem.get("link")
        child = child_elem.get("link")
        if parent is None or child is None:
            raise ValueError(f"Joint '{joint_name}' contains empty parent/child link attribute")
        origin = _parse_origin(joint_elem.find("origin"))
        axis_elem = joint_elem.find("axis")
        axis = _parse_float_tuple(axis_elem.get("xyz")) if axis_elem is not None else _default_axis(joint_type)
        limit = _parse_joint_limit(joint_elem.find("limit"))
        joints[joint_name] = URDFJoint(
            name=joint_name,
            type=joint_type,
            parent=parent,
            child=child,
            origin=origin,
            axis=axis,  # type: ignore[arg-type]
            limit=limit,
        )

    return URDFModel(name=name, links=links, joints=joints)

