"""Static checks for URDF + mesh assets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

from .model import URDFJoint, URDFLink, URDFModel
from .package_utils import PackageContext


@dataclass
class Issue:
    severity: str  # "error", "warning", "info"
    code: str
    message: str
    data: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        payload = dict(self.data)
        payload.update({"severity": self.severity, "code": self.code, "message": self.message})
        return payload


class IssueCollector:
    def __init__(self) -> None:
        self._issues: List[Issue] = []

    def add(self, severity: str, code: str, message: str, **data: object) -> None:
        self._issues.append(Issue(severity=severity, code=code, message=message, data=dict(data)))

    def extend(self, issues: Iterable[Issue]) -> None:
        self._issues.extend(issues)

    def to_list(self) -> List[Issue]:
        return list(self._issues)


SUPPORTED_JOINT_TYPES = {"fixed", "revolute", "continuous", "prismatic", "planar"}


def run_checks(
    model: URDFModel,
    context: PackageContext,
    *,
    expected_mesh_scale: Optional[float] = None,
) -> List[Issue]:
    collector = IssueCollector()

    root = model.find_root_link()
    if root is None:
        collector.add("error", "topology.no_root", "Could not determine root link; joint graph may be cyclic")
    else:
        unreachable = _find_unreachable_links(model, root)
        if unreachable:
            collector.add(
                "error",
                "topology.disconnected",
                f"{len(unreachable)} links are disconnected from root '{root}'",
                links=sorted(unreachable),
            )

    collector.extend(_check_links(model))
    collector.extend(_check_joints(model))
    collector.extend(_check_meshes(model, context, expected_mesh_scale=expected_mesh_scale))

    return collector.to_list()


def _find_unreachable_links(model: URDFModel, root: str) -> Set[str]:
    visited = {root}
    queue = [root]
    while queue:
        current = queue.pop(0)
        for joint in model.children_for(current):
            child = joint.child
            if child not in model.links:
                continue
            if child not in visited:
                visited.add(child)
                queue.append(child)
    return set(model.links) - visited


def _check_links(model: URDFModel) -> List[Issue]:
    issues: List[Issue] = []
    for link in model.links.values():
        if link.inertial is None:
            issues.append(
                Issue(
                    severity="warning",
                    code="link.missing_inertial",
                    message=f"Link '{link.name}' is missing inertial parameters",
                )
            )
        else:
            if link.inertial.mass <= 0.0:
                issues.append(
                    Issue(
                        severity="error",
                        code="link.invalid_mass",
                        message=f"Link '{link.name}' has non-positive mass {link.inertial.mass}",
                        data={"mass": link.inertial.mass},
                    )
                )
        if not link.visuals:
            issues.append(
                Issue(
                    severity="warning",
                    code="link.missing_visual",
                    message=f"Link '{link.name}' has no visual geometry",
                )
            )
        if not link.collisions:
            issues.append(
                Issue(
                    severity="warning",
                    code="link.missing_collision",
                    message=f"Link '{link.name}' has no collision geometry",
                )
            )
    return issues


def _check_joints(model: URDFModel) -> List[Issue]:
    issues: List[Issue] = []
    for joint in model.joints.values():
        if joint.type not in SUPPORTED_JOINT_TYPES:
            issues.append(
                Issue(
                    severity="warning",
                    code="joint.unsupported_type",
                    message=f"Joint '{joint.name}' uses unsupported type '{joint.type}'",
                )
            )
        if joint.parent not in model.links:
            issues.append(
                Issue(
                    severity="error",
                    code="joint.missing_parent",
                    message=f"Joint '{joint.name}' references missing parent link '{joint.parent}'",
                )
            )
        if joint.child not in model.links:
            issues.append(
                Issue(
                    severity="error",
                    code="joint.missing_child",
                    message=f"Joint '{joint.name}' references missing child link '{joint.child}'",
                )
            )
        if joint.type in {"revolute", "prismatic"} and joint.limit is None:
            issues.append(
                Issue(
                    severity="warning",
                    code="joint.missing_limit",
                    message=f"Joint '{joint.name}' ({joint.type}) is missing limits",
                )
            )
    return issues


def _check_meshes(
    model: URDFModel,
    context: PackageContext,
    *,
    expected_mesh_scale: Optional[float],
) -> List[Issue]:
    collector = IssueCollector()
    already_checked: Set[Path] = set()
    scale_warned: Set[str] = set()

    def _inspect(mesh_path: Path, *, geom_type: str, link_name: str) -> None:
        if mesh_path in already_checked:
            return
        already_checked.add(mesh_path)
        if not mesh_path.exists():
            collector.add(
                "error",
                "mesh.missing_file",
                f"Referenced {geom_type} mesh not found: {mesh_path}",
                link=link_name,
                path=str(mesh_path),
            )
        elif mesh_path.suffix.lower() not in {".stl", ".obj", ".dae", ".ply", ".usd", ".usdz"}:
            collector.add(
                "warning",
                "mesh.format",
                f"Mesh '{mesh_path.name}' uses untested format '{mesh_path.suffix}'",
                link=link_name,
            )

    def _maybe_warn_scale(mesh_filename: str, link_name: str) -> None:
        if expected_mesh_scale is None:
            return
        if mesh_filename in scale_warned:
            return
        scale_warned.add(mesh_filename)
        collector.add(
            "warning",
            "mesh.missing_scale",
            (
                f"Mesh '{mesh_filename}' referenced by link '{link_name}' has no scale. "
                f"Expected uniform scale {expected_mesh_scale} for CAD units -> metres"
            ),
            link=link_name,
        )

    for link in model.links.values():
        for visual in link.visuals:
            if visual.mesh is None:
                continue
            mesh_path = context.resolve_uri(visual.mesh.filename)
            _inspect(mesh_path, geom_type="visual", link_name=link.name)
            if visual.mesh.scale is None:
                _maybe_warn_scale(visual.mesh.filename, link.name)
        for collision in link.collisions:
            if collision.mesh is None:
                continue
            mesh_path = context.resolve_uri(collision.mesh.filename)
            _inspect(mesh_path, geom_type="collision", link_name=link.name)
            if collision.mesh.scale is None:
                _maybe_warn_scale(collision.mesh.filename, link.name)

    return collector.to_list()

