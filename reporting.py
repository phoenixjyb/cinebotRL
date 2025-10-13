"""High level summarisation utilities for asset inspection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from .checks import Issue, run_checks
from .model import URDFModel
from .package_utils import PackageContext


@dataclass
class LinkSummary:
    name: str
    has_inertial: bool
    has_visual: bool
    has_collision: bool
    visual_meshes: List[str]
    collision_meshes: List[str]


@dataclass
class InspectionReport:
    package_name: str
    urdf_path: str
    root_link: Optional[str]
    link_count: int
    joint_count: int
    issues: List[Issue] = field(default_factory=list)
    link_summaries: List[LinkSummary] = field(default_factory=list)
    kinematic_tree: str = ""

    def issue_stats(self) -> Dict[str, int]:
        stats: Dict[str, int] = {"error": 0, "warning": 0, "info": 0}
        for issue in self.issues:
            stats[issue.severity] = stats.get(issue.severity, 0) + 1
        return stats


def build_report(
    model: URDFModel,
    context: PackageContext,
    *,
    expected_mesh_scale: Optional[float] = None,
) -> InspectionReport:
    root = model.find_root_link()
    issues = run_checks(model, context, expected_mesh_scale=expected_mesh_scale)
    link_summaries = _collect_link_summaries(model)
    tree_repr = format_kinematic_tree(model, root_name=root)

    return InspectionReport(
        package_name=context.package_name,
        urdf_path=str(context.urdf_path),
        root_link=root,
        link_count=len(model.links),
        joint_count=len(model.joints),
        issues=issues,
        link_summaries=link_summaries,
        kinematic_tree=tree_repr,
    )


def _collect_link_summaries(model: URDFModel) -> List[LinkSummary]:
    summaries: List[LinkSummary] = []
    for link in model.links.values():
        summaries.append(
            LinkSummary(
                name=link.name,
                has_inertial=link.inertial is not None,
                has_visual=bool(link.visuals),
                has_collision=bool(link.collisions),
                visual_meshes=[visual.mesh.filename for visual in link.visuals if visual.mesh],
                collision_meshes=[collision.mesh.filename for collision in link.collisions if collision.mesh],
            )
        )
    return sorted(summaries, key=lambda summary: summary.name)


def format_kinematic_tree(model: URDFModel, root_name: Optional[str]) -> str:
    if root_name is None:
        return "<unknown root>"

    lines: List[str] = []

    def _walk(link_name: str, prefix: str, is_root: bool) -> None:
        if is_root:
            lines.append(f"{prefix}{link_name}")
        children = model.children_for(link_name)
        for index, joint in enumerate(children):
            is_last = index == len(children) - 1
            branch = "└─" if is_last else "├─"
            joint_line = f"{prefix}{branch}{joint.name}:{joint.type} -> {joint.child}"
            lines.append(joint_line)
            child_prefix = prefix + ("  " if is_last else "│ ")
            _walk(joint.child, child_prefix, is_root=False)

    _walk(root_name, "", is_root=True)
    return "\n".join(lines)

