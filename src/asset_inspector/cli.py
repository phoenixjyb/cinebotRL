"""CLI entry-point for asset inspection utilities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from .parser import parse_urdf
from .package_utils import PackageContext
from .reporting import InspectionReport, build_report
from .visualize import build_scene, export_scene, require_trimesh


def _add_common_package_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--package-path",
        type=Path,
        required=True,
        help="Path to the ROS-style package directory (contains package.xml)",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=None,
        help="Optional explicit URDF file to inspect (defaults to the only URDF found)",
    )
    parser.add_argument(
        "--mesh-scale",
        type=float,
        default=None,
        help="Expected uniform mesh scale to convert CAD units to metres (e.g. 0.001 for mm)",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect and visualise mobile robot assets")
    subparsers = parser.add_subparsers(dest="command", required=True)

    report_parser = subparsers.add_parser("report", help="Generate textual/JSON inspection report")
    _add_common_package_args(report_parser)
    report_parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write full report as JSON",
    )

    vis_parser = subparsers.add_parser("visualize", help="Export a trimesh scene of the robot")
    _add_common_package_args(vis_parser)
    vis_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output file for the scene (e.g. scene.glb, scene.html)",
    )
    vis_parser.add_argument(
        "--no-axes",
        action="store_true",
        help="Disable world axes helper geometry",
    )

    args = parser.parse_args(argv)

    try:
        context = PackageContext.from_directory(args.package_path, urdf_path=args.urdf)
    except Exception as exc:  # pragma: no cover - CLI friendly error path
        parser.error(str(exc))

    model = parse_urdf(context.urdf_path)

    if args.command == "report":
        return _handle_report(args, model, context)
    if args.command == "visualize":
        return _handle_visualize(args, model, context)
    parser.error(f"Unknown command {args.command}")
    return 1


def _handle_report(args: argparse.Namespace, model, context: PackageContext) -> int:
    report = build_report(model, context, expected_mesh_scale=args.mesh_scale)
    _print_report(report)
    if args.json_out:
        _write_json_report(report, args.json_out)
    return 0


def _handle_visualize(args: argparse.Namespace, model, context: PackageContext) -> int:
    try:
        scene = build_scene(
            model,
            context,
            default_mesh_scale=args.mesh_scale or 1.0,
            include_axes=not args.no_axes,
        )
    except RuntimeError as exc:
        require_trimesh()  # re-raise clearer message
        raise exc
    export_scene(scene, args.output)
    print(f"Scene exported to {args.output}")
    return 0


def _print_report(report: InspectionReport) -> None:
    print(f"Package: {report.package_name}")
    print(f"URDF: {report.urdf_path}")
    print(f"Root link: {report.root_link or '<unknown>'}")
    print(f"Links: {report.link_count} | Joints: {report.joint_count}")

    stats = report.issue_stats()
    print("Issues:")
    for severity in ("error", "warning", "info"):
        count = stats.get(severity, 0)
        print(f"  {severity.title()}: {count}")

    if report.issues:
        print("\nDetailed findings:")
        for issue in report.issues:
            details = ", ".join(f"{key}={value}" for key, value in issue.data.items())
            suffix = f" ({details})" if details else ""
            print(f"- [{issue.severity.upper()}] {issue.code}: {issue.message}{suffix}")

    print("\nKinematic tree:")
    print(report.kinematic_tree)

    print("\nLink coverage summary:")
    for summary in report.link_summaries:
        attributes = []
        attributes.append("inertial" if summary.has_inertial else "no inertial")
        attributes.append("visual" if summary.has_visual else "no visual")
        attributes.append("collision" if summary.has_collision else "no collision")
        print(f"- {summary.name}: {', '.join(attributes)}")


def _write_json_report(report: InspectionReport, output_path: Path) -> None:
    payload: Dict[str, Any] = {
        "package_name": report.package_name,
        "urdf_path": report.urdf_path,
        "root_link": report.root_link,
        "link_count": report.link_count,
        "joint_count": report.joint_count,
        "issues": [issue.as_dict() for issue in report.issues],
        "link_summaries": [summary.__dict__ for summary in report.link_summaries],
        "kinematic_tree": report.kinematic_tree,
    }
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"JSON report written to {output_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())

