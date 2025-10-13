"""Geometry-centric helpers that rely on trimesh for quick previews."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import numpy as np

from .model import (
    IDENTITY_MATRIX,
    Matrix4,
    URDFModel,
    compute_joint_transform,
    matmul,
)
from .package_utils import PackageContext

try:
    import trimesh
except ImportError as exc:  # pragma: no cover - optional dependency guard
    trimesh = None  # type: ignore[assignment]
    IMPORT_ERROR = exc
else:  # pragma: no cover - only executed when trimesh is available
    IMPORT_ERROR = None


def require_trimesh() -> None:
    if trimesh is None:
        raise RuntimeError(
            "trimesh dependency is required for visualization. Install with `pip install trimesh`"
        ) from IMPORT_ERROR


def compute_link_transforms(
    model: URDFModel,
    joint_positions: Optional[Mapping[str, float]] = None,
) -> Dict[str, Matrix4]:
    joint_positions = joint_positions or {}
    root = model.find_root_link()
    if root is None:
        raise ValueError("Cannot compute transforms without a root link")

    transforms: Dict[str, Matrix4] = {root: IDENTITY_MATRIX}
    queue = [root]
    while queue:
        current = queue.pop(0)
        current_tf = transforms[current]
        for joint in model.children_for(current):
            position = joint_positions.get(joint.name, 0.0)
            local_tf = compute_joint_transform(joint, position)
            child_tf = matmul(current_tf, local_tf)
            transforms[joint.child] = child_tf
            queue.append(joint.child)
    return transforms


@lru_cache(maxsize=64)
def _load_mesh(mesh_path: Path) -> "trimesh.base.Trimesh":
    require_trimesh()
    mesh = trimesh.load_mesh(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type loaded from {mesh_path}: {type(mesh)}")
    return mesh


def _apply_scale(mesh: "trimesh.Trimesh", scale: Optional[Iterable[float]]) -> "trimesh.Trimesh":
    if scale is None:
        return mesh
    scaled = mesh.copy()
    scaled.apply_scale(scale)
    return scaled


def _to_numpy(matrix: Matrix4) -> "np.ndarray":
    return np.array(matrix, dtype=float)


def build_scene(
    model: URDFModel,
    context: PackageContext,
    *,
    joint_positions: Optional[Mapping[str, float]] = None,
    default_mesh_scale: float = 1.0,
    include_axes: bool = True,
) -> "trimesh.Scene":
    require_trimesh()
    transforms = compute_link_transforms(model, joint_positions=joint_positions)
    scene = trimesh.Scene()

    for link_name, link in model.links.items():
        link_tf = transforms.get(link_name)
        if link_tf is None:
            continue
        for visual_idx, visual in enumerate(link.visuals):
            if visual.mesh is None:
                continue
            mesh_path = context.resolve_uri(visual.mesh.filename)
            mesh = _load_mesh(mesh_path)
            scale = visual.mesh.scale or (default_mesh_scale,) * 3
            if scale == (1.0, 1.0, 1.0):
                scaled_mesh = mesh
            else:
                scaled_mesh = _apply_scale(mesh, scale)
            total_tf = matmul(link_tf, visual.origin.as_matrix())
            node_name = f"{link_name}_visual_{visual_idx}"
            scene.add_geometry(
                scaled_mesh,
                node_name=node_name,
                transform=_to_numpy(total_tf),
            )

    if include_axes:
        axes = trimesh.creation.axis(axis_length=0.25)
        scene.add_geometry(axes, node_name="world_axes")

    return scene


def export_scene(
    scene: "trimesh.Scene",
    output_path: Path,
) -> None:
    require_trimesh()
    export_path = output_path.resolve()
    export_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(export_path)

