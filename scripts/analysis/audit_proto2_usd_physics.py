"""Audit the active Proto2 USD physics profile.

This checks the failure modes that made rendered rollouts misleading:
- robot visuals missing or detached from articulation joints
- joint local positions left in millimetres after mesh scaling
- importer/default masses, zero inertias, invalid centers of mass

Run with Isaac/Kit Python, for example on the .98 Windows/WSL host:
    PYTHONUTF8=1 NO_PROXY='*' no_proxy='*' \
      /mnt/g/isaaclab_venv/Scripts/python.exe -X utf8 \
      scripts/analysis/audit_proto2_usd_physics.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEFAULT_USD = PROJECT_ROOT / "assets_own" / "recomoProto2-1190_moveit" / "recomoProto2-1190_moveit_wrapper_baked_scaled.usd"
EXPECTED_MASS_KG = 40.0

ARM_JOINT_MIN_OFFSETS_M = {
    "arm_base_joint": 0.5,
    "joint5_arm_pitch": 0.03,
    "joint4_elbow_pitch": 0.20,
    "joint3_gimbal_yaw": 0.20,
    "joint2_gimbal_roll": 0.03,
    "joint1_gimbal_pitch": 0.03,
    "camera_optical_center": 0.03,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit active Proto2 USD mass, inertia, COM, meshes, and joint scale.")
    parser.add_argument("--usd", default=str(DEFAULT_USD), help="USD stage to audit.")
    parser.add_argument("--expected_mass", type=float, default=EXPECTED_MASS_KG)
    parser.add_argument("--mass_tolerance", type=float, default=0.5, help="Allowed total mass error in kg.")
    parser.add_argument("--json", action="store_true", help="Print JSON to stdout.")
    parser.add_argument("--output", default=None, help="Optional path for a clean JSON report.")
    return parser.parse_args()


def as_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    if hasattr(value, "GetReal") and hasattr(value, "GetImaginary"):
        imag = value.GetImaginary()
        return [float(value.GetReal()), *[float(x) for x in imag]]
    if hasattr(value, "__len__") and not isinstance(value, str):
        return [float(x) for x in value]
    return None


def finite_vec(values: list[float] | None) -> bool:
    return values is not None and all(math.isfinite(v) for v in values)


def vec_norm(values: list[float] | None) -> float:
    if values is None:
        return 0.0
    return math.sqrt(sum(float(v) * float(v) for v in values))


def rel_targets(joint, name: str) -> list[str]:
    rel = joint.GetPrim().GetRelationship(name)
    return [target.pathString for target in rel.GetTargets()]


def audit_stage(usd_path: Path, expected_mass: float, mass_tolerance: float) -> dict[str, Any]:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"failed to open USD: {usd_path}")

    bodies = []
    total_mass = 0.0
    zero_inertia_bodies = []
    invalid_com_bodies = []
    duplicate_mass_api_names = {}
    for prim in stage.Traverse():
        has_rigid = prim.HasAPI(UsdPhysics.RigidBodyAPI)
        has_mass = prim.HasAPI(UsdPhysics.MassAPI)
        if not has_rigid and not has_mass:
            continue
        mass_api = UsdPhysics.MassAPI(prim)
        mass = mass_api.GetMassAttr().Get()
        inertia = as_list(mass_api.GetDiagonalInertiaAttr().Get())
        com = as_list(mass_api.GetCenterOfMassAttr().Get())
        name = prim.GetName()
        if isinstance(mass, (int, float)) and math.isfinite(float(mass)):
            total_mass += float(mass)
        if not finite_vec(inertia) or vec_norm(inertia) <= 1e-9:
            zero_inertia_bodies.append(prim.GetPath().pathString)
        if not finite_vec(com):
            invalid_com_bodies.append(prim.GetPath().pathString)
        if has_mass:
            duplicate_mass_api_names.setdefault(name, []).append(prim.GetPath().pathString)
        bodies.append(
            {
                "path": prim.GetPath().pathString,
                "name": name,
                "rigid": bool(has_rigid),
                "mass_api": bool(has_mass),
                "mass": float(mass) if isinstance(mass, (int, float)) else None,
                "diagonal_inertia": inertia,
                "center_of_mass": com,
            }
        )

    duplicate_mass_api_names = {k: v for k, v in duplicate_mass_api_names.items() if len(v) > 1}

    mesh_count = 0
    visual_mesh_count = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh_count += 1
            if "/visuals/" in prim.GetPath().pathString:
                visual_mesh_count += 1

    joint_rows = []
    underscaled_joint_offsets = []
    for prim in stage.Traverse():
        if not (
            prim.IsA(UsdPhysics.Joint)
            or prim.IsA(UsdPhysics.RevoluteJoint)
            or prim.IsA(UsdPhysics.PrismaticJoint)
            or prim.IsA(UsdPhysics.FixedJoint)
        ):
            continue
        joint = UsdPhysics.Joint(prim)
        name = prim.GetName()
        local_pos0 = as_list(joint.GetLocalPos0Attr().Get())
        local_pos1 = as_list(joint.GetLocalPos1Attr().Get())
        max_offset = max(vec_norm(local_pos0), vec_norm(local_pos1))
        min_expected = ARM_JOINT_MIN_OFFSETS_M.get(name)
        if min_expected is not None and max_offset < min_expected:
            underscaled_joint_offsets.append(
                {
                    "joint": prim.GetPath().pathString,
                    "max_offset_m": max_offset,
                    "expected_min_m": min_expected,
                    "localPos0": local_pos0,
                    "localPos1": local_pos1,
                }
            )
        joint_rows.append(
            {
                "path": prim.GetPath().pathString,
                "name": name,
                "type": prim.GetTypeName(),
                "body0": rel_targets(joint, "physics:body0"),
                "body1": rel_targets(joint, "physics:body1"),
                "localPos0": local_pos0,
                "localPos1": local_pos1,
                "max_offset_m": max_offset,
            }
        )

    failures = []
    if abs(total_mass - expected_mass) > mass_tolerance:
        failures.append(f"total mass {total_mass:.3f} kg outside expected {expected_mass:.3f}+/-{mass_tolerance:.3f} kg")
    if zero_inertia_bodies:
        failures.append(f"{len(zero_inertia_bodies)} bodies have zero/invalid inertia")
    if invalid_com_bodies:
        failures.append(f"{len(invalid_com_bodies)} bodies have invalid COM")
    if duplicate_mass_api_names:
        failures.append(f"nested/duplicate MassAPI names found: {sorted(duplicate_mass_api_names)}")
    if visual_mesh_count < 8:
        failures.append(f"too few visual meshes: {visual_mesh_count}")
    if underscaled_joint_offsets:
        failures.append(f"{len(underscaled_joint_offsets)} key arm/gimbal joints have underscaled local offsets")

    return {
        "usd": str(usd_path),
        "expected_mass_kg": expected_mass,
        "mass_tolerance_kg": mass_tolerance,
        "total_mass_kg": total_mass,
        "body_count": len(bodies),
        "mesh_count": mesh_count,
        "visual_mesh_count": visual_mesh_count,
        "joint_count": len(joint_rows),
        "zero_inertia_bodies": zero_inertia_bodies,
        "invalid_com_bodies": invalid_com_bodies,
        "duplicate_mass_api_names": duplicate_mass_api_names,
        "underscaled_joint_offsets": underscaled_joint_offsets,
        "key_bodies": [
            b for b in bodies if b["name"] in {"base_link", "arm_base_link", "arm_link_1", "arm_link_2", "arm_link_3", "arm_link_4", "arm_link_5", "arm_link_6", "cam_link"}
        ],
        "key_joints": [j for j in joint_rows if j["name"] in set(ARM_JOINT_MIN_OFFSETS_M)],
        "failures": failures,
        "ok": not failures,
    }


def main() -> int:
    args = parse_args()
    usd_path = Path(args.usd)
    if not usd_path.is_absolute():
        usd_path = PROJECT_ROOT / usd_path

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    try:
        result = audit_stage(usd_path, args.expected_mass, args.mass_tolerance)
        if args.output:
            output_path = Path(args.output)
            if not output_path.is_absolute():
                output_path = PROJECT_ROOT / output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

        if args.json or not args.output:
            print(json.dumps(result, indent=2))
        else:
            status = "PASS" if result["ok"] else "FAIL"
            print(
                f"{status}: mass={result['total_mass_kg']:.3f}kg "
                f"bodies={result['body_count']} meshes={result['mesh_count']} "
                f"visual_meshes={result['visual_mesh_count']} joints={result['joint_count']}"
            )
            if result["failures"]:
                for failure in result["failures"]:
                    print(f"- {failure}")
        if not result["ok"]:
            return 1
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
