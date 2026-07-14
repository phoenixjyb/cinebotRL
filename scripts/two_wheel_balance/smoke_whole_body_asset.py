#!/usr/bin/env python3
"""Machine-readable USD audit for the two-wheel whole-body articulation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from xml.etree import ElementTree as ET

os.environ.setdefault("ACCEPT_EULA", "YES")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--usd", type=Path, required=True)
parser.add_argument("--urdf", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from pxr import Usd, UsdPhysics


PHYSICAL_ARM_JOINTS = {
    "joint6_arm_yaw",
    "joint5_arm_pitch",
    "joint4_elbow_pitch",
    "joint3_gimbal_yaw",
    "joint2_gimbal_roll",
    "joint1_gimbal_pitch",
}
VIRTUAL_FRAME_JOINTS = {
    "ee1_level_pitch",
    "ee1_rot_z",
    "ee1_rot_y",
    "ee1_rot_x",
}
STAGE0_LOCKED_JOINTS = {
    "joint3_gimbal_yaw",
    "joint2_gimbal_roll",
    "joint1_gimbal_pitch",
    *VIRTUAL_FRAME_JOINTS,
}


def main() -> int:
    stage = Usd.Stage.Open(str(args.usd.resolve()))
    if stage is None:
        raise RuntimeError(f"cannot open USD: {args.usd}")
    prims = list(stage.Traverse())
    joints = [prim for prim in prims if prim.IsA(UsdPhysics.Joint)]
    joint_names = {prim.GetName() for prim in joints}
    rigid_bodies = [prim for prim in prims if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    articulation_roots = [
        prim for prim in prims if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    masses = {}
    positive_inertia = True
    for prim in rigid_bodies:
        mass_api = UsdPhysics.MassAPI.Get(stage, prim.GetPath())
        mass = mass_api.GetMassAttr().Get() if mass_api else None
        inertia = mass_api.GetDiagonalInertiaAttr().Get() if mass_api else None
        if mass is not None:
            masses[str(prim.GetPath())] = float(mass)
        positive_inertia &= inertia is not None and all(float(value) > 0.0 for value in inertia)

    urdf_root = ET.parse(args.urdf.resolve()).getroot()
    links = {item.attrib["name"]: item for item in urdf_root.findall("link")}
    urdf_joints = {
        item.attrib["name"]: item for item in urdf_root.findall("joint")
    }
    wheel_origins = {}
    wheel_radii = {}
    wheel_axes = {}
    for side in ("left", "right"):
        joint = urdf_root.find(f"./joint[@name='{side}_wheel_joint']")
        link = links[f"{side}_wheel_link"]
        wheel_origins[side] = [
            float(value) for value in joint.find("origin").attrib["xyz"].split()
        ]
        wheel_axes[side] = [
            float(value) for value in joint.find("axis").attrib["xyz"].split()
        ]
        wheel_radii[side] = float(
            link.find("./collision/geometry/cylinder").attrib["radius"]
        )

    wheel_drives = {}
    for joint in joints:
        if joint.GetName() not in {"left_wheel_joint", "right_wheel_joint"}:
            continue
        drive = UsdPhysics.DriveAPI.Get(joint, "angular")
        wheel_drives[joint.GetName()] = {
            "present": bool(drive),
            "stiffness": float(drive.GetStiffnessAttr().Get() or 0.0) if drive else 0.0,
            "damping": float(drive.GetDampingAttr().Get() or 0.0) if drive else 0.0,
        }

    total_mass = sum(masses.values())
    checks = {
        "default_prim": bool(stage.GetDefaultPrim()),
        "single_articulation_root": len(articulation_roots) == 1,
        "twenty_rigid_bodies": len(rigid_bodies) == 20,
        "nineteen_joints": len(joints) == 19,
        "all_rigid_bodies_have_mass": len(masses) == len(rigid_bodies),
        "positive_inertia": positive_inertia,
        "total_mass_28kg": abs(total_mass - 28.0) < 0.01,
        "wheel_joints_present": {"left_wheel_joint", "right_wheel_joint"} <= joint_names,
        "physical_arm_joints_present": PHYSICAL_ARM_JOINTS <= joint_names,
        "virtual_frame_joints_present": VIRTUAL_FRAME_JOINTS <= joint_names,
        "stage0_gimbal_and_virtual_joints_fixed": all(
            urdf_joints[name].attrib["type"] == "fixed"
            for name in STAGE0_LOCKED_JOINTS
        ),
        "physical_cam_link_present": any(prim.GetName() == "cam_link" for prim in rigid_bodies),
        "no_planar_base_joints": not {
            "base_joint_vx",
            "base_joint_vy",
            "base_joint_wz",
        }
        & joint_names,
        "wheel_position_drives_disabled": all(
            item["stiffness"] == 0.0 for item in wheel_drives.values()
        ),
        "wheel_track_620mm": abs(wheel_origins["left"][1] - wheel_origins["right"][1] - 0.620)
        < 1e-9,
        "wheel_diameter_8in": abs(2.0 * wheel_radii["left"] - 0.2032) < 1e-9,
        "wheel_axes_positive_y": all(
            wheel_axes[side] == [0.0, 1.0, 0.0] for side in ("left", "right")
        ),
    }
    result = {
        "schema": "recomo_two_wheel_whole_body_asset_audit_v2",
        "usd": str(args.usd.resolve()),
        "urdf": str(args.urdf.resolve()),
        "default_prim": str(stage.GetDefaultPrim().GetPath()) if stage.GetDefaultPrim() else None,
        "joint_names": sorted(joint_names),
        "rigid_body_names": sorted(prim.GetName() for prim in rigid_bodies),
        "rigid_body_masses_kg": masses,
        "total_mass_kg": total_mass,
        "wheel_drives": wheel_drives,
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


exit_code = 1
try:
    exit_code = main()
except Exception:
    import traceback

    traceback.print_exc()
finally:
    app.close()
raise SystemExit(exit_code)
