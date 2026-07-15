#!/usr/bin/env python3
"""Machine-readable URDF/USD contract audit for the riser robot."""

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


EXPECTED_MOVABLE = {
    "left_wheel_joint",
    "right_wheel_joint",
    "riser_joint",
    "joint3_gimbal_yaw",
    "joint2_gimbal_roll",
    "joint1_gimbal_pitch",
}
REMOVED_ARM_JOINTS = {
    "joint6_arm_yaw",
    "joint5_arm_pitch",
    "joint4_elbow_pitch",
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
        positive_inertia &= inertia is not None and all(
            float(value) > 0.0 for value in inertia
        )

    urdf_root = ET.parse(args.urdf.resolve()).getroot()
    urdf_joints = {
        item.attrib["name"]: item for item in urdf_root.findall("joint")
    }
    urdf_links = {
        item.attrib["name"]: item for item in urdf_root.findall("link")
    }
    movable = {
        name
        for name, joint in urdf_joints.items()
        if joint.attrib["type"] != "fixed"
    }
    riser_limit = urdf_joints["riser_joint"].find("limit")
    left_origin = [
        float(value)
        for value in urdf_joints["left_wheel_joint"].find("origin").attrib["xyz"].split()
    ]
    right_origin = [
        float(value)
        for value in urdf_joints["right_wheel_joint"].find("origin").attrib["xyz"].split()
    ]
    wheel_radius = float(
        urdf_links["left_wheel_link"]
        .find("./collision/geometry/cylinder")
        .attrib["radius"]
    )

    drives = {}
    for joint in joints:
        linear = joint.GetName() == "riser_joint"
        drive = UsdPhysics.DriveAPI.Get(joint, "linear" if linear else "angular")
        drives[joint.GetName()] = {
            "present": bool(drive),
            "stiffness": float(drive.GetStiffnessAttr().Get() or 0.0) if drive else 0.0,
            "damping": float(drive.GetDampingAttr().Get() or 0.0) if drive else 0.0,
        }

    riser_prim = next(joint for joint in joints if joint.GetName() == "riser_joint")
    riser_usd = UsdPhysics.PrismaticJoint(riser_prim)
    riser_usd_contract = {
        "lower_limit": float(riser_usd.GetLowerLimitAttr().Get()),
        "upper_limit": float(riser_usd.GetUpperLimitAttr().Get()),
        "local_pos0": [float(value) for value in riser_usd.GetLocalPos0Attr().Get()],
        "local_pos1": [float(value) for value in riser_usd.GetLocalPos1Attr().Get()],
    }

    total_mass = sum(masses.values())
    checks = {
        "default_prim": bool(stage.GetDefaultPrim()),
        "single_articulation_root": len(articulation_roots) == 1,
        "fourteen_rigid_bodies": len(rigid_bodies) == 14,
        "thirteen_joints": len(joints) == 13,
        "usd_joint_names_match_urdf": joint_names == set(urdf_joints),
        "all_rigid_bodies_have_mass": len(masses) == len(rigid_bodies),
        "positive_inertia": positive_inertia,
        "total_mass_28kg": abs(total_mass - 28.0) < 0.01,
        "exact_movable_joint_contract": movable == EXPECTED_MOVABLE,
        "arm_joints_absent": not REMOVED_ARM_JOINTS & joint_names,
        "camera_frames_present": {"cam_link", "ee_tool", "ee1_tool"}
        <= {prim.GetName() for prim in rigid_bodies},
        "riser_range_and_speed": float(riser_limit.attrib["lower"]) == 0.0
        and float(riser_limit.attrib["upper"]) == 1.2
        and float(riser_limit.attrib["velocity"]) == 1.0,
        "usd_riser_range_1p2m": abs(riser_usd_contract["lower_limit"]) < 1e-9
        and abs(riser_usd_contract["upper_limit"] - 1.2) < 1e-6,
        "usd_riser_origin_in_metres": abs(
            riser_usd_contract["local_pos0"][2] - 0.5735305656633361
        )
        < 1e-6,
        "imported_drives_disabled": all(
            not item["present"] or item["stiffness"] == 0.0
            for item in drives.values()
        ),
        "wheel_track_620mm": abs(left_origin[1] - right_origin[1] - 0.620) < 1e-9,
        "wheel_diameter_8in": abs(2.0 * wheel_radius - 0.2032) < 1e-9,
    }
    result = {
        "schema": "recomo_two_wheel_riser_asset_audit_v1",
        "usd": str(args.usd.resolve()),
        "urdf": str(args.urdf.resolve()),
        "joint_names": sorted(joint_names),
        "movable_joint_names": sorted(movable),
        "rigid_body_names": sorted(prim.GetName() for prim in rigid_bodies),
        "rigid_body_masses_kg": masses,
        "total_mass_kg": total_mass,
        "drives": drives,
        "riser_usd_contract": riser_usd_contract,
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
