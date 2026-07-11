#!/usr/bin/env python3
"""Machine-readable USD contract audit for the two-wheel asset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("ACCEPT_EULA", "YES")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")
os.environ.setdefault("GYMNASIUM_DISABLE_PLUGIN_ENTRYPOINTS", "1")

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--usd", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from pxr import Usd, UsdPhysics


def main() -> int:
    stage = Usd.Stage.Open(str(args.usd.resolve()))
    if stage is None:
        raise RuntimeError(f"cannot open USD: {args.usd}")

    prims = list(stage.Traverse())
    joints = [p for p in prims if p.IsA(UsdPhysics.Joint)]
    wheel_joints = [p for p in joints if p.GetName() in {"left_wheel_joint", "right_wheel_joint"}]
    revolute_joints = [p for p in joints if p.IsA(UsdPhysics.RevoluteJoint)]
    articulation_roots = [p for p in prims if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    masses = []
    inertia_positive = True
    for prim in prims:
        mass_api = UsdPhysics.MassAPI.Get(stage, prim.GetPath())
        if not mass_api:
            continue
        mass = mass_api.GetMassAttr().Get()
        if mass is not None and mass > 0:
            masses.append(float(mass))
        inertia = mass_api.GetDiagonalInertiaAttr().Get()
        if inertia is not None:
            inertia_positive &= all(float(v) > 0.0 for v in inertia)

    drives = {}
    for joint in wheel_joints:
        drive = UsdPhysics.DriveAPI.Get(joint, "angular")
        drives[joint.GetName()] = {
            "present": bool(drive),
            "stiffness": float(drive.GetStiffnessAttr().Get() or 0.0) if drive else 0.0,
            "damping": float(drive.GetDampingAttr().Get() or 0.0) if drive else 0.0,
        }

    layer_text = "\n".join(layer.ExportToString() for layer in stage.GetLayerStack())
    checks = {
        "default_prim": bool(stage.GetDefaultPrim()),
        "articulation_root": len(articulation_roots) == 1,
        "floating_no_world_fixed_joint": "world" not in layer_text.lower(),
        "two_wheel_joints": len(wheel_joints) == 2,
        "only_two_revolute_joints": len(revolute_joints) == 2,
        "no_planar_virtual_joints": all(token not in layer_text for token in ("base_joint_vx", "base_joint_vy", "base_joint_wz")),
        "no_caster": "caster" not in layer_text.lower(),
        "no_legacy_stl": "base_link.STL" not in layer_text,
        "mass_near_26kg": abs(sum(masses) - 26.0) < 0.1,
        "positive_inertia": inertia_positive,
        "wheel_position_drives_disabled": all(v["stiffness"] == 0.0 for v in drives.values()),
    }
    result = {
        "schema": "recomo_two_wheel_asset_audit_v1",
        "usd": str(args.usd.resolve()),
        "default_prim": str(stage.GetDefaultPrim().GetPath()) if stage.GetDefaultPrim() else None,
        "articulation_roots": [str(p.GetPath()) for p in articulation_roots],
        "joints": [p.GetName() for p in joints],
        "revolute_joints": [p.GetName() for p in revolute_joints],
        "wheel_drives": drives,
        "total_mass_kg": sum(masses),
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


exit_code = 1
try:
    exit_code = main()
except Exception:
    import traceback

    traceback.print_exc()
finally:
    import threading

    shutdown_watchdog = threading.Timer(10.0, lambda: os._exit(exit_code))
    shutdown_watchdog.daemon = True
    shutdown_watchdog.start()
    app.close()
    shutdown_watchdog.cancel()
raise SystemExit(exit_code)
