# Recomo Proto2 Two-Wheel Balance Asset Audit

Audit date: 2026-07-11

## Contract

- Source is the expanded plain URDF in this directory.
- Geometry uses SI metres and primitive collision shapes.
- Import scale is `1.0`.
- Root is floating; there is no world, planar, or caster joint.
- Actuated joints are exactly `left_wheel_joint` and `right_wheel_joint`.
- Both wheel joints are externally effort-controlled with imported drive stiffness and damping set to zero.
- Modeled mass is 26.0 kg. Mass, COM, inertia, wheel width, and 20 Nm effort limit remain provisional.

## Conversion

```bash
/mnt/g/isaaclab_venv/Scripts/python.exe -X utf8 \
  G:\\wSpace\\cinebotRL-two-wheel-balance\\scripts\\convert_urdf_to_usd.py \
  --urdf assets_own/recomoProto2_two_wheel_balance/recomoProto2_two_wheel_balance.urdf \
  --usd assets_own/recomoProto2_two_wheel_balance/recomoProto2_two_wheel_balance.usd \
  --mesh-scale 1.0 \
  --default-drive-type none \
  --headless
```

The converter's default remains `position` for compatibility with existing assets. This asset must explicitly use `none`.

## Result

The machine-readable audit passes all checks. See `docs/03_training/two_wheel_balance/evidence_20260711/asset_audit.json`.

Isaac Sim 5.1 emits unresolved visual-scope warnings for the empty fixed `upper_imu_link` and `arm_mount_link`. Physics, mass, articulation, joint, contact, and rendering checks pass, but Fabric cloning is disabled for this layered imported USD. The warnings must be revisited when the provisional primitive asset is replaced by production CAD.
