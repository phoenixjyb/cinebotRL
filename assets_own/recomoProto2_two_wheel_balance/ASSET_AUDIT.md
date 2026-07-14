# Recomo Proto2 Two-Wheel Balance Asset Audit

Audit date: 2026-07-14

## Contract

- Source is the Xacro and its expanded plain URDF in this directory.
- Geometry uses SI metres and primitive collision shapes.
- Import scale is `1.0`.
- Root is floating; there is no world, planar, or caster joint.
- Actuated joints are exactly `left_wheel_joint` and `right_wheel_joint`.
- Wheel track is 0.620 m and wheel diameter is 0.2032 m (8 inches).
- Both wheel joint axes are `[0, +1, 0]`, so positive wheel velocity drives chassis `+X`.
- Both wheel joints are externally effort-controlled with imported drive stiffness and damping set to zero.
- Modeled mass is exactly 28.0 kg. All seven rigid bodies have explicit positive inertials and the live articulation resolves to 27.999998 kg.
- The `base_link` mass is an aggregate rigid travel-pose approximation. COM, inertia, wheel width, and the 20 Nm effort limit remain provisional.

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

The current machine-readable audit passes all checks. See `docs/03_training/two_wheel_balance/evidence_20260714_28kg/asset_audit.json`.

Isaac Sim 5.1 emits unresolved visual-scope warnings for the empty fixed `upper_imu_link` and `arm_mount_link`. Physics, mass, articulation, joint, contact, and rendering checks pass, but Fabric cloning is disabled for this layered imported USD. The warnings must be revisited when the provisional primitive asset is replaced by production CAD.
