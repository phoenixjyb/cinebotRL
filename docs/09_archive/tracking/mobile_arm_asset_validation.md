# mobile_arm_whole_body asset validation

This note bundles the new inspection tooling with the current state of the `mobile_arm_whole_body` package. Use it whenever STL/URDF assets are updated so the Isaac Sim / Isaac Lab pipelines stay clean.

## Quick start: generate a health report

```powershell
# From the repo root
$env:PYTHONPATH = "src"
python -m asset_inspector.cli report `
    --package-path assets_own/mobile_arm_whole_body `
    --mesh-scale 0.001 `
    --json-out assets/processed/mobile_arm_whole_body/inspection_report.json
```

- `--mesh-scale 0.001` enforces millimetre → metre scaling for CAD meshes (per `robot_spec.md`).
- The CLI prints a readable summary and emits a machine-friendly JSON file for automation.

## Optional: export a visual sanity check

Install the lightweight dependency once:

```powershell
pip install trimesh
```

Then export a scene (GLB, OBJ, HTML) using the same CLI:

```powershell
$env:PYTHONPATH = "src"
python -m asset_inspector.cli visualize `
    --package-path assets_own/mobile_arm_whole_body `
    --mesh-scale 0.001 `
    --output assets/processed/mobile_arm_whole_body/mobile_arm_scene.glb
```

Open the exported file inside Isaac Sim, MeshLab, or any GLB viewer to confirm transforms before regenerating USD assets.

## Findings that must be addressed before Isaac integration

- **Chassis inertial + geometry missing**: `chassis_center_link` is a placeholder. Add inertial mass/inertia, visual shell (`meshes/cr_no_V.stl`) and simplified collision (boxes/cylinders for diff-drive kinematics).
- **Mesh scale absent in URDF**: every STL reference lacks a `scale="0.001 0.001 0.001"`. Add the scale or convert meshes to metres before exporting URDF/USDA for Isaac.
- **Controller metadata**: MoveIt config is still using the fake controller. Add diff-drive and arm joint controllers (Isaac Sim expects articulated body + articulation controllers or ROS2 bridges).
- **Wheel / drive data**: wheel radii, track width, drive inertias, and effort/velocity limits are still placeholders. Populate them so the reinforcement learning environment can clamp commands correctly.
- **Sensor payload TBD**: extend the URDF with camera frames and inertia shifts once the end-effector tooling is final.

Track progress by re-running the report—warnings will clear as each gap is filled.
