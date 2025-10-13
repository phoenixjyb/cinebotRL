# EE Frame Alignment Notes (2025-09-23)

Context: CAD/URDF home configuration positions the camera tool frame at the
"Home EE" marker, while trajectory generation software emits the first waypoint
("Desired EE") in the camera optical frame. To avoid offset when starting a
shot we need explicit transforms recorded here.

## Known transforms
- `T_base_ee_home` (from URDF): TBD — obtain from current URDF export.
- `T_base_desired_start` (from trajectory planner): TBD — record from first
  calibrated trajectory in Isaac Sim once assets are imported.
- `T_ee_home_desired = T_base_ee_home^{-1} * T_base_desired_start` — this delta
  should be ≈ identity after re-alignment.

## Action items
1. When the digital twin is available, log the URDF home pose for the camera
   frame (pull from Isaac Sim stage inspector or `ros2 run tf2_ros tf2_echo`).
2. Load a representative trajectory and capture the first waypoint pose. Use
   the alignment script to compute `T_ee_home_desired`.
3. Update trajectory loader to subtract the offset so the policy starts on the
   desired frame; document the applied transform in this file.
4. Once validated, bake the corrected camera mount pose into the URDF/USD so
   both planners and policies share the same reference.

## Open questions
- Does the hardware have mechanical hard-stops guaranteeing repeatable home?
- Should the policy receive the alignment offset as part of the observation
  vector for robustness?
- How frequently does the trajectory tool recompute the first waypoint? If it
  varies per shot, we may need a calibration routine per load-in.
