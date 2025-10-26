# MATLAB Reachability Map Tools

FK-based reachability map generation and visualization for mobile manipulator.

## Directory Structure

```
matlab/
├── README.md                          # This file
├── build_reachability_map_FK.m        # Main build script (FK-based)
├── run_build.m                        # Wrapper to run build with error handling
├── load_robot_with_meshes.m           # Robot loading utility
├── quick_viz.m                        # Quick visualization helper
├── reach_map_mobile_mm_arm_only.mat   # Generated reachability map (arm workspace)
│
├── visualize_fk_map.m                 # Visualize FK reachability map (with robot)
├── visualize_reachability.m           # Simple reachability visualization
├── visualize_stored_configs.m         # Comprehensive collision verification tool
│
├── docs/                              # Documentation files
│   ├── README_BUILD_VISUALIZE.md      # Main build and visualization guide
│   ├── FK_VS_IK_APPROACH.md          # FK vs IK comparison
│   ├── COORDINATE_FRAMES_EXPLAINED.md # Frame transformation details
│   └── ...                            # Other technical docs
│
├── tests/                             # Test and verification scripts
│   ├── test_chassis_collision.m       # Collision detection validation
│   ├── verify_no_collisions.m         # Verify all stored configs
│   └── check_body_order.m             # Robot body structure check
│
├── tools/                             # Utility tools
│   ├── calculate_optimal_grid.m       # Grid parameter optimization
│   ├── export_reachability_formats.m  # Export map to other formats
│   ├── monitor_build.py               # Monitor build progress
│   └── ...                            # Other utilities
│
└── archive/                           # Deprecated/old files
```

## Quick Start

### Build Reachability Map

```matlab
cd matlab
run_build  % Builds reach_map_mobile_mm_arm_only.mat
```

**Build time**: 2-4 minutes  
**Output**: `reach_map_mobile_mm_arm_only.mat` (~50 MB)

### Visualize Map

```matlab
% Quick visualization (point cloud only)
quick_viz

% Full visualization (with robot)
visualize_fk_map('reach_map_mobile_mm_arm_only.mat')
```

### Verify Collision Detection

```matlab
% Comprehensive check of all stored configurations
cd tests
verify_no_collisions  % Should report 0 collisions

% Visual inspection of worst-case configs
visualize_stored_configs  % Shows 9 most critical configurations
```

## Map Contents

The `.mat` file contains:

- **`reachScore`** (32×40×28): Binary reachability grid (0 or 1)
- **`manipMax`** (32×40×28): Best manipulability at each voxel
- **`qExample`** (32×40×28×6): Best joint config (6 arm joints) per voxel
- **`config`**: Grid parameters (origin, size, voxel_size, etc.)
- **`metadata`**: Build info (date, samples, collision_mode, etc.)

## Extract Surface Mesh

`tools/extract_reachability_surface.m` converts the dense voxel grid into a lightweight surface representation that is easier to visualise and consume from Python.

```matlab
cd matlab
addpath tools
surface = extract_reachability_surface('reach_map_mobile_mm_arm_only.mat', ...
    'output_mat', 'exports/reach_surface.mat', ...
    'output_ply', 'exports/reach_surface.ply', ...
    'smooth_kernel', 3, ...      % set 0 to skip smoothing
    'reduce_fraction', 0.4, ...  % 0<r<=1, 1 keeps full mesh
    'min_cluster', 16);          % drop tiny voxel islands
```

Outputs:

- `.mat` file with a `surface` struct containing `vertices`, `faces`, `normals`, per-voxel boundary points, and metadata.
- Optional ASCII `.ply` mesh for immediate use in Python (`trimesh`, `open3d`, `pyvista`, Blender, etc.).

Minimal Python loader for the `.mat` surface (requires `scipy`):

```python
import scipy.io
import numpy as np

data = scipy.io.loadmat("exports/reach_surface.mat", squeeze_me=True)
surface = data["surface"]
vertices = np.asarray(surface["vertices"])
faces = np.asarray(surface["faces"], dtype=np.int32) - 1  # MATLAB -> 0-based
normals = np.asarray(surface["normals"])
```

If `scipy` is unavailable, load the `.ply` output instead:

```python
import trimesh
mesh = trimesh.load("exports/reach_surface.ply")
```

Tune the options to trade fidelity for mesh size. The defaults smooth the binary grid, remove voxel speckles smaller than 32 cells, and decimate the mesh to roughly 35% of the original faces.

## Key Features

✅ **Collision Detection**: Uses MATLAB's `checkCollision` with 'adjacent' mode  
✅ **Frame Transforms**: Correct rotation (-90° Z) + translation from URDF  
✅ **Validated Workspace**: All 12,646 stored configs verified collision-free  
✅ **Metadata Tracking**: Build parameters, rejection rates, collision mode  

## Configuration

Edit `build_reachability_map_FK.m` to customize:

- **Grid parameters**: `GRID_ORIGIN`, `GRID_SIZE`, `VOXEL_SIZE`
- **Sampling**: `N_SAMPLES`, `ATTEMPTS_PER_VOXEL`
- **Collision**: `CHECK_COLLISIONS`, collision mode
- **Performance**: `USE_PARFOR` (parallel processing)

## Python Integration

Load map in Python for RL training:

```python
from scripts.reachability_utils import ReachabilityMap

map = ReachabilityMap('matlab/reach_map_mobile_mm_arm_only.mat')
scores = map.query_batch(points_world, in_mobile_frame=False)
```

## Documentation

See `docs/` folder for detailed technical documentation:

- **Build Guide**: `README_BUILD_VISUALIZE.md`
- **Frame Transforms**: `COORDINATE_FRAMES_EXPLAINED.md`
- **FK Approach**: `FK_VS_IK_APPROACH.md`
- **Collision Fix**: `COLLISION_FIX_ARM_ONLY.md`
- **Grid Parameters**: `GRID_PARAMETERS_EXPLAINED.md`

## Troubleshooting

**Build fails with collision error?**  
→ Check URDF meshes are accessible  
→ Try `USE_PARFOR = false` in build script

**Visualization shows points inside chassis?**  
→ Run `tests/visualize_stored_configs` to verify  
→ Check frame transformation has -90° Z-rotation

**Map file too large?**  
→ Increase `VOXEL_SIZE` (e.g., 0.03 → 0.05)  
→ Reduce `GRID_SIZE` if possible

## Contributing

When modifying the build/visualization:

1. Test with `tests/test_chassis_collision.m`
2. Verify with `tests/verify_no_collisions.m`
3. Check frame transforms are correct
4. Update documentation in `docs/`
5. Commit changes with descriptive message

## Status

- ✅ FK-based build working
- ✅ Collision detection validated
- ✅ Frame transformations corrected
- ✅ All configs verified collision-free
- ✅ Ready for RL training (Session 8)
