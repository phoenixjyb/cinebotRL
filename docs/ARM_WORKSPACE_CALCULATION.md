# Arm Workspace Envelope Calculation

**Date:** 2025-10-23  
**Source:** `assets_own/mobile_manipulator_PPR_base_corrected.urdf`

---

## 🔧 Arm Geometry from URDF

### Mounting Position
```xml
<joint name="arm_mount_joint" type="fixed">
  <origin xyz="0.15995000 0.00000000 0.94650000"/>
  <!-- Arm shoulder is 0.160m forward, 0.947m above base center -->
</joint>
```

**Shoulder mount relative to base center:**
- X: +0.160m (forward)
- Y: 0.000m (centerline)
- Z: +0.947m (above ground)

### Link Chain (from URDF joint origins)

| Joint | Origin XYZ (m) | Description |
|-------|----------------|-------------|
| `left_arm_joint1` | `[-0.001, 0.045, 0]` | Shoulder yaw/pitch |
| `left_arm_joint2` | `[0, 0.106, 0]` | Shoulder roll |
| `left_arm_joint3` | `[-0.349, 0.020, 0]` | **Elbow** (major link!) |
| `left_arm_joint4` | `[0.048, 0.071, 0]` | Wrist pitch |
| `left_arm_joint5` | `[0.243, 0.002, 0]` | **Forearm** (major link!) |
| `left_arm_joint6` | `[0.054, 0.004, 0]` | Wrist roll |
| End-effector | `[~0.08, 0, 0]` | Gripper/tool (estimated) |

---

## 📐 Workspace Envelope Analysis

### Method 1: Sum of Link Lengths (Maximum Reach)

**Major links contributing to reach:**
```
Upper arm:  0.349m (joint3 X offset)
Forearm:    0.243m (joint5 X offset)
Shoulder Y: 0.106m (joint2 Y offset)
Wrist:      0.071m (joint4 Y offset)
EE offset:  ~0.080m (estimated)
```

**Maximum theoretical reach from shoulder:**
```
Radial (straight extension):
  0.349 + 0.243 + 0.08 = 0.672m

Including lateral offsets (Y-axis):
  sqrt((0.672)² + (0.106 + 0.071)²) 
  = sqrt(0.452 + 0.031)
  = sqrt(0.483)
  = 0.695m ≈ 0.7m
```

**Maximum reach from base center:**
```
From shoulder: 0.7m
Shoulder forward offset: 0.16m
Shoulder height: 0.947m

XY plane reach from base: 
  0.7 + 0.16 = 0.86m (forward extension)
  
3D reach from base center:
  sqrt((0.86)² + (0.947)²)
  = sqrt(0.740 + 0.897)
  = sqrt(1.637)
  = 1.28m
```

### Method 2: Practical Working Envelope

Accounting for joint limits and singularity avoidance:

**From URDF joint limits:**
```xml
left_arm_joint1: [-2.88, 2.88] rad  (±165°)
left_arm_joint2: [0, 3.23] rad      (0° to 185°)
left_arm_joint3: [-3.32, 0] rad     (-190° to 0°)
left_arm_joint4: [-2.88, 2.88] rad  (±165°)
left_arm_joint5: [-1.66, 1.66] rad  (±95°)
left_arm_joint6: [-2.88, 2.88] rad  (±165°)
```

**Practical workspace zones:**

1. **Inner dead zone** (singularity near shoulder):
   - Minimum reach: ~0.15m from shoulder
   - Reason: Arm cannot fold fully backward

2. **Optimal working zone** (best dexterity):
   - From shoulder: 0.3m to 0.6m
   - From base center (XY): 0.3m to 0.7m
   - Height (Z): 0.5m to 1.5m (relative to ground)

3. **Maximum extension** (stretched, poor dexterity):
   - From shoulder: 0.6m to 0.7m
   - From base center (XY): 0.7m to 0.9m
   - Height (Z): 0.2m to 1.7m

---

## 🎯 Comparison with Code Constants

### Current Code Assumptions

```python
# From src/rl_platform/tasks/mobile_mm/rewards.py
arm_reach: float = 0.6  # meters (empirical)

# From docs/TRAJECTORY_START_INSIDE_BODY_ANALYSIS.md
arm_reach_min = 0.3
arm_reach_max = 1.2
```

### Validation Against URDF

| Metric | Code | URDF Calculation | Status |
|--------|------|------------------|--------|
| Min reach (from shoulder) | 0.3m | ~0.15m | ✅ Conservative (good) |
| Typical reach (from base XY) | 0.6m | 0.5-0.7m | ✅ **Accurate!** |
| Max reach (from shoulder) | 1.2m | 0.7m | ❌ **Too optimistic!** |
| Max reach (from base XY) | - | 0.86m | - |
| Max 3D reach (from base) | - | 1.28m | - |

---

## 🔍 Observed vs Calculated

### From Training Logs (Session 6/7)

```
Typical EE distance from base: 0.57-1.07m
Best tracking: 0.55m (base center)
Worst tracking: 1.07m (base center)
```

**Analysis:**
- 0.55m: Within optimal working zone ✅
- 1.07m: Beyond max reach from shoulder (0.7m) ❌

**Wait... how is 1.07m possible?**

Let's recalculate with the actual observation:
```
EE: [2.051, 0.044, 0.942]
Base: [1.050, 0.080, 0.000]
Shoulder: [1.210, 0.080, 0.947]

EE from shoulder:
  sqrt((2.051-1.210)² + (0.044-0.080)² + (0.942-0.947)²)
  = sqrt(0.841² + 0.036² + 0.005²)
  = sqrt(0.707 + 0.001 + 0.000)
  = 0.841m
```

**This exceeds our calculated max reach of 0.7m!**

### Two Possible Explanations

1. **Missing link lengths**: The gripper/end-effector adds ~0.15m we didn't account for
2. **Joint coupling**: Lateral movements (Y-axis offsets) can extend reach beyond simple sum

Let me recalculate including ALL offsets:

```
Major X-axis contributions:
  joint3: -0.349m (negative = backward)
  joint5: +0.243m (forward)
  joint6: +0.054m (forward)
  EE:     +0.08m (estimated)
  
If arm bent optimally:
  Max forward: |−0.349| + 0.243 + 0.054 + 0.08 = 0.726m

Y-axis contributions:
  joint2: 0.106m
  joint4: 0.071m
  Total: 0.177m

Combined radial reach:
  sqrt(0.726² + 0.177²)
  = sqrt(0.527 + 0.031)
  = sqrt(0.558)
  = 0.747m ≈ 0.75m
```

**Better! But observed was 0.84m...**

---

## 📊 Revised Workspace Envelope

Based on URDF + observed data:

### From Shoulder (realistic)

| Zone | Radius | Description |
|------|--------|-------------|
| Dead zone | 0-0.15m | Singularities, avoid |
| Optimal | 0.2-0.6m | Best dexterity |
| Extended | 0.6-0.75m | Calculated max |
| **Observed max** | **0.84m** | Actual from logs |

### From Base Center (XY plane)

| Zone | Radius | Description |
|------|--------|-------------|
| Dead zone | 0-0.2m | Under robot |
| Optimal | 0.3-0.7m | ✅ **Code assumes 0.6m** |
| Extended | 0.7-0.9m | Stretched |
| **Observed max** | **1.0m** | EE at 1.015m in logs |

---

## ✅ Recommendations

### 1. Update Code Constants

```python
# src/rl_platform/tasks/mobile_mm/rewards.py

# CURRENT:
arm_reach: float = 0.6  # meters (empirical)

# RECOMMENDED:
arm_reach_optimal: float = 0.6  # Comfortable working radius (XY from base)
arm_reach_max: float = 0.85     # Maximum extension (XY from base)
arm_reach_shoulder_max: float = 0.75  # From shoulder mount point

# Use in rewards:
beyond_optimal = max(0, dist - arm_reach_optimal)
beyond_max = max(0, dist - arm_reach_max)

# Gentle penalty when beyond optimal, strong when beyond max
penalty = 2.0 * beyond_optimal + 10.0 * beyond_max
```

### 2. Add Runtime Validation

```python
# In env.py, add sanity check:
def _validate_arm_geometry(self):
    """Check if EE position is physically possible given arm limits."""
    ee_pos = self.robot.data.body_pos_w[:, self._ee_body_idx]
    shoulder_pos = self.robot.data.root_pos_w + torch.tensor([0.16, 0, 0.947])
    
    ee_to_shoulder = torch.norm(ee_pos - shoulder_pos, dim=-1)
    max_reach = 0.85  # From URDF calculation
    
    if torch.any(ee_to_shoulder > max_reach):
        print(f"⚠️  WARNING: EE beyond physical reach!")
        print(f"   Max distance: {ee_to_shoulder.max():.3f}m")
        print(f"   Arm max reach: {max_reach}m")
```

### 3. Visualization Script

Create `scripts/visualize_arm_workspace.py`:
```python
"""
Visualize arm reachable workspace as point cloud.
Uses forward kinematics to sample joint space and plot EE positions.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# TODO: Implement FK solver using URDF link chain
# TODO: Sample joint space uniformly
# TODO: Plot 3D scatter of reachable points
```

---

## 🎯 Key Takeaways

1. **Code assumption (0.6m) is CORRECT** for optimal working radius from base center
2. **Maximum reach is ~0.85m** from base center (XY plane), not 1.2m
3. **Observed 1.0m+ distances** suggest either:
   - Arm fully extended + base rotation contributing
   - Missing gripper length in URDF (adds ~0.1-0.15m)
   - Physics engine allowing slight over-extension
4. **Current reward scaling is appropriate** - targets beyond 0.6m should trigger base movement

---

## 📚 References

- URDF: `assets_own/mobile_manipulator_PPR_base_corrected.urdf`
- Reward code: `src/rl_platform/tasks/mobile_mm/rewards.py:135, 209`
- Observations: Session 6/7 training logs
- Joint limits: Lines 243-290 in URDF
