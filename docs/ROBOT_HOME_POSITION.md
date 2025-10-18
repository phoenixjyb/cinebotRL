# Robot Home Position Configuration

**Date:** October 17, 2025  
**Location:** `src/rl_platform/tasks/mobile_mm/env.py` (lines 134-145)

---

## 🏠 Home Position Definition

The robot's **home position** (also called **default position** or **initial state**) is defined in the `ArticulationCfg.InitialStateCfg`:

```python
init_state=ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),  # Base position at world origin
    joint_pos={
        # Arm joints (mid-range initialization)
        "left_arm_joint1": 0.0,    # Shoulder rotation
        "left_arm_joint2": 1.6,    # Shoulder lift (radians)
        "left_arm_joint3": -1.6,   # Elbow (radians)
        "left_arm_joint4": 0.0,    # Wrist roll
        "left_arm_joint5": 0.0,    # Wrist pitch
        "left_arm_joint6": 0.0,    # Wrist yaw
    },
),
```

---

## 📊 Home Position Breakdown

### Mobile Base (PPR - Planar Prismatic Revolute):
```python
pos=(0.0, 0.0, 0.0)  # (x, y, z) in world coordinates
```
- **X:** 0.0 m (forward/backward)
- **Y:** 0.0 m (left/right)
- **Z:** 0.0 m (up/down, though typically on ground)

The base has 3 DOF but is not explicitly initialized in joint_pos because it uses `pos` instead:
- `joint_x`: 0.0 (implicit from pos)
- `joint_y`: 0.0 (implicit from pos)
- `joint_theta`: 0.0 (facing forward, implicit)

### Arm Joints (6-DOF Manipulator):
| Joint | Name | Home Value | Description | Configuration |
|-------|------|------------|-------------|---------------|
| 1 | `left_arm_joint1` | **0.0 rad** | Shoulder rotation (yaw) | Neutral/centered |
| 2 | `left_arm_joint2` | **1.6 rad** (~91.7°) | Shoulder lift (pitch) | ~Right angle up |
| 3 | `left_arm_joint3` | **-1.6 rad** (~-91.7°) | Elbow (pitch) | ~Right angle down |
| 4 | `left_arm_joint4` | **0.0 rad** | Wrist roll | Neutral |
| 5 | `left_arm_joint5` | **0.0 rad** | Wrist pitch | Neutral |
| 6 | `left_arm_joint6` | **0.0 rad** | Wrist yaw | Neutral |

---

## 🎯 Why This Configuration?

### Mid-Range Position Strategy:
The home position is designed as a **"mid-range"** configuration:

1. **Avoids Joint Limits:**
   - Keeps joints away from extreme positions
   - Provides maximum mobility in all directions
   - Reduces risk of singularities

2. **Balanced Configuration:**
   - Shoulder lift + elbow creates roughly vertical arm
   - End-effector positioned at reasonable working height
   - Good starting point for diverse trajectories

3. **Symmetry:**
   - Joint1, Joint4, Joint5, Joint6 all at 0.0 (neutral)
   - Joint2 and Joint3 mirror each other (+1.6, -1.6)
   - Creates balanced torque distribution

---

## 🔄 Reset Behavior

### During Environment Reset:

**With Randomization (Default):**
```python
if self.task_cfg.randomize_initial_joint_positions:  # Default: True
    noise = torch.randn(len(env_ids), num_joints, device=device) * 0.1  # ±0.1 radians
    default_joint_pos = self.robot.data.default_joint_pos[env_ids]
    self.robot.set_joint_position_target(default_joint_pos + noise, env_ids=env_ids)
```

- Each reset adds Gaussian noise (std=0.1 rad ≈ 5.7°)
- Home position becomes: `home ± random_noise`
- Increases training robustness and exploration

**Without Randomization:**
```python
randomize_initial_joint_positions: bool = False  # In config.py
```
- Robot always resets to exact home position
- More predictable but less robust training

---

## 📐 Visual Representation

```
        Joint2: 1.6 rad (~91.7° up)
              ↗
             /
    Joint1: 0.0 (centered)
            |
     Base: (0, 0, 0)
            |
           ↙ Joint3: -1.6 rad (~91.7° down)
```

**Rough Pose:**
- Shoulder points upward (~90°)
- Elbow bends downward (~90°)
- Result: Arm roughly vertical
- Wrist neutral (all 0.0)

---

## 🔧 Configuration Location

**File:** `src/rl_platform/tasks/mobile_mm/env.py`

```python
def _setup_scene(self):
    """Configure and spawn scene elements."""
    
    robot_cfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=robot_usd_path,
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "left_arm_joint1": 0.0,
                "left_arm_joint2": 1.6,
                "left_arm_joint3": -1.6,
                "left_arm_joint4": 0.0,
                "left_arm_joint5": 0.0,
                "left_arm_joint6": 0.0,
            },
        ),
        # ... actuators config ...
    )
```

---

## 🎛️ Customization

### To Change Home Position:

**1. Edit the configuration:**
```python
# In env.py, modify joint_pos dictionary:
joint_pos={
    "left_arm_joint1": 0.0,    # Your desired value
    "left_arm_joint2": 1.6,    # Your desired value
    # ... etc
}
```

**2. Disable randomization (optional):**
```python
# In config.py or task config:
randomize_initial_joint_positions: bool = False
```

**3. Change noise level (if keeping randomization):**
```python
# In config.py:
initial_joint_noise_std: float = 0.05  # Reduce from 0.1 (less random)
# or
initial_joint_noise_std: float = 0.2   # Increase from 0.1 (more random)
```

---

## 📊 Comparison with Other Common Poses

| Pose Type | Joint2 | Joint3 | Description | Use Case |
|-----------|--------|--------|-------------|----------|
| **Home (Current)** | 1.6 | -1.6 | Mid-range vertical | ✅ Training default |
| Zero Pose | 0.0 | 0.0 | All joints at 0 | Initial testing |
| Tucked | -1.57 | -1.57 | Arm folded back | Storage/transit |
| Extended | 0.0 | 0.0 | Arm straight out | Max reach testing |
| Ready | 0.79 | -0.79 | Partial bend | Manipulation tasks |

---

## 🔍 Verification

### To Check Current Home Position:

**1. In Code:**
```python
print(f"Default joint pos: {self.robot.data.default_joint_pos}")
```

**2. At Runtime:**
```python
# After environment creation:
env.robot.data.default_joint_pos
# Output: tensor([[0.0, 1.6, -1.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], ...])
#                 [x,  y,  θ,   j1,  j2,  j3,  j4,  j5,  j6]
```

**3. Visual Inspection:**
- Run environment without headless mode
- Observe robot pose at episode start
- Should see vertically-oriented arm configuration

---

## 💡 Design Rationale

### Why 1.6 rad (not π/2 = 1.57)?

While π/2 ≈ 1.57 rad would be exactly 90°, using **1.6 rad** provides:
- Slight offset from singularity at exact 90°
- Better numerical stability
- Still roughly vertical for practical purposes
- Avoids potential gimbal lock issues

### Why Symmetric (Joint2 = -Joint3)?

This creates a **balanced vertical configuration**:
- Equal and opposite torques
- Roughly cancels out gravitational moments
- End-effector at predictable height
- Good starting point for reaching tasks in all directions

---

## 📝 Related Configuration

**Randomization Settings** (`config.py`):
```python
@dataclass
class MobileMMTrackConfig:
    # ... other settings ...
    
    # Initial pose randomization
    randomize_initial_joint_positions: bool = True
    initial_joint_noise_std: float = 0.1  # radians (~5.7°)
```

**Actuator Settings** (`env.py`):
```python
actuators={
    "arm": ImplicitActuatorCfg(
        joint_names_expr=["left_arm_joint[1-6]"],
        stiffness=400.0,
        damping=40.0,
    ),
    "base": ImplicitActuatorCfg(
        joint_names_expr=["joint_x", "joint_y", "joint_theta"],
        stiffness=10000.0,
        damping=1000.0,
    ),
}
```

---

## 🎯 Summary

**Robot Home Position:**
- **Base:** Origin (0, 0, 0) in world coordinates
- **Arm Joints:** Mid-range configuration
  - Shoulder lift: **1.6 rad** (~91.7° up)
  - Elbow: **-1.6 rad** (~91.7° down)
  - Other joints: **0.0 rad** (neutral)
- **Randomization:** ±0.1 rad Gaussian noise on reset (default)
- **Purpose:** Balanced starting configuration avoiding joint limits and singularities

**To Modify:** Edit `joint_pos` dictionary in `env.py` `ArticulationCfg.InitialStateCfg`
