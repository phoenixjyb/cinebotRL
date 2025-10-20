# PPR Control Architecture Explanation

**Date**: October 21, 2025  
**Context**: Clarifying (v_x, ω_z) velocity commands → PPR position control

## Your Control Flow

### 1. Policy Action Space (8D)
```python
action = [
    arm_joint1_pos,  # 6 arm joints (position targets)
    arm_joint2_pos,
    arm_joint3_pos,
    arm_joint4_pos,
    arm_joint5_pos,
    arm_joint6_pos,
    v_x,             # Linear velocity (m/s) in robot frame
    ω_z              # Angular velocity (rad/s)
]
```

**Action space is HYBRID**:
- Arm: Position targets (direct joint angles)
- Base: Velocity commands (v_x, ω_z)

### 2. Velocity Integration in env.py

**Code** (`src/rl_platform/tasks/mobile_mm/env.py` lines 710-724):
```python
# Policy outputs normalized velocities [-1, 1]
base_vx = actions[:, 6:7]  # Normalized
base_wz = actions[:, 7:8]  # Normalized

# Scale to physical limits
base_vx_scaled = base_vx * 1.5      # [-1.5, +1.5] m/s
base_wz_scaled = base_wz * 2.5      # [-2.5, +2.5] rad/s

# Get current orientation (for differential drive)
theta = current_base_pos[:, 2]  # Current yaw angle

# Integrate velocities → position deltas (differential drive kinematics)
dt = 0.02  # 50Hz control frequency
dx = base_vx_scaled * torch.cos(theta) * dt       # World frame X
dy = base_vx_scaled * torch.sin(theta) * dt       # World frame Y
dtheta = base_wz_scaled * dt                      # Rotation

# Compute new PPR position targets
position_deltas = torch.stack([dx, dy, dtheta], dim=1)
new_base_targets = current_base_pos + position_deltas

# Apply as POSITION commands to ALL PPR joints
robot.set_joint_position_target(
    target=new_base_targets,          # [num_envs, 3]: [x, y, theta]
    joint_ids=self._base_joint_ids    # [joint_x, joint_y, joint_theta]
)
```

### 3. PhysX Execution

**ALL PPR joints use Position Control**:

```
Policy: v_x = 1.0 m/s, ω_z = 0.5 rad/s
   ↓
env.py integration (dt=0.02s):
   dx = 1.0 * cos(theta) * 0.02 = 0.02m
   dy = 1.0 * sin(theta) * 0.02 = ~0m (if theta≈0)
   dtheta = 0.5 * 0.02 = 0.01 rad
   ↓
PPR position targets:
   joint_x: current + 0.02 → NEW POSITION
   joint_y: current + 0.0  → HOLD
   joint_theta: current + 0.01 → NEW POSITION
   ↓
PhysX PD controllers:
   - joint_x: Apply spring force toward target (10kN/m stiffness)
   - joint_y: Apply spring force toward target
   - joint_theta: Apply torque toward target
   ↓
Result: Base moves forward 0.02m and rotates 0.01 rad in 20ms
```

## Why This Design?

### Advantages:
1. **Policy outputs intuitive velocities**: Easier to learn (v_x, ω_z) than position deltas
2. **Differential drive kinematics**: Matches mobile robot conventions (cmd_vel)
3. **Position control prevents drift**: PhysX spring-damper provides stability
4. **Smooth control**: PD controller smooths discrete velocity commands

### Implementation Details:
- **Rate limiting**: Velocity changes clamped to prevent jerky motion
- **Orientation coupling**: dx/dy computed in world frame using current theta
- **Zero-mass helpers**: PPR intermediate links (base_link_x/y) have mass=0.0 for stability
- **Spring-damper**: 10kN/m stiffness + 1kN·s/m damping (from env.py)

## Isaac Sim USD Import Settings

### ❌ YOUR SCREENSHOT IS WRONG:

Your screenshot shows:
```
joint_theta: Target = "Velocity"  ← INCORRECT!
```

### ✅ CORRECT SETTINGS:

**All PPR joints should be "Position" targets**:
```
joint_x:     Target = "Position"  ← Correct
joint_y:     Target = "Position"  ← Correct
joint_theta: Target = "Position"  ← Change from "Velocity"!
```

**Why**:
- Your code calls `set_joint_position_target()` for ALL three joints
- Even though policy outputs velocities, they are integrated to positions before PhysX
- PhysX receives position commands, not velocity commands

## Comparison: Position vs Velocity Control in USD

### If using Position Control (CORRECT for your code):
```python
# env.py sends:
robot.set_joint_position_target(new_theta_pos, joint_ids=[joint_theta])

# PhysX executes:
torque = K_p * (target_pos - current_pos) - K_d * current_vel
# Where K_p = 10,000 N/m (stiffness), K_d = 1,000 N·s/m (damping)
```

### If using Velocity Control (WRONG for your code):
```python
# env.py would need to send:
robot.set_joint_velocity_target(omega_z, joint_ids=[joint_theta])

# PhysX would execute:
torque = K_v * (target_vel - current_vel)
# Velocity PD controller (different dynamics)
```

**Your code uses Position Control**, so USD must match!

## Action Required

### In Isaac Sim URDF Importer Dialog:

1. **Find the joint configuration table** (shown in your screenshot)
2. **Locate row 1**: `joint_theta`
3. **Change "Target" dropdown**: "Velocity" → **"Position"**
4. **Verify rows 2-3**: `joint_x` and `joint_y` are already "Position" ✅
5. **Verify rows 4+**: All arm joints are "Position" ✅

### Why This Matters:

**If joint_theta stays as "Velocity" control**:
- Your `set_joint_position_target()` call will be misinterpreted
- PhysX will treat position commands as velocity setpoints
- Base rotation will be unstable or frozen
- Training will fail due to control mismatch

**After fixing to "Position" control**:
- Position commands interpreted correctly
- PD spring-damper (10kN/m) provides smooth, stable control
- Base can rotate freely within ±2π limits
- Training should work correctly

## Summary

| Component | Value | Notes |
|-----------|-------|-------|
| **Policy action space** | 8D: [arm(6), v_x, ω_z] | Hybrid: positions + velocities |
| **env.py integration** | v_x, ω_z → dx, dy, dtheta | Differential drive kinematics |
| **PhysX command** | `set_joint_position_target()` | ALL PPR joints |
| **USD joint targets** | **Position** (all 3 PPR) | Must match `set_joint_position_target()` |
| **Control frequency** | 50 Hz (dt=0.02s) | Velocity integration timestep |
| **Spring stiffness** | 10,000 N/m | From env.py PD controller |
| **Damping** | 1,000 N·s/m | From env.py PD controller |

**Critical Fix**: Change `joint_theta` from "Velocity" → "Position" in Isaac Sim import dialog!

---

**References**:
- env.py implementation: `src/rl_platform/tasks/mobile_mm/env.py:710-724`
- Action space definition: `src/task_spec.py:40-43`
- USD import settings: Screenshot analysis
