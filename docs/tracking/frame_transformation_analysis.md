# Frame Transformation Analysis - Base Movement Issue

**Date**: 2024-10-27  
**Context**: Investigating why base doesn't move despite policy commanding movement

## Critical Bug Discovery

Example from Session 7b (Env 1478, worst case):
```
🚗 Base Pos (root_pos_w):  [  1.050,   0.080,   0.000]  ← World position
🔧 PPR offsets (joint_pos): [ -0.251,  -6.303,   1.361]  ← Joint positions
🚗 Base Moved:             0.0000 m  ← Displacement from episode start

Discrepancy: 6.38m difference between joint_y and root Y position!
```

**Symptom**: PPR joints show large values (joint_y = -6.3m) but `root_pos_w` stays nearly frozen (Y = 0.08m).

---

## Three Coordinate Frames in System

### 1. **World Frame** (Isaac Sim Global)
- **Purpose**: Ground truth for physics simulation and rewards
- **Variables**: 
  - `root_pos_w`: Robot root body position in world [X, Y, Z]
  - `body_pos_w`: All body positions including EE in world
  - `root_lin_vel_w`, `root_ang_vel_w`: Velocities in world
- **Usage**: 
  - Observations use `root_pos_w` for base position (line 911)
  - Rewards calculate EE error using `body_pos_w - target_pos` (both world frame)
- **Properties**: Absolute coordinates, never resets

### 2. **Base Frame (Chassis)**
- **Purpose**: Robot's local coordinate system
- **In URDF**: `abstract_chassis_link` is the upper structure body
- **Relationship to World**: 
  - Position: `root_pos_w` (the articulation root)
  - Orientation: `root_quat_w`
- **Usage**: 
  - Policy actions are in body frame: [vx, vy, ω] where vx is "forward"
  - Transformed to world via: `dx = vx*cos(θ)`, `dy = vx*sin(θ)`

### 3. **Arm Base Frame**
- **Purpose**: Origin for reachability map queries
- **In URDF**: `left_arm_base_link` attached to chassis via `arm_mount_joint`
- **Position**: Fixed offset from chassis at `[0.160, 0.0, 0.947]` meters
- **Usage**: 
  - `reach_map.world_to_arm_frame()` transforms target positions
  - KD-tree query checks if target reachable from arm base
- **Properties**: Moves with chassis, offset never changes

---

## Kinematic Chain (from URDF)

```
world (Isaac Sim origin)
  │
  └─ base (root link) ← PhysX articulation root, THIS IS root_pos_w
       │
       └─ [joint_x: prismatic X] ± 50m limits
            │
            └─ base_link_x (1.0kg helper link)
                 │
                 └─ [joint_y: prismatic Y] ± 50m limits
                      │
                      └─ base_link_y (1.0kg helper link)
                           │
                           └─ [joint_theta: revolute Z] ± 2π limits
                                │
                                └─ abstract_chassis_link (31kg main body)
                                     │
                                     └─ [arm_mount_joint: fixed]
                                          │
                                          └─ left_arm_base_link
                                               │
                                               └─ [left_arm_joint1...6]
```

**KEY INSIGHT**: The `base` link is the PhysX root. PPR joints (joint_x, joint_y, joint_theta) connect intermediate helper links that ultimately position the `abstract_chassis_link`.

---

## The Fundamental Problem

### Current Implementation (WRONG)

**In `_apply_actions()` (lines 821-830)**:
```python
# Calculate displacement in world frame
theta = robot.data.joint_pos[:, 2]  # joint_theta (orientation)
dx = base_vx_scaled * cos(theta) * dt  # World X displacement
dy = base_vx_scaled * sin(theta) * dt  # World Y displacement
dtheta = base_wz_scaled * dt

# Add to current joint positions
current_base_pos = robot.data.joint_pos[:, _base_joint_ids]  # [joint_x, joint_y, joint_theta]
new_base_targets = current_base_pos + [dx, dy, dtheta]

# Command joints
robot.set_joint_position_target(new_base_targets, joint_ids=[0,1,2])
```

**In `_reset_idx()` (lines 1310-1340)**:
```python
# Set root state directly to trajectory start
new_root_state[:, 0:2] = first_target_pos[:, 0:2]  # World XY
robot.write_root_state_to_sim(new_root_state, env_ids)

# ALSO set joint positions to same values
base_joint_pos[:, 0:2] = first_target_pos[:, 0:2]  # joint_x, joint_y
robot.set_joint_position_target(base_joint_pos, joint_ids=[0,1,2], env_ids)
```

### Why This Breaks

1. **Dual Control Conflict**:
   - `write_root_state_to_sim()` directly teleports the `base` (root) link to world position
   - `set_joint_position_target()` commands PPR joints to position intermediate links
   - **These fight each other!** PhysX tries to satisfy both constraints simultaneously

2. **Incorrect Assumption**:
   - Code assumes `joint_x` position = world X position (1:1 mapping)
   - **Reality**: `joint_x` is a local displacement from parent link `base`
   - When `base` (root) is teleported via `write_root_state_to_sim()`, joints measure **relative offsets**
   - But code treats them as **absolute world coordinates**!

3. **Accumulation Error**:
   - Every step: `new_joint_pos = current_joint_pos + dx`
   - If `base` (root) doesn't move (locked by previous `write_root_state_to_sim()`), joints accumulate displacement
   - Result: `joint_x` goes to -0.251m, `joint_y` goes to -6.303m, but `root_pos_w` stays at [1.050, 0.080]
   - This explains the 6.38m discrepancy!

4. **Comment Confirms Design Flaw** (line 905-906):
   ```python
   # PPR joints are used for COMMANDING movement, but root_pos_w reflects 
   # the actual simulated position
   ```
   - Previous developer acknowledged joints and root_pos can differ
   - But this is fundamentally wrong for a mobile base!

---

## Frame Transformation Flow (Current vs Expected)

### Current (BROKEN) Flow:

```
Policy Action [vx, vy, ω] (body frame)
  ↓
Transform to World: dx = vx*cos(θ), dy = vx*sin(θ)
  ↓
Add to joint positions: joint_x += dx, joint_y += dy  ← WRONG: treats joints as world coords
  ↓
set_joint_position_target(new_joint_pos)
  ↓
PhysX tries to move intermediate links (base_link_x, base_link_y)
  ↓
BUT: root link "base" locked/constrained by previous write_root_state_to_sim()
  ↓
RESULT: Joints move but root_pos_w doesn't → discrepancy grows
```

### Expected (CORRECT) Flow Option A - Direct Root Control:

```
Policy Action [vx, vy, ω] (body frame)
  ↓
Transform to World: dx = vx*cos(θ), dy = vx*sin(θ)
  ↓
Integrate velocity to root state: root_vel_w = [dx/dt, dy/dt, 0]
  ↓
write_root_link_velocity_to_sim(root_vel_w)  ← Direct velocity control
  ↓
PhysX integrates velocity → root_pos_w updates
  ↓
RESULT: Base actually moves, root_pos_w changes
```

### Expected (CORRECT) Flow Option B - Pure Joint Control:

```
Policy Action [vx, vy, ω] (body frame)
  ↓
Transform to World: dx = vx*cos(θ), dy = vx*sin(θ)
  ↓
Set joint VELOCITIES: joint_vel_x = dx/dt, joint_vel_y = dy/dt
  ↓
set_joint_velocity_target(joint_vels)
  ↓
PhysX simulates joint motion → kinematic chain propagates → root_pos_w updates
  ↓
RESULT: Base moves via joint actuation, root_pos_w follows kinematics
```

---

## Evidence from Code

### 1. Displacement Tracking Confirms Frozen Base (Session 7c addition)
```python
# Line 558-565
base_displacement = torch.norm(
    base_pos_world[:, :2] - self._episode_start_base_pos[:, :2], 
    dim=-1
)
print(f"🚗 Base Movement from Start (m): mean={base_displacement.mean():.4f}")
```
**Result**: Mean displacement = 0.002m (2mm) over entire episode → **base frozen**

### 2. Joint Positions Show Large Accumulated Error
```python
# Line 522
base_ppr = self.robot.data.joint_pos[:, 0:3]  # [joint_x, joint_y, joint_theta]
```
**Result**: Values like [-0.251, -6.303, 1.361] → **joints drifting wildly**

### 3. Action Integration Uses World Frame Math
```python
# Lines 821-822
dx = base_vx_scaled.squeeze(-1) * torch.cos(theta) * dt  # X displacement in global frame
dy = base_vx_scaled.squeeze(-1) * torch.sin(theta) * dt  # Y displacement in global frame
```
**Assumption**: `dx`, `dy` are in world frame, can be added to joint positions  
**Reality**: Joint positions are NOT world coordinates!

### 4. Reset Does Dual Teleport
```python
# Lines 1315-1318: Teleport root
new_root_state[:, 0:2] = first_target_pos[:, 0:2]
robot.write_root_state_to_sim(new_root_state, env_ids)

# Lines 1335-1338: Teleport joints
base_joint_pos[:, 0:2] = first_target_pos[:, 0:2]
robot.set_joint_position_target(base_joint_pos, joint_ids=_base_joint_ids, env_ids)
```
**Problem**: Setting same values via two different mechanisms creates constraint conflict

---

## Proposed Fix: Option A - Direct Root Velocity Control

**Rationale**: 
- Simplest and most reliable
- Avoids joint/root desynchronization
- PhysX handles integration correctly
- No kinematic chain complexity

### Implementation Changes

#### 1. Remove Joint-Based Base Control

**In `_apply_actions()`** (lines 750-850):

```python
# REMOVE: Joint position integration
# current_base_pos = self.robot.data.joint_pos[:, _base_joint_ids]
# new_base_targets = current_base_pos + position_deltas
# self.robot.set_joint_position_target(new_base_targets, joint_ids=_base_joint_ids)

# NEW: Direct root velocity control
root_vel_w = torch.zeros(self.num_envs, 6, device=self.device)
root_vel_w[:, 0] = base_vx_scaled * torch.cos(theta)  # World vx
root_vel_w[:, 1] = base_vx_scaled * torch.sin(theta)  # World vy
root_vel_w[:, 5] = base_wz_scaled  # Angular velocity around Z

self.robot.write_root_link_velocity_to_sim(root_vel_w)
```

#### 2. Simplify Reset (Remove Dual Control)

**In `_reset_idx()`** (lines 1310-1360):

```python
# Set root state (position + zero velocity)
new_root_state[:, 0:2] = first_target_pos[env_ids, 0:2]
new_root_state[:, 7:13] = 0.0  # Zero velocities
self.robot.write_root_state_to_sim(new_root_state, env_ids)

# REMOVE: Joint position reset
# base_joint_pos[:, 0:2] = first_target_pos[:, 0:2]
# self.robot.set_joint_position_target(base_joint_pos, ...)

# NEW: Set joint positions to ZERO (they're offsets from root)
base_joint_pos = torch.zeros(len(env_ids), 3, device=self.device)
self.robot.set_joint_position_target(
    base_joint_pos, 
    joint_ids=self._base_joint_ids,
    env_ids=env_ids
)
```

#### 3. Update Observations (Already Correct)

Observations already use `root_pos_w` (line 911), so no changes needed:
```python
base_pos = self.robot.data.root_pos_w.clone()  # ✓ Correct
```

#### 4. Update Comments

Remove misleading comment at lines 905-906:
```python
# OLD: PPR joints are used for COMMANDING movement, but root_pos_w reflects 
#      the actual simulated position

# NEW: Base position controlled directly via root state velocity commands.
#      PPR joints remain at zero offset (kinematic helpers only).
```

---

## Alternative Fix: Option B - Pure Joint Control

**Rationale**:
- More physically realistic (joints actually drive motion)
- Better for sim-to-real transfer
- Requires careful joint velocity control

### Implementation (Sketch)

```python
# In _apply_actions():
# Get current root orientation
theta = self.robot.data.root_quat_w  # Use quaternion for proper rotation

# Transform body-frame velocities to world frame
world_vel = transform_velocity(body_vel=[vx, vy, wz], orientation=theta)

# Set joint VELOCITIES (not positions!)
joint_vel_targets = torch.zeros(self.num_envs, 3, device=self.device)
joint_vel_targets[:, 0] = world_vel[:, 0]  # joint_x velocity
joint_vel_targets[:, 1] = world_vel[:, 1]  # joint_y velocity  
joint_vel_targets[:, 2] = world_vel[:, 2]  # joint_theta velocity

self.robot.set_joint_velocity_target(joint_vel_targets, joint_ids=_base_joint_ids)

# In _reset_idx():
# DO NOT use write_root_state_to_sim() - let joints position the root
# Just set joint positions to trajectory start
base_joint_pos[:, 0:2] = first_target_pos[:, 0:2]
base_joint_pos[:, 2] = 0.0
self.robot.set_joint_position_target(base_joint_pos, joint_ids=_base_joint_ids, env_ids)
```

**Challenges**:
- Need to verify Isaac Lab supports joint velocity targets for prismatic joints
- May introduce oscillations (needs tuning of joint damping/stiffness)
- Root state initialization more complex

---

## Recommendation

**Choose Option A (Direct Root Velocity Control)** for these reasons:

1. **Simplicity**: One control mechanism, no synchronization issues
2. **Reliability**: PhysX velocity integration is well-tested
3. **Performance**: No joint solver overhead
4. **Debugging**: Easier to verify base movement
5. **Precedent**: Other IsaacLab mobile bases likely use this approach

**Implementation priority**:
1. Fix `_apply_actions()` to use `write_root_link_velocity_to_sim()` ✅ HIGH
2. Fix `_reset_idx()` to remove dual control ✅ HIGH  
3. Add debug logging to verify root_vel_w is set correctly ✅ MEDIUM
4. Test with 4 envs, 1000 steps, constant vx=0.5 m/s ✅ HIGH
5. Verify displacement >0.5m after 1 second ✅ HIGH

---

## Testing Plan

### Phase 1: Small-Scale Validation (30 minutes)

Create `scripts/test_base_movement_fix.py`:
```python
# 4 envs, simple forward motion
# Command: vx = 0.5 m/s, vy = 0, wz = 0
# Duration: 1000 steps (10 seconds at 100Hz)
# Expected: displacement >5.0m in 10 seconds
# Monitor: root_pos_w, root_vel_w, joint_pos
```

Success criteria:
- `root_pos_w[:, 0]` increases by ~5.0m ± 0.5m
- `root_vel_w[:, 0]` ≈ 0.5 m/s during motion
- `joint_pos` stays near zero (no accumulation)

### Phase 2: Session 7c Launch (After validation)

Monitor metrics:
- Base displacement mean >0.1m (vs 0.002m in 7b)
- Base alignment positive (vs -0.053 in 7b)
- Mean error <1.0m (vs 2.45m in 7b, 1.4m in 7)
- Environment health <50% broken

---

## Session History Context

| Session | Base Displacement | Mean Error | Issue |
|---------|------------------|------------|-------|
| 7       | 0.0008-0.0017m   | 1.3-1.6m   | Base frozen, arm overextending |
| 7b      | 0.002m           | 2.45m      | Base frozen, reward gaming |
| 7c      | **TBD**          | **TBD**    | **Awaiting base movement fix** |

**Root cause of ALL failures**: Base never moved due to incorrect frame transformations and dual control mechanism conflict.

**Expected outcome after fix**: Base will move toward targets (>0.1m mean), arm stays within reach limits, mean error drops below 1.0m.

---

## References

- URDF: `assets_own/mobile_manipulator_PPR_base_corrected.urdf`
- Environment: `src/rl_platform/tasks/mobile_mm/env.py`
- Config: `src/rl_platform/tasks/mobile_mm/config.py`
- Session logs: `H:\wSpace\cinebotRL\logs\sb3\mobilemmtrackee_v0\`

---

**Next Steps**: Implement Option A fix, test with small validation script, then launch Session 7c.
