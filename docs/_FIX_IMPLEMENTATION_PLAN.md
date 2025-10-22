# Critical Fixes Implementation Plan

**Date:** 2025-10-22  
**Based on:** Code Review Validation Analysis  
**Target:** Fix 3 critical issues blocking base learning

---

## Fix Priority Matrix

| Issue | Impact | Effort | Priority | ETA |
|-------|--------|--------|----------|-----|
| 1. Jerk Penalty Too Harsh | CRITICAL - Blocks base movement | 15 mins | 🔴 P0 | Immediate |
| 2. Contact Forces Zero | CRITICAL - No collision avoidance | 1-2 hours | 🔴 P0 | Same day |
| 3. USD Bundle Verification | CRITICAL - Base rotation may not work | 30 mins | 🔴 P0 | Same day |
| 4. Lateral Motion Frame | MEDIUM - Affects movement quality | 30 mins | 🟡 P1 | Next |
| 5. prev_joint_vel Bug | LOW - Only first step | 10 mins | 🟢 P2 | Nice to have |

---

## Fix 1: Jerk Penalty Too Harsh (P0 - CRITICAL)

### Problem
Current jerk penalty is unrealistic:
- Max jerk limit: 5 m/s³
- Actual jerk during normal acceleration: 100 m/s³
- Penalty: ~900 points (dominates all other rewards)
- **Result:** Policy learns to freeze base

### Root Cause
```python
# src/rl_platform/tasks/mobile_mm/rewards.py:481-483
jerk = (current_accel - prev_accel) / dt  # Uses COMMANDED accelerations
jerk_mag = torch.norm(jerk, dim=-1)
violation = torch.clamp(jerk_mag - max_jerk, min=0.0)  # max_jerk = 5.0 too low
return scale * violation ** 2  # scale = 0.1
```

### Solution Options

#### Option A: Raise Jerk Limit (RECOMMENDED - Quick Fix)
**Effort:** 5 minutes  
**Risk:** Low

**Change in `src/task_spec.py`:**
```python
# Current
"max_linear_jerk": 5.0,  # m/s^3

# Fixed
"max_linear_jerk": 50.0,  # m/s^3 (realistic for mobile robots)
```

**Justification:**
- Mobile robots can handle 50-100 m/s³ jerk
- TurtleBot typical: ~80 m/s³
- Our robot (30kg): 50 m/s³ is reasonable

#### Option B: Lower Penalty Scale
**Effort:** 5 minutes  
**Risk:** Low

**Change in `src/rl_platform/tasks/mobile_mm/rewards.py`:**
```python
# Around line 677 (reward weights)
"jerk_limit_penalty": 0.001,  # Was 0.1
```

#### Option C: Use PhysX Accelerations (Better Long-Term)
**Effort:** 1 hour  
**Risk:** Medium (need to ensure PhysX data is correct)

**Changes needed:**
1. Pass actual `base_lin_vel` and `prev_base_lin_vel` to jerk_penalty
2. Compute accel = (current_vel - prev_vel) / dt
3. Use these accelerations instead of commanded

### Recommended Fix (Hybrid)
1. **Immediate (5 mins):** Raise limit to 50.0 m/s³
2. **Next week:** Switch to PhysX accelerations (more accurate)

**Code Change:**
```python
# src/task_spec.py (line ~35)
def get_robot_limits() -> dict[str, float]:
    """Physical limits of the mobile manipulator."""
    return {
        "max_linear_velocity": 1.5,        # m/s
        "max_angular_velocity": 2.0,       # rad/s
        "max_linear_acceleration": 1.0,    # m/s^2
        "max_angular_acceleration": 5.0,   # rad/s^2
        "max_linear_jerk": 50.0,           # m/s^3 (was 5.0) ✅ CHANGED
        "arm_reach": 0.6,                  # m
    }
```

---

## Fix 2: Contact Forces Reading Zero (P0 - CRITICAL)

### Problem
- PhysX reports `Max contact force: 0.0000 N` every step
- No self-collision penalties
- No termination on arm-chassis collisions
- Policy never learns to avoid bad base positions

### Root Cause Analysis Needed
Need to check 3 potential issues:

#### Issue 2a: Wrong PhysX Array
**Check:** `src/rl_platform/tasks/mobile_mm/env.py` (contact force reading)

**Current code location:** Search for `contact_forces` or `net_contact_forces`

**Action:**
```python
# Add debug print in env.py after physics step
print(f"Available contact data:")
print(f"  - net_contact_forces: {self.robot.data.net_contact_forces.shape}")
print(f"  - contact_force_matrix: {self.robot.data.contact_force_matrix.shape if hasattr(...) else 'N/A'}")
print(f"  - Max force: {self.robot.data.net_contact_forces.abs().max()}")
```

#### Issue 2b: Chassis Collision Disabled in USD
**Check:** USD file collision properties

**Action:**
```bash
# Open USD in Isaac Sim and check:
1. Select "abstract_chassis_link" prim
2. Check Physics → Collision → Enabled
3. Check collision shape exists
4. Check collision group/mask
```

#### Issue 2c: Contact Sensors Not Configured
**Check:** Do we need explicit contact sensors?

**Action:**
1. Review IsaacLab docs for contact force access
2. Check if contact sensors need to be added to robot config
3. Verify PhysX contact reporting is enabled

### Debugging Script
Create `scripts/debug_contact_forces.py`:
```python
"""Debug contact force reading."""
import torch
from rl_platform.tasks.mobile_mm import MobileMMTrackEEEnv, MobileMMTrackEEEnvCfg

# Create environment
cfg = MobileMMTrackEEEnvCfg()
cfg.scene.num_envs = 1
env = MobileMMTrackEEEnv(cfg=cfg)

# Reset
env.reset()

# Move arm to deliberately hit chassis
for i in range(100):
    # Action: move arm joints toward chassis
    actions = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=env.device)
    obs, reward, done, truncated, info = env.step(actions)
    
    # Print contact forces
    if i % 10 == 0:
        contact_forces = env.robot.data.net_contact_forces
        max_force = contact_forces.abs().max()
        print(f"Step {i}: Max contact force = {max_force.item():.4f} N")
        
        if max_force > 0:
            print(f"  Contact detected! Forces shape: {contact_forces.shape}")
            print(f"  Forces: {contact_forces[0, :5, :]}")  # First 5 bodies
            break
else:
    print("❌ No contacts detected after 100 steps")

env.close()
```

### Implementation Steps

1. **Run Debug Script** (15 mins)
   - Identify if forces are truly zero or just not being accessed correctly
   
2. **Fix Based on Finding** (30-60 mins)
   - If wrong array → use correct PhysX data
   - If collision disabled → enable in USD and re-export
   - If sensors needed → add contact sensor configuration

3. **Test Fix** (15 mins)
   - Run debug script again
   - Verify forces > 0 when arm hits chassis
   - Check reward penalties fire correctly

---

## Fix 3: USD Bundle Verification (P0 - CRITICAL)

### Problem
Need to verify the USD Isaac Lab loads has all URDF fixes:
- PPR helper masses: 1.0 kg ✅
- joint_theta limits: ±6.283185 rad ❓ (VERIFY)

### Verification Steps

#### Step 1: Check Which USD is Loaded (5 mins)
```python
# Add to env.py __init__ (after robot creation)
print(f"[DEBUG] Robot asset path: {self.robot.cfg.prim_path}")
print(f"[DEBUG] USD file: {self.robot.cfg.usd_path if hasattr(...) else 'Unknown'}")
```

#### Step 2: Verify Joint Limits at Runtime (5 mins)
```python
# Add to env.py after robot initialization
joint_limits = self.robot.data.soft_joint_pos_limits[0, :, :]  # [num_joints, 2]
print(f"[DEBUG] Joint limits (lower, upper):")
for i, name in enumerate(self.robot.joint_names[:3]):  # PPR joints
    lower, upper = joint_limits[i, 0].item(), joint_limits[i, 1].item()
    print(f"  {name}: [{lower:.4f}, {upper:.4f}]")
    if name == "joint_theta":
        assert abs(lower - (-6.28)) < 0.01, f"joint_theta lower limit wrong: {lower}"
        assert abs(upper - 6.28) < 0.01, f"joint_theta upper limit wrong: {upper}"
        print(f"  ✅ joint_theta limits correct!")
```

#### Step 3: Re-export USD if Needed (30 mins)
If limits are wrong, re-export USD:

```bash
# 1. Open Isaac Sim 5.0
# 2. File → Import → URDF
# 3. Select: assets_own/mobile_manipulator_PPR_base_corrected.urdf
# 4. Settings:
#    - Mesh scale: 0.001 (mm to m)
#    - Joint control: Position
#    - Moveable base: Yes
# 5. Save as: assets_own/usd/mobile_manipulator_PPR_base_corrected.usd
# 6. Verify in USD:
#    - joint_theta → Properties → Physics → Revolute Joint → Lower/Upper Limit
#    - Should show: -6.283185 / 6.283185
```

### Test Script
Create `scripts/test_usd_limits.py`:
```python
"""Verify USD has correct joint limits."""
from rl_platform.tasks.mobile_mm import MobileMMTrackEEEnv, MobileMMTrackEEEnvCfg

# Create environment
cfg = MobileMMTrackEEEnvCfg()
cfg.scene.num_envs = 1
env = MobileMMTrackEEEnv(cfg=cfg)
env.reset()

# Check joint limits
joint_limits = env.robot.data.soft_joint_pos_limits[0, :, :]
for i, name in enumerate(env.robot.joint_names[:3]):
    lower, upper = joint_limits[i, 0].item(), joint_limits[i, 1].item()
    print(f"{name}: [{lower:.4f}, {upper:.4f}]")
    
    if name == "joint_theta":
        if abs(lower - (-6.28)) < 0.01 and abs(upper - 6.28) < 0.01:
            print("✅ joint_theta limits CORRECT: ±6.28 rad")
        else:
            print(f"❌ joint_theta limits WRONG: expected ±6.28, got [{lower:.4f}, {upper:.4f}]")
            print("   ACTION: Re-export USD from corrected URDF")

env.close()
```

---

## Fix 4: Lateral Motion Penalty Frame (P1 - MEDIUM)

### Problem
Lateral motion penalty might penalize forward motion during yaw changes.

### Investigation Needed
```python
# Check: src/rl_platform/tasks/mobile_mm/rewards.py:426-462
# Look for rotation logic in lateral_motion_penalty()
```

### Potential Fix
If rotation is wrong, use world-frame velocities:
```python
# Instead of rotating body-frame velocity
lateral_vel_world = base_lin_vel_w[:, 1]  # Y component in world frame
penalty = torch.abs(lateral_vel_world) * scale
```

---

## Fix 5: prev_joint_vel Initialization (P2 - LOW)

### Problem
```python
# env.py initialization (6 columns)
self.prev_joint_vel = torch.zeros((self.num_envs, 6), device=self.device)

# Later usage (slice 3:9 = 6 elements, but from wrong base)
prev_joint_vel=self.prev_joint_vel[:, 3:9]  # Bug: only gets last 3 elements
```

### Fix
```python
# env.py initialization
self.prev_joint_vel = torch.zeros((self.num_envs, 9), device=self.device)  # Full robot DOF

# Usage (now correct)
prev_joint_vel=self.prev_joint_vel[:, 3:9]  # Gets arm joints 3-8
```

---

## Implementation Order

### Phase 1: Immediate (Today - 1 hour total)

1. ✅ **Fix Jerk Penalty** (5 mins)
   - Edit `src/task_spec.py`
   - Change `max_linear_jerk: 5.0 → 50.0`
   - Git commit

2. ✅ **Verify USD Limits** (30 mins)
   - Create `scripts/test_usd_limits.py`
   - Run test
   - If wrong, re-export USD
   - Git commit

3. ✅ **Debug Contact Forces** (30 mins)
   - Create `scripts/debug_contact_forces.py`
   - Run test
   - Identify issue (wrong array / disabled collision / missing sensors)

### Phase 2: Same Day (2-3 hours total)

4. ✅ **Fix Contact Forces** (1-2 hours)
   - Implement fix based on debug findings
   - Test with debug script
   - Verify collision penalties fire
   - Git commit

5. ✅ **Fix prev_joint_vel** (10 mins)
   - Quick win, low risk
   - Git commit

### Phase 3: Next (1-2 hours total)

6. ✅ **Review Lateral Motion Penalty** (30 mins)
   - Check current implementation
   - Test with turning base
   - Fix if needed

7. ✅ **Test Full System** (1 hour)
   - Run short training (1M steps)
   - Verify base moves without excessive penalties
   - Check contact forces trigger on collisions
   - Monitor jerk penalties (should be < 10, not 900)

---

## Validation Checklist

After all fixes:

- [ ] Jerk penalty < 10 during normal acceleration
- [ ] Base moves in training (vx, ωz both non-zero)
- [ ] Contact forces > 0 when arm hits chassis
- [ ] joint_theta changes when base rotates
- [ ] joint_theta limits show ±6.28 rad
- [ ] No massive reward penalties for base movement
- [ ] Collision termination fires when appropriate

---

## Expected Impact

**Before Fixes:**
- Jerk penalty: ~900 points → Policy freezes base
- Contact forces: 0 N → No collision avoidance
- Base rotation: Maybe locked → Can't turn

**After Fixes:**
- Jerk penalty: ~10 points → Base can move
- Contact forces: > 0 N → Collision avoidance works
- Base rotation: ±6.28 rad → Can turn freely

**Training Impact:**
- Base should start learning to reposition
- Self-collision avoidance emerges
- Whole-body coordination improves

---

## Next Steps

1. **Implement Phase 1 fixes** (today, 1 hour)
2. **Create testing plan document** (next)
3. **Run validation tests** (after fixes)
4. **Launch Session 6** with fixes (after validation)

---

**End of Implementation Plan**
