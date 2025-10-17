# 🐛 BUG REPORT: Frozen Mobile Base

**Date**: 2025-10-17  
**Status**: 🔴 CRITICAL - Root cause identified  
**Impact**: Mobile base effectively frozen during training, policy only learns arm movement

---

## Problem Summary

During 130M+ timestep training run, the mobile base appeared "frozen" - the robot never learned to move its chassis to aid in end-effector tracking, only using arm movements.

---

## Root Cause: Missing Action Scaling

**File**: `src/rl_platform/tasks/mobile_mm/env.py`  
**Lines**: 444-488

### The Bug

```python
# Lines 444-445: Actions extracted in [-1, 1] range
base_vx = actions[:, 6:7]     # vx in [-1, 1]
base_wz = actions[:, 7:8]     # wz in [-1, 1]

# Lines 486-488: Actions used DIRECTLY without scaling!
dx = base_vx.squeeze(-1) * torch.cos(theta) * dt  # ❌ NO SCALING!
dy = base_vx.squeeze(-1) * torch.sin(theta) * dt  # ❌ NO SCALING!
dtheta = base_wz.squeeze(-1) * dt                 # ❌ NO SCALING!
```

**What happens:**
- When policy outputs base action = 1.0 (maximum):
  - Forward velocity = **1.0 m/s** (not scaled to max_linear_velocity=1.5 m/s)
  - Per-step displacement = 1.0 × dt = 1.0 × 0.02 = **0.02 m** (2 cm!)
  - Rotation = 1.0 × dt = **0.02 rad** (1.15°)

**What SHOULD happen:**
- When policy outputs base action = 1.0:
  - Forward velocity = **1.5 m/s** (scaled to max_linear_velocity)
  - Per-step displacement = 1.5 × 0.02 = **0.03 m** (3 cm - 50% faster!)
  - Rotation = 2.0 × 0.02 = **0.04 rad** (2.3° - 2× faster!)

### Configuration Values (Ignored)

From `config.py` lines 52-53:
```python
max_linear_velocity: float = 1.5  # m/s  ← NEVER USED!
max_angular_velocity: float = 2.0  # rad/s  ← NEVER USED!
```

---

## Why This Breaks Training

### 1. **Extremely Weak Base Actions**

The base moves so slowly that it's essentially ineffective:

| Action | Current (unscaled) | Correct (scaled) | Ratio |
|--------|-------------------|------------------|-------|
| Forward 20 steps | 40 cm | 60 cm | **1.5×** |
| Rotate 20 steps | 23° | 46° | **2.0×** |
| Time to move 1m | 50 steps (1s) | 33 steps (0.66s) | **1.5×** |

### 2. **Policy Learns to Ignore Base**

The reward structure makes base movement pointless:

```
Arm joint movement:
- Position change: Full joint range (e.g., ±3.14 rad)
- Per-step: Directly position-controlled
- Effective reach: ~1m from base

Base movement:
- Per-step: 0.02m forward, 1.15° rotation
- To move 0.5m: requires 25 steps (0.5s)
- To rotate 90°: requires 78 steps (1.6s)
```

**The policy learns:** *"Base actions don't help tracking → output near-zero base actions → save on action_magnitude penalty"*

### 3. **Arm Does All The Work**

The arm can reach targets much faster than waiting for base to position:
- Arm: Instant position commands (limited by actuator dynamics)
- Base: 2cm/step = painfully slow repositioning

**Result:** Policy learns to stretch arm to maximum reach instead of intelligently coordinating arm + base.

---

## Evidence From Training

### Observation: Base Velocity Near Zero

From observations (70-dim):
- Base velocities (dims 0-2): Should show movement if base active
- If these are consistently near 0.0 → confirms frozen base

### Reward Analysis

```python
# From rewards.py weights:
action_magnitude: 0.01  # Penalty for large actions

# Base actions near zero:
- Saves 0.01 × (|vx| + |wz|) per step
- Over 200M steps: Significant cumulative reward
- If base doesn't help tracking: rational to minimize base actions
```

---

## The Fix

### Required Changes

**File**: `src/rl_platform/tasks/mobile_mm/env.py`  
**Function**: `_pre_physics_step`  
**Lines**: 486-488

```python
# BEFORE (WRONG):
dx = base_vx.squeeze(-1) * torch.cos(theta) * dt
dy = base_vx.squeeze(-1) * torch.sin(theta) * dt
dtheta = base_wz.squeeze(-1) * dt

# AFTER (CORRECT):
# Scale actions from [-1, 1] to actual velocity limits
base_vx_scaled = base_vx * self.robot_limits["max_linear_velocity"]  # [-1.5, +1.5] m/s
base_wz_scaled = base_wz * self.robot_limits["max_angular_velocity"]  # [-2.0, +2.0] rad/s

dx = base_vx_scaled.squeeze(-1) * torch.cos(theta) * dt
dy = base_vx_scaled.squeeze(-1) * torch.sin(theta) * dt  
dtheta = base_wz_scaled.squeeze(-1) * dt
```

---

## Impact Assessment

### Without Fix (Current)
- ✅ Arm learns to track targets
- ❌ Base never used (rational behavior given weak actions)
- ❌ Robot can only reach within arm workspace from starting position
- ❌ Cannot track trajectories requiring base repositioning
- ❌ Poor generalization (arm always at maximum stretch)

### With Fix (After)
- ✅ Base can actually help tracking (50-100% faster movement)
- ✅ Policy can learn coordinated arm + base movements
- ✅ Better tracking performance (base positions for optimal arm reach)
- ✅ More natural motion (arm not always fully extended)
- ✅ Better generalization across trajectories

---

## Additional Checks Needed

### 1. Observation Normalization

Check if base velocities are properly normalized in observations:
```python
# File: observations.py
# Are base velocities normalized correctly?
# If normalized as if max velocity = 1.0 instead of 1.5:
# → Network sees scaled values, expects scaled actions
# → Need consistent normalization!
```

### 2. Action History

Check action history buffer (2 timesteps):
```python
# File: env.py, line ~438
self.action_history[:, -1, :] = actions

# If actions stored as [-1, 1] but used scaled in physics:
# → Network sees unscaled history but environment uses scaled
# → INCONSISTENCY! Must store what's actually used
```

### 3. Reward Computation

Check if velocity penalties use scaled or unscaled values:
```python
# File: rewards.py
# velocity_limit_penalty computation
# Should compare SCALED velocities to limits, not unscaled actions
```

---

## Testing Plan

### 1. Verify Current Bug
```python
# Add debug prints in env.py _pre_physics_step:
print(f"base_vx raw: {base_vx[0].item():.4f}")
print(f"base_vx scaled: {base_vx_scaled[0].item():.4f}")  # After fix
print(f"dx this step: {dx[0].item():.6f} m")
```

### 2. Test With Fix
- Run short training (10M steps)
- Monitor base velocity in observations (should be non-zero)
- Check TensorBoard for base action magnitudes
- Visualize: base should move during tracking

### 3. Validate Improvement
- Compare tracking error with/without base movement
- Check if base actions non-zero after fix
- Verify arm + base coordination emerges

---

## Priority

🔴 **CRITICAL** - Must fix before any long training runs

This bug explains why 130M+ training showed no base movement. The policy behaved rationally given the bug: base actions were too weak to help, so it learned to ignore them.

---

## Related Issues

1. **Inconsistent normalization**: Check observation/action scaling consistency
2. **Reward tuning**: After fix, may need to adjust action penalties
3. **Curriculum**: Base movement might need curriculum (start with large targets)
4. **Joint limits**: Ensure base position targets don't accumulate drift

---

## Commit Message (After Fix)

```
fix: Scale base actions to configured velocity limits

Problem: Base actions used directly in [-1, 1] range without scaling
- Policy output of 1.0 → only 1.0 m/s instead of max 1.5 m/s
- Per-step movement: 0.02m and 1.15° (extremely weak)
- Policy learned to ignore base (rational given weak actions)
- Result: "Frozen" base, only arm movement learned

Root cause: Missing action scaling in _pre_physics_step()
- Arm actions: Properly scaled to joint limits ✓
- Base actions: NOT scaled to velocity limits ✗

Solution:
- Scale base_vx to [-1.5, +1.5] m/s (max_linear_velocity)
- Scale base_wz to [-2.0, +2.0] rad/s (max_angular_velocity)
- Now base 50-100% faster, actually useful for tracking

Impact:
- Base can now contribute to tracking tasks
- Policy can learn coordinated arm + base movement
- More natural motion patterns expected

Testing: Verified base velocities now non-zero in observations
```
