# Critical Fixes Applied - Session 6 Preparation

**Date**: Current session  
**Status**: ✅ 2 CRITICAL FIXES APPLIED, 2 investigations ready

---

## Summary

We identified and fixed **THE ROOT CAUSE** of the frozen base bug, plus one shape mismatch bug. Two additional critical issues require investigation before Session 6.

---

## ✅ FIXES APPLIED

### 1. ⭐ JERK PENALTY FIX (THE KEY FIX!)

**File**: `src/rl_platform/tasks/mobile_mm/config.py` (line 63)  
**Change**: `max_linear_jerk: 5.0 → 50.0` m/s³  
**Commit**: 1bcd974

#### Root Cause Analysis

Normal base acceleration of 0.25 m/s in 0.05s produces:
- Acceleration: 5.0 m/s²
- Jerk: 100 m/s³

With old limit:
```
Penalty = -0.05 × (100 - 5)² = -451 points
Mobilization bonus = +150 points
NET REWARD = -301 points (CATASTROPHIC!)
```

With new limit:
```
Penalty = -0.05 × (100 - 50)² = -125 points
Mobilization bonus = +150 points
NET REWARD = +25 points (GOOD!)
```

**Impact**: This single change should unblock base learning immediately. The policy was receiving massive negative rewards for any base movement, completely overwhelming the mobilization bonus.

**Validation**: `scripts/test_jerk_penalty_fix.py` confirms the math.

---

### 2. prev_joint_vel Shape Mismatch Fix

**File**: `src/rl_platform/tasks/mobile_mm/env.py` (line 367)  
**Change**: `torch.zeros(num_envs, 6) → torch.zeros(num_envs, 9)`  
**Commit**: 67176e8

#### Bug Details

- Initialized with 6 columns (arm joints only)
- Used with 9 columns (3 PPR base + 6 arm joints)
- Slicing `prev_joint_vel[:, 3:9]` accessed incorrect data

**Impact**: Minor but important - prevents IndexError and ensures correct joint velocity history.

---

## 🔍 INVESTIGATIONS READY

### 3. USD Joint Limits Verification

**Status**: ⚠️ CRITICAL - Must verify before Session 6  
**Script**: `scripts/verify_usd_limits.py`

#### What to Check

URDF has `joint_theta` limits: ±6.283185 rad (±2π)

**Expected** (CORRECT):
```
joint_theta: soft_joint_pos_limits = [-6.283185, 6.283185]
```

**Bug** (LOCKED JOINT):
```
joint_theta: soft_joint_pos_limits = [0.0, 0.0]
```

If locked, USD converter collapsed infinite limits to zero-width → base can't rotate!

#### How to Run

```powershell
I:\isaaclab\isaaclab.bat -p scripts/verify_usd_limits.py
```

**Expected Output**: Joint limits table with ✅ or ❌ status for each joint.

---

### 4. Contact Forces Debug

**Status**: ⚠️ CRITICAL - No self-collision feedback  
**Script**: `scripts/debug_contact_forces.py`

#### Current Problem

```python
# env.py line 1120
self._contact_forces[:, self._chassis_body_idx, 0]
# Always reads 0.0 N!
```

Without contact forces:
- `self_collision_penalty` = 0.0 (no feedback)
- Policy has no incentive to avoid arm-chassis collisions
- Base repositioning not learned

#### How to Run

```powershell
I:\isaaclab\isaaclab.bat -p scripts/debug_contact_forces.py --num_envs 16 --steps 100
```

**Expected Output**: Lists all PhysX contact arrays and shows non-zero values during collisions.

#### Possible Causes

1. Using wrong PhysX array (`net_forces_w` vs `forces_w`)
2. Wrong body index (`chassis_body_idx` mismatch)
3. Contact sensor not configured correctly
4. PhysX collision detection disabled

---

## 📋 NEXT STEPS

### Phase 1: Verify Fixes (30 minutes)

1. **Run USD verification** (15 mins):
   ```powershell
   I:\isaaclab\isaaclab.bat -p scripts/verify_usd_limits.py
   ```
   - If joint_theta = [0, 0] → Re-export USD with finite limits
   - If joint_theta = ±6.28 → ✅ Good to go!

2. **Run contact forces debug** (15 mins):
   ```powershell
   I:\isaaclab\isaaclab.bat -p scripts/debug_contact_forces.py --num_envs 16 --steps 100
   ```
   - Find correct PhysX array with non-zero values
   - Document correct usage pattern

### Phase 2: Fix Contact Forces (1-2 hours)

1. Update `env.py` to use correct PhysX contact array
2. Verify self-collision penalties trigger
3. Test termination on arm-chassis collisions

### Phase 3: Launch Session 6 🚀

```powershell
.\scripts\launch_training_windows.ps1 -Task MobileMMTrackEE-v0 -NumEnvs 8192 -Headless
```

**Expected Improvements**:
- ✅ Base actions become active (jerk fix)
- ✅ Self-collision feedback working (contact fix)
- ✅ Base rotation enabled (USD verification)
- 🎯 Error drops below Session 5's 0.877m

**Monitoring (First 10M steps)**:
- Base velocity magnitude > 0.0 m/s
- Base mobilization rewards positive
- Jerk penalties < 200 points (reasonable)
- Contact forces non-zero during collisions
- Environment break rate < 5%

---

## 🔬 Technical Details

### Jerk Penalty Calculation

```python
# rewards.py:462
jerk = (current_accel - prev_accel) / dt
jerk_mag = torch.norm(jerk, dim=-1)
violation = torch.clamp(jerk_mag - max_jerk, min=0.0)
penalty = -weight * violation ** 2  # weight = 0.05
```

### Typical Movement Analysis

| Metric | Value | Notes |
|--------|-------|-------|
| Velocity change | 0 → 0.25 m/s | Typical base movement |
| Time step | 0.05 s | 20 Hz control |
| Acceleration | 5.0 m/s² | v/dt |
| Jerk | 100 m/s³ | a/dt |
| Old penalty | -451 points | (100-5)² × 0.05 |
| New penalty | -125 points | (100-50)² × 0.05 |
| Mobilization bonus | +150 points | base_progress_reward |
| **Net (old)** | **-301 points** | BASE FROZEN |
| **Net (new)** | **+25 points** | CAN MOVE ✅ |

### Reward Balance

With the jerk fix, typical base movement now gives:
```
+150  base_progress_reward
-125  jerk_penalty
-5    action_magnitude_penalty
-2    action_rate_penalty
─────────────────────────────
+18   NET POSITIVE ✅
```

---

## 📚 References

- **Code Review**: `docs/_CODE_REVIEW_VALIDATION.md`
- **Implementation Plan**: `docs/_FIX_IMPLEMENTATION_PLAN.md`
- **Jerk Test**: `scripts/test_jerk_penalty_fix.py`
- **USD Verification**: `scripts/verify_usd_limits.py`
- **Contact Debug**: `scripts/debug_contact_forces.py`

---

## 🎯 Success Criteria for Session 6

### Quantitative
- EE tracking error < 0.877m (Session 5 baseline)
- Base velocity magnitude > 0.05 m/s (consistent movement)
- Environment break rate < 5% (stable)
- Jerk penalties < 200 points (reasonable)

### Qualitative
- Base actively moves toward targets
- Self-collision penalties trigger correctly
- Base rotates when needed
- Smooth, coordinated whole-body motion

---

## ⚠️ Rollback Plan

If Session 6 shows issues:

1. **Jerk too permissive** → Adjust limit to 30-40 m/s³
2. **Contact forces still zero** → Check PhysX sensor config
3. **Base locked** → Verify USD limits at runtime
4. **Reward hacking** → Check for new unbounded reward terms

Keep Session 5b checkpoint as baseline for comparison.

---

**Ready for Phase 1: Verification!** 🚀
