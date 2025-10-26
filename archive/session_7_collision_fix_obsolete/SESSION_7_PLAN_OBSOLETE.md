# Session 7: Fix Self-Collision Catastrophe

**Status:** 📋 PLANNING  
**Date:** 2025-10-23  
**Priority:** 🔥 CRITICAL - Session 6 revealed catastrophic self-collision issue

---

## 🎯 Objective

Fix the self-collision penalty that's destroying the learning signal in Session 6. The policy learned decent tracking (0.5-2.5m errors) and the base IS moving, but self-collision penalties (-30K per step) overwhelm all other rewards, making the final episode reward -11.7M.

---

## 📊 Session 6 Analysis Summary

### What Worked ✅
1. **Base movement:** Mobilization rewards 0.02-0.20 (non-zero = moving!)
2. **ContactSensor:** Detecting 635.66 N forces correctly
3. **Tracking:** Best 0.49m error, typical 0.5-2.5m range
4. **Training:** Completed 100M steps, no crashes

### The Problem ❌
```
[DEBUG Step 300] Total reward: -30,308.07
  self_collision_penalty: +30,264.8711  ← 1000x TOO LARGE!
  position_tracking: +0.5647
  base_mobilization: +0.0770
  
⚠️ [COLLISION DETECTED] Step 302
   Max contact force: 635.66 N
   Collision on body: base
```

**Episode Reward Calculation:**
- ~30,000 collision penalty per step
- × 399 steps per episode
- = -11.97M episode reward
- **This is 1000x larger than best position tracking reward (+39)!**

---

## 🔍 Root Cause Investigation

### Why is Self-Collision So High?

**Theory 1: Penalty Weight Too Large**
- Current: `self_collision_penalty_weight = 1000.0`
- Position tracking max: 50 points
- Base mobilization max: 30 points
- **Ratio: 1000:50:30 is absurdly imbalanced!**

**Theory 2: Constant Base Collisions**
- Base colliding with legs/arms during movement
- Collision geometry too conservative
- No collision filtering between base and attached parts

**Theory 3: Math Error**
- Penalty displayed as "+30264" (positive)
- Should be negative to reduce reward
- Possible sign flip in reward calculation?

### Evidence from Logs

```python
# From evaluate.py output Step 300:
self_collision_penalty: +30,264.8711  # Shown as POSITIVE
position_tracking: +0.5647
base_mobilization: +0.0770

# But total reward is NEGATIVE:
Total reward: -30,308.07
```

**This suggests:** The penalty IS being subtracted (total is negative), but the logging shows component values before sign flip.

---

## 🛠️ Proposed Fixes

### Option A: Reduce Penalty Weight (RECOMMENDED)
**Fastest, lowest risk**

**Change:**
```python
# In src/rl_platform/tasks/mobile_mm/config.py
class RewardWeights:
    self_collision_penalty: float = 1000.0  # OLD
    self_collision_penalty: float = 5.0     # NEW (200x reduction)
```

**Rationale:**
- Makes collision penalty ~10% of position tracking max (5 vs 50)
- Still penalizes collisions, but doesn't dominate reward
- Policy can learn from other reward signals

**Test Plan:**
1. Train for 1M steps (5 minutes)
2. Check episode reward: Should be -1000 to +500 range (not -11M!)
3. Verify base still moves: mobilization > 0
4. If successful, continue to 100M steps

---

### Option B: Investigate Collision Geometry
**More thorough, but time-consuming**

**Action Items:**
1. Visualize robot in Isaac Sim GUI (non-headless)
2. Enable collision visualization
3. Watch where base collides during movement
4. Check URDF collision meshes:
   - `assets_own/mobile_manipulator_PPR_base_corrected.urdf`
   - Are base collision bounds too large?
   - Should base collide with legs?

**Possible Fixes:**
- Adjust collision mesh sizes
- Add collision filtering (base ↔ legs = no collision)
- Separate collision groups for self vs environment

---

### Option C: Verify Reward Math
**Quick sanity check**

**Code to Review:**
```python
# In src/rl_platform/tasks/mobile_mm/task.py
# Find where self_collision_penalty is applied

# Check for sign errors:
reward -= self_collision_penalty * weight  # Should be minus!
# NOT:
reward += self_collision_penalty * weight  # Would be wrong
```

**Test:**
```python
# Add debug print in compute_rewards():
print(f"self_collision penalty sign: {self_collision_penalty_value}")
print(f"Before penalty: {reward_before}")
print(f"After penalty: {reward_after}")
# Verify reward_after < reward_before when collision occurs
```

---

## 📋 Session 7 Implementation Plan

### Phase 1: Quick Fix (Option A) - 10 minutes
**Goal:** Make training viable by reducing penalty weight

1. **Modify config:**
   ```python
   # src/rl_platform/tasks/mobile_mm/config.py
   self_collision_penalty: float = 5.0  # Was 1000.0
   ```

2. **Test short run (1M steps, 5 min):**
   ```powershell
   I:\isaaclab\isaaclab.bat -p scripts/reinforcement_learning/sb3/train.py `
       --task MobileMMTrackEE-v0 `
       --num_envs 4096 `
       --total_timesteps 1000000 `
       --headless
   ```

3. **Check metrics:**
   - Episode reward: Should be -1000 to +500 (not -11M!)
   - Base mobilization: Should still be > 0
   - Position tracking: Should improve
   - If good → proceed to Phase 2

### Phase 2: Full Training - 9 hours
**Goal:** Complete 100M steps with fixed penalty

```powershell
I:\isaaclab\isaaclab.bat -p scripts/reinforcement_learning/sb3/train.py `
    --task MobileMMTrackEE-v0 `
    --num_envs 4096 `
    --n_steps 128 `
    --batch_size 1024 `
    --total_timesteps 100000000 `
    --learning_rate 3e-4 `
    --ent_coef 0.001 `
    --enable_entropy_decay `
    --final_ent_coef 1e-4 `
    --decay_start_timestep 50000000 `
    --decay_duration_timesteps 50000000 `
    --enable_kl_schedule `
    --kl_warmup 0.25 `
    --kl_main 0.15 `
    --kl_finetune 0.07 `
    --target_kl 1.0 `
    --trajectory_type multi_recorded `
    --use_all_trajectories `
    --headless
```

**Monitor every 10M steps:**
- Episode reward trending positive?
- Base actively moving?
- Tracking error improving?

### Phase 3: Evaluation
**Goal:** Verify the fix worked

```powershell
I:\isaaclab\isaaclab.bat -p scripts/reinforcement_learning/sb3/evaluate.py `
    --checkpoint logs/sb3/mobilemmtrackee_v0/SESSION7/final_model.zip `
    --num_envs 16 `
    --num_episodes 10 `
    --headless `
    --trajectory_type multi_recorded `
    --use_all_trajectories
```

**Success Criteria:**
- ✅ Mean episode reward: -500 to +2000 (not -11M!)
- ✅ Mean tracking error: < 1.0m
- ✅ Base mobilization: > 0.1 (actively used)
- ✅ Self-collision penalty: < 50 per step (was 30K!)

---

## 🎯 Expected Outcomes

### If Option A Works (Likely):
- **Episode reward:** -500 to +2000 range
- **Tracking:** 0.3-0.8m errors (better than Session 6's 0.5-2.5m)
- **Base movement:** 5-15cm adjustments (coordinated)
- **Training time:** Same as Session 6 (~9 hours)

### If Option A Fails:
- Episode reward still very negative
- Base doesn't move OR moves wildly
- → Proceed to Option B (collision geometry investigation)

---

## 📁 Files to Modify

### Primary Change:
```
src/rl_platform/tasks/mobile_mm/config.py
```

**Specific Line:**
```python
class RewardWeights:
    # ...
    self_collision_penalty: float = 5.0  # Changed from 1000.0
```

### Optional Investigation (if needed):
```
src/rl_platform/tasks/mobile_mm/task.py  # Check reward calculation
assets_own/mobile_manipulator_PPR_base_corrected.urdf  # Collision meshes
```

---

## 📊 Comparison: Sessions 6 vs 7 (Expected)

| Metric | Session 6 | Session 7 (Target) | Improvement |
|--------|-----------|-------------------|-------------|
| Episode Reward | -11.7M | +500 to +2000 | **99.98%** 🎉 |
| Collision Penalty/Step | 30,000 | 150 (5.0 × 30 contacts) | **99.5%** |
| Tracking Error | 0.5-2.5m | 0.3-0.8m | **40-68%** |
| Base Movement | YES ✅ | YES ✅ | Maintained |
| Training Stability | Stable | Stable | Maintained |

---

## 🔧 Detailed Code Change

### Before (Session 6):
```python
# src/rl_platform/tasks/mobile_mm/config.py, line ~XX
@configclass
class RewardWeights:
    """Reward component weights for the mobile manipulator task."""
    
    # Tracking rewards
    position_tracking: float = 50.0
    orientation_tracking: float = 10.0
    
    # Penalties
    self_collision_penalty: float = 1000.0  # ← TOO LARGE!
    joint_limit_penalty: float = 10.0
    velocity_limit_penalty: float = 5.0
    
    # ...
```

### After (Session 7):
```python
# src/rl_platform/tasks/mobile_mm/config.py, line ~XX
@configclass
class RewardWeights:
    """Reward component weights for the mobile manipulator task."""
    
    # Tracking rewards
    position_tracking: float = 50.0
    orientation_tracking: float = 10.0
    
    # Penalties
    self_collision_penalty: float = 5.0  # FIXED: 200x reduction (was 1000.0)
    joint_limit_penalty: float = 10.0
    velocity_limit_penalty: float = 5.0
    
    # ...
```

**Commit Message:**
```
Session 7: Reduce self-collision penalty 1000.0 → 5.0

Session 6 revealed catastrophic self-collision penalties (-30K/step)
overwhelming all other rewards. Episode rewards were -11.7M despite
decent tracking (0.5-2.5m errors).

Changes:
- self_collision_penalty: 1000.0 → 5.0 (200x reduction)
- Rationale: Make collision penalty ~10% of position tracking max
- Still penalizes collisions, but doesn't dominate learning signal

Expected: Episode rewards should be -500 to +2000 (not -11M!)
Tracking should improve as policy can now learn from position rewards.

Issue: Session 6 showed base IS moving (mobilization 0.02-0.20) and
ContactSensor works (635N detected), but self-collision penalty was
1000x larger than all other rewards combined.

Testing: Train 1M steps first to verify episode rewards are reasonable.
```

---

## ⚠️ Risk Assessment

### Low Risk ✅
- **Change:** Single parameter (penalty weight)
- **Rollback:** Easy (revert config.py)
- **Test time:** 5 minutes (1M steps)

### What Could Go Wrong?
1. **Penalty too small:**
   - Robot ignores collisions completely
   - Wild base movements return
   - **Mitigation:** If this happens, try 10.0 or 20.0 instead

2. **Still negative rewards:**
   - Other penalties also too high
   - Tracking not improving
   - **Next step:** Investigate Option B (collision geometry)

3. **Base stops moving:**
   - Reducing collision penalty shouldn't affect base movement
   - But if it does, adjust jerk_penalty back to 5.0
   - **Unlikely:** Session 6 showed base moves fine at 50.0

---

## 📝 Success Definition

### Session 7 is SUCCESSFUL if:
1. ✅ **Episode reward:** -500 to +2000 (99.98% improvement!)
2. ✅ **Tracking:** Mean error < 1.0m
3. ✅ **Base moves:** Mobilization reward > 0.1
4. ✅ **Training stable:** No crashes, no reward explosions
5. ✅ **Collision penalty:** < 50 per step (99.5% reduction from 30K)

### Metrics to Track:
```python
# Every 10M steps, check:
mean_episode_reward = ?  # Should be positive eventually
mean_tracking_error = ?  # Should be < 1.0m
base_mobilization = ?    # Should be > 0.1
self_collision_pen = ?   # Should be < 50/step
```

---

## 🎓 Lessons from Session 6

### What We Learned:
1. **Penalty scaling matters!** 1000x weight destroys learning
2. **Base CAN move** with jerk_penalty=50.0
3. **ContactSensor works** perfectly (635N detected)
4. **Policy learned tracking** despite corrupted rewards (0.5-2.5m)
5. **Reward balance critical** for RL to work

### Best Practices Going Forward:
- Keep penalties proportional to rewards (max 20% of max positive reward)
- Test short runs (1M steps) after major config changes
- Monitor reward components, not just total reward
- Balance > magnitude for reward design

---

**Created:** 2025-10-23  
**Author:** AI Analysis  
**Status:** Ready for Session 7 launch  
**Estimated Fix Time:** 10 minutes + 9 hour training
