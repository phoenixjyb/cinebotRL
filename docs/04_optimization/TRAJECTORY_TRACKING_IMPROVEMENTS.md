# Trajectory Tracking System: Architecture Analysis & Improvement Recommendations

**Date:** October 21, 2025  
**Training Session:** Session 4b (8192 envs, 25.7M/100M steps)  
**Context:** Post-physics fixes, investigating frequent reset loops and trajectory tracking design

---

## Executive Summary

After fixing 6 critical URDF physics bugs and optimizing control to 20Hz, training shows **exceptional metrics** (0.958 explained variance, 0.00443 value loss). However, ~0.1-1% of environments are stuck in **infinite reset loops** due to overly aggressive termination criteria. 

Investigation revealed the current RL architecture uses a **hybrid "time-based moving target"** approach that is fundamentally sound for trajectory tracking, but needs tactical fixes for termination policy and strategic enhancements for velocity tracking.

---

## Current Architecture Analysis

### 1. Observation Space (What Policy Sees)

✅ **Good: Policy HAS trajectory awareness via lookahead**

```python
# From src/rl_platform/tasks/mobile_mm/observations.py
Observation components (55 dims total):
├─ Base state: pos(3) + quat(4) + lin_vel(3) + ang_vel(3) = 13 dims
├─ Arm joints: pos(6) + vel(6) = 12 dims
├─ EE state: pos(3) + quat(4) + lin_vel(3) + ang_vel(3) = 13 dims
├─ Tracking error: pos_error(3) + quat_error(4) = 7 dims
└─ Lookahead: NEXT 3 waypoints @ 0.1s intervals = 9 dims  ← ENABLED!
```

**Key insight:** Policy sees current target position + next 3 waypoints (0.3s into future), giving it trajectory direction and timing awareness.

### 2. Target Advancement (Time-Based)

```python
# From src/rl_platform/tasks/mobile_mm/trajectories.py
- Waypoint spacing: 0.1s (waypoint_dt)
- Control frequency: 0.05s (20Hz, 2 control steps per waypoint)
- Interpolation: Linear for positions, SLERP for orientations
- Advancement: Time-based (independent of robot progress)

def step(self):
    self._recorded_time_accum += 0.05  # 20Hz control
    if accumulated_time >= 0.1s:
        current_waypoint_idx += 1  # Advance regardless of robot position
```

**Design philosophy:** Target moves like a timed trajectory in real-world execution, regardless of whether robot keeps up. Robot must learn to anticipate and follow.

### 3. Reward Structure (Position-Focused)

```python
# From src/rl_platform/tasks/mobile_mm/rewards.py
Active rewards:
├─ position_tracking: exp(-scale * error²)           # Instantaneous position
├─ orientation_tracking: exp(-scale * angular_dist²) # Instantaneous orientation
├─ base_mobilization: Reward base motion that reduces distance
├─ action_smoothness: Penalize jerky actions
├─ joint_limit_penalty: Penalize approaching limits
└─ collision_penalty: Penalize self-collision forces
```

❌ **Missing:** No velocity or acceleration tracking rewards! Policy only optimizes for instantaneous position/orientation matching, not trajectory timing adherence.

---

## Problem: Infinite Reset Loops

### Root Cause

```python
# From src/rl_platform/tasks/mobile_mm/config.py
terminate_on_tracking_error: True
max_tracking_error: 2.0  # meters
```

**What happens:**
1. Env spawns with broken IK → EE 3.3m from base (max reach ~0.65m)
2. `tracking_error = 3.3m > 2.0m` → TERMINATE immediately (step 0 or 1)
3. Reset → broken spawn again → terminate → **infinite loop**

**Evidence:** Between tracking displays (50 steps = 2.5s), dozens of `[RESET] Env X: Base moved to trajectory start` messages show envs resetting every 1-5 steps.

**Why metrics still good:** Only affects ~0.1-1% of 8192 envs. The other 99% learn normally, dominating training metrics (0.958 explained variance).

### Why 2.0m Threshold Is Wrong for Trajectory Tracking

**Goal-reaching paradigm** (what threshold assumes):
- Target is static
- 2.0m error = catastrophic failure (robot can't reach)
- Appropriate to terminate and retry

**Trajectory-tracking paradigm** (what we actually have):
- Target is moving (advances every 0.1s)
- Robot might legitimately fall 2.0m behind and need time to catch up
- Termination prevents recovery/learning from difficult states

---

## Recommendations

### Priority 1: TACTICAL FIX (Termination Policy) ⚡ URGENT

**Problem:** Infinite reset loops prevent ~1% of envs from learning.

**Solution A: Grace Period** (RECOMMENDED)

Add consecutive error counting to prevent instant termination:

```python
# In src/rl_platform/tasks/mobile_mm/config.py
@configclass
class MobileMMTaskConfig:
    # Termination
    terminate_on_tracking_error: bool = True
    max_tracking_error: float = 2.0  # meters
    tracking_error_grace_period: int = 10  # consecutive steps @ 20Hz = 0.5s
```

```python
# In src/rl_platform/tasks/mobile_mm/env.py
def _initialize_buffers(self):
    # ... existing buffers ...
    # Track consecutive high-error steps per env
    self.high_error_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

def _get_dones(self):
    terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    
    if self.task_cfg.terminate_on_tracking_error:
        ee_pos = self.robot.data.body_pos_w[:, self._ee_body_idx, :]
        target_pos, _ = self.trajectory_manager.get_target_pose()
        tracking_error = torch.norm(target_pos - ee_pos, dim=-1)
        
        # Increment counter for envs with high error
        high_error_mask = tracking_error > self.task_cfg.max_tracking_error
        self.high_error_count[high_error_mask] += 1
        self.high_error_count[~high_error_mask] = 0  # Reset counter when error drops
        
        # Only terminate after consecutive high errors (grace period)
        grace_period = self.task_cfg.tracking_error_grace_period
        terminated |= self.high_error_count >= grace_period
    
    # ... rest of termination logic ...
    return terminated, time_out

def _reset_idx(self, env_ids):
    # Reset error counter for resetting envs
    self.high_error_count[env_ids] = 0
    # ... rest of reset logic ...
```

**Benefits:**
- ✅ Broken spawns get 10 steps (0.5s) to recover before termination
- ✅ Prevents instant termination from transient spikes
- ✅ Still terminates if persistently bad (safety mechanism)
- ✅ Minimal code changes (1 config param, 1 buffer, 5 lines logic)

**Solution B: Increase Threshold** (QUICK FIX)

```python
max_tracking_error: 5.0  # meters (up from 2.0)
```

**Pros:** One-line change  
**Cons:** Doesn't address fundamental issue (some broken spawns are 3.3m+ away)

**Solution C: Disable Termination** (MOST PERMISSIVE)

```python
terminate_on_tracking_error: False  # Only timeout at 400 steps
```

**Pros:** Envs never stuck in reset loops, learn from all states  
**Cons:** No safety mechanism for catastrophically bad behavior

**RECOMMENDATION:** Implement Solution A (grace period). It's the best balance of safety and learning.

---

### Priority 2: STRATEGIC ENHANCEMENT (Velocity Tracking) 🎯 MEDIUM PRIORITY

**Problem:** Policy only cares about position matching, not trajectory *timing*. A robot that lags 2 steps behind but maintains perfect position tracking gets full reward, even though it's not following the timed trajectory.

**Solution: Add velocity and acceleration tracking rewards**

```python
# In src/rl_platform/tasks/mobile_mm/rewards.py

def velocity_tracking_reward(
    ee_vel: torch.Tensor,
    target_vel: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Reward for matching trajectory velocity (timing adherence).
    
    Encourages policy to follow trajectory at correct speed, not just positions.
    
    Args:
        ee_vel: Current EE velocity [num_envs, 3]
        target_vel: Target velocity from trajectory [num_envs, 3]
        scale: Scaling factor for error sensitivity
        
    Returns:
        Reward values [num_envs]
    """
    vel_error = torch.norm(target_vel - ee_vel, dim=-1)
    return torch.exp(-scale * vel_error ** 2)


def acceleration_tracking_reward(
    ee_accel: torch.Tensor,
    target_accel: torch.Tensor,
    scale: float = 0.5,
) -> torch.Tensor:
    """Reward for matching trajectory acceleration (smooth following).
    
    Args:
        ee_accel: Current EE acceleration [num_envs, 3]
        target_accel: Target acceleration from trajectory [num_envs, 3]
        scale: Scaling factor for error sensitivity
        
    Returns:
        Reward values [num_envs]
    """
    accel_error = torch.norm(target_accel - ee_accel, dim=-1)
    return torch.exp(-scale * accel_error ** 2)
```

**Implementation in env.py:**

```python
# In src/rl_platform/tasks/mobile_mm/trajectories.py
def get_target_velocity(self) -> torch.Tensor:
    """Compute target velocity from waypoint spacing and timing.
    
    Returns:
        target_velocity: [num_envs, 3] in m/s
    """
    # Numerical derivative: (next_pos - current_pos) / waypoint_dt
    current_idx = self.current_waypoint_idx
    next_idx = (current_idx + 1) % max_length
    
    current_pos = self.recorded_positions[batch_indices, current_idx]
    next_pos = self.recorded_positions[batch_indices, next_idx]
    
    velocity = (next_pos - current_pos) / self.waypoint_dt  # 0.1s spacing
    return velocity
```

```python
# In src/rl_platform/tasks/mobile_mm/env.py _get_rewards()
def _get_rewards(self):
    # ... existing reward calculation ...
    
    # NEW: Velocity tracking
    if self.task_cfg.rewards.velocity_tracking > 0:
        target_vel = self.trajectory_manager.get_target_velocity()
        ee_vel = self.robot.data.body_lin_vel_w[:, self._ee_body_idx, :]
        vel_reward = velocity_tracking_reward(ee_vel, target_vel, scale=1.0)
        rewards += self.task_cfg.rewards.velocity_tracking * vel_reward
    
    # ... rest of rewards ...
```

**Configuration:**

```python
# In src/rl_platform/tasks/mobile_mm/config.py
@configclass
class RewardsCfg:
    position_tracking: float = 1.0
    orientation_tracking: float = 0.5
    velocity_tracking: float = 0.3  # NEW: Timing adherence
    acceleration_tracking: float = 0.1  # OPTIONAL: Smooth following
    # ... rest of rewards ...
```

**Benefits:**
- ✅ Policy learns to follow trajectory *timing*, not just positions
- ✅ Reduces "lag behind but still track" behaviors
- ✅ More accurate trajectory execution
- ✅ Better generalization to time-critical tasks

**Tradeoffs:**
- ⚠️ Slightly more complex reward landscape (may slow initial learning)
- ⚠️ Requires retuning reward weights
- ⚠️ May need to restart training from scratch (architecture change)

**RECOMMENDATION:** Add after fixing termination policy. Not critical for current task (position tracking works well), but valuable for future time-critical trajectories.

---

### Priority 3: OPTIONAL ENHANCEMENTS 💡 LOW PRIORITY

#### 3.1: Increase Lookahead Horizon

**Current:** 3 waypoints (0.3s into future)  
**Proposed:** 5 waypoints (0.5s into future)

```python
# In src/rl_platform/tasks/mobile_mm/config.py
lookahead_steps: int = 5  # Up from 3
```

**Benefits:**
- Longer planning horizon for complex trajectories
- Better anticipation of sharp turns

**Tradeoffs:**
- Larger observation space (15 dims instead of 9)
- May not matter if policy already learns well with 3 steps

---

#### 3.2: Trajectory Phase Information

Add explicit trajectory progress to observations:

```python
# In observations.py compose_observation()
trajectory_phase = current_waypoint_idx / total_waypoints  # [0, 1]
components.append(trajectory_phase.unsqueeze(-1))  # +1 dim
```

**Benefits:**
- Policy knows if at start, middle, or end of trajectory
- May help with trajectory-specific strategies

**Tradeoffs:**
- Assumes fixed-length trajectories (current loop handling makes this ambiguous)

---

#### 3.3: Spatial Progress Rewards

Reward progress along trajectory path, not just tracking error:

```python
def trajectory_progress_reward(
    prev_waypoint_idx: torch.Tensor,
    current_waypoint_idx: torch.Tensor,
) -> torch.Tensor:
    """Reward for advancing through trajectory waypoints."""
    return (current_waypoint_idx - prev_waypoint_idx).float()
```

**Benefits:**
- Encourages keeping up with trajectory advancement
- Complements position tracking

**Tradeoffs:**
- May conflict with position tracking (robot might rush ahead)
- Needs careful weight balancing

---

## Implementation Roadmap

### Phase 1: Critical Fixes (DO NOW) ⚡

**Timeline:** 1-2 hours implementation + testing

1. ✅ **Implement grace period for termination** (Solution A)
   - Add `tracking_error_grace_period: 10` to config
   - Add `high_error_count` buffer to env
   - Modify `_get_dones()` logic
   - Update `_reset_idx()` to reset counter

2. ✅ **Test with small run** (256 envs, 10 minutes)
   - Verify no infinite reset loops
   - Check [RESET] message frequency
   - Monitor enhanced tracking display statistics

3. ✅ **Restart full training** (8192 envs)
   - Continue from 25.7M steps OR restart from scratch
   - Monitor for improvement in "Broken (>2.0m)" percentage
   - Expect: <0.05% broken (down from 0.1-1%)

**Expected outcome:** Infinite reset loops eliminated, training more stable.

---

### Phase 2: Strategic Enhancements (AFTER PHASE 1 SUCCESS) 🎯

**Timeline:** 1-2 days implementation + retraining

1. ✅ **Implement velocity tracking rewards**
   - Add `get_target_velocity()` to trajectories.py
   - Add `velocity_tracking_reward()` to rewards.py
   - Add velocity reward term to `_get_rewards()`
   - Add config parameter

2. ✅ **Tune reward weights**
   - Start with `velocity_tracking: 0.2`
   - Balance against `position_tracking: 1.0`
   - Monitor training curves for conflicts

3. ✅ **Retrain from scratch** (architectural change)
   - 8192 envs, 100M steps
   - Compare with position-only baseline
   - Evaluate trajectory timing accuracy

**Expected outcome:** Tighter trajectory timing adherence, better temporal tracking.

---

### Phase 3: Optional Polish (IF NEEDED) 💡

**Timeline:** Optional, based on performance needs

1. Increase lookahead horizon (5 waypoints)
2. Add trajectory phase information
3. Experiment with spatial progress rewards

**Decision criteria:** Only implement if Phase 1+2 results show specific deficiencies.

---

## Architecture Verdict

### ✅ Current Design is FUNDAMENTALLY SOUND

**Strengths:**
1. ✅ **Lookahead awareness:** Policy sees 3 future waypoints → trajectory direction known
2. ✅ **Smooth interpolation:** Eliminates step-wise jumps confusing for learning
3. ✅ **Time-based advancement:** Matches real-world trajectory execution paradigm
4. ✅ **Base mobilization:** Explicitly rewards chassis movement to reach distant targets
5. ✅ **Proven results:** 0.958 explained variance shows architecture learns extremely well

**Weaknesses:**
1. ❌ **Termination too aggressive:** 2.0m threshold assumes goal-reaching, not moving targets
2. ❌ **Position-only rewards:** No velocity tracking → ignores trajectory timing
3. ⚠️ **Limited lookahead:** 0.3s may be insufficient for complex maneuvers

### 🎯 Key Insight: This is NOT "Chasing a Moving Goal"

Your concern was: "chasing a simple goal point should be very different with tracking a series of waypoints, right?"

**Answer:** Your implementation is ALREADY trajectory-aware, not simple goal-chasing:
- Policy sees **waypoint sequence** via lookahead (knows where trajectory is going)
- Target **interpolates smoothly** between waypoints (not discrete jumps)
- Time-based advancement creates **temporal structure** (must keep pace with trajectory)

**The issue is NOT fundamental design** — it's tactical execution (termination policy) and strategic refinement (velocity rewards).

---

## Comparison: Goal-Reaching vs Trajectory-Tracking

| Aspect | Goal-Reaching | Your Implementation | Ideal Trajectory-Tracking |
|--------|---------------|---------------------|---------------------------|
| **Target** | Static position | Smoothly moving (interpolated) | ✅ Correct |
| **Observations** | Current target only | Current + 3 lookahead waypoints | ✅ Good (could increase to 5) |
| **Advancement** | N/A (target fixed) | Time-based (every 0.1s) | ✅ Correct |
| **Rewards** | Position error | Position error only | ❌ Add velocity tracking |
| **Termination** | 2.0m error reasonable | 2.0m too strict for moving target | ❌ Add grace period |
| **Temporal awareness** | N/A | Via lookahead | ✅ Present |

**Verdict:** 3/5 excellent, 2/5 need fixes. **NOT a fundamental redesign.**

---

## Training Impact Analysis

### Current Session 4b (25.7M/100M steps)

**Metrics (EXCELLENT):**
- Explained variance: 0.958 (near-perfect value function!)
- Value loss: 0.00443 (extremely low)
- FPS: 3134 (efficient)

**Issues (MINOR):**
- ~0.1-1% envs in infinite reset loops (tactical fix needed)
- No velocity tracking (strategic enhancement opportunity)

### Expected Impact of Fixes

**After Phase 1 (Grace Period):**
- Infinite loops eliminated → 100% envs learning
- Slight FPS decrease (~2-3%) from extra logic
- Training stability improved
- **Can continue from 25.7M steps** (tactical fix, no architecture change)

**After Phase 2 (Velocity Rewards):**
- Tighter trajectory timing adherence
- May slow initial learning (more complex reward)
- Better final performance on time-critical tasks
- **MUST restart from 0** (architectural change)

---

## Conclusion

Your trajectory tracking architecture is **well-designed and proven effective** (0.958 explained variance is outstanding!). The issues you're seeing are:

1. **Termination policy bug** (tactical, easy fix)
2. **Missing velocity rewards** (strategic, nice-to-have)

**NOT fundamental design flaws.**

### Next Steps:

1. ✅ **Implement grace period** (1-2 hours)
2. ✅ **Test with small run** (10 minutes)
3. ✅ **Restart training** from 25.7M OR from scratch
4. ⏸️ **Wait for results** before considering velocity rewards

The current architecture already handles trajectory tracking correctly — you just need to fix the termination policy to stop the reset loops.

---

## References

**Code Files:**
- `src/rl_platform/tasks/mobile_mm/env.py` - Main environment logic
- `src/rl_platform/tasks/mobile_mm/trajectories.py` - Trajectory manager
- `src/rl_platform/tasks/mobile_mm/observations.py` - Observation composition
- `src/rl_platform/tasks/mobile_mm/rewards.py` - Reward functions
- `src/rl_platform/tasks/mobile_mm/config.py` - Task configuration

**Related Docs:**
- `README.md` - Quick-start and architecture overview
- `docs/reference/reward_cheatsheet.md` - Reward system reference
- `docs/workflows/multi_trajectory_training.md` - Training procedures
- `docs/tracking/ee_frame_alignment.md` - EE tracking details

**Training Sessions:**
- Session 1: 44.9M steps (base frozen - physics bugs)
- Session 2: 450 steps (base moving 6mm vs 23mm - spring too stiff)
- Session 3: ~500 steps (base mobilization verified - 20Hz control)
- Session 4a: 10M steps (4096 envs - excellent results)
- **Session 4b: 25.7M/100M steps** (8192 envs - current) ← THIS ANALYSIS

---

**Document Version:** 1.0  
**Last Updated:** October 21, 2025  
**Status:** Ready for implementation (Phase 1 recommended)
