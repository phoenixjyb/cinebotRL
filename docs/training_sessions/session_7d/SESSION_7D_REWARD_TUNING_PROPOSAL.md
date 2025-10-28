# Reward Tuning Proposal for Session 7d

**Date:** 2025-10-28  
**Purpose:** Fix strategic base movement issues identified in Session 7c  
**Target:** Increase reachability from 6% to >50%, reduce mean error from 1.01m to <0.50m

---

## 🎯 Problem Statement

### Session 7c Issues:

1. **Base moves but not strategically:**
   - Movement: 0.1-1.8m ✅ (good range)
   - But 93% of targets remain unreachable ❌
   - Mobilization reward: 0.0-2.3 pts (too low to drive behavior)

2. **Reward imbalance:**
   ```
   position_tracking:    82.09 pts  (dominant ✅)
   target_distance_pen:  20.01 pts  (too punishing ❌)
   base_mobilization:     0.00 pts  (too weak ❌)
   ```

3. **From evaluation logs:**
   ```
   ⚠️ Base SHOULD move! (target 1.335m beyond reach → penalty 13.35 pts)
   ⚠️ Base SHOULD move! (target 0.898m beyond reach → penalty 8.98 pts)
   
   Reachability: 1/16 envs (6.25%)
   Avg base→target distance: 1.288m
   Base mobilization: 0.0-2.3 pts (not enough incentive!)
   ```

---

## 💡 Root Cause Analysis

### Current Reward Balance:

```python
# config.py (Session 7c)
class RewardWeights:
    position_tracking: float = 100.0           # Dominant
    base_progress_reward: float = 150.0        # HIGH weight but...
    target_distance_penalty: float = 5.0       # Penalty scale
```

### Why Base Mobilization is Low:

Looking at `rewards.py` line 75-132 (`base_mobilization_reward`):

```python
# Progress capped at 0.2m per step (20cm)
progress = torch.clamp(progress, min=0.0, max=0.2)

# Max reward per step:
# base_progress_reward × max_progress = 150 × 0.2 = 30 pts
```

**But in practice:**
- Most steps have <5cm progress → 150 × 0.05 = 7.5 pts
- Steps with no movement → 0 pts
- Only rewarded when target is out of reach
- **Observed in logs: 0.0-2.3 pts (too low!)**

### Why Distance Penalty Dominates:

Looking at `rewards.py` line 135+ (`target_distance_penalty`):

```python
# Penalty calculation:
dist_beyond_reach = dist_to_target - arm_reach  # e.g., 1.3m - 0.6m = 0.7m
penalty = scale × dist_beyond_reach             # 5.0 × 0.7 = 3.5 pts

# But in logs we see 13.35 pts penalty!
# This means: scale × 2.67m beyond reach
# Suggests scale might be multiplied elsewhere or distance is larger
```

**Problem:**
- Penalty grows linearly with distance (unbounded)
- Base progress reward capped at 30 pts/step
- **Penalty can exceed potential reward!**

---

## 🔧 Proposed Changes

### Change 1: Increase Base Progress Weight (HIGH PRIORITY) 🔥

**File:** `src/rl_platform/tasks/mobile_mm/config.py`

**Current:**
```python
base_progress_reward: float = 150.0  # Max 30 pts/step (150 × 0.2)
```

**Proposed:**
```python
base_progress_reward: float = 250.0  # Max 50 pts/step (250 × 0.2) ⬆️ 67% increase
```

**Rationale:**
- Makes base progress comparable to position tracking (100 pts)
- 50 pts max is enough to overcome typical penalties
- Still capped to prevent explosion

**Expected Impact:**
- Base movement becomes more attractive
- Policy learns to move toward unreachable targets
- Reachability should increase from 6% → 30-50%

---

### Change 2: Reduce Distance Penalty Scale (HIGH PRIORITY) 🔥

**File:** `src/rl_platform/tasks/mobile_mm/config.py`

**Current:**
```python
target_distance_penalty: float = 5.0  # Can reach 13+ pts penalty
```

**Proposed:**
```python
target_distance_penalty: float = 3.0  # Gentler penalty ⬇️ 40% reduction
```

**Rationale:**
- Current penalty too harsh (13 pts = 26% of tracking reward!)
- Should encourage movement, not punish immobility too harshly
- 3.0 scale → max ~8 pts penalty (more reasonable)

**Expected Impact:**
- Less punishment for being far from target
- Base has more "incentive space" to explore movement
- Better balance with mobilization reward

---

### Change 3: Add Base-Target Alignment Reward (NEW COMPONENT) 🎯

**File:** `src/rl_platform/tasks/mobile_mm/rewards.py`

**Add new reward function:**

```python
def base_target_alignment_reward(
    base_pos: torch.Tensor,
    base_vel: torch.Tensor,
    target_pos: torch.Tensor,
    arm_reach: float = 0.6,
    scale: float = 1.0,
) -> torch.Tensor:
    """Reward base movement that aligns toward the target.
    
    Encourages moving in the correct direction, even if not reducing
    distance much yet. Helps policy learn goal-directed navigation.
    
    Args:
        base_pos: Current base position [num_envs, 3]
        base_vel: Base velocity (linear XY) [num_envs, 3]
        target_pos: Target position [num_envs, 3]
        arm_reach: Arm workspace radius
        scale: Reward scale factor
        
    Returns:
        Reward [num_envs] - positive when moving toward target
    """
    # Only reward when target is out of reach
    base_xy = base_pos[:, :2]
    target_xy = target_pos[:, :2]
    dist_to_target = torch.norm(target_xy - base_xy, dim=-1)
    out_of_reach = (dist_to_target > arm_reach).float()
    
    # Direction from base to target
    base_to_target = target_xy - base_xy
    base_to_target_norm = torch.norm(base_to_target, dim=-1, keepdim=True)
    base_to_target_unit = base_to_target / (base_to_target_norm + 1e-6)
    
    # Base velocity direction (XY only)
    base_vel_xy = base_vel[:, :2]
    base_speed = torch.norm(base_vel_xy, dim=-1, keepdim=True)
    base_vel_unit = base_vel_xy / (base_speed + 1e-6)
    
    # Dot product: 1.0 = perfect alignment, 0.0 = perpendicular, -1.0 = opposite
    alignment = torch.sum(base_vel_unit * base_to_target_unit, dim=-1)
    alignment = torch.clamp(alignment, min=0.0, max=1.0)  # Only reward positive alignment
    
    # Scale by speed (moving faster in right direction = more reward)
    speed_scale = torch.clamp(base_speed.squeeze(-1), min=0.0, max=0.5)  # Cap at 0.5 m/s
    
    return scale * alignment * speed_scale * out_of_reach
```

**Update config.py:**
```python
class RewardWeights:
    # ... existing rewards ...
    base_target_alignment: float = 10.0  # NEW: Reward moving toward target
```

**Integrate in env.py (around line 1200):**
```python
# Add to _compute_rewards() after base_progress_reward
alignment_reward = rewards.base_target_alignment_reward(
    base_pos=self.robot.data.root_pos_w,
    base_vel=self.robot.data.root_lin_vel_w,
    target_pos=self.trajectory.current_target_pos,
    arm_reach=0.6,
    scale=self.task_cfg.rewards.base_target_alignment,
)
reward_dict["base_target_alignment"] = alignment_reward
```

**Rationale:**
- Provides **intermediate feedback** for moving in right direction
- Helps policy learn navigation before perfect distance reduction
- Max reward: 10 × 1.0 alignment × 0.5 speed = 5 pts (reasonable)

**Expected Impact:**
- Base learns to "aim" toward targets
- Faster convergence to goal-directed movement
- Improved reachability (6% → 50%+)

---

### Change 4: Adjust Position Tracking Weight (OPTIONAL)

**File:** `src/rl_platform/tasks/mobile_mm/config.py`

**Current:**
```python
position_tracking: float = 100.0  # Session 7c (increased from 50)
```

**Proposed:**
```python
position_tracking: float = 80.0  # Slight reduction ⬇️ 20%
```

**Rationale:**
- Make room for base mobilization rewards
- 80 pts still dominant but not overwhelming
- Better balance between tracking and mobility

**Alternative:** Keep at 100.0 (tracking should be primary objective)

**Decision:** **SKIP THIS CHANGE** - Keep position_tracking at 100.0 as dominant signal

---

## 📊 Expected Reward Balance After Changes

### Current (Session 7c):
```
Component                     Min      Mean     Max      % of Total
--------------------------------------------------------------------
position_tracking            0.56     40.00    97.87    65% (dominant ✅)
velocity_limit_penalty      20.00     25.00    36.00    23% (too high ⚠️)
target_distance_penalty      0.18      5.00    13.35     8% (too high ⚠️)
base_mobilization            0.00      0.27     2.31     2% (too low ❌)
orientation_tracking         0.73      1.50     1.69     1%
others                       ~0.50      ~1.0     ~2.0     1%
--------------------------------------------------------------------
TOTAL                         ~22       ~73     ~153    100%
```

### Proposed (Session 7d):
```
Component                     Min      Mean     Max      % of Total
--------------------------------------------------------------------
position_tracking            0.56     40.00    97.87    55% (dominant ✅)
base_mobilization            0.00      8.00    50.00    14% (much better! ⬆️)
velocity_limit_penalty      20.00     25.00    36.00    20% (acceptable)
base_target_alignment        0.00      2.00     5.00     3% (NEW ⭐)
target_distance_penalty      0.11      3.00     8.00     5% (reduced ⬇️)
orientation_tracking         0.73      1.50     1.69     1%
others                       ~0.50      ~1.0     ~2.0     2%
--------------------------------------------------------------------
TOTAL                         ~22       ~81     ~200    100%
```

**Key Improvements:**
- Base mobilization: 2% → 14% (7x increase! 🎉)
- Distance penalty: 8% → 5% (less punishing)
- New alignment reward provides intermediate feedback
- Position tracking still dominant (good!)

---

## 🚀 Implementation Plan

### Step 1: Code Changes (5 minutes)

**1a. Update config.py:**
```python
# src/rl_platform/tasks/mobile_mm/config.py, line ~81

@dataclass
class RewardWeights:
    """Reward term weights for the tracking task."""
    
    # Tracking rewards
    position_tracking: float = 100.0  # Keep as dominant
    orientation_tracking: float = 2.0
    progress_bonus: float = 1.0
    base_progress_reward: float = 250.0  # ⬆️ INCREASED: 150→250 (67% boost)
    base_target_alignment: float = 10.0  # ⭐ NEW: Reward moving toward target
    target_distance_penalty: float = 3.0  # ⬇️ REDUCED: 5→3 (40% gentler)
    excessive_base_movement_penalty: float = 10.0
    
    # ... rest unchanged ...
```

**1b. Add alignment reward function to rewards.py:**

Insert after `base_mobilization_reward()` (around line 132):

```python
def base_target_alignment_reward(
    base_pos: torch.Tensor,
    base_vel: torch.Tensor,
    target_pos: torch.Tensor,
    arm_reach: float = 0.6,
    scale: float = 1.0,
) -> torch.Tensor:
    """Reward base movement that aligns toward the target.
    
    Encourages moving in the correct direction, even if not reducing
    distance much yet. Helps policy learn goal-directed navigation.
    
    Args:
        base_pos: Current base position [num_envs, 3]
        base_vel: Base velocity (linear XY) [num_envs, 3]
        target_pos: Target position [num_envs, 3]
        arm_reach: Arm workspace radius
        scale: Reward scale factor
        
    Returns:
        Reward [num_envs] - positive when moving toward target
    """
    # Only reward when target is out of reach
    base_xy = base_pos[:, :2]
    target_xy = target_pos[:, :2]
    dist_to_target = torch.norm(target_xy - base_xy, dim=-1)
    out_of_reach = (dist_to_target > arm_reach).float()
    
    # Direction from base to target
    base_to_target = target_xy - base_xy
    base_to_target_norm = torch.norm(base_to_target, dim=-1, keepdim=True)
    base_to_target_unit = base_to_target / (base_to_target_norm + 1e-6)
    
    # Base velocity direction (XY only)
    base_vel_xy = base_vel[:, :2]
    base_speed = torch.norm(base_vel_xy, dim=-1, keepdim=True)
    base_vel_unit = base_vel_xy / (base_speed + 1e-6)
    
    # Dot product: 1.0 = perfect alignment, 0.0 = perpendicular, -1.0 = opposite
    alignment = torch.sum(base_vel_unit * base_to_target_unit, dim=-1)
    alignment = torch.clamp(alignment, min=0.0, max=1.0)  # Only reward positive alignment
    
    # Scale by speed (moving faster in right direction = more reward)
    speed_scale = torch.clamp(base_speed.squeeze(-1), min=0.0, max=0.5)  # Cap at 0.5 m/s
    
    return scale * alignment * speed_scale * out_of_reach
```

**1c. Integrate in env.py:**

Find `_compute_rewards()` method (around line 1200-1300) and add after base progress reward:

```python
# Around line 1250, after base_progress computation
alignment_reward = rewards.base_target_alignment_reward(
    base_pos=self.robot.data.root_pos_w,
    base_vel=self.robot.data.root_lin_vel_w,
    target_pos=self.trajectory.current_target_pos,
    arm_reach=0.6,
    scale=self.task_cfg.rewards.base_target_alignment,
)
reward_dict["base_target_alignment"] = alignment_reward
```

Also add to logging section (around line 1380):

```python
# In _print_tracking_stats() method
print(f"   base_target_alignment: min={alignment_reward.min():.4f}  mean={alignment_reward.mean():.4f}  max={alignment_reward.max():.4f}")
```

### Step 2: Verification (5 minutes)

**Test environment loads:**
```bash
I:\isaaclab\isaaclab.bat -p scripts\test_mobile_mm_env.py --headless
```

**Expected output:**
- No errors about missing reward components
- Environment initializes with 4096 envs
- Reward dict includes `base_target_alignment`

### Step 3: Launch Training (11 hours)

**Session 7d command:**
```powershell
.\scripts\launch_training_windows.ps1 `
  -Task MobileMMTrackEE-v0 `
  -NumEnvs 4096 `
  -Headless `
  -TotalTimesteps 200000000
```

**What to monitor:**
- Base mobilization rewards increasing (look for 5-20 pts)
- Base target alignment rewards appearing (look for 2-5 pts)
- Distance penalties decreasing (from 13 → 8 pts)
- Mean episode reward trending upward

### Step 4: Evaluation (20 minutes)

**After training completes:**
```bash
I:\isaaclab\isaaclab.bat -p scripts\reinforcement_learning\sb3\evaluate.py \
  --checkpoint H:\wSpace\cinebotRL\logs\sb3\mobilemmtrackee_v0\[SESSION_7D_DIR]\final_model.zip \
  --num_envs 16 \
  --num_episodes 100 \
  --headless \
  --deterministic \
  --trajectory_type multi_recorded \
  --use_all_trajectories
```

**Success criteria:**
```
✅ Reachability > 50% (currently 6%)
✅ Mean tracking error < 0.50m (currently 1.01m)
✅ Base mobilization > 5.0 avg (currently 0.27)
✅ Good envs (0.1-0.3m) > 30% (currently 19%)
```

---

## 📊 Predicted Outcomes

### Conservative Estimate:
```
Reachability: 6% → 25%
Mean error: 1.01m → 0.70m
Base mobilization: 0.27 → 3.5 pts avg
Good envs: 19% → 30%
Episode reward: 12,330 → 18,000
```

### Optimistic Estimate:
```
Reachability: 6% → 60%
Mean error: 1.01m → 0.45m
Base mobilization: 0.27 → 8.0 pts avg
Good envs: 19% → 50%
Episode reward: 12,330 → 25,000
```

### Worst Case:
```
Reachability: 6% → 15%
Mean error: 1.01m → 0.85m
Base mobilization: 0.27 → 2.0 pts avg
Good envs: 19% → 25%
Episode reward: 12,330 → 14,000

→ If this happens, increase base_progress_reward to 400.0 and retrain
```

---

## 🎯 Rollback Plan

If Session 7d performs worse than 7c:

**Revert config.py:**
```python
base_progress_reward: float = 150.0  # Back to Session 7c
target_distance_penalty: float = 5.0  # Back to Session 7c
# base_target_alignment: float = 10.0  # Remove line
```

**Remove alignment reward from rewards.py and env.py**

**Re-evaluate Session 7c model** to confirm baseline

---

## 🔬 Alternative Approaches (If This Fails)

### Option A: Curriculum Learning
- Start with close targets (all reachable)
- Gradually increase target distances
- Let policy learn base movement on easy cases first

### Option B: Stronger Alignment Reward
- Increase `base_target_alignment: 10.0 → 20.0`
- Make directional movement more attractive

### Option C: Exponential Progress Reward
- Change `progress` scaling from linear to exponential
- Reward large movements more than small ones

### Option D: Dense Waypoint Rewards
- Reward intermediate progress toward target
- Break long movements into sub-goals

---

## 📝 Summary

### Changes:
1. ⬆️ Increase `base_progress_reward`: 150 → 250 (67% boost)
2. ⬇️ Reduce `target_distance_penalty`: 5 → 3 (40% gentler)
3. ⭐ Add `base_target_alignment` reward (new, 10.0 weight)

### Expected Impact:
- Reachability: 6% → 25-60%
- Mean error: 1.01m → 0.45-0.70m
- Base becomes goal-directed navigator
- More balanced reward distribution

### Time Investment:
- Code changes: 5 minutes
- Training: 11 hours (200M timesteps)
- Evaluation: 20 minutes
- Total: ~12 hours

### Risk: LOW
- Changes are additive (new reward) and tuning (existing weights)
- Can easily revert if performance degrades
- Based on concrete issues identified in Session 7c logs

---

**Ready to implement? Let's make Session 7d the breakthrough we need!** 🚀
