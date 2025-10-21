# Critical Analysis: Why Base Might Not Learn to Move

**Date:** October 21, 2025  
**Context:** Training at 25.7M/100M steps with excellent metrics (0.958 explained variance) but minimal base movement

---

## 🔍 Hypothesis: Observation Space Missing Critical Information

### Problem Statement

After 25.7M training steps with excellent value function learning (0.958 explained variance, 0.00443 value loss), the base shows minimal movement. This suggests the policy HAS learned to optimize rewards, but the reward landscape may not require base movement OR the observations don't provide sufficient information to trigger base movement learning.

---

## 🐛 Potential Bug #1: Missing Base-to-Target Distance in Observations

### Current Observation Composition

From `src/rl_platform/tasks/mobile_mm/observations.py` lines 50-75:

```python
components = []

# Base state (13 dims)
components.extend([base_pos, base_quat, base_lin_vel, base_ang_vel])

# Arm joints (12 dims)
arm_joint_pos = joint_pos[:, 3:9]
arm_joint_vel = joint_vel[:, 3:9]
components.extend([arm_joint_pos, arm_joint_vel])

# End-effector state (13 dims)
components.extend([ee_pos, ee_quat, ee_lin_vel, ee_ang_vel])

# Tracking error (7 dims) ← ONLY EE-TO-TARGET!
pos_error = target_pos - ee_pos  # EE error, NOT base-to-target distance!
quat_error = quat_diff(ee_quat, target_quat)
components.extend([pos_error, quat_error])

# Lookahead (9 dims if enabled)
if lookahead_pos is not None:
    components.append(lookahead_flat)
```

### The Critical Gap

**What policy sees:**
- ✅ Base position (world frame)
- ✅ Target position (via EE error: `target - ee`)
- ✅ EE position
- ❌ **Missing: base-to-target distance!**
- ❌ **Missing: "is target within arm reach?" signal**

**Why this matters:**

The policy can compute EE-to-target error, but to know "should I move the base?", it needs to know:
1. **Distance from base to target** (not just EE to target)
2. **Is this distance beyond arm reach?** (explicit signal)

Currently, the policy must:
1. Observe `base_pos` (world frame)
2. Observe `ee_pos` (world frame)  
3. Infer `target_pos` from `pos_error = target_pos - ee_pos`
4. Compute `base_to_target = target_pos - base_pos` (requires 2 arithmetic operations in latent space!)
5. Compare `||base_to_target||` with arm reach (~0.6m) to decide if base should move

**This is a complex inference chain!** The policy might not learn this multi-step reasoning, especially when arm-only strategies give good rewards.

---

## 🐛 Potential Bug #2: Reward Landscape May Not Require Base Movement

### Current Reward Structure

From `src/rl_platform/tasks/mobile_mm/rewards.py` lines 660-680:

```python
total_reward = (
    pos_reward                    # 50.0 * exp(-error²)
    + ori_reward                  # 2.0 * exp(-angular_error²)
    + prog_bonus                  # 1.0 * improvement
    + base_mob_reward             # 50.0 * (progress when out of reach)
    - dist_penalty                # 10.0 * (distance - arm_reach)
    - action_mag_penalty          # 0.01 * ||actions||²
    - action_rt_penalty           # 0.01 * ||Δactions||²
    - action_smooth_penalty       # 0.05 * ||ΔΔactions||²
    - vel_limit_penalty           # 5.0 * violations
    - accel_limit_penalty         # 5.0 * violations
    - jerk_penalty                # 0.1 * ||jerk||²
    - joint_limit_penalty         # 10.0 * violations
    - lateral_penalty             # 2.0 * sideways motion
    - self_coll_penalty           # 50.0 * collision forces
    - stab_penalty                # 0.1 * oscillation
)
```

### Reward Analysis

**When target is within arm reach (<0.6m):**
- `pos_reward`: Maximum ~50.0 (exponential, very high when error small)
- `base_mob_reward`: **~0** (sigmoid suppresses reward when in reach)
- `dist_penalty`: **~0** (no penalty within arm reach)
- Action penalties: ~-0.1 to -1.0 (small)

**Total when in reach: ~45-50** (arm-only strategy works!)

**When target is out of reach (>0.6m):**
- `pos_reward`: Drops exponentially (e.g., at 2.0m error → ~0.0002)
- `base_mob_reward`: **+50.0 * progress** (IF base moves toward target)
- `dist_penalty`: **-10.0 * (dist - 0.6)** (e.g., at 2.0m → -14.0)
- Action penalties: ~-0.1 to -1.0

**Total when out of reach:**
- If base doesn't move: `0.0002 - 14.0 = -14.0` (BAD!)
- If base moves 0.1m closer: `0.0002 + 5.0 - 14.0 = -9.0` (LESS BAD!)

**The problem:**
Even when base DOES move, reward is still negative (-9.0)! The policy might learn:
- "Avoid trajectories that go out of reach" (stay close to base)
- "Stretch arm as far as possible" (max 0.6m)
- "Accept some tracking error to avoid base movement penalties"

---

## 🐛 Potential Bug #3: Base Movement vs. Arm Movement Efficiency

### Energy Efficiency Comparison

**Arm movement:**
- Reward: `pos_reward = 50.0 * exp(-error²)` → ~40-50 when error <0.1m
- Cost: `action_penalties ~0.1` (small joint motions)
- Net: **+40 to +50**

**Base movement:**
- Reward: `base_mob = 50.0 * progress` → ~5-10 for 0.1m movement
- Cost: 
  - `dist_penalty = -10.0 * (1.0 - 0.6) = -4.0` (while moving)
  - `action_penalties ~0.2` (2 base DOFs)
  - `accel_penalty ~1.0` (accelerating base)
  - `jerk_penalty ~0.2` (changing base velocity)
- Net: **+5 -4 -1.5 = -0.5** (NEGATIVE!)

**Conclusion:** Moving the base is LESS rewarding than just using the arm, even when target is far!

---

## 🐛 Potential Bug #4: Lookahead Might Show Targets Return to Reach

### Lookahead Behavior

Config shows:
```python
use_lookahead: bool = True
lookahead_steps: int = 3  # Next 3 waypoints
lookahead_dt: float = 0.1  # 0.1s per waypoint
```

**Scenario:**
1. Current target at 0.8m from base (out of reach)
2. Lookahead shows target moves back to 0.4m in 0.3s
3. Policy learns: "Wait, target will come back in reach soon, don't bother moving base!"

This would be SMART but prevents base mobilization learning.

---

## 🐛 Potential Bug #5: Trajectory Characteristics

Let me check if trajectories actually require base movement:

**Questions:**
1. What is the maximum distance any target goes from the starting base position?
2. Do trajectories stay within 0.6m arm reach most of the time?
3. Is there enough "dwell time" at far targets to make base movement worthwhile?

If trajectories are mostly within arm reach, the policy CORRECTLY learns to not move the base!

---

## 🔬 Diagnostic Tests

### Test 1: Check Observation Space

**Question:** Can policy distinguish "target within reach" vs "target out of reach"?

**Test:**
```python
# In observations.py, add explicit base-to-target info
base_to_target = target_pos - base_pos  # [num_envs, 3]
base_to_target_dist = torch.norm(base_to_target[:, :2], dim=-1, keepdim=True)  # [num_envs, 1]
out_of_reach_signal = (base_to_target_dist > 0.6).float()  # [num_envs, 1] binary flag

components.extend([base_to_target, base_to_target_dist, out_of_reach_signal])
```

**Expected impact:** Policy immediately sees "I'm out of reach, need to move base!"

---

### Test 2: Check Reward Balance

**Question:** Is base movement reward competitive with arm-only strategy?

**Test:**
```python
# Increase base mobilization reward
base_progress_reward: float = 100.0  # Up from 50.0

# Reduce distance penalty (too harsh)
target_distance_penalty: float = 5.0  # Down from 10.0

# Or even better: only penalize if NOT moving base
def smart_distance_penalty(base_pos, prev_base_pos, target_pos, arm_reach=0.6):
    dist = torch.norm(target_pos[:, :2] - base_pos[:, :2], dim=-1)
    base_moved = torch.norm(base_pos[:, :2] - prev_base_pos[:, :2], dim=-1) > 0.01
    
    # Only penalize if out of reach AND not moving
    penalty = (dist - arm_reach).clamp(min=0.0)
    penalty = torch.where(base_moved, penalty * 0.1, penalty)  # 10x less penalty if moving
    return penalty
```

---

### Test 3: Analyze Trajectory Statistics

**Question:** Do trajectories require base movement?

**Test script:**
```python
import json
import numpy as np

# Load trajectory
with open('trajectoryToLearn/1_pull_world_scaled.json') as f:
    traj = json.load(f)

# Assume starting base at [0, 0, 0]
base_pos = np.array([0, 0, 0.86])  # Typical base height
positions = np.array([wp['ee_pos'] for wp in traj['trajectory']])

# Compute distances
distances = np.linalg.norm(positions[:, :2] - base_pos[:2], axis=1)

print(f"Trajectory distance statistics (2D):")
print(f"  Min: {distances.min():.3f}m")
print(f"  Mean: {distances.mean():.3f}m")
print(f"  Max: {distances.max():.3f}m")
print(f"  % within arm reach (<0.6m): {(distances < 0.6).mean() * 100:.1f}%")
print(f"  % out of reach (>0.6m): {(distances > 0.6).mean() * 100:.1f}%")
```

---

## 🎯 Recommended Fixes (Priority Order)

### Priority 1: Add Explicit Base-to-Target Information to Observations ⚡

**Why:** Policy needs clear signal about when to move base

**Implementation:**
```python
# In src/rl_platform/tasks/mobile_mm/observations.py

def compose_observation(...):
    components = []
    
    # ... existing components ...
    
    # NEW: Base-to-target information (CRITICAL FOR BASE MOBILIZATION!)
    base_to_target_xy = target_pos[:, :2] - base_pos[:, :2]  # [num_envs, 2]
    base_to_target_dist = torch.norm(base_to_target_xy, dim=-1, keepdim=True)  # [num_envs, 1]
    
    # Explicit "out of reach" signal (binary flag)
    arm_reach = 0.6  # meters
    out_of_reach = (base_to_target_dist > arm_reach).float()  # [num_envs, 1]
    
    # Add to observations
    components.extend([base_to_target_xy, base_to_target_dist, out_of_reach])
    # Adds 4 dims: [dx, dy, distance, out_of_reach_flag]
    
    # ... rest of composition ...
```

**Expected improvement:** Policy immediately learns "when this flag is 1, move base!"

---

### Priority 2: Rebalance Rewards to Make Base Movement Worthwhile 🎯

**Why:** Current reward structure penalizes base movement even when it's necessary

**Implementation:**
```python
# In src/rl_platform/tasks/mobile_mm/config.py

class RewardWeights:
    position_tracking: float = 50.0  # Keep high
    base_progress_reward: float = 150.0  # INCREASE from 50.0 (3x stronger!)
    target_distance_penalty: float = 3.0  # DECREASE from 10.0 (less harsh)
    
    # Reduce action penalties for base (encourage exploration)
    action_magnitude: float = 0.005  # Half of current
    jerk_limit_penalty: float = 0.05  # Half of current
```

**Alternative: Smart distance penalty**
```python
# In src/rl_platform/tasks/mobile_mm/rewards.py

def target_distance_penalty(
    base_pos: torch.Tensor,
    prev_base_pos: torch.Tensor,
    target_pos: torch.Tensor,
    arm_reach: float = 0.6,
    scale: float = 10.0,
) -> torch.Tensor:
    """Penalize distance beyond arm reach, BUT reduce penalty if base is moving."""
    
    dist = torch.norm(target_pos[:, :2] - base_pos[:, :2], dim=-1)
    excess_dist = (dist - arm_reach).clamp(min=0.0)
    
    # Check if base actually moved (>1cm threshold to ignore noise)
    base_moved = torch.norm(base_pos[:, :2] - prev_base_pos[:, :2], dim=-1) > 0.01
    
    # Apply full penalty if not moving, 10% penalty if moving (encouragement!)
    penalty = torch.where(base_moved, excess_dist * 0.1, excess_dist)
    
    return scale * penalty
```

---

### Priority 3: Analyze Trajectory Characteristics 🔍

**Why:** If trajectories don't need base movement, policy is correct!

**Action:**
1. Run trajectory analysis script (see Test 3 above)
2. Check what % of waypoints are >0.6m from base
3. If <20% out of reach → **trajectories don't require base movement!**
4. Solution: Use trajectories with larger workspace or reduce arm reach in config

---

### Priority 4: Add Curriculum Learning 💡

**Why:** Policy might need gradual introduction to base movement

**Implementation:**
```python
# Gradually increase workspace size
class CurriculumConfig:
    stage_1: use trajectories with 80% within reach (learn arm control)
    stage_2: use trajectories with 50% out of reach (introduce base movement)
    stage_3: use trajectories with 20% within reach (require base movement)
```

---

## 📊 Validation Tests

After implementing fixes, check:

### Metric 1: Base Action Distribution
```python
base_actions = policy_output[:, 6:8]  # vx, wz
print(f"Base action std: {base_actions.std(dim=0)}")
# Should see std > 0.3 (active use of base actions)
```

### Metric 2: Base Movement Statistics
```python
base_pos_delta = base_pos - prev_base_pos
movement_magnitude = torch.norm(base_pos_delta[:, :2], dim=-1)
print(f"Base movement per step: {movement_magnitude.mean():.4f}m")
# Should see >0.001m average (real movement, not noise)
```

### Metric 3: Reward Component Balance
```python
print(f"base_mobilization reward: {base_mob_reward.mean():.2f}")
print(f"position_tracking reward: {pos_reward.mean():.2f}")
# base_mobilization should be comparable magnitude to position_tracking
```

---

## 🎯 Immediate Action Plan

1. **[30 min]** Run trajectory analysis (Test 3) - understand if base movement is needed
2. **[1 hour]** Implement Priority 1 fix (add base-to-target obs)
3. **[30 min]** Implement Priority 2 fix (rebalance rewards)
4. **[10 min]** Test with 256 envs, 10k steps - check base action distribution
5. **[Decision]** If base actions active → full retrain; if not → investigate further

---

## 🔬 Next Investigation If Fixes Don't Work

If base still doesn't move after fixes:

1. **Check action masking:** Is PPO masking base actions somehow?
2. **Check action initialization:** Does policy initialize with zero base actions?
3. **Check gradient flow:** Are base action gradients non-zero?
4. **Check USD physics:** Is base actually frozen in simulation despite commands?
5. **Check joint limits:** Are base joint limits preventing movement?

---

**Status:** Analysis complete, awaiting user decision on which test to run first.
