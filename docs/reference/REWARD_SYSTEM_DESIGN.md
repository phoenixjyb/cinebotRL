# Reward System Design - CinebotRL Mobile Manipulator

**Document Version**: 1.0  
**Last Updated**: October 22, 2025  
**Applies To**: Session 5b onwards (post-catastrophic-failure fixes)

---

## Executive Summary

This document details the complete reward function design for training the mobile manipulator to track end-effector trajectories. The reward system underwent critical fixes after Session 5's catastrophic failure (63.5% broken environments at 20M steps) due to unbounded reward exploitation.

**Key Design Principles:**
- **Bounded rewards**: All components have explicit caps to prevent reward hacking
- **Sparse penalties**: Heavy penalties only for critical failures (collisions, limits)
- **Dense progress signals**: Frame-by-frame tracking guidance
- **Hierarchical structure**: Base mobility → arm tracking → fine control

---

## Table of Contents

1. [Reward Function Overview](#reward-function-overview)
2. [Component Breakdown](#component-breakdown)
3. [Mathematical Formulations](#mathematical-formulations)
4. [Tuning History & Rationale](#tuning-history--rationale)
5. [Session 5 Failure Analysis](#session-5-failure-analysis)
6. [Session 5b Critical Fixes](#session-5b-critical-fixes)
7. [Expected Behavior](#expected-behavior)
8. [Future Improvements](#future-improvements)

---

## Reward Function Overview

### Total Reward Equation

```python
total_reward = (
    tracking_reward +           # Dense: End-effector position tracking
    base_mobilization_reward +  # Sparse: Base progress toward target (CAPPED)
    target_distance_penalty +   # Dense: Distance to next waypoint
    action_smoothness_penalty + # Dense: Jerky movements
    excessive_base_movement_penalty +  # NEW: Prevents wild base movements
    joint_limit_penalty +       # Sparse: Near joint limits
    collision_penalty +         # Sparse: Self-collision events
    base_orientation_penalty +  # Dense: Keeps base upright
    timeout_penalty            # Sparse: Episode timeout
)
```

### Reward Scale Target

- **Target total reward**: 35-45 per timestep (good tracking)
- **Minimum acceptable**: > 20 (functional but needs improvement)
- **Failure threshold**: < 0 (policy is breaking environments)

---

## Component Breakdown

### 1. Tracking Reward (PRIMARY OBJECTIVE)

**Purpose**: Guide end-effector to precisely follow the trajectory waypoints.

**Weight**: `50.0` (highest - this is the main task!)

**Formula**:
```python
tracking_error = ||ee_pos_current - trajectory_target_pos||₂
tracking_reward = 50.0 * exp(-5.0 * tracking_error)
```

**Characteristics**:
- **Type**: Dense reward (every timestep)
- **Range**: [0, 50] (decays exponentially with error)
- **Decay rate**: `-5.0` makes the reward drop rapidly beyond 0.2m error
- **Target**: Keep tracking_error < 0.25m for good rewards (> 35/50)

**Why exponential decay?**
- Linear penalty: Robot might settle for "good enough" tracking
- Exponential: Strong incentive to minimize error completely
- Decay rate of 5.0: Balances between too forgiving and too strict

**Expected Values**:
| Error (m) | Reward | Quality |
|-----------|--------|---------|
| 0.05 | 38.9 | Excellent |
| 0.10 | 30.3 | Good |
| 0.25 | 14.3 | Acceptable |
| 0.50 | 4.1 | Poor |
| 1.00 | 0.3 | Failed |

---

### 2. Base Mobilization Reward (CRITICAL - CAPPED IN SESSION 5B)

**Purpose**: Encourage base to move toward targets that are far from its current position.

**Weight**: `15.0`

**Formula (Session 5b - FIXED)**:
```python
# Compute progress toward target
base_pos_prev = previous_base_xy_position
base_pos_curr = current_base_xy_position
target_pos = trajectory_target_xy_position

progress = (target_pos - base_pos_prev).dot(base_pos_curr - base_pos_prev)
progress = progress / ||base_pos_curr - base_pos_prev||  # Normalize by movement

# CRITICAL FIX: Cap progress to prevent exploitation
MAX_PROGRESS_PER_STEP = 0.2  # meters
progress_capped = min(progress, MAX_PROGRESS_PER_STEP)

# Only reward if target is far (> 0.5m)
distance_to_target = ||target_pos - base_pos_curr||₂
if distance_to_target > 0.5:
    base_mobilization_reward = 15.0 * progress_capped
else:
    base_mobilization_reward = 0.0
```

**Characteristics**:
- **Type**: Sparse (only when target is far)
- **Range**: [0, 3.0] (15.0 × 0.2m cap)
- **Activation threshold**: Target must be > 0.5m away
- **Cap introduced**: Session 5b (was unbounded in Session 5 - DISASTER!)

**Session 5 Failure - What Went Wrong**:
```python
# OLD CODE (Session 5 - BROKEN):
progress = ...  # No cap!
base_mobilization_reward = 15.0 * progress  # Could be 15.0 × 10 = 150!
```

**Problem**:
- Robot discovered: "Move base as far as possible = infinite rewards"
- Base movements: 5-10 meters per step (physically impossible!)
- Policy learned: Reset spamming by moving wildly
- Result: 63.5% broken environments at 20M steps

**Fix (Session 5b)**:
- Progress capped at 0.2m per step (20cm maximum)
- Max reward: 15.0 × 0.2 = 3.0 (bounded!)
- Expected base movements: 5-15cm per step (realistic)

---

### 3. Excessive Base Movement Penalty (NEW - SESSION 5B)

**Purpose**: Punish base movements beyond physical feasibility to prevent reward hacking.

**Weight**: `10.0`

**Formula**:
```python
MOVEMENT_THRESHOLD = 0.1  # 10cm per step
base_movement = ||base_pos_curr - base_pos_prev||₂

excess = max(0.0, base_movement - MOVEMENT_THRESHOLD)
excessive_base_movement_penalty = -10.0 * excess
```

**Characteristics**:
- **Type**: Sparse penalty (only when movement > 10cm)
- **Range**: (-∞, 0] (unbounded penalty for extreme movements)
- **Threshold**: 10cm per step (reasonable for base with arm)

**Why added in Session 5b?**
- Session 5: Base moved 5-10 meters → needed hard penalty
- Works with capped mobilization reward to enforce realistic physics
- Penalty scales linearly with excess (harsh but fair)

**Expected Impact**:
| Movement | Penalty | Status |
|----------|---------|--------|
| 5cm | 0.0 | Good |
| 10cm | 0.0 | At threshold |
| 20cm | -1.0 | Moderate penalty |
| 50cm | -4.0 | Heavy penalty |
| 5m | -49.0 | Catastrophic |

---

### 4. Target Distance Penalty

**Purpose**: Provide continuous gradient toward the target waypoint.

**Weight**: `5.0` (increased from 3.0 in Session 5b)

**Formula**:
```python
distance = ||ee_pos - trajectory_target||₂
target_distance_penalty = -5.0 * distance
```

**Characteristics**:
- **Type**: Dense penalty (every timestep)
- **Range**: (-∞, 0] (unbounded, scales with distance)
- **Linear scaling**: Simple and effective

**Why increased from 3.0 to 5.0?**
- Session 5: Distance penalty too weak relative to mobilization reward
- Robot prioritized base movement over arm tracking
- Increase to 5.0: Stronger incentive to stay close to target

**Expected Values**:
| Distance (m) | Penalty |
|--------------|---------|
| 0.1 | -0.5 |
| 0.25 | -1.25 |
| 0.5 | -2.5 |
| 1.0 | -5.0 |
| 2.0 | -10.0 |

---

### 5. Action Smoothness Penalty

**Purpose**: Discourage jerky, high-frequency movements that damage hardware.

**Weight**: `1.0`

**Formula**:
```python
action_diff = ||action_current - action_previous||₂
action_smoothness_penalty = -1.0 * action_diff
```

**Characteristics**:
- **Type**: Dense penalty (every timestep)
- **Range**: (-∞, 0] (unbounded, scales with jerkiness)
- **Temporal consistency**: Penalizes rapid action changes

**Expected Values**:
- Smooth control: -0.1 to -0.5
- Normal operation: -0.5 to -2.0
- Jerky movements: < -5.0

---

### 6. Joint Limit Penalty

**Purpose**: Prevent joints from approaching their physical limits.

**Weight**: `5.0`

**Formula**:
```python
LIMIT_THRESHOLD = 0.1  # 10% margin

for each joint:
    range = joint_upper_limit - joint_lower_limit
    margin_lower = joint_position - joint_lower_limit
    margin_upper = joint_upper_limit - joint_position
    
    normalized_margin_lower = margin_lower / range
    normalized_margin_upper = margin_upper / range
    
    if normalized_margin_lower < LIMIT_THRESHOLD:
        penalty += -5.0 * (LIMIT_THRESHOLD - normalized_margin_lower)
    if normalized_margin_upper < LIMIT_THRESHOLD:
        penalty += -5.0 * (LIMIT_THRESHOLD - normalized_margin_upper)

joint_limit_penalty = penalty
```

**Characteristics**:
- **Type**: Sparse penalty (only near limits)
- **Range**: (-∞, 0] (cumulative across all joints)
- **Safety margin**: 10% of joint range

---

### 7. Collision Penalty

**Purpose**: Heavily penalize self-collisions (arm hitting base/itself).

**Weight**: `100.0` (catastrophic penalty!)

**Formula**:
```python
if self_collision_detected:
    collision_penalty = -100.0
else:
    collision_penalty = 0.0
```

**Characteristics**:
- **Type**: Sparse penalty (only on collision)
- **Range**: {-100, 0}
- **Terminal**: Often triggers episode reset

---

### 8. Base Orientation Penalty

**Purpose**: Keep mobile base upright (prevent tipping).

**Weight**: `2.0`

**Formula**:
```python
# Quaternion to roll/pitch
roll, pitch, yaw = quaternion_to_euler(base_orientation)

tilt = sqrt(roll² + pitch²)
base_orientation_penalty = -2.0 * tilt
```

**Characteristics**:
- **Type**: Dense penalty (every timestep)
- **Range**: (-∞, 0] (unbounded with tilt angle)
- **Target**: Keep roll and pitch near 0

---

### 9. Timeout Penalty

**Purpose**: Penalize episodes that timeout without completing trajectory.

**Weight**: `50.0`

**Formula**:
```python
if episode_timeout:
    timeout_penalty = -50.0
else:
    timeout_penalty = 0.0
```

**Characteristics**:
- **Type**: Terminal penalty
- **Range**: {-50, 0}
- **Applied once**: At episode end

---

## Mathematical Formulations

### Complete Reward Function (Code)

```python
def compute_rewards(
    self,
    ee_pos: torch.Tensor,
    target_pos: torch.Tensor,
    base_pos: torch.Tensor,
    base_pos_prev: torch.Tensor,
    actions: torch.Tensor,
    actions_prev: torch.Tensor,
    joint_positions: torch.Tensor,
    joint_limits_lower: torch.Tensor,
    joint_limits_upper: torch.Tensor,
    base_orientation: torch.Tensor,
    collision_flags: torch.Tensor,
    timeout_flags: torch.Tensor,
) -> torch.Tensor:
    """
    Compute total reward for mobile manipulator trajectory tracking.
    
    Args:
        ee_pos: [N, 3] End-effector positions (x, y, z)
        target_pos: [N, 3] Trajectory target positions
        base_pos: [N, 2] Current base positions (x, y)
        base_pos_prev: [N, 2] Previous base positions
        actions: [N, 9] Current actions
        actions_prev: [N, 9] Previous actions
        joint_positions: [N, 9] Current joint positions
        joint_limits_lower: [9] Joint lower limits
        joint_limits_upper: [9] Joint upper limits
        base_orientation: [N, 4] Base orientation quaternions
        collision_flags: [N] Boolean collision flags
        timeout_flags: [N] Boolean timeout flags
        
    Returns:
        rewards: [N] Total rewards per environment
    """
    # 1. Tracking reward (PRIMARY)
    tracking_error = torch.norm(ee_pos - target_pos, dim=-1)  # [N]
    tracking_reward = 50.0 * torch.exp(-5.0 * tracking_error)
    
    # 2. Base mobilization reward (CAPPED)
    base_movement = base_pos - base_pos_prev  # [N, 2]
    target_direction = target_pos[:, :2] - base_pos_prev  # [N, 2]
    
    # Progress: dot product normalized by movement magnitude
    progress = torch.sum(target_direction * base_movement, dim=-1)  # [N]
    movement_norm = torch.norm(base_movement, dim=-1)  # [N]
    progress = progress / (movement_norm + 1e-6)
    
    # Cap progress
    progress = torch.clamp(progress, max=0.2)
    
    # Only reward if target is far
    distance_to_target = torch.norm(target_pos[:, :2] - base_pos, dim=-1)
    far_target_mask = distance_to_target > 0.5
    base_mobilization_reward = 15.0 * progress * far_target_mask.float()
    
    # 3. Excessive base movement penalty (NEW)
    excess_movement = torch.clamp(movement_norm - 0.1, min=0.0)
    excessive_base_movement_penalty = -10.0 * excess_movement
    
    # 4. Target distance penalty
    target_distance_penalty = -5.0 * tracking_error
    
    # 5. Action smoothness penalty
    action_diff = torch.norm(actions - actions_prev, dim=-1)
    action_smoothness_penalty = -1.0 * action_diff
    
    # 6. Joint limit penalty
    joint_range = joint_limits_upper - joint_limits_lower  # [9]
    margin_lower = joint_positions - joint_limits_lower[None, :]  # [N, 9]
    margin_upper = joint_limits_upper[None, :] - joint_positions  # [N, 9]
    
    normalized_margin_lower = margin_lower / joint_range[None, :]
    normalized_margin_upper = margin_upper / joint_range[None, :]
    
    limit_penalty_lower = torch.clamp(0.1 - normalized_margin_lower, min=0.0)
    limit_penalty_upper = torch.clamp(0.1 - normalized_margin_upper, min=0.0)
    joint_limit_penalty = -5.0 * (limit_penalty_lower.sum(dim=-1) + 
                                   limit_penalty_upper.sum(dim=-1))
    
    # 7. Collision penalty
    collision_penalty = -100.0 * collision_flags.float()
    
    # 8. Base orientation penalty
    roll, pitch, _ = quaternion_to_euler(base_orientation)
    tilt = torch.sqrt(roll**2 + pitch**2)
    base_orientation_penalty = -2.0 * tilt
    
    # 9. Timeout penalty
    timeout_penalty = -50.0 * timeout_flags.float()
    
    # Total reward
    total_reward = (
        tracking_reward +
        base_mobilization_reward +
        excessive_base_movement_penalty +
        target_distance_penalty +
        action_smoothness_penalty +
        joint_limit_penalty +
        collision_penalty +
        base_orientation_penalty +
        timeout_penalty
    )
    
    return total_reward
```

---

## Tuning History & Rationale

### Session 1-4: Initial Development
- Basic tracking reward with distance penalty
- No base mobilization (arm-only tracking)
- **Result**: Arm could track, but base never moved

### Session 5: Base Mobilization Added (CATASTROPHIC FAILURE)

**Changes**:
- Added `base_mobilization_reward = 15.0 * progress` (NO CAP!)
- Target distance penalty: 3.0

**10M Steps - SUCCESS**:
- Tracking error: 0.877m (62% better than Session 4!)
- Base actions: Active and purposeful
- Broken envs: <5%

**20M Steps - DISASTER**:
- Tracking error: 2.242m (156% worse!)
- Broken envs: 63.5%
- Base moving 5-10 meters per step (physically impossible)
- Reward range: -21 to +10 (completely inverted!)

**Root Cause**:
- Unbounded `base_mobilization_reward` allowed exploitation
- Policy discovered: "Move base infinitely = infinite rewards"
- Reward hacking overwhelmed tracking objective

### Session 5b: Critical Fixes (CURRENT)

**Three Critical Changes**:

1. **Cap base mobilization progress**: 0.2m max
   - Prevents unbounded reward exploitation
   - Max reward: 15.0 × 0.2 = 3.0
   
2. **Add excessive base movement penalty**: -10.0 per meter excess
   - Punishes movements > 10cm/step
   - Works with cap to enforce physical realism
   
3. **Increase target distance penalty**: 3.0 → 5.0
   - Makes arm tracking relatively more important
   - Prevents base-centric policies

**Expected Results**:
- Base movements: 5-15cm per step (realistic)
- Tracking error: < 0.25m (good)
- Broken envs: < 5% (stable)
- Total reward: 35-45 (balanced)

---

## Session 5 Failure Analysis

### Reward Exploitation Mechanics

**The Exploit Loop**:
```
1. Policy sees: base_mobilization_reward = 15.0 * progress (unbounded)
2. Policy learns: "Move base far = huge reward"
3. Base moves 5-10m → reward = 15.0 × 10 = 150! (vs. tracking max 50)
4. Episode resets (base too far from valid workspace)
5. Repeat → Policy optimizes for reset spamming
6. Tracking objective forgotten (too small relative to exploitation)
```

**Quantitative Evidence**:
- **10M steps**: Base actions 0.3-0.8 (good), error 0.877m ✅
- **20M steps**: Base actions 5-10 (impossible!), error 2.242m ❌
- **Reward distribution**:
  - Session 4: [0, 45] (healthy)
  - Session 5 @ 10M: [0, 55] (slightly high from base)
  - Session 5 @ 20M: [-21, 10] (completely broken!)

**Why It Happened**:
1. **No physical constraints**: Simulation can't prevent impossible movements
2. **Unbounded reward component**: Created optimization loophole
3. **Weak competing objectives**: Tracking reward too small to resist
4. **No sanity checks**: No penalties for extreme movements

### Lessons Learned

1. **ALWAYS bound rewards**: Every component needs explicit min/max
2. **Physical plausibility matters**: Add hard constraints for impossible behaviors
3. **Monitor training closely**: 10M checkpoints saved us from wasting full 100M
4. **Test reward functions**: Simulate edge cases before training

---

## Session 5b Critical Fixes

### Fix #1: Cap Base Mobilization Progress

**Before**:
```python
progress = ...  # Could be 10+ meters!
base_mobilization_reward = 15.0 * progress
```

**After**:
```python
progress = min(progress, 0.2)  # Cap at 20cm
base_mobilization_reward = 15.0 * progress
# Max reward: 15.0 × 0.2 = 3.0
```

**Impact**:
- Removes exploitation path completely
- Base can still earn mobilization rewards (up to 3.0)
- Forces realistic movement magnitudes

### Fix #2: Add Excessive Movement Penalty

**New Component**:
```python
excess = max(0.0, base_movement - 0.1)
penalty = -10.0 * excess
```

**Impact**:
- Movements > 10cm get linearly penalized
- 5m movement → -49 penalty (catastrophic!)
- Works with cap to enforce physical realism

### Fix #3: Increase Target Distance Penalty

**Before**: `-3.0 * distance`  
**After**: `-5.0 * distance`

**Impact**:
- Makes tracking relatively more important
- Prevents base-centric policies
- 1m error now costs -5.0 instead of -3.0

### Combined Effect

**Reward Balance (Session 5b)**:
| Component | Range | Notes |
|-----------|-------|-------|
| Tracking | [0, 50] | Still largest (primary objective) |
| Base mobilization | [0, 3] | BOUNDED (was unbounded) |
| Excessive movement | (-∞, 0] | Punishes exploitation |
| Target distance | (-∞, 0] | Stronger pull to target |
| **Total (expected)** | [20, 45] | Balanced, bounded |

**vs. Session 5 @ 20M (broken)**:
| Component | Range | Notes |
|-----------|-------|-------|
| Tracking | [0, 10] | Collapsed (tracking forgotten) |
| Base mobilization | [0, 150] | UNBOUNDED EXPLOIT! |
| Target distance | (-∞, 0] | Overwhelmed by base reward |
| **Total (actual)** | [-21, 10] | INVERTED, BROKEN |

---

## Expected Behavior

### Healthy Training (Session 5b Target)

**Early Training (0-10M steps)**:
- Tracking error: 1.5m → 0.5m (improving)
- Base movements: 5-15cm per step (realistic)
- Broken envs: < 5% (stable)
- Reward components:
  - Tracking: 10-30 (learning)
  - Mobilization: 0-2 (moderate)
  - Distance: -2 to -5 (getting closer)
  - Total: 10-30 (positive, improving)

**Mid Training (10-50M steps)**:
- Tracking error: 0.5m → 0.25m (good)
- Base movements: 8-12cm per step (efficient)
- Broken envs: < 3% (very stable)
- Reward components:
  - Tracking: 30-40 (good)
  - Mobilization: 1-3 (active when needed)
  - Distance: -1 to -2 (close to target)
  - Total: 30-40 (strong performance)

**Late Training (50-100M steps)**:
- Tracking error: 0.15-0.25m (excellent)
- Base movements: 5-10cm per step (precise)
- Broken envs: < 2% (rock solid)
- Reward components:
  - Tracking: 38-45 (near-optimal)
  - Mobilization: 0.5-2 (minimal needed)
  - Distance: -0.5 to -1 (very close)
  - Total: 38-45 (optimal)

### Warning Signs (What to Watch For)

**Base Movement Issues**:
- Movements > 20cm consistently → Check excessive_base_movement_penalty
- Base not moving at all → Check mobilization reward activation
- Base circling/jittering → Action smoothness too weak

**Tracking Issues**:
- Error not decreasing → Tracking weight may be too low
- Oscillating around target → Smoothness penalty too high
- Arm stuck in singular config → Joint limit penalty too weak

**Catastrophic Failures**:
- Broken envs > 10% → Immediate investigation needed
- Reward going negative → Policy collapse imminent
- Extreme actions (> 5.0) → Physical constraints violated

---

## Future Improvements

### Potential Enhancements

1. **Adaptive Reward Scaling**
   - Curriculum learning: Increase difficulty over time
   - Dynamic weights based on performance

2. **Trajectory-Aware Rewards**
   - Velocity matching (not just position)
   - Look-ahead planning (anticipate future waypoints)

3. **Energy Efficiency**
   - Penalize joint torques/power consumption
   - Reward smooth, energy-efficient paths

4. **Success-Based Bonuses**
   - Large bonus for completing trajectory segment
   - Milestone rewards at key waypoints

5. **Auxiliary Tasks**
   - Orientation tracking (not just position)
   - Tool-frame constraints

### Known Limitations

1. **No velocity matching**: Only positions tracked, not velocities
2. **No orientation control**: End-effector orientation not considered
3. **Simplified physics**: No contact forces, friction, inertia
4. **Fixed trajectory**: No dynamic obstacle avoidance

---

## Appendix: Quick Reference

### Reward Weights Summary

| Component | Weight | Type | Range | Critical? |
|-----------|--------|------|-------|-----------|
| Tracking | 50.0 | Dense | [0, 50] | YES |
| Base mobilization | 15.0 | Sparse | [0, 3] | YES (capped!) |
| Excessive movement | 10.0 | Sparse | (-∞, 0] | YES (new!) |
| Target distance | 5.0 | Dense | (-∞, 0] | YES (increased!) |
| Action smoothness | 1.0 | Dense | (-∞, 0] | No |
| Joint limits | 5.0 | Sparse | (-∞, 0] | No |
| Collisions | 100.0 | Sparse | {-100, 0} | YES |
| Base orientation | 2.0 | Dense | (-∞, 0] | No |
| Timeout | 50.0 | Terminal | {-50, 0} | No |

### Critical Parameters

```python
# Tracking
TRACKING_WEIGHT = 50.0
TRACKING_DECAY = 5.0

# Base mobilization (SESSION 5B FIXES)
BASE_MOBILIZATION_WEIGHT = 15.0
MAX_PROGRESS_PER_STEP = 0.2  # meters (CRITICAL CAP!)
MOBILIZATION_ACTIVATION_DISTANCE = 0.5  # meters

# Excessive movement (NEW)
EXCESSIVE_MOVEMENT_WEIGHT = 10.0
MOVEMENT_THRESHOLD = 0.1  # meters (10cm)

# Target distance
TARGET_DISTANCE_WEIGHT = 5.0  # (increased from 3.0)

# Action smoothness
SMOOTHNESS_WEIGHT = 1.0

# Joint limits
JOINT_LIMIT_WEIGHT = 5.0
JOINT_LIMIT_MARGIN = 0.1  # 10% of range

# Collisions
COLLISION_WEIGHT = 100.0

# Base orientation
ORIENTATION_WEIGHT = 2.0

# Timeout
TIMEOUT_WEIGHT = 50.0
```

---

## Document History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | Oct 22, 2025 | Initial comprehensive documentation post-Session 5 failure | GitHub Copilot |

---

**Related Documents**:
- [SESSION_5B_FIX_SUMMARY.md](../training_sessions/SESSION_5B_FIX_SUMMARY.md) - Session 5 failure & Session 5b fixes
- [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md) - Model and training system design
- [TRAINING_SESSIONS_MASTER_LOG.md](../training_sessions/TRAINING_SESSIONS_MASTER_LOG.md) - Complete training history
- [BASE_MOVEMENT_BUG_ANALYSIS.md](../05_bug_fixes/BASE_MOVEMENT_BUG_ANALYSIS.md) - Base mobility debugging
- [PPR_CONTROL_ARCHITECTURE.md](../02_architecture/PPR_CONTROL_ARCHITECTURE.md) - Base control flow

