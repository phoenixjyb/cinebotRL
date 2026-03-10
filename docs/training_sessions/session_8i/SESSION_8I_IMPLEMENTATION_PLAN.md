# Session 8i Implementation Plan

**Date**: November 6, 2025  
**Goal**: Fix orientation tracking while maintaining 8h@40M position accuracy (237.3 cm)  
**Strategy**: Distance-gated orientation rewards + curriculum redesign + observation enhancement  

---

## 🎯 Core Objectives

### Primary Goal
- **Maintain Position**: Keep position error ~237-250 cm (match 8h@40M baseline)
- **Improve Orientation**: Target 80-100° (from current 135°, 40% improvement)
- **Early Stopping**: Monitor and stop at optimal checkpoint (avoid 8h@100M regression)

### Key Insight from Session 8h
- **40M checkpoint best** (237.3 cm position, 135.1° orientation)
- **100M regressed** (302.4 cm position, 119.1° orientation)
- **Curriculum transition caused harm** despite gradual approach
- **Solution**: Distance-gated rewards + trajectory curriculum + enhanced observations

---

## 📋 Implementation Phases

### Phase 1: Observation Space Enhancement ✅ PRIORITY
**Goal**: Add orientation-awareness features (+6 dims)

**New Observations**:
1. **Axis-angle error** (+3 dims): Rotation axis × angle magnitude
   - Gives "shortest rotation path" to target orientation
   - More intuitive than quaternion for learning

2. **EE angular velocity** (+3 dims): ω_x, ω_y, ω_z
   - Already available but not added to observations yet
   - Enables damping rewards and smoothness penalties

**Total New Dims**: +6 (current 70 → 76 dims)

**Files to Modify**:
- `src/rl_platform/tasks/mobile_mm/observations.py`
- `src/rl_platform/tasks/mobile_mm/config.py` (observation dimensions)

---

### Phase 2: Distance-Gated Reward System ✅ PRIORITY
**Goal**: Separate position-reaching from orientation-alignment

**Core Logic**:
```python
distance_to_target = norm(ee_pos - target_pos)
distance_threshold = 0.7  # meters

if distance < threshold:  # COMFORT ZONE - focus on orientation
    position_weight = 4.0       # Maintain baseline
    orientation_weight = 30.0   # High attention (from 8h stage-2)
    orientation_progress_bonus = 2.0  # NEW: reward angle reduction
    angular_velocity_penalty = 1.0    # NEW: smoothness
else:  # FAR ZONE - focus on reaching
    position_weight = 4.0       
    orientation_weight = 4.0    # Low (defer orientation)
    orientation_progress_bonus = 0.0  # Disabled
    angular_velocity_penalty = 0.0    # Disabled
```

**New Reward Components**:
1. **Orientation Progress Bonus**:
   ```python
   def orientation_progress_bonus(prev_ori_error, current_ori_error):
       improvement = prev_ori_error - current_ori_error
       return torch.clamp(improvement, min=0.0)
   ```

2. **Angular Velocity Penalty** (smoothness):
   ```python
   def angular_velocity_penalty(ee_ang_vel, joint_vel):
       # Penalize excessive angular velocity
       ee_ang_speed = torch.norm(ee_ang_vel, dim=-1)
       joint_vel_penalty = torch.norm(joint_vel[:, 3:9], dim=-1)  # Arm only
       return ee_ang_speed + 0.5 * joint_vel_penalty
   ```

**Files to Modify**:
- `src/rl_platform/tasks/mobile_mm/rewards.py` (new functions)
- `src/rl_platform/tasks/mobile_mm/config.py` (new weights)
- `src/rl_platform/tasks/mobile_mm/env.py` (distance-gated logic)

---

### Phase 3: Trajectory Curriculum ⚠️ MEDIUM PRIORITY
**Goal**: Progressive orientation difficulty

**Trajectory Phases**:
```
Phase 1: "Easy Orientation" (0-40M steps)
- Criteria: max_ori_change < 30°/s, distance < 1.5m
- Focus: Pure orientation alignment in comfort zone
- File: trajectoryToLearn/orientation_easy.txt

Phase 2: "Medium Orientation" (40-80M steps)  
- Criteria: max_ori_change < 60°/s, distance < 3.0m
- Focus: Mixed position + orientation
- File: trajectoryToLearn/orientation_medium.txt

Phase 3: "Full Complexity" (80M+ steps)
- Criteria: All trajectories
- Focus: Complete task distribution
- File: trajectoryToLearn/all_trajectories.txt
```

**Implementation**:
- Create trajectory filters
- Add curriculum loading logic in env.py
- Switch trajectories at specified step thresholds

**Files to Create**:
- `trajectoryToLearn/orientation_easy.txt`
- `trajectoryToLearn/orientation_medium.txt`
- `data/trajectory_filters/filter_by_orientation.py` (helper script)

**Files to Modify**:
- `src/rl_platform/tasks/mobile_mm/env.py` (trajectory loading)
- `src/rl_platform/tasks/mobile_mm/config.py` (curriculum steps)

---

### Phase 4: Training Script & Auto-Stop 🔄 MEDIUM PRIORITY
**Goal**: Safe training with early stopping

**Auto-Stop Criteria**:
```python
# Every 5M steps
if orientation_error > previous_best * 1.15:  # 15% degradation
    stop_and_save("orientation_degraded")
elif position_error > 250:  # Absolute threshold
    stop_and_save("position_regressed")
elif training_steps >= 40_000_000:  # First milestone
    evaluate_and_decide()
```

**Evaluation Schedule**:
- Evaluate every 10M steps (10M, 20M, 30M, 40M, ...)
- Save checkpoints every 2M (keep 1M intervals after cleanup)
- Compare with 8h@40M baseline

**Files to Create**:
- `scripts/launch_session_8i.ps1` (training launcher)
- `scripts/reinforcement_learning/sb3/callbacks/orientation_monitor.py` (callback)

---

## 🔧 Detailed Implementation Steps

### Step 1: Observation Enhancement (30 min)

#### A. Add Axis-Angle Error Computation
**File**: `src/rl_platform/tasks/mobile_mm/observations.py`

**Add new function after `quat_diff`**:
```python
def quat_to_axis_angle(quat_diff: torch.Tensor) -> torch.Tensor:
    """Convert relative quaternion to axis-angle representation.
    
    Args:
        quat_diff: Relative quaternion [num_envs, 4] (w, x, y, z)
        
    Returns:
        axis_angle: Rotation axis * angle magnitude [num_envs, 3]
                   Direction = rotation axis, Magnitude = angle in radians
    """
    # Extract components
    w, x, y, z = quat_diff[:, 0], quat_diff[:, 1], quat_diff[:, 2], quat_diff[:, 3]
    
    # Angle (always positive)
    angle = 2 * torch.acos(torch.clamp(w.abs(), 0.0, 1.0))  # [num_envs]
    
    # Axis (normalized)
    axis_norm = torch.sqrt(x**2 + y**2 + z**2 + 1e-8)  # Avoid division by zero
    axis_x = x / axis_norm
    axis_y = y / axis_norm
    axis_z = z / axis_norm
    
    # Axis-angle: axis * angle (encodes both direction and magnitude)
    axis_angle = torch.stack([
        axis_x * angle,
        axis_y * angle,
        axis_z * angle
    ], dim=-1)  # [num_envs, 3]
    
    return axis_angle
```

#### B. Modify `compose_observation` Function
**Location**: After line 70 (`components.extend([pos_error, quat_error])`)

**Add**:
```python
# NEW (Session 8i): Axis-angle error for better orientation learning
# Gives "shortest rotation path" - more intuitive than quaternion
axis_angle_error = quat_to_axis_angle(quat_error)  # [num_envs, 3]
components.append(axis_angle_error)

# NEW (Session 8i): End-effector angular velocity (already computed, now added to obs)
# Enables smoothness rewards and damping penalties
components.append(ee_ang_vel)  # [num_envs, 3]
```

#### C. Update Dimension Calculation
**File**: `src/rl_platform/tasks/mobile_mm/observations.py`
**Function**: `get_observation_dimensions` (line ~220)

**Modify**:
```python
# Tracking error: pos_error(3) + quat_error(4) + axis_angle_error(3) + ee_ang_vel(3) = 13
dim += 13  # Was 7, now 13
```

---

### Step 2: Distance-Gated Rewards (45 min)

#### A. Add New Reward Functions
**File**: `src/rl_platform/tasks/mobile_mm/rewards.py`
**Location**: After `progress_bonus` function (~line 80)

```python
def orientation_progress_bonus(
    prev_ori_error: torch.Tensor,
    current_ori_error: torch.Tensor,
) -> torch.Tensor:
    """Bonus for reducing orientation error (like position progress_bonus).
    
    Args:
        prev_ori_error: Previous orientation error in radians [num_envs]
        current_ori_error: Current orientation error in radians [num_envs]
        
    Returns:
        Bonus values [num_envs]
    """
    improvement = prev_ori_error - current_ori_error
    return torch.clamp(improvement, min=0.0)


def angular_velocity_penalty(
    ee_ang_vel: torch.Tensor,
    joint_vel: torch.Tensor,
    scale_ee: float = 1.0,
    scale_joint: float = 0.5,
) -> torch.Tensor:
    """Penalty for excessive angular velocity (smoothness).
    
    Penalizes both end-effector angular velocity and arm joint velocities
    to encourage smooth orientation changes.
    
    Args:
        ee_ang_vel: End-effector angular velocity [num_envs, 3]
        joint_vel: All joint velocities [num_envs, 9] (only uses arm [3:9])
        scale_ee: Weight for EE angular velocity
        scale_joint: Weight for joint velocities
        
    Returns:
        Penalty values [num_envs]
    """
    # EE angular speed (magnitude)
    ee_ang_speed = torch.norm(ee_ang_vel, dim=-1)
    
    # Arm joint velocity magnitude (extract arm joints [3:9])
    arm_joint_vel = joint_vel[:, 3:9]  # Only arm, not base
    joint_speed = torch.norm(arm_joint_vel, dim=-1)
    
    return scale_ee * ee_ang_speed + scale_joint * joint_speed
```

#### B. Add Distance Gating to `compute_combined_reward`
**File**: `src/rl_platform/tasks/mobile_mm/rewards.py`
**Function**: `compute_combined_reward` (~line 900)

**Find the section**:
```python
# Tracking rewards
pos_reward = weights["position_tracking"] * position_tracking_reward(
    current_ee_pos, target_pos, scale=1.0
)

ori_reward = weights["orientation_tracking"] * orientation_tracking_reward(
    current_ee_quat, target_quat, scale=0.5
)
```

**Replace with**:
```python
# ========================================
# SESSION 8i: DISTANCE-GATED ORIENTATION REWARDS
# ========================================
# When far from target: focus on reaching (low orientation weight)
# When close to target: focus on alignment (high orientation weight)
current_error = torch.norm(target_pos - current_ee_pos, dim=-1)  # [num_envs]
distance_threshold = weights.get("distance_gate_threshold", 0.7)  # meters

# Compute distance gate mask
in_comfort_zone = current_error < distance_threshold  # [num_envs] boolean

# Position tracking (always same weight)
pos_reward = weights["position_tracking"] * position_tracking_reward(
    current_ee_pos, target_pos, scale=1.0
)

# Orientation tracking (GATED by distance)
# Far: use low weight, Close: use high weight
ori_weight_far = weights.get("orientation_tracking_far", 4.0)
ori_weight_close = weights.get("orientation_tracking_close", 30.0)

# Blend weights based on distance
ori_weight_current = torch.where(
    in_comfort_zone,
    torch.full_like(current_error, ori_weight_close),
    torch.full_like(current_error, ori_weight_far)
)  # [num_envs]

ori_reward_base = orientation_tracking_reward(
    current_ee_quat, target_quat, scale=0.5
)  # [num_envs]
ori_reward = ori_weight_current * ori_reward_base  # Element-wise multiply

# Orientation progress bonus (ONLY in comfort zone)
if weights.get("orientation_progress_bonus", 0.0) > 0:
    ori_progress = orientation_progress_bonus(
        prev_ori_error, current_ori_error
    )  # [num_envs]
    
    # Only apply in comfort zone
    ori_progress_masked = torch.where(
        in_comfort_zone,
        ori_progress,
        torch.zeros_like(ori_progress)
    )
    ori_progress_reward = weights["orientation_progress_bonus"] * ori_progress_masked
else:
    ori_progress_reward = torch.zeros_like(current_error)

# Angular velocity penalty (ONLY in comfort zone - for smoothness)
if weights.get("angular_velocity_penalty", 0.0) > 0:
    ang_vel_pen = angular_velocity_penalty(
        ee_ang_vel, joint_vel, scale_ee=1.0, scale_joint=0.5
    )  # [num_envs]
    
    # Only apply in comfort zone
    ang_vel_pen_masked = torch.where(
        in_comfort_zone,
        ang_vel_pen,
        torch.zeros_like(ang_vel_pen)
    )
    ang_vel_penalty = -weights["angular_velocity_penalty"] * ang_vel_pen_masked
else:
    ang_vel_penalty = torch.zeros_like(current_error)
```

#### C. Add to reward_components dictionary
**Same function, near the end** (~line 1050):
```python
reward_components = {
    # ... existing components ...
    "orientation_progress_bonus": ori_progress_reward.mean().item(),
    "angular_velocity_penalty": ang_vel_penalty.mean().item(),
}
```

---

### Step 3: Configuration Updates (15 min)

#### A. Add New Weights
**File**: `src/rl_platform/tasks/mobile_mm/config.py`
**Location**: In `RewardWeights` class, after orientation_tracking (~line 105)

```python
# ========================================
# SESSION 8i: DISTANCE-GATED ORIENTATION
# ========================================
distance_gate_threshold: float = 0.7  # Distance threshold for gating (meters)
orientation_tracking_far: float = 4.0  # Low weight when far from target
orientation_tracking_close: float = 30.0  # High weight when close to target
orientation_progress_bonus: float = 2.0  # Reward for orientation improvement
angular_velocity_penalty: float = 1.0  # Smoothness penalty for orientation changes
```

#### B. Update Observation Dimensions
**File**: `src/rl_platform/tasks/mobile_mm/config.py`
**Find**: Observation space size calculation

**Update comment**:
```python
# Session 8i: 76 dims = 70 (session 8h) + 6 (axis_angle_error + ee_ang_vel)
#   - Axis-angle error: +3 (rotation axis * angle)
#   - EE angular velocity: +3 (already computed, now in obs)
```

---

### Step 4: Environment Modifications (30 min)

#### A. Store Previous Orientation Error
**File**: `src/rl_platform/tasks/mobile_mm/env.py`
**Function**: `_get_rewards` (~line 1700)

**Find**:
```python
self.ee_ori_error_buf = 2 * torch.acos(dot_product)  # [num_envs] - angular error in radians
```

**Add after**:
```python
# SESSION 8i: Store previous orientation error for progress bonus
if not hasattr(self, 'prev_ee_ori_error'):
    self.prev_ee_ori_error = self.ee_ori_error_buf.clone()
```

#### B. Pass to Reward Function
**Same function, find compute_combined_reward call** (~line 1800):

**Add parameter**:
```python
prev_ori_error=self.prev_ee_ori_error,  # NEW
current_ori_error=self.ee_ori_error_buf,  # NEW
```

#### C. Update Previous Error
**Same function, at the end** (~line 1900):

**Add**:
```python
self.prev_ee_ori_error = self.ee_ori_error_buf.clone()  # Update for next step
```

---

## 📊 Expected Outcomes

### Optimistic (Best Case) 🎉
```
Session 8i @ 40M:
- Position: 230-240 cm (match or beat 8h@40M)
- Orientation: 80-100° (40% improvement from 135°)
- Episode Reward: -600 to -700 (improvement from -784)
Status: DEPLOY READY
```

### Realistic (Likely) ✅
```
Session 8i @ 40M:
- Position: 240-250 cm (slight regression acceptable)
- Orientation: 100-110° (25% improvement)
- Episode Reward: -750 to -850
Status: GOOD PROGRESS, continue to 80M
```

### Pessimistic (Needs Debug) ⚠️
```
Session 8i @ 40M:
- Position: 260-280 cm (significant regression)
- Orientation: 130-140° (no improvement)
- Episode Reward: <-1000
Status: STOP, debug reward balance
```

---

## 🚀 Training Plan

### Configuration
```yaml
Session: 8i
Total Steps: 120M (3 × 40M cycles)
Learning Rate: 2e-4 (keep from 8h)
Num Envs: 16384
Evaluation Frequency: 10M steps
Checkpoint Frequency: 2M steps (keep 1M after cleanup)
```

### Timeline
```
Phase 1: 0-40M steps (~3-4 hours)
- Trajectory: orientation_easy.txt
- Expected: Position 240-250cm, Orientation 110-120°
- Decision: If orientation < 115°, continue. Else, debug.

Phase 2: 40-80M steps (~3-4 hours)
- Trajectory: orientation_medium.txt  
- Expected: Position 230-240cm, Orientation 90-100°
- Decision: If orientation < 100°, continue. Else, stop.

Phase 3: 80-120M steps (~3-4 hours)
- Trajectory: all_trajectories.txt
- Expected: Position 220-230cm, Orientation 70-90°
- Decision: Evaluate deployment readiness
```

### Evaluation Checkpoints
- **10M, 20M, 30M, 40M**: Quick evaluation (16 envs, 50 episodes)
- **40M**: Full evaluation (64 envs, 200 episodes) - MILESTONE
- **60M, 80M, 100M, 120M**: Full evaluations

---

## 📝 Implementation Checklist

### Phase 1: Observation Enhancement
- [ ] Add `quat_to_axis_angle` function to observations.py
- [ ] Modify `compose_observation` to include axis-angle + ee_ang_vel
- [ ] Update `get_observation_dimensions` to reflect +6 dims
- [ ] Test observation space: Run env and print obs.shape (should be 76)

### Phase 2: Distance-Gated Rewards
- [ ] Add `orientation_progress_bonus` function to rewards.py
- [ ] Add `angular_velocity_penalty` function to rewards.py
- [ ] Modify `compute_combined_reward` with distance gating logic
- [ ] Add new components to reward_components dict
- [ ] Test reward computation: Run env and check reward components

### Phase 3: Configuration
- [ ] Add distance gating weights to RewardWeights class
- [ ] Update observation dimension comments
- [ ] Verify all new parameters are present
- [ ] Test config loading: Import and print config

### Phase 4: Environment Logic
- [ ] Store prev_ee_ori_error in env._get_rewards
- [ ] Pass prev/current ori error to compute_combined_reward
- [ ] Update prev_ee_ori_error at end of _get_rewards
- [ ] Test env reset and step: Check all buffers exist

### Phase 5: Trajectory Curriculum (OPTIONAL)
- [ ] Create orientation_easy.txt filter
- [ ] Create orientation_medium.txt filter
- [ ] Add curriculum logic to env.py
- [ ] Test trajectory loading at different steps

### Phase 6: Launch Script
- [ ] Create scripts/launch_session_8i.ps1
- [ ] Add orientation monitoring callback
- [ ] Test dry-run launch
- [ ] Verify all paths and configs

---

## 🎯 Success Criteria

### Minimum Requirements (Deploy-Ready)
- Position error: < 250 cm (16.7% above target, but acceptable)
- Orientation error: < 100° (66.7% above target, but 26% improvement)
- No training divergence or auto-pause triggers
- Stable performance across evaluations

### Stretch Goals (Exceptional)
- Position error: < 230 cm (match 8h@40M)
- Orientation error: < 80° (33.3% above target, 41% improvement)
- Smooth training without regression
- Deploy-ready by 40-60M steps

---

## 📚 References

- **Baseline**: Session 8h @ 40M (237.3 cm, 135.1°)
- **Playbook**: `ref_codes/mobile_mm_training_playbook.md`
- **Previous Sessions**: `docs/training_sessions/session_8h/`
- **Evaluation Plots**: `evaluation_plots/session_8h_README.md`

---

## 🔄 Next Steps

1. **Implement Phase 1** (Observations) - START HERE
2. **Implement Phase 2** (Rewards) - Core functionality
3. **Implement Phase 3-4** (Config + Env) - Integration
4. **Test locally** (single env, verify observations/rewards)
5. **Launch training** (full scale)
6. **Monitor and evaluate** (every 10M steps)

Let's build this! 🚀
