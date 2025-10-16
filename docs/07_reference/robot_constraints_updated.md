# Robot Physical Constraints & Self-Collision System

## Overview 🤖

Updated reward system with **realistic robot constraints** and **critical self-collision detection** for safe mobile manipulator training.

---

## Physical Constraints Implemented ✅

### 1. Mobile Base Limits (Differential Drive)

```python
@dataclass
class RobotLimits:
    # Linear motion
    max_linear_velocity: float = 1.5  # m/s
    max_linear_acceleration: float = 5.0  # m/s²
    max_linear_jerk: float = 5.0  # m/s³
    
    # Angular motion (yaw only)
    max_angular_velocity: float = 2.0  # rad/s
```

**Key constraint**: **NO LATERAL MOTION** - Differential drive robots cannot move sideways!

### 2. Arm Joint Limits

```python
    # Motor constraints
    max_joint_velocity: float = 2.0  # rad/s (motor speed limit)
    max_joint_acceleration: float = 10.0  # rad/s²
    
    # Position constraints
    enforce_joint_limits: bool = True
    joint_limit_margin: float = 0.1  # radians (safety margin)
```

Position limits are **automatically extracted** from your USD file!

### 3. Control Frequency

```python
    decimation: int = 20  # Control @ 10Hz
    # Physics runs at 200Hz, control at 200/20 = 10Hz
```

**Matches trajectory timing**: 100ms waypoint spacing = 10Hz control

---

## Self-Collision Detection 🚨

### Why Critical?

Mobile manipulators have **high risk** of self-collision:
- Arm hitting mobile base
- Arm links colliding with each other
- Gripper hitting robot body

**Unlike external obstacles**, self-collision **MUST BE PREVENTED**!

### Implementation

#### Contact Sensors Enabled

```python
# env.py - _create_scene_config()
robot_cfg = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=robot_usd_path,
        activate_contact_sensors=True,  # ← ENABLED for self-collision
    ),
)
```

#### Self-Collision Penalty Function

```python
def self_collision_penalty(
    net_contact_forces: torch.Tensor,  # [num_envs, num_bodies, 3]
    threshold: float = 1.0,  # Newtons
    scale: float = 1.0,
    continuous: bool = True,
) -> torch.Tensor:
    """Penalty for robot hitting itself."""
    
    # Compute contact force magnitude for each body
    contact_force_mag = torch.norm(net_contact_forces, dim=-1)
    
    if continuous:
        # Soft penalty: scales with force magnitude
        violation = clamp(contact_force_mag - threshold, min=0.0)
        return scale * sum(violation)
    else:
        # Hard penalty: binary trigger
        has_collision = any(contact_force_mag > threshold)
        return scale * has_collision
```

**Two modes**:
1. **Continuous** (default): Penalty increases smoothly with contact force
   - Gentler learning signal
   - Allows agent to learn "how close is too close"

2. **Binary**: Fixed large penalty for any contact
   - Harsher constraint
   - Clearer boundary but harder to learn from

#### Configuration

```python
@dataclass
class RewardWeights:
    # Self-collision (CRITICAL!)
    self_collision_penalty: float = 50.0  # High weight!
    self_collision_threshold: float = 1.0  # Newtons
    self_collision_continuous: bool = True  # Smooth penalty
```

**Weight = 50.0** - Much higher than other penalties to strongly discourage self-collision!

#### Termination on Severe Self-Collision

```python
@dataclass
class MobileMMTrackConfig:
    # Termination
    terminate_on_self_collision: bool = True
    self_collision_termination_threshold: float = 10.0  # Newtons
```

**Two-tier system**:
- **Penalty threshold (1.0 N)**: Start penalizing light contact
- **Termination threshold (10.0 N)**: End episode on hard collision

This allows agent to learn from "close calls" without immediately terminating.

---

## Constraint Penalties Summary 📊

### Updated Reward Components

| Component | Weight | Purpose |
|-----------|--------|---------|
| **Position tracking** | 10.0 | Primary objective |
| **Orientation tracking** | 2.0 | Secondary objective |
| **Progress bonus** | 1.0 | Improvement incentive |
| **Action magnitude** | 0.01 | Energy efficiency |
| **Action rate** | 0.01 | Smoothness (1st derivative) |
| **Action smoothness** | 0.05 | Jerk penalty (2nd derivative) |
| **Velocity limit** | 5.0 | Exceed 1.5 m/s or 2 rad/s |
| **Acceleration limit** | 5.0 | Exceed 5 m/s² |
| **Jerk limit** | 3.0 | Exceed 5 m/s³ |
| **Joint limit** | 10.0 | Too close to joint limits |
| **Lateral motion** | 2.0 | Sideways movement (impossible for diff drive) |
| **🚨 Self-collision** | **50.0** | **Robot hitting itself** |
| **Stability** | 0.1 | Excessive base motion |

---

## How Penalties Work 🔧

### 1. Velocity Limit Penalty

```python
def velocity_limit_penalty(
    base_lin_vel,  # [num_envs, 3]
    joint_vel,  # [num_envs, 6]
    max_linear_vel=1.5,  # m/s
    max_joint_vel=2.0,  # rad/s
):
    # Base forward velocity (x-direction)
    base_violation = clamp(|base_vel_x| - 1.5, min=0.0)² 
    
    # Joint velocities
    joint_violation = sum(clamp(|joint_vel| - 2.0, min=0.0)²)
    
    return scale * (base_violation + joint_violation)
```

**Behavior**:
- Velocity < limit: No penalty
- Velocity = limit: Starting to penalize
- Velocity > limit: Quadratic penalty (strongly discouraged)

### 2. Acceleration Limit Penalty

```python
def acceleration_limit_penalty(
    current_vel,
    prev_vel,
    dt=0.1,  # 10Hz control
    max_accel=5.0,  # m/s²
):
    accel = (current_vel - prev_vel) / dt
    violation = clamp(|accel| - max_accel, min=0.0)
    return scale * violation²
```

**Computed from velocity differences** - no direct acceleration measurement needed!

### 3. Jerk Limit Penalty

```python
def jerk_penalty(
    current_accel,
    prev_accel,
    dt=0.1,
    max_jerk=5.0,  # m/s³
):
    jerk = (current_accel - prev_accel) / dt
    violation = clamp(|jerk| - max_jerk, min=0.0)
    return scale * violation²
```

**Prevents sudden acceleration changes** - ensures smooth motion.

### 4. Joint Limit Penalty

```python
def joint_limit_penalty(
    joint_pos,  # [num_envs, 6]
    joint_lower,  # [6] from USD
    joint_upper,  # [6] from USD
    margin=0.1,  # radians
):
    # Distance from lower limit
    lower_violation = clamp(margin - (joint_pos - joint_lower), min=0.0)²
    
    # Distance from upper limit
    upper_violation = clamp(margin - (joint_upper - joint_pos), min=0.0)²
    
    return scale * sum(lower_violation + upper_violation)
```

**Soft boundary**: Starts penalizing 0.1 radians before hitting hard limits.

### 5. Lateral Motion Penalty

```python
def lateral_motion_penalty(
    base_lin_vel,  # [num_envs, 3]
):
    # Y-direction velocity (sideways)
    lateral_vel = base_lin_vel[:, 1]
    return scale * lateral_vel²
```

**Physical impossibility**: Differential drive cannot move sideways. Any y-velocity indicates:
- Simulation instability
- Unrealistic motion
- Potential collision/sliding

---

## Termination Conditions 🛑

### 1. Severe Self-Collision

```python
if terminate_on_self_collision:
    max_contact_force = max(contact_force_magnitudes)
    if max_contact_force > 10.0:  # Newtons
        TERMINATE_EPISODE()
```

**Why terminate?**
- Hard collision indicates catastrophic failure
- No point continuing from unsafe state
- Forces agent to learn avoidance early

### 2. Excessive Tracking Error

```python
if terminate_on_tracking_error:
    tracking_error = ||ee_pos - target_pos||
    if tracking_error > 2.0:  # meters
        TERMINATE_EPISODE()
```

**Prevents wandering**: If robot loses target by >2m, reset.

---

## Reward History Tracking 📈

The environment tracks state history for derivative calculations:

```python
# Stored in env.__init__
self.prev_actions  # t-1
self.prev_prev_actions  # t-2
self.prev_base_lin_vel  # For acceleration
self.prev_joint_vel  # For joint acceleration
self.prev_base_accel  # For jerk

# Updated in _get_rewards()
base_accel = (base_lin_vel - prev_base_lin_vel) / dt
jerk = (base_accel - prev_base_accel) / dt

# Reset on episode reset
def _reset_idx(env_ids):
    self.prev_actions[env_ids] = 0.0
    self.prev_prev_actions[env_ids] = 0.0
    self.prev_base_lin_vel[env_ids] = 0.0
    # ... etc
```

---

## Expected Training Behavior 🎯

### Phase 1: Learning Tracking (First 100K steps)

Agent will:
- ✅ Focus on reducing position/orientation error
- ⚠️ Violate velocity/acceleration limits frequently
- ⚠️ Experience occasional self-collisions
- ⚠️ Show jerky motion

**Reward**: ~5-10 per step (penalties dominate)

### Phase 2: Constraint Satisfaction (100K-500K steps)

Agent learns:
- ✅ Respect velocity limits (1.5 m/s, 2 rad/s)
- ✅ Avoid self-collision (weight 50.0 teaches this fast!)
- ✅ Smoother acceleration profiles
- ⚠️ Still occasional jerk violations

**Reward**: ~8-12 per step (improving)

### Phase 3: Smooth Tracking (500K-2M steps)

Agent achieves:
- ✅ Accurate tracking (<0.2m error)
- ✅ Zero self-collisions
- ✅ All constraints satisfied
- ✅ Smooth, natural motion

**Reward**: ~12-15 per step (near optimal)

---

## Monitoring During Training 📊

### Key Metrics to Watch

```python
# TensorBoard / Weights & Biases
extras["reward_components"] = {
    "self_collision_penalty": -X,  # Should decrease to ~0
    "velocity_limit_penalty": -X,  # Should decrease rapidly
    "acceleration_limit_penalty": -X,
    "jerk_penalty": -X,
    "joint_limit_penalty": -X,  # Should be near 0 if limits set correctly
    "lateral_motion_penalty": -X,  # Should be near 0 (diff drive)
}

# Episode terminations
extras["termination_reasons"] = {
    "self_collision": count,  # Should decrease to 0
    "tracking_error": count,
    "timeout": count,  # Should increase (completing episodes)
}
```

### Warning Signs 🚩

1. **Self-collision penalty not decreasing**: 
   - Weight too low? Increase to 100.0
   - Termination threshold too high?

2. **Constant velocity violations**:
   - Limits too restrictive for task?
   - Need curriculum learning (start relaxed, tighten)

3. **High lateral motion penalty**:
   - Simulation issue
   - Check physics stability

---

## Tuning Guide 🎛️

### If Agent Can't Learn

**Relax constraints temporarily**:
```python
max_linear_velocity: float = 2.0  # Was 1.5
max_linear_acceleration: float = 10.0  # Was 5.0
velocity_limit_penalty: float = 1.0  # Was 5.0
```

Then gradually tighten during training.

### If Self-Collisions Persist

**Increase penalty**:
```python
self_collision_penalty: float = 100.0  # Was 50.0
terminate_on_self_collision: bool = True  # Ensure enabled
self_collision_termination_threshold: float = 5.0  # Was 10.0 (stricter)
```

### If Motion is Too Jerky

**Increase smoothness penalties**:
```python
action_smoothness: float = 0.1  # Was 0.05
jerk_limit_penalty: float = 5.0  # Was 3.0
```

---

## Implementation Files 📁

| File | Changes |
|------|---------|
| `config.py` | Added `RobotLimits`, updated `RewardWeights` |
| `rewards.py` | Added 6 new penalty functions + `self_collision_penalty` |
| `env.py` | Contact sensors enabled, history tracking, updated reward computation |

---

## Summary ✅

**Physical Constraints Added**:
- ✅ Linear velocity: 1.5 m/s
- ✅ Angular velocity: 2.0 rad/s
- ✅ Acceleration: 5.0 m/s²
- ✅ Jerk: 5.0 m/s³
- ✅ Joint velocity: 2.0 rad/s
- ✅ Joint limits: Read from USD + 0.1 rad margin
- ✅ No lateral motion (differential drive)

**Self-Collision System**:
- ✅ Contact sensors enabled
- ✅ Continuous penalty function (weight 50.0)
- ✅ Termination on severe collision (>10 N)
- ✅ Two-tier threshold (1 N penalty, 10 N terminate)

**Control Frequency**:
- ✅ 10 Hz control (matches 100ms trajectory waypoints)
- ✅ 200 Hz physics simulation

Your robot will now learn **safe, smooth, physically-realistic motion**! 🚀
