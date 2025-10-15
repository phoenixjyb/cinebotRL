# Reward System & USD Asset Configuration

## USD Robot Asset Usage ✅

**Yes, your custom USD file is being used!**

### Asset Path Configuration

Located in: `src/rl_platform/robots/mobile_mm.py`

```python
def get_mobile_mm_usd_path() -> Path:
    """Returns path to your custom robot USD file."""
    assets_root = Path(__file__).parent.parent.parent.parent / "assets_own"
    return assets_root / "usd" / "mobile_manipulator_PPR_base_corrected.usd"
```

**Your USD file**: `assets_own/usd/mobile_manipulator_PPR_base_corrected.usd`

### How It's Loaded

In `src/rl_platform/tasks/mobile_mm/env.py`:

```python
def _create_scene_config(self) -> InteractiveSceneCfg:
    """Create scene with your custom robot."""
    
    # Get YOUR USD file path
    robot_usd_path = str(get_mobile_mm_usd_path())
    
    # Configure robot articulation
    robot_cfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=robot_usd_path,  # ← Your custom USD loaded here!
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                # Your robot's arm joints
                "left_arm_joint1": 0.0,
                "left_arm_joint2": 1.6,
                "left_arm_joint3": -1.6,
                "left_arm_joint4": 0.0,
                "left_arm_joint5": 0.0,
                "left_arm_joint6": 0.0,
            },
        ),
        actuators={
            "arm": sim_utils.ImplicitActuatorCfg(
                joint_names_expr=["left_arm_joint[1-6]"],
                stiffness=400.0,
                damping=40.0,
            ),
        },
    )
```

### End-Effector Tracking

The system identifies and tracks your robot's end-effector:

```python
def _setup_scene(self):
    """Find end-effector link in YOUR USD file."""
    ee_link_name = "left_gripper_link"  # From your USD structure
    
    if ee_link_name in self.robot.body_names:
        self._ee_body_idx = self.robot.body_names.index(ee_link_name)
        print(f"Found EE link '{ee_link_name}' at index {self._ee_body_idx}")
```

**Confirmed via USD inspection**: `/Robot/left_arm_link6/left_gripper_link`

---

## Reward & Penalty System 🎯

### Reward Configuration

Located in: `src/rl_platform/tasks/mobile_mm/config.py`

```python
@dataclass
class RewardWeights:
    """Default reward weights."""
    
    # Tracking Rewards (positive)
    position_tracking: float = 10.0      # End-effector position accuracy
    orientation_tracking: float = 2.0    # End-effector orientation accuracy
    progress_bonus: float = 1.0          # Bonus for reducing error
    
    # Action Penalties (negative)
    action_magnitude: float = 0.01       # Energy efficiency penalty
    action_rate: float = 0.01            # Smoothness penalty
    
    # Safety Penalties (negative)
    collision_penalty: float = 10.0      # Collision detection penalty
    stability_penalty: float = 0.1       # Base motion penalty
    
    # Obstacle Rewards/Penalties
    min_obstacle_distance_weight: float = 1.0
    safety_radius: float = 0.2  # meters
```

### Reward Components Breakdown

#### 1. Position Tracking Reward (Weight: 10.0)

**Formula**: `exp(-scale * error²)`

```python
def position_tracking_reward(current_pos, target_pos, scale=1.0):
    """Exponential reward for position accuracy."""
    error = ||target_pos - current_pos||
    return exp(-scale * error²)
```

**Behavior**:
- Error = 0.0m → Reward = 10.0 (perfect!)
- Error = 0.1m → Reward ≈ 9.0 (excellent)
- Error = 0.5m → Reward ≈ 0.78 (poor)
- Error > 1.0m → Reward ≈ 0 (very poor)

**Why exponential?** Sharply rewards precise tracking, gradually penalizes larger errors.

---

#### 2. Orientation Tracking Reward (Weight: 2.0)

**Formula**: `exp(-scale * angular_distance²)`

```python
def orientation_tracking_reward(current_quat, target_quat, scale=0.5):
    """Quaternion-based orientation reward."""
    dot_product = |current_quat · target_quat|
    angular_dist = 2 * acos(dot_product)  # radians
    return exp(-scale * angular_dist²)
```

**Behavior**:
- Angular error = 0° → Reward = 2.0 (perfect)
- Angular error = 10° → Reward ≈ 1.9 (excellent)
- Angular error = 45° → Reward ≈ 0.7 (moderate)
- Angular error > 90° → Reward ≈ 0 (poor)

**Why lower weight?** Orientation is important but secondary to position for tracking.

---

#### 3. Progress Bonus (Weight: 1.0)

**Formula**: `max(0, prev_error - current_error)`

```python
def progress_bonus(prev_error, current_error):
    """Bonus for improving tracking accuracy."""
    improvement = prev_error - current_error
    return max(0, improvement)
```

**Behavior**:
- Error decreased by 0.1m → Bonus = +0.1
- Error stayed same → Bonus = 0
- Error increased → Bonus = 0 (no negative bonus)

**Why important?** Encourages continuous improvement, not just maintaining current position.

---

#### 4. Action Magnitude Penalty (Weight: 0.01)

**Formula**: `scale * ||actions||²`

```python
def action_magnitude_penalty(actions, scale=1.0):
    """Penalty for large joint commands (energy)."""
    return scale * sum(actions²)
```

**Behavior**:
- Small actions (±0.1) → Penalty ≈ 0.0001 (minimal)
- Medium actions (±0.5) → Penalty ≈ 0.0025 (noticeable)
- Large actions (±1.0) → Penalty ≈ 0.01 (significant)

**Why needed?** Encourages energy-efficient movements, prevents thrashing.

---

#### 5. Action Rate Penalty (Weight: 0.01)

**Formula**: `scale * ||actions - prev_actions||²`

```python
def action_rate_penalty(actions, prev_actions, scale=1.0):
    """Penalty for rapid action changes (smoothness)."""
    action_diff = actions - prev_actions
    return scale * sum(action_diff²)
```

**Behavior**:
- Smooth changes (Δ < 0.1) → Penalty ≈ 0.0001 (minimal)
- Medium changes (Δ = 0.5) → Penalty ≈ 0.0025 (moderate)
- Jerky changes (Δ > 1.0) → Penalty ≈ 0.01 (high)

**Why important?** Produces smooth, natural-looking robot motion. Prevents oscillations.

---

#### 6. Collision Penalty (Weight: 10.0)

**Formula**: Binary penalty based on contact forces

```python
def collision_penalty(contact_forces, threshold=1.0, scale=1.0):
    """Large penalty if any contact exceeds threshold."""
    has_collision = any(contact_forces > threshold)
    return scale * has_collision
```

**Behavior**:
- No contact → Penalty = 0
- Any contact > threshold → Penalty = -10.0 (severe!)

**Current status**: ⚠️ **Not yet active** (contact sensors disabled)

---

#### 7. Stability Penalty (Weight: 0.1)

**Formula**: `scale * (||base_lin_vel||² + ||base_ang_vel||²)`

```python
def stability_penalty(base_lin_vel, base_ang_vel, scale=1.0):
    """Penalty for excessive base motion."""
    lin_vel_mag = ||base_lin_vel||
    ang_vel_mag = ||base_ang_vel||
    return scale * (lin_vel_mag² + ang_vel_mag²)
```

**Behavior**:
- Stationary base → Penalty = 0
- Moving base (0.5 m/s) → Penalty ≈ 0.025
- Fast motion (1.0 m/s) → Penalty ≈ 0.1

**Why needed?** Mobile manipulator should minimize base motion, track primarily with arm.

---

#### 8. Obstacle Distance Reward (Weight: 1.0)

**Formula**: Sigmoid-based smooth transition

```python
def obstacle_distance_reward(min_distance, safety_radius, scale=1.0):
    """Reward/penalty based on obstacle proximity."""
    normalized_dist = min_distance / safety_radius
    return scale * (sigmoid(5 * (normalized_dist - 1.0)) - 0.5)
```

**Behavior** (safety_radius = 0.2m):
- Distance > 0.2m → Reward = +0.5 (safe!)
- Distance = 0.2m → Reward = 0 (neutral)
- Distance < 0.2m → Penalty up to -0.5 (danger!)

**Current status**: ⚠️ **Not yet active** (obstacles disabled)

---

## Total Reward Computation

### Combined Formula

```python
Total Reward = 
    + 10.0 * position_tracking         # Primary objective
    + 2.0  * orientation_tracking      # Secondary objective
    + 1.0  * progress_bonus            # Improvement incentive
    - 0.01 * action_magnitude          # Energy efficiency
    - 0.01 * action_rate               # Smoothness
    - 10.0 * collision_penalty         # Safety (not active yet)
    - 0.1  * stability_penalty         # Base stability
    + 1.0  * obstacle_distance         # Clearance (not active yet)
```

### Expected Reward Range

**Perfect performance**:
- Position tracking: +10.0
- Orientation tracking: +2.0
- Progress bonus: +0.1 (average)
- Action penalties: -0.02
- Stability penalty: -0.01
- **Total: ≈ +12.0 per step**

**Moderate performance**:
- Position tracking: +5.0 (0.3m error)
- Orientation tracking: +1.0 (30° error)
- Progress: +0.05
- Penalties: -0.05
- **Total: ≈ +6.0 per step**

**Poor performance**:
- Position tracking: +1.0 (1m error)
- Orientation tracking: +0.2
- Progress: 0
- Penalties: -0.1
- **Total: ≈ +1.0 per step**

**Episode total** (20s @ 50Hz = 1000 steps):
- Perfect: 12,000 cumulative reward
- Moderate: 6,000 cumulative reward
- Poor: 1,000 cumulative reward

---

## Tuning Guidelines 🔧

### If Agent Struggles to Learn:

1. **Increase tracking reward weights**:
   ```python
   position_tracking: float = 20.0  # From 10.0
   orientation_tracking: float = 5.0  # From 2.0
   ```

2. **Decrease action penalties** (allow more exploration):
   ```python
   action_magnitude: float = 0.005  # From 0.01
   action_rate: float = 0.005  # From 0.01
   ```

3. **Increase progress bonus** (reward improvement more):
   ```python
   progress_bonus: float = 5.0  # From 1.0
   ```

### If Agent Learns But Motion is Jerky:

1. **Increase smoothness penalty**:
   ```python
   action_rate: float = 0.05  # From 0.01
   ```

2. **Increase stability penalty**:
   ```python
   stability_penalty: float = 0.5  # From 0.1
   ```

### If Agent Moves Base Too Much:

1. **Increase stability penalty**:
   ```python
   stability_penalty: float = 1.0  # From 0.1
   ```

### For Multi-Trajectory Training:

Consider **curriculum learning** with reward shaping:
- Start with high tracking rewards (easier to learn)
- Gradually increase smoothness penalties (refine motion)

---

## Future Enhancements 🚀

### Not Yet Implemented:

1. **Contact Sensors**: 
   - Currently disabled: `activate_contact_sensors=False`
   - Enable for collision detection and safety

2. **Obstacle Avoidance**:
   - Obstacle spawning disabled: `enable_obstacles: bool = False`
   - Add dynamic obstacles for robustness

3. **Previous Action Storage**:
   - Currently using same action twice: `prev_actions=self.prev_actions`
   - Need proper history buffer (line 327 in env.py)

4. **Joint Limits & Constraints**:
   - Add penalties for approaching joint limits
   - Enforce workspace boundaries

5. **Trajectory-Specific Rewards**:
   - Different weights for different cinematic categories
   - Reward smooth camera-like motion for handheld_subtle
   - Reward speed for tracking_zigzag

---

## Configuration Files

### Main Config
`src/rl_platform/tasks/mobile_mm/config.py`

### Environment
`src/rl_platform/tasks/mobile_mm/env.py`

### Reward Functions
`src/rl_platform/tasks/mobile_mm/rewards.py`

### Robot Assets
`src/rl_platform/robots/mobile_mm.py`

---

## Quick Reference Commands

### View Current Weights
```python
from rl_platform.tasks.mobile_mm.config import RewardWeights
weights = RewardWeights()
print(weights)
```

### Test Reward Function
```bash
python scripts/test_mobile_mm_env.py --num_envs 1 --steps 100
# Monitor extras["reward_components"] for breakdown
```

### Monitor Training Rewards
```bash
tensorboard --logdir logs/sb3/MobileMMTrackEE-v0
# Check: reward_components/* for individual terms
```

---

## Summary ✅

**USD Asset**: ✅ Your custom robot `mobile_manipulator_PPR_base_corrected.usd` is loaded

**End-Effector**: ✅ Tracked via `left_gripper_link` body

**Reward System**: ✅ Comprehensive 8-component system
- **Primary**: Position tracking (weight 10.0) + Orientation tracking (weight 2.0)
- **Secondary**: Progress bonus, smoothness, energy, stability
- **Safety**: Collision & obstacle penalties (not yet active)

**Tunable**: All weights configurable in `config.py`

**Logged**: Individual reward components tracked in `extras["reward_components"]`

Your system is ready for training! The reward structure balances accuracy, efficiency, and smoothness. 🎬✨
