# Model Architecture & Training System

**Version:** 1.0  
**Last Updated:** 2025-01-08  
**Related Documents:** [REWARD_SYSTEM_DESIGN.md](REWARD_SYSTEM_DESIGN.md), [TRAINING_SESSIONS_MASTER_LOG.md](../training_sessions/TRAINING_SESSIONS_MASTER_LOG.md)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Observation Space (46 dimensions)](#2-observation-space-46-dimensions)
3. [Action Space (8 dimensions)](#3-action-space-8-dimensions)
4. [PPO Network Architecture](#4-ppo-network-architecture)
5. [Training Hyperparameters](#5-training-hyperparameters)
6. [Hardware & System Requirements](#6-hardware--system-requirements)
7. [IsaacLab → Stable-Baselines3 Integration](#7-isaaclab--stable-baselines3-integration)
8. [Connection to Reward System](#8-connection-to-reward-system)
9. [Code References](#9-code-references)

---

## 1. Executive Summary

### 1.1 System Overview

The CinebotRL training system is designed to teach a **mobile manipulator** (PPR base + 6-DOF arm) to perform **precise end-effector trajectory tracking** using **PPO** (Proximal Policy Optimization). The system combines:

- **IsaacLab/Isaac Sim** (NVIDIA Omniverse): High-fidelity physics simulation at 200 Hz
- **Stable-Baselines3**: PPO implementation with custom callbacks
- **8,192 parallel environments**: Massive parallelization for sample efficiency
- **46-dimensional observations**: Complete robot state + trajectory errors + lookahead
- **8-dimensional continuous actions**: Joint velocities (6 arm + 2 base)
- **~235K trainable parameters**: Enhanced 3-layer MLP for both actor and critic

### 1.2 Key Design Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **Algorithm** | PPO with GAE(λ=0.95) | Stable on-policy RL, proven for robotics |
| **Observation** | 46 dims (state + error + lookahead) | Balance between information richness and sample efficiency |
| **Action Space** | Continuous [-1, 1]^8 | Direct velocity control, smooth trajectories |
| **Network Size** | 3-layer [256, 256, 128] | 16× larger than default; handles complex state |
| **Physics Rate** | 200 Hz (dt=0.005s) | High-fidelity contacts, stable simulation |
| **Parallel Envs** | 8,192 | Maximize GPU utilization (RTX 3090) |
| **Rollout Length** | 128 steps | ~0.64s episodes, good GAE estimation |

### 1.3 Training Performance (Session 5b Baseline)

- **Total Steps:** 100M+ timesteps
- **Wall-Clock Time:** ~18 hours (RTX 3090, 8192 envs)
- **Throughput:** ~1,500 steps/sec (~12M env interactions/sec)
- **Final Reward:** ~0.85 (normalized, see [REWARD_SYSTEM_DESIGN.md](REWARD_SYSTEM_DESIGN.md))
- **GPU Memory:** ~18GB/24GB (leaves headroom for visualization)

### 1.4 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        IsaacLab Simulation                          │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Physics: PhysX @ 200 Hz (dt=0.005s)                         │  │
│  │  Rendering: RTX raytracing (optional)                        │  │
│  │  Environments: 8,192 parallel instances                      │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                             ↓                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  MobileMMTrackEEEnv (DirectRLEnv)                            │  │
│  │  - Observations: 46-dim Box (normalized)                     │  │
│  │  - Actions: 8-dim Box [-1, 1]                                │  │
│  │  - Rewards: 9 components (position, orientation, velocity...) │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│              IsaacLabToSB3VecEnvWrapper                             │
│  - Converts dict observations → numpy arrays                        │
│  - Handles old/new Gymnasium reset/step API                         │
│  - Updates observation_space after first reset                      │
└─────────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                  Stable-Baselines3 PPO                              │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Actor Network (Policy π):                                   │  │
│  │    Input(46) → Linear(256) → ReLU → Linear(256) → ReLU →    │  │
│  │    → Linear(128) → ReLU → Linear(8) → Tanh                   │  │
│  │    Parameters: ~118K                                         │  │
│  └──────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Critic Network (Value V):                                   │  │
│  │    Input(46) → Linear(256) → ReLU → Linear(256) → ReLU →    │  │
│  │    → Linear(128) → ReLU → Linear(1)                          │  │
│  │    Parameters: ~117K                                         │  │
│  └──────────────────────────────────────────────────────────────┘  │
│  - Total Parameters: ~235K (16× default)                            │
│  - Orthogonal initialization (better RL convergence)                │
│  - Action distribution: Diagonal Gaussian (log_std_init=-1.0)      │
└─────────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      Custom Callbacks                               │
│  - EntropyDecayCallback: Exponential decay (τ=10M steps)            │
│  - AdaptiveKLCallback: Dynamic target_kl adjustment                 │
│  - CheckpointCallback: Save every 1M steps                          │
│  - TensorBoard logging: Scalars, histograms, gradients             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Observation Space (46 dimensions)

### 2.1 Overview

The observation space is a **continuous Box space** of 46 dimensions (expandable to 70+ with optional features). All observations are normalized to facilitate learning.

```python
observation_space = gym.spaces.Box(
    low=-np.inf,
    high=np.inf,
    shape=(46,),
    dtype=np.float32
)
```

**Design Philosophy:**
- **Markovian state**: Include velocities + errors to satisfy MDP assumptions
- **Task-relevant**: Position errors, lookahead, base-to-target signals
- **Normalized**: Scale to [-1, 1] or [0, 1] ranges where appropriate

### 2.2 Observation Breakdown (Default: 46 dims)

| Component | Dims | Range | Description |
|-----------|------|-------|-------------|
| **Base State** | **13** | | |
| `base_pos` | 3 | Real (m) | Base position in world frame (x, y, z) |
| `base_quat` | 4 | Unit quat | Base orientation (w, x, y, z) |
| `base_lin_vel` | 3 | [0, 1] | Normalized linear velocity (÷ 1.5 m/s) |
| `base_ang_vel` | 3 | [0, 1] | Normalized angular velocity (÷ 2.0 rad/s) |
| **Arm Joint State** | **12** | | |
| `arm_joint_pos` | 6 | Real (rad) | Arm joint positions (joints 3-8) |
| `arm_joint_vel` | 6 | Real (rad/s) | Arm joint velocities (joints 3-8) |
| **End-Effector State** | **13** | | |
| `ee_pos` | 3 | Real (m) | EE position in world frame |
| `ee_quat` | 4 | Unit quat | EE orientation (w, x, y, z) |
| `ee_lin_vel` | 3 | Real (m/s) | EE linear velocity |
| `ee_ang_vel` | 3 | Real (rad/s) | EE angular velocity |
| **Tracking Error** | **7** | | |
| `pos_error` | 3 | Real (m) | `target_pos - ee_pos` |
| `quat_error` | 4 | Unit quat | Relative quaternion (ee → target) |
| **Base-to-Target** | **4** | | **CRITICAL for base motion!** |
| `dx`, `dy` | 2 | Real (m) | Horizontal distance (base → target) |
| `distance` | 1 | Real (m) | Euclidean distance (base → target) |
| `out_of_reach` | 1 | {0, 1} | Binary flag: 1 if dist > 0.6m (arm reach) |
| **Optional: Lookahead** | **+9** | | (if `use_lookahead=True`) |
| `lookahead_pos` | 9 | Real (m) | Next 3 target positions (3 steps × 3) |
| **Optional: Action History** | **+16** | | (if `use_action_history=True`) |
| `action_history` | 16 | [-1, 1] | Last 2 actions (2 × 8 dims) |
| **Total (default)** | **46** | | |
| **Total (with optionals)** | **71** | | |

### 2.3 Key Design Decisions

#### 2.3.1 Why Exclude Base PPR Joints from Observations?

The robot has **9 total DOF**:
- **Joints 0-2:** PPR base (prismatic X, prismatic Y, revolute Z) — **NOT observed**
- **Joints 3-8:** 6-DOF arm — **observed**

**Rationale:**
- Base PPR joints are **virtual actuators** used internally by PhysX for commanding movement
- The **actual base state** is captured by `base_pos`, `base_quat`, `base_lin_vel`, `base_ang_vel`
- Observing PPR joints would be redundant and confuse the policy

**Code Reference:**
```python
# src/rl_platform/tasks/mobile_mm/observations.py, lines 61-63
arm_joint_pos = joint_pos[:, 3:9]  # Only arm joints
arm_joint_vel = joint_vel[:, 3:9]  # Only arm joints
components.extend([arm_joint_pos, arm_joint_vel])
```

#### 2.3.2 Base-to-Target Information (CRITICAL)

The **4-dimensional base-to-target signal** was added in Session 5b to fix the "lazy base" problem:

```python
# src/rl_platform/tasks/mobile_mm/observations.py, lines 77-84
base_to_target_xy = target_pos[:, :2] - base_pos[:, :2]  # [num_envs, 2]
base_to_target_dist = torch.norm(base_to_target_xy, dim=-1, keepdim=True)
arm_reach = 0.6  # meters (empirical)
out_of_reach = (base_to_target_dist > arm_reach).float()  # Binary flag
components.extend([base_to_target_xy, base_to_target_dist, out_of_reach])
```

**Impact:**
- **Before (Session 5a):** Policy relied only on `pos_error` → base rarely moved
- **After (Session 5b):** Explicit `out_of_reach` flag → base actively repositions

See [Session 5b Fix Summary](../training_sessions/SESSION_5B_FIX_SUMMARY.md) for details.

#### 2.3.3 Velocity Normalization

Base velocities are **normalized** to [0, 1] to match action space semantics:

```python
# src/rl_platform/tasks/mobile_mm/env.py, lines 909-910
base_lin_vel_obs = base_lin_vel / 1.5  # max_linear_velocity = 1.5 m/s
base_ang_vel_obs = base_ang_vel / 2.0  # max_angular_velocity = 2.0 rad/s
```

**Rationale:**
- Actions are in [-1, 1] → observations should be similarly scaled
- Avoids large magnitude discrepancies between state components
- Improves neural network conditioning

### 2.4 Observation Composition (Code Walkthrough)

The full observation is assembled in `compose_observation()`:

```python
# src/rl_platform/tasks/mobile_mm/observations.py, lines 8-119

def compose_observation(...) -> torch.Tensor:
    components = []
    
    # 1. Base state (13 dims)
    components.extend([base_pos, base_quat, base_lin_vel, base_ang_vel])
    
    # 2. Arm joint state (12 dims: 6 pos + 6 vel)
    arm_joint_pos = joint_pos[:, 3:9]
    arm_joint_vel = joint_vel[:, 3:9]
    components.extend([arm_joint_pos, arm_joint_vel])
    
    # 3. End-effector state (13 dims)
    components.extend([ee_pos, ee_quat, ee_lin_vel, ee_ang_vel])
    
    # 4. Tracking error (7 dims)
    pos_error = target_pos - ee_pos
    quat_error = quat_diff(ee_quat, target_quat)
    components.extend([pos_error, quat_error])
    
    # 5. Base-to-target (4 dims) - CRITICAL for base motion!
    base_to_target_xy = target_pos[:, :2] - base_pos[:, :2]
    base_to_target_dist = torch.norm(base_to_target_xy, dim=-1, keepdim=True)
    out_of_reach = (base_to_target_dist > 0.6).float()
    components.extend([base_to_target_xy, base_to_target_dist, out_of_reach])
    
    # 6. Optional: Lookahead (+9 dims if enabled)
    if lookahead_pos is not None:
        lookahead_flat = lookahead_pos.view(batch_size, -1)  # [N, 3, 3] -> [N, 9]
        components.append(lookahead_flat)
    
    # 7. Optional: Action history (+16 dims if enabled)
    if action_history is not None:
        history_flat = action_history.view(batch_size, -1)  # [N, 2, 8] -> [N, 16]
        components.append(history_flat)
    
    # Concatenate and return
    observations = torch.cat(components, dim=-1)  # [num_envs, 46 or 71]
    return observations
```

### 2.5 Observation Validation

Observation dimensions are computed at initialization:

```python
# src/rl_platform/tasks/mobile_mm/env.py, lines 84-92
self.num_observations = get_observation_dimensions(
    num_joints=6,  # Only arm joints
    use_lookahead=self.task_cfg.use_lookahead,
    lookahead_steps=self.task_cfg.lookahead_steps,
    use_action_history=self.task_cfg.use_action_history,
    action_history_length=self.task_cfg.action_history_length,
    action_dim=self.num_actions,
)
print(f"  - Observation dim: {self.cfg.num_observations}")  # Logs: 46 or 71
```

---

## 3. Action Space (8 dimensions)

### 3.1 Overview

The action space is a **continuous Box space** of 8 dimensions, representing **joint-level velocity commands** for the mobile manipulator.

```python
action_space = gym.spaces.Box(
    low=-1.0,
    high=1.0,
    shape=(8,),
    dtype=np.float32
)
```

**Design Philosophy:**
- **Normalized actions**: All actions in [-1, 1] for algorithm stability
- **Velocity control**: Arm joint positions, base velocities
- **Safety margins**: Joint limits respect 5% buffer to avoid hard stops

### 3.2 Action Breakdown

| Index | DOF | Type | Physical Meaning | Scaling | Physical Range |
|-------|-----|------|------------------|---------|----------------|
| 0 | Arm Joint 1 | Position | Shoulder rotation | Joint limits + 5% margin | ~[-π, π] rad |
| 1 | Arm Joint 2 | Position | Shoulder pitch | Joint limits + 5% margin | ~[-π/2, π/2] rad |
| 2 | Arm Joint 3 | Position | Elbow pitch | Joint limits + 5% margin | ~[-π, 0] rad |
| 3 | Arm Joint 4 | Position | Wrist roll | Joint limits + 5% margin | ~[-π, π] rad |
| 4 | Arm Joint 5 | Position | Wrist pitch | Joint limits + 5% margin | ~[-π/2, π/2] rad |
| 5 | Arm Joint 6 | Position | Wrist yaw | Joint limits + 5% margin | ~[-π, π] rad |
| 6 | Base v_x | Velocity | Forward/backward | × 1.5 m/s | [-1.5, 1.5] m/s |
| 7 | Base ω_z | Velocity | Angular rotation | × 2.0 rad/s | [-2.0, 2.0] rad/s |

**Note:** The differential-drive base **cannot move sideways** (v_y = 0 always). This is a physical constraint of the mobile platform.

### 3.3 Action Scaling & Application

Actions are processed in `_pre_physics_step()` before each physics simulation:

#### 3.3.1 Arm Joint Actions (Indices 0-5)

**Position Control** with safety margins:

```python
# src/rl_platform/tasks/mobile_mm/env.py, lines 713-717
arm_actions = actions[:, :6]  # Extract arm commands [-1, 1]
arm_actions_scaled = self._scale_actions_to_joint_limits(arm_actions)
self.robot.set_joint_position_target(arm_actions_scaled, joint_ids=self._arm_joint_ids)
```

**Scaling Function:**

```python
# src/rl_platform/tasks/mobile_mm/env.py, lines 821-852
def _scale_actions_to_joint_limits(self, actions: torch.Tensor) -> torch.Tensor:
    """Scale normalized actions from [-1, 1] to actual joint limits with safety margins."""
    lower = self.joint_lower_limits  # From URDF
    upper = self.joint_upper_limits
    
    # Add 5% safety margin from each limit
    range_size = upper - lower
    safety_margin = 0.05 * range_size
    lower_safe = lower + safety_margin
    upper_safe = upper - safety_margin
    
    # Scale from [-1, 1] to [lower_safe, upper_safe]
    actions_normalized = (actions + 1.0) * 0.5  # [-1, 1] → [0, 1]
    scaled_actions = actions_normalized * (upper_safe - lower_safe) + lower_safe
    
    return scaled_actions
```

**Example:**
- Joint limit: [-π, π] rad
- Safety margin: 0.05 × 2π = 0.314 rad
- Safe range: [-2.827, 2.827] rad
- Action = -0.5 → Scaled = -1.414 rad

#### 3.3.2 Base Velocity Actions (Indices 6-7)

**Velocity Control** with acceleration limiting:

```python
# src/rl_platform/tasks/mobile_mm/env.py, lines 714-716, 765-773
base_vx = actions[:, 6:7]     # Forward/backward [-1, 1]
base_wz = actions[:, 7:8]     # Angular rotation [-1, 1]

# Scale to physical limits
base_vx_desired = base_vx * 1.5   # m/s
base_wz_desired = base_wz * 2.0   # rad/s

# Rate limiting for smooth motion (respects acceleration limits)
dt = 0.005 * decimation  # Typically 0.02s (4 substeps)
max_vel_delta_linear = 1.0 * dt   # Max acceleration: 1.0 m/s²
max_vel_delta_angular = 5.0 * dt  # Max angular accel: 5.0 rad/s²

# Clamp velocity change per timestep
base_vx_clamped = torch.clamp(base_vx_desired - prev_vx, -max_vel_delta_linear, max_vel_delta_linear) + prev_vx
base_wz_clamped = torch.clamp(base_wz_desired - prev_wz, -max_vel_delta_angular, max_vel_delta_angular) + prev_wz
```

**Why Velocity Control for Base?**
- **Arm:** Position control → Precise end-effector placement
- **Base:** Velocity control → Smooth mobile navigation
- Mixing control modes is standard in mobile manipulation

### 3.4 Action History Buffer

Actions are stored for **2 timesteps** (if `use_action_history=True`) to enable:
- **Jerk penalty**: Detect sudden changes (3rd derivative)
- **Smoothness reward**: Encourage gradual motion
- **Policy consistency**: Agent learns temporal dependencies

```python
# src/rl_platform/tasks/mobile_mm/env.py, lines 350-353, 710-712
self.action_history = torch.zeros(
    self.num_envs, 2, self.cfg.num_actions, device=self.device
)

# Update history each step (roll + insert new action)
self.action_history = torch.roll(self.action_history, shifts=-1, dims=1)
self.action_history[:, -1, :] = actions  # Store raw [-1, 1] actions
```

### 3.5 Differential Drive Constraints

The mobile base uses **differential drive kinematics**:

```
         ┌─────────┐
         │  Robot  │
         │    ↑    │  Forward direction
         │    │    │
         └─────────┘
         Left  Right
         Wheel Wheel

v_left = v_x - (wheelbase/2) * ω_z
v_right = v_x + (wheelbase/2) * ω_z
```

**Constraints:**
- **No lateral motion:** v_y = 0 (can't move sideways)
- **Holonomic planning:** Base must reorient to move in desired direction
- **Wheelbase:** ~0.4m (typical for differential drive platforms)

This is why the action space is **8D** (not 9D):
- 6 arm joints + v_x + ω_z = 8 DOF
- ~~v_y~~ is not controllable (physical constraint)

### 3.6 Action Space Configuration

```python
# src/rl_platform/tasks/mobile_mm/env.py, lines 72, 105-111
num_actions: int = 8  # 6 arm joints + 2 base DOF (v_x, omega_z)

self.action_space = gym.spaces.Box(
    low=-1.0,
    high=1.0,
    shape=(self.num_actions,),
    dtype=np.float32
)
```

### 3.7 Action Clipping & Safety

Multiple safety mechanisms ensure actions don't violate physical constraints:

1. **Tanh activation:** Policy network outputs are passed through `tanh` → guaranteed [-1, 1]
2. **Joint limit margins:** 5% buffer prevents hard stops
3. **Acceleration limits:** Base velocities are rate-limited to respect max acceleration
4. **Collision detection:** Episode terminates if robot collides with obstacles

**Code References:**
- Action scaling: `_scale_actions_to_joint_limits()` (lines 821-852)
- Base rate limiting: `_pre_physics_step()` (lines 765-773)
- Safety checks: `_check_termination()` (checks joint limits, collisions)

---

## 4. PPO Network Architecture

### 4.1 Overview

The policy is implemented using **PPO** (Proximal Policy Optimization) with an **enhanced 3-layer MLP** architecture. The network is **16× larger** than Stable-Baselines3's default to handle the complex 46-dimensional observation space.

**Total Parameters:** ~235,000 (~118K actor + ~117K critic)

### 4.2 Actor Network (Policy π)

The **actor** (policy network) maps observations to action distributions:

```
Input: observations [batch, 46]
   ↓
Linear(46 → 256) + ReLU
   ↓
Linear(256 → 256) + ReLU
   ↓
Linear(256 → 128) + ReLU
   ↓
Linear(128 → 8)  [mean μ]
   ↓
Tanh  (squash to [-1, 1])
   ↓
Output: action_mean [batch, 8]

Separate learnable: log_std [8]  (diagonal Gaussian)
```

**Layer Sizes:**
- Layer 1: 46 × 256 + 256 = **11,776** params
- Layer 2: 256 × 256 + 256 = **65,792** params
- Layer 3: 256 × 128 + 128 = **32,896** params
- Output: 128 × 8 + 8 = **1,032** params
- **Shared log_std:** 8 params (diagonal covariance)
- **Total Actor:** ~**111,504** params

**Code:**
```python
# scripts/reinforcement_learning/sb3/train.py, lines 828-836
policy_kwargs = dict(
    net_arch=dict(
        pi=[256, 256, 128],  # Actor: 3-layer network
        vf=[256, 256, 128]   # Critic: 3-layer network
    ),
    activation_fn=torch.nn.ReLU,
    ortho_init=True,         # Orthogonal weight initialization
    log_std_init=-1.0,       # Initial log(std) = -1.0 → std ≈ 0.37
)
```

### 4.3 Critic Network (Value V)

The **critic** (value network) estimates state value for advantage calculation:

```
Input: observations [batch, 46]
   ↓
Linear(46 → 256) + ReLU
   ↓
Linear(256 → 256) + ReLU
   ↓
Linear(256 → 128) + ReLU
   ↓
Linear(128 → 1)
   ↓
Output: value [batch, 1]
```

**Layer Sizes:**
- Layer 1: 46 × 256 + 256 = **11,776** params
- Layer 2: 256 × 256 + 256 = **65,792** params
- Layer 3: 256 × 128 + 128 = **32,896** params
- Output: 128 × 1 + 1 = **129** params
- **Total Critic:** ~**110,593** params

### 4.4 Action Distribution

PPO uses a **diagonal Gaussian distribution** for continuous action spaces:

```python
π(a | s) = N(μ(s), σ²I)

where:
  μ(s) = policy_network(s)  [8-dimensional mean]
  σ = exp(log_std)           [8-dimensional std, learned]
  I = identity matrix        [diagonal covariance]
```

**Initial Standard Deviation:**
```python
log_std_init = -1.0
σ_init = exp(-1.0) ≈ 0.368
```

**Why σ = 0.368?**
- **High exploration early:** Actions sample from N(μ, 0.368²) → ~68% within ±0.37 of mean
- **Entropy decay:** σ decreases over time via entropy coefficient scheduling
- **Prevents collapse:** Ensures policy doesn't become deterministic too early

**Action Sampling:**
```python
# Training: sample from distribution
action = μ + σ * ε,  where ε ~ N(0, 1)

# Inference: use mean (deterministic)
action = μ
```

### 4.5 Network Initialization

**Orthogonal Initialization** (`ortho_init=True`) is used for all layers:

```python
# For each weight matrix W:
W = orthogonal_matrix * gain

where:
  gain = sqrt(2)  for ReLU layers (He initialization)
  gain = 0.01     for output layer (small initial actions)
```

**Benefits for RL:**
- **Preserves gradient norms:** Prevents vanishing/exploding gradients
- **Faster convergence:** Empirically shown to improve PPO performance
- **Better exploration:** Initial policy is more diverse

**Reference:**
> "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks" (Saxe et al., 2013)

### 4.6 Network Architecture Comparison

| Version | Actor Layers | Critic Layers | Total Params | Performance |
|---------|--------------|---------------|--------------|-------------|
| **SB3 Default** | [64, 64] | [64, 64] | ~14K | Poor (struggles with 46D obs) |
| **CinebotRL** | [256, 256, 128] | [256, 256, 128] | ~235K | Good (Session 5b: ~0.85 reward) |

**Why Larger Network?**
- **46-dimensional observations:** Need capacity to extract features
- **Continuous 8D actions:** Requires rich representations for smooth control
- **Trajectory tracking:** Temporal correlations demand deeper networks

**Training Time Impact:**
- **Forward pass:** ~1.5× slower (but GPU parallelism masks overhead)
- **Backward pass:** ~2× slower (more gradients to compute)
- **Overall:** ~10% wall-clock time increase (dominated by physics simulation)

### 4.7 Training Stability Features

1. **Gradient Clipping:**
   ```python
   max_grad_norm = 0.5  # Clip gradients to prevent explosive updates
   ```

2. **Value Function Clipping:**
   ```python
   clip_range_vf = 1.0  # Prevent value function from diverging
   ```

3. **Entropy Regularization:**
   ```python
   ent_coef = 0.01  # Initial entropy coefficient
   # Decays exponentially: ent_coef(t) = 0.01 * exp(-t / 10M)
   ```

4. **KL Divergence Constraint:**
   ```python
   target_kl = None  # Disabled by default (can enable for conservative updates)
   ```

**Code Reference:**
```python
# scripts/reinforcement_learning/sb3/train.py, lines 846-864
model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    learning_rate=3e-4,
    n_steps=128,
    batch_size=512,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    clip_range_vf=1.0,      # Value function clipping
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,      # Gradient clipping
    target_kl=None,
    tensorboard_log=args.log_dir,
    device=device,
)
```

---

## 5. Training Hyperparameters

### 5.1 Core PPO Hyperparameters

| Hyperparameter | Value | Description | Tuning Notes |
|----------------|-------|-------------|--------------|
| **Learning Rate** | `3e-4` | Adam optimizer step size | Default for PPO; stable across tasks |
| **Rollout Length** (`n_steps`) | `128` | Steps per env before update | 128 × 8192 = 1.05M timesteps/rollout |
| **Batch Size** | `512` | Minibatch size for SGD | Good balance for 1.05M rollout buffer |
| **Epochs** (`n_epochs`) | `10` | SGD epochs per rollout | 10 × (1.05M / 512) ≈ 20K gradient steps |
| **Discount** (`γ`) | `0.99` | Future reward discount | Standard for continuous control |
| **GAE Lambda** (`λ`) | `0.95` | Advantage estimation bias/variance | Standard for robotics |
| **Clip Range** | `0.2` | PPO policy update clipping | Prevents large policy shifts |
| **Entropy Coefficient** | `0.01` (initial) | Exploration bonus | Decays exponentially (see below) |
| **Value Coefficient** | `0.5` | Value loss weight | Standard SB3 default |
| **Gradient Clipping** | `0.5` | Max gradient norm | Prevents explosive updates |

**Code:**
```python
# scripts/reinforcement_learning/sb3/train.py, lines 74-127
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--n_steps", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--clip_range", type=float, default=0.2)
parser.add_argument("--ent_coef", type=float, default=0.01)
```

### 5.2 Rollout & Update Mechanics

**Rollout Phase** (data collection):
```
For each of 128 steps:
    For each of 8,192 environments:
        action = policy(observation)
        next_obs, reward, done, info = env.step(action)
        store (obs, action, reward, value, log_prob)

Total timesteps collected: 128 × 8,192 = 1,048,576 (~1M)
```

**Update Phase** (policy optimization):
```
For each of 10 epochs:
    Shuffle rollout buffer
    For each minibatch of 512 samples:
        Compute PPO loss:
            L_policy = -min(ratio * A, clip(ratio, 0.8, 1.2) * A)
            L_value = MSE(V_pred, V_target)
            L_entropy = -H(π)
            L_total = L_policy + 0.5 * L_value - 0.01 * L_entropy
        
        Backpropagate and update weights

Total gradient steps per update: 10 × (1,048,576 / 512) = 20,480
```

**Timeline:**
- **Rollout time:** ~0.8s (1M steps @ ~1,300 steps/sec)
- **Update time:** ~1.2s (20K gradient steps on RTX 3090)
- **Total iteration time:** ~2.0s
- **Iterations for 100M steps:** 100M / 1.05M ≈ 95 iterations
- **Total training time:** 95 × 2s ≈ 190s ≈ **3.2 minutes**... **Wait, that's wrong!**

**Actual Training Time (Session 5b):**
- **100M steps in ~18 hours** → ~1,500 steps/sec wall-clock
- **Discrepancy:** Physics simulation (200 Hz) dominates, not RL updates

**Corrected Breakdown:**
- Physics simulation: 1M steps @ 200 Hz = 5,000s of simulation time
- RL overhead: Negligible (~5% of wall-clock)
- **Bottleneck:** GPU physics, not gradient computation

### 5.3 Entropy Decay Schedule

To prevent **late-stage policy divergence**, entropy coefficient decays exponentially:

```python
ent_coef(t) = ent_coef_init * exp(-t / τ)

where:
  ent_coef_init = 0.01  (initial exploration)
  τ = 10,000,000        (decay timescale: 10M steps)
  t = current timestep
```

**Decay Curve:**

| Timestep | ent_coef | Behavior |
|----------|----------|----------|
| 0 | 0.0100 | High exploration |
| 10M | 0.0037 | Moderate exploration |
| 50M | 0.0007 | Low exploration |
| 100M | 0.00005 | Near-deterministic |

**Implementation:**
```python
# scripts/reinforcement_learning/sb3/train.py (EntropyDecayCallback)
new_ent_coef = self.initial_ent_coef * np.exp(-timesteps / self.decay_timesteps)
self.model.ent_coef = new_ent_coef
```

**Why Decay?**
- **Early:** High entropy → diverse actions → exploration
- **Late:** Low entropy → deterministic policy → exploitation
- **Prevents:** Policy from "forgetting" learned behaviors (common in long training)

### 5.4 Adaptive KL Scheduling (Optional)

**AdaptiveKLCallback** dynamically adjusts `target_kl` based on policy updates:

```python
if kl_divergence > target_kl_high:
    target_kl *= 1.1  # Increase tolerance (allow bigger updates)
elif kl_divergence < target_kl_low:
    target_kl *= 0.9  # Decrease tolerance (conservative updates)
```

**Default:** Disabled (`target_kl=None`) in Session 5b.

**When to enable:**
- Policy diverges after convergence
- Need more conservative updates
- Fine-tuning from checkpoint

### 5.5 GAE (Generalized Advantage Estimation)

PPO uses **GAE** to compute advantages with bias-variance tradeoff:

```python
# Temporal difference error
δ_t = r_t + γ * V(s_{t+1}) - V(s_t)

# GAE advantage
A_t = Σ_{l=0}^∞ (γλ)^l δ_{t+l}

where:
  γ = 0.99   (discount factor)
  λ = 0.95   (GAE lambda)
```

**λ = 0.95 Trade-off:**
- **λ = 0:** Low variance, high bias (1-step TD)
- **λ = 1:** High variance, low bias (Monte Carlo)
- **λ = 0.95:** Sweet spot for robotics

**Reference:**
> "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (Schulman et al., 2015)

### 5.6 Checkpoint & Logging

**Checkpointing:**
```python
save_freq = 1,000,000 // num_envs  # Save every 1M total steps
checkpoint_dir = "checkpoints/"
```

**Saved every 1M steps:**
- `rl_model_1000000_steps.zip` (policy + value networks)
- `vecnormalize.pkl` (observation normalization stats, if enabled)

**TensorBoard Logging:**
```python
tensorboard_log = f"logs/sb3/{task}/{timestamp}/"

Logged metrics:
  - rollout/ep_rew_mean       (average episode reward)
  - rollout/ep_len_mean       (average episode length)
  - train/policy_loss         (PPO policy loss)
  - train/value_loss          (value function MSE)
  - train/entropy_loss        (policy entropy)
  - train/approx_kl           (KL divergence)
  - train/clip_fraction       (fraction of clipped updates)
  - custom/entropy_coef       (current entropy coefficient)
  - custom/base_activation    (base velocity statistics)
```

**View Logs:**
```powershell
tensorboard --logdir logs/sb3/MobileMMTrackEE-v0/
```

### 5.7 Hardware-Specific Optimizations

**GPU Settings:**
```python
# scripts/reinforcement_learning/sb3/train.py, lines 690-703
torch.backends.cudnn.benchmark = True  # Auto-tune cuDNN kernels
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TensorFloat-32
torch.backends.cudnn.allow_tf32 = True
```

**Benefits:**
- **cuDNN benchmark:** 10-20% speedup for fixed input sizes
- **TF32:** 8× faster matrix multiplications on RTX 30XX (minimal precision loss)

**Memory Management:**
- **Gradient accumulation:** Disabled (batch_size = 512 fits in memory)
- **Mixed precision:** Not used (TF32 sufficient)

### 5.8 Complete Hyperparameter Yaml (Reference)

```yaml
# Hypothetical PPO config (actual code uses argparse)
algorithm: PPO
policy: MlpPolicy

network:
  pi: [256, 256, 128]
  vf: [256, 256, 128]
  activation: ReLU
  ortho_init: true
  log_std_init: -1.0

hyperparameters:
  learning_rate: 3.0e-4
  n_steps: 128
  batch_size: 512
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  clip_range_vf: 1.0
  ent_coef: 0.01
  vf_coef: 0.5
  max_grad_norm: 0.5
  target_kl: null

training:
  total_timesteps: 100_000_000
  num_envs: 8192
  save_freq: 1_000_000
  log_interval: 1
  device: cuda

callbacks:
  - EntropyDecay:
      initial: 0.01
      tau: 10_000_000
  - AdaptiveKL:
      enabled: false
  - Checkpoint:
      save_freq: 1_000_000
```

---

## 6. Hardware & System Requirements

### 6.1 Reference Hardware (Session 5b)

| Component | Specification | Utilization |
|-----------|---------------|-------------|
| **GPU** | NVIDIA RTX 3090 (24GB) | ~18GB/24GB (~75%) |
| **CPU** | Intel Xeon W-2145 (8C/16T) | ~40% (physics offloaded to GPU) |
| **RAM** | 64GB DDR4 | ~12GB (8192 envs + trajectories) |
| **Storage** | NVMe SSD | ~500MB (checkpoints + logs) |
| **OS** | Windows 11 | Isaac Sim runs natively |

**Throughput:**
- **Steps/sec:** ~1,500 (wall-clock)
- **Env interactions/sec:** 8,192 × 1,500 = **12.3M interactions/sec**
- **100M steps:** ~18 hours

### 6.2 Minimum Requirements

**For Training:**
- **GPU:** RTX 3060 (12GB) or better — CUDA 11.8+, compute capability 7.5+
- **CPU:** 6-core CPU (physics sim preprocessing)
- **RAM:** 32GB (4096 envs)
- **Storage:** 100GB (Isaac Sim + assets)

**For Inference:**
- **GPU:** GTX 1660 (6GB) — 1-256 envs
- **CPU:** 4-core CPU
- **RAM:** 16GB

### 6.3 Scaling Recommendations

| Num Envs | GPU Memory | Training Time (100M) | Recommended GPU |
|----------|------------|----------------------|-----------------|
| 512 | ~4GB | ~7 days | RTX 3060 (12GB) |
| 2048 | ~8GB | ~2 days | RTX 3070 (8GB) |
| 4096 | ~12GB | ~1 day | RTX 3080 (10GB) |
| 8192 | ~18GB | ~18 hours | RTX 3090 (24GB) |
| 16384 | ~32GB | ~10 hours | RTX A6000 (48GB) |

**Scaling Formula:**
```
GPU_memory ≈ 2.2 × num_envs / 1000  (GB)
Training_time ≈ 100M / (num_envs × 1500)  (seconds)
```

### 6.4 Software Stack

| Component | Version | Purpose |
|-----------|---------|---------|
| **Isaac Sim** | 2023.1.1+ | Physics simulation |
| **IsaacLab** | Latest (I:\isaaclab) | RL environment wrapper |
| **Python** | 3.10 | Runtime |
| **PyTorch** | 2.0.1 (CUDA 11.8) | Neural networks |
| **Stable-Baselines3** | 2.5.0+ | PPO implementation |
| **Gymnasium** | 0.29.1 | Environment API |

**Installation:** See [docs/setup/TRAIN_ON_WINDOWS.md](../01_setup/TRAIN_ON_WINDOWS.md)

---

## 7. IsaacLab → Stable-Baselines3 Integration

### 7.1 The Wrapper Problem

**Challenge:** IsaacLab uses `DirectRLEnv` (returns dict observations + torch tensors), but Stable-Baselines3 expects **Gymnasium API** (numpy arrays).

**Solution:** `IsaacLabToSB3VecEnvWrapper` converts between the two interfaces.

### 7.2 Wrapper Architecture

```python
# scripts/reinforcement_learning/sb3/train.py, lines 200-350

class IsaacLabToSB3VecEnvWrapper:
    """Wrap IsaacLab environment to be compatible with Stable-Baselines3."""
    
    def __init__(self, env):
        self.env = env
        self.num_envs = env.num_envs
        
        # Placeholder observation space (updated after first reset)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )
        self.action_space = env.action_space
    
    def reset(self):
        """Reset all environments."""
        obs_dict, _ = self.env.reset()  # New Gymnasium API
        
        # Extract observations from dict
        obs = obs_dict["policy"]  # Tensor [num_envs, obs_dim]
        
        # Update observation space on first reset
        if self.observation_space.shape == (1,):
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(obs.shape[-1],),  # [obs_dim]
                dtype=np.float32
            )
        
        # Convert torch → numpy
        return obs.cpu().numpy()
    
    def step(self, actions):
        """Step all environments."""
        # Convert numpy → torch
        actions_tensor = torch.from_numpy(actions).to(self.env.device)
        
        # Step environment
        obs_dict, rewards, dones, truncated, infos = self.env.step(actions_tensor)
        
        # Extract and convert observations
        obs = obs_dict["policy"].cpu().numpy()
        rewards = rewards.cpu().numpy()
        dones = dones.cpu().numpy()
        
        return obs, rewards, dones, infos
```

### 7.3 Key Conversion Details

**1. Observation Space Discovery:**
- Isaac Lab doesn't expose `observation_space.shape` until first reset
- Wrapper initializes with placeholder `shape=(1,)`, then updates dynamically

**2. Tensor → Numpy Conversion:**
```python
# IsaacLab returns torch.Tensor on GPU
obs_torch = env.step(actions)["policy"]  # [8192, 46] on CUDA

# SB3 expects numpy.ndarray on CPU
obs_numpy = obs_torch.cpu().numpy()  # [8192, 46] on RAM
```

**3. Reward Aggregation:**
- IsaacLab: Component rewards stored in `info` dict
- SB3: Logs total reward in TensorBoard
- Wrapper: Preserves `info` dict for custom logging

**4. Termination Handling:**
```python
# Old Gymnasium API (SB3 < 2.0)
dones = terminated | truncated

# New Gymnasium API (SB3 >= 2.0)
return obs, rewards, terminated, truncated, infos
```

Wrapper handles both APIs transparently.

### 7.4 Performance Impact

| Operation | Time (8192 envs) | Notes |
|-----------|------------------|-------|
| `env.reset()` | ~50ms | GPU → CPU transfer |
| `env.step()` | ~0.7ms | Simulation step (200 Hz physics) |
| `obs.cpu().numpy()` | ~0.1ms | Tensor copy (async) |
| **Total overhead** | **~2%** | Negligible vs physics |

**Optimization:** Use `torch.from_numpy()` with `copy=False` to avoid unnecessary copies.

---

## 8. Connection to Reward System

The model architecture is **tightly coupled** to the reward function design. See [REWARD_SYSTEM_DESIGN.md](REWARD_SYSTEM_DESIGN.md) for full details.

### 8.1 Reward → Observation Mapping

| Reward Component | Required Observations | Dimension |
|------------------|------------------------|-----------|
| **Position Error** | `pos_error` (target - ee_pos) | 3 |
| **Orientation Error** | `quat_error` (quat_diff) | 4 |
| **Velocity Tracking** | `ee_lin_vel`, `ee_ang_vel` | 6 |
| **Base Motion** | `base_to_target_dist`, `out_of_reach` | 2 |
| **Joint Smoothness** | `action_history` (for jerk) | 16 |
| **Collision Avoidance** | (future) `min_obstacle_dist` | 1 |

**Total:** 32 / 46 dimensions are **directly used** in reward computation.

### 8.2 Reward Shaping → Policy Behavior

**Example: Base Activation**

**Before (Session 5a):**
- **Reward:** Only `pos_error` (no base-specific signal)
- **Observation:** No `out_of_reach` flag
- **Policy:** Rarely moves base (relies on arm only)

**After (Session 5b):**
- **Reward:** Added `base_motion_reward` (+0.2 for moving when `out_of_reach=1`)
- **Observation:** Added `base_to_target_dist`, `out_of_reach` (4 dims)
- **Policy:** Actively repositions base when target is far

**Code:**
```python
# src/rl_platform/tasks/mobile_mm/rewards.py, lines 250-260
out_of_reach = (base_to_target_dist > arm_reach).float()
base_motion_reward = torch.where(
    out_of_reach > 0.5,  # If target is out of reach
    0.2 * (base_lin_vel.norm(dim=-1) / max_linear_vel),  # Reward base motion
    torch.zeros_like(out_of_reach)  # Otherwise, no reward
)
```

### 8.3 Reward Weights → Network Capacity

**Why 235K parameters?**

The reward function has **9 components** with varying scales:
- Position: ~10× more important than velocity
- Orientation: ~5× more important than smoothness
- Base motion: Binary signals (hard to learn)

**Network must:**
- **Disentangle** reward components (separate heads would help, but MLP is simpler)
- **Balance** conflicting objectives (e.g., speed vs. smoothness)
- **Generalize** across trajectories (1000+ unique trajectories)

**Empirical finding:**
- **[64, 64]** (14K params): Fails to track (underfits)
- **[128, 128]** (35K params): Partial tracking (struggles with base motion)
- **[256, 256, 128]** (235K params): Good tracking (Session 5b: ~0.85 reward)

**Hypothesis:** Larger networks better approximate complex value functions for multi-objective RL.

---

## 9. Code References

### 9.1 Key Files

| File | Purpose | Lines of Interest |
|------|---------|-------------------|
| **src/rl_platform/tasks/mobile_mm/env.py** | Environment definition | 670-940 (_pre_physics_step, _get_observations) |
| **src/rl_platform/tasks/mobile_mm/observations.py** | Observation composition | 8-216 (compose_observation, get_observation_dimensions) |
| **src/rl_platform/tasks/mobile_mm/rewards.py** | Reward computation | Full file (9 reward components) |
| **scripts/reinforcement_learning/sb3/train.py** | Training script | 200-350 (wrapper), 820-865 (PPO config) |
| **scripts/launch_training_windows.ps1** | Launcher (preferred) | Full file (CLI flags) |

### 9.2 Environment Initialization

```python
# src/rl_platform/tasks/mobile_mm/env.py, lines 84-111

# Compute observation dimensions
self.num_observations = get_observation_dimensions(
    num_joints=6,
    use_lookahead=self.task_cfg.use_lookahead,
    lookahead_steps=self.task_cfg.lookahead_steps,
    use_action_history=self.task_cfg.use_action_history,
    action_history_length=self.task_cfg.action_history_length,
    action_dim=self.num_actions,
)

# Define observation space
self.observation_space = gym.spaces.Box(
    low=-np.inf, high=np.inf,
    shape=(self.num_observations,),
    dtype=np.float32
)

# Define action space
self.action_space = gym.spaces.Box(
    low=-1.0, high=1.0,
    shape=(self.num_actions,),
    dtype=np.float32
)
```

### 9.3 Training Invocation

**Preferred (PowerShell Launcher):**
```powershell
.\scripts\launch_training_windows.ps1 `
    -Task MobileMMTrackEE-v0 `
    -NumEnvs 8192 `
    -Headless `
    -TotalTimesteps 100000000 `
    -LearningRate 3e-4 `
    -NSteps 128 `
    -BatchSize 512
```

**Direct (Isaac Lab):**
```powershell
I:\isaaclab\isaaclab.bat -p scripts/reinforcement_learning/sb3/train.py `
    --task MobileMMTrackEE-v0 `
    --num_envs 8192 `
    --headless `
    --total_timesteps 100000000
```

### 9.4 Reproducing Session 5b

**Complete Command:**
```powershell
.\scripts\launch_training_windows.ps1 `
    -Task MobileMMTrackEE-v0 `
    -NumEnvs 8192 `
    -Headless `
    -TotalTimesteps 100073472 `
    -LearningRate 3e-4 `
    -NSteps 128 `
    -BatchSize 512 `
    -NEpochs 10 `
    -Gamma 0.99 `
    -GAELambda 0.95 `
    -ClipRange 0.2 `
    -EntCoef 0.01 `
    -EnableEntropyDecay
```

**Expected Results:**
- **Training time:** ~18 hours (RTX 3090, 8192 envs)
- **Final reward:** ~0.85 (normalized, see REWARD_SYSTEM_DESIGN.md)
- **Checkpoints:** `checkpoints/rl_model_{1M,2M,...,100M}_steps.zip`
- **Logs:** `logs/sb3/MobileMMTrackEE-v0/{timestamp}/`

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2025-01-08 | 1.0 | Initial documentation created |

---

## References

1. **PPO Algorithm:**  
   Schulman et al., "Proximal Policy Optimization Algorithms" (2017)  
   https://arxiv.org/abs/1707.06347

2. **GAE:**  
   Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (2015)  
   https://arxiv.org/abs/1506.02438

3. **Orthogonal Initialization:**  
   Saxe et al., "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks" (2013)  
   https://arxiv.org/abs/1312.6120

4. **IsaacLab Documentation:**  
   https://isaac-sim.github.io/IsaacLab/

5. **Stable-Baselines3 Documentation:**  
   https://stable-baselines3.readthedocs.io/

---

**End of Document**
