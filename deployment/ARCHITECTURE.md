# Deployment Architecture: CinebotRL → NVIDIA Orin + ROS2 Humble

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING (Windows PC)                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Isaac Lab Sim (16,384 envs)                                                 │
│         ↓                                                                     │
│  PPO Training (200M steps)                                                   │
│         ↓                                                                     │
│  Checkpoint: rl_model_200000000_steps.zip                                    │
│         ↓                                                                     │
│  [scripts/export_policy_for_deployment.py]                                   │
│         ↓                                                                     │
│  ┌─────────────────────────────────┐                                         │
│  │  policy_session_7d.onnx         │  (235 KB, optimized)                   │
│  │  normalization_stats.npz        │  (observation scaling)                  │
│  └─────────────────────────────────┘                                         │
│                                                                               │
└───────────────────────────┬──────────────────────────────────────────────────┘
                            │ SCP / USB Transfer
                            ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT (NVIDIA Orin + ROS2 Humble)                     │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                      ROS2 Policy Inference Node                         │ │
│  │                                                                          │ │
│  │  ┌──────────────┐            ┌─────────────────────────────┐          │ │
│  │  │ ONNX Runtime │            │   Observation Builder        │          │ │
│  │  │ (GPU/CUDA)   │◀───────────│   - Joint states (9)        │          │ │
│  │  │              │            │   - Joint velocities (9)     │          │ │
│  │  │ Input: (74)  │            │   - Base pose (3)            │          │ │
│  │  │ Output: (8)  │            │   - Target pose (7)          │          │ │
│  │  └──────────────┘            │   - Relative states          │          │ │
│  │         ↓                     │   - Normalization (stats.npz)│          │ │
│  │  ┌──────────────┐            └─────────────────────────────┘          │ │
│  │  │ Action Scale │                       ↑                               │ │
│  │  │ - Base: [-1.5, 1.5] m/s              │                               │ │
│  │  │ - Rot:  [-2.0, 2.0] rad/s            │                               │ │
│  │  │ - Arm:  [-1.0, 1.0] rad/s            │                               │ │
│  │  └──────────────┘                       │                               │ │
│  │         ↓                                 │                               │ │
│  └─────────┼─────────────────────────────────┼───────────────────────────┘ │
│            │                                 │                               │
│            ↓ /joint_commands                 ↑ /joint_states                │
│                                               ↑ /base_pose                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Robot Hardware Controllers                        │   │
│  │                                                                       │   │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │   │
│  │  │ Base Motion  │    │ Arm Motion   │    │  State Publishers    │  │   │
│  │  │ Controller   │    │ Controller   │    │  - Joint encoders    │  │   │
│  │  │              │    │              │    │  - Base odometry     │  │   │
│  │  │ Diff. Drive/ │    │ 6-DOF Arm    │    │  - Localization      │  │   │
│  │  │ Omni-wheel   │    │ (Trajectory) │    │                      │  │   │
│  │  └──────────────┘    └──────────────┘    └──────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│            │                     │                        │                  │
│            ↓                     ↓                        ↑                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Physical Robot                                │   │
│  │                                                                       │   │
│  │  [Base: 3-DOF mobile platform]  +  [Arm: 6-DOF manipulator]         │   │
│  │                                                                       │   │
│  │                    Tracking Target: Camera End-Effector              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
│  Additional ROS2 Nodes:                                                      │
│  ┌────────────────────┐  ┌─────────────────────┐  ┌──────────────────────┐│
│  │ Trajectory Player  │  │ Safety Monitor      │  │ Performance Logger   ││
│  │ - Load JSON files  │  │ - Joint limits      │  │ - Tracking error     ││
│  │ - Publish targets  │  │ - Collision detect  │  │ - Inference latency  ││
│  │ - 10 Hz update     │  │ - E-stop handler    │  │ - Velocity smoothness││
│  └────────────────────┘  └─────────────────────┘  └──────────────────────┘│
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow (20 Hz control loop)

```
1. Robot State → ROS2 Topics
   /joint_states:      [joint_x, joint_y, joint_theta, arm_j1...arm_j6]
   /base_pose:         [x, y, yaw] from odometry/AMCL
   
2. Trajectory Playback → Target
   /camera_target_pose: [x, y, z, qx, qy, qz, qw] from JSON trajectory

3. Policy Inference (every 50ms)
   obs = build_observation()        # 74-dim vector
   obs_norm = normalize(obs)         # Apply saved stats
   action = onnx_session.run(obs_norm)  # 8-dim output [base_vx, base_vy, base_wz, arm_v1...arm_v6]
   
4. Action → Robot Commands
   /joint_commands:     trajectory_msgs/JointTrajectory
                        → Joint position targets (integrated from velocities)

5. Robot Execution
   Controllers execute joint commands → Robot moves → State updates → Loop
```

## Performance Targets

| Metric | Training (Sim) | Deployment (Real) | Notes |
|--------|----------------|-------------------|-------|
| Control Freq | 20 Hz | 20 Hz | Must match! |
| Inference Time | N/A | 2-5 ms (GPU) | ONNX on Orin |
| Loop Latency | N/A | <10 ms | Total E2E |
| Observation Dim | 74 | 74 | Exact match required |
| Action Dim | 8 | 8 | Velocity commands |
| Max Base Speed | 1.5 m/s | 1.5 m/s | Safety limits |
| Max Arm Speed | 1.0 rad/s | 1.0 rad/s | Joint limits |

## File Structure on Orin

```
~/cinebot_ws/
├── policy_session_7d.onnx              # Trained policy (235 KB)
├── normalization_stats.npz              # Observation scaling
├── src/
│   └── cinebot_control/
│       ├── scripts/
│       │   └── ros2_policy_node.py     # Main inference node
│       ├── launch/
│       │   └── policy_inference.launch.py
│       └── package.xml
└── install/
    └── setup.bash                       # Source this!
```

## Quick Start Commands

```bash
# 1. On Windows (after training completes)
python scripts/export_policy_for_deployment.py \
    --checkpoint logs/sb3/.../rl_model_200000000_steps.zip \
    --output deployment/policy_session_7d.onnx

# 2. Transfer to Orin
scp deployment/* orin@orin-hostname:~/cinebot_ws/

# 3. On Orin - Install dependencies
ssh orin@orin-hostname
pip3 install onnxruntime-gpu numpy

# 4. Build ROS2 workspace
cd ~/cinebot_ws
colcon build --packages-select cinebot_control
source install/setup.bash

# 5. Launch policy inference
ros2 launch cinebot_control policy_inference.launch.py

# 6. Monitor performance
ros2 topic hz /joint_commands  # Should be ~20 Hz
ros2 topic echo /joint_commands # Check command values
```

## Safety Considerations

⚠️ **Before deploying on real hardware:**

1. ✅ Test in simulation first (Isaac Sim or Gazebo)
2. ✅ Start with reduced velocities (scale actions by 0.3x)
3. ✅ Implement emergency stop (hardware + software)
4. ✅ Add joint limit checking in ROS2 node
5. ✅ Test with simple trajectories before complex ones
6. ✅ Monitor for oscillations/instability
7. ✅ Have manual override ready

## Sim-to-Real Transfer Tips

Expected challenges and solutions:

| Issue | Cause | Solution |
|-------|-------|----------|
| Jerky motion | Control freq mismatch | Verify 20 Hz loop |
| Poor tracking | Action scaling wrong | Tune velocity multipliers |
| Overshoot | PID gains different | Add damping term |
| Localization drift | Odometry quality | Use wheel + IMU fusion |
| Latency spikes | CPU overload | Use GPU inference |

## Next Steps After Deployment

1. **Collect real-world data** - Log tracking errors and robot states
2. **Fine-tune if needed** - Use collected data for domain adaptation
3. **Add trajectory library** - Integrate 1,038 cinematic trajectories
4. **Performance monitoring** - Track success rate, smoothness metrics
5. **Safety validation** - Test edge cases and failure modes
