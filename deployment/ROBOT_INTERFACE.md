# Robot Interface Specification - ROS2 Topics

**Based on actual robot documentation (Section 2.3)**

---

## 📋 Actual Robot ROS2 Interface

### Input Topics (Robot → Policy Node)

| # | Topic Name | Message Type | Description | Details |
|---|------------|--------------|-------------|---------|
| 1 | `/motion_target/target_joint_state_arm_left` | `sensor_msgs::msg::JointState` | 机械臂控制信号<br>Arm control signal | `float64[] position` - 6 DOF arm joint target positions |
| 2 | `/hdas/feedback_arm_left` | `sensor_msgs::msg::JointState` | 机械臂反馈信号<br>Arm feedback signal | `float64[] position` - 6 DOF arm joint current positions |
| 3 | `/odom_wheel` | `nav_msgs::msg::Odometry` | 底盘定位信息<br>Base odometry | Wheel odometry for mobile base |
| 4 | `/mobile_base/commands/velocity` | `geometry_msgs::msg::Twist` | 底盘速度控制信号<br>Base velocity commands | `cmd_vel.linear.x` - forward speed (m/s)<br>`cmd_vel.angular.z` - yaw rate (rad/s) |

### Key Findings:

1. **Arm Commands are POSITIONS, not velocities!**
   - Robot expects `JointState.position` (6 DOF)
   - We need to integrate velocity actions → positions
   
2. **Base Commands use Twist message**
   - `linear.x` - forward velocity (m/s)
   - `linear.y` - lateral velocity (m/s) - **if omnidirectional**
   - `angular.z` - yaw rate (rad/s)

3. **Odometry is nav_msgs/Odometry, not Pose2D**
   - Includes full pose (position + orientation quaternion)
   - Includes twist (linear + angular velocities)

---

## 🔧 Updated Deployment Files

Created new files matching robot interface:
1. **`ros2_policy_node_robot.py`** - Updated node with correct topics/types
2. **`policy_inference_robot.launch.py`** - Updated launch file

---

## 📊 Topic Mapping Comparison

### Original Deployment vs. Robot Interface

| Function | Original | Robot Actual | Change Required |
|----------|----------|--------------|-----------------|
| **Arm Feedback** | `/joint_states` | `/hdas/feedback_arm_left` | ✅ Topic name |
| **Arm Command** | `/joint_commands` (JointTrajectory) | `/motion_target/target_joint_state_arm_left` (JointState) | ⚠️ Topic + Type |
| **Base Odometry** | `/base_pose` (Pose2D) | `/odom_wheel` (Odometry) | ⚠️ Topic + Type |
| **Base Command** | `/joint_commands` (JointTrajectory) | `/mobile_base/commands/velocity` (Twist) | ⚠️ Topic + Type |
| **Target Pose** | `/camera_target_pose` (PoseStamped) | `/camera_target_pose` (assumed) | ✅ OK |

---

## 🚨 Critical Implementation Changes

### 1. Arm Control: Velocity → Position Integration

**Training:** Policy outputs joint velocities (rad/s)  
**Robot:** Expects joint positions (rad)

**Solution:**
```python
# In control loop:
dt = 1.0 / control_frequency  # e.g., 0.05 s for 20 Hz

# Get velocity actions from policy
arm_velocities = policy_output[3:9] * arm_velocity_scale  # [6] rad/s

# Integrate: position(t) = position(t-1) + velocity * dt
integrated_arm_pos += arm_velocities * dt

# Publish positions
arm_cmd = JointState()
arm_cmd.position = integrated_arm_pos.tolist()
arm_cmd_pub.publish(arm_cmd)
```

### 2. Base Control: Direct Twist Message

**Training:** Policy outputs base velocities (vx, vy, vyaw)  
**Robot:** Expects `geometry_msgs/Twist`

**Solution:**
```python
# Get base velocity actions from policy
base_vx = policy_output[0] * base_linear_scale  # m/s
base_vy = policy_output[1] * base_linear_scale  # m/s (if omnidirectional)
base_vyaw = policy_output[2] * base_angular_scale  # rad/s

# Publish Twist
base_cmd = Twist()
base_cmd.linear.x = base_vx
base_cmd.linear.y = base_vy  # Set to 0.0 if differential drive
base_cmd.angular.z = base_vyaw
base_cmd_pub.publish(base_cmd)
```

### 3. Odometry: Extract from nav_msgs/Odometry

**Original:** Simple Pose2D (x, y, theta)  
**Robot:** Full Odometry with quaternion

**Solution:**
```python
def odom_callback(msg: Odometry):
    # Extract position
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    
    # Convert quaternion to yaw
    qx, qy, qz, qw = msg.pose.pose.orientation.x/y/z/w
    yaw = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
    
    base_pos = [x, y, yaw]
    
    # Extract velocities
    vx = msg.twist.twist.linear.x
    vy = msg.twist.twist.linear.y
    vyaw = msg.twist.twist.angular.z
    
    base_vel = [vx, vy, vyaw]
```

---

## 📝 Observation Vector Construction

**Training expects 74-dimensional observation:**

1. **Joint Positions (9):** [base_x, base_y, base_yaw, arm_j1, ..., arm_j6]
2. **Joint Velocities (9):** [base_vx, base_vy, base_vyaw, arm_v1, ..., arm_v6]
3. **Base Pose (3):** [x, y, yaw] (redundant but matches training)
4. **Target Pose (7):** [x, y, z, qx, qy, qz, qw]
5. **Target Relative to EE (7):** Relative transform (requires forward kinematics)
6. **Previous Action (8):** Last policy output
7. **Additional Features:** Time, phase, etc. (to reach 74 total)

**Important:** You need to verify the exact observation structure from your training code (`src/task_spec.py` or environment definition).

---

## 🔍 Verification Steps

### 1. Check Joint Names

The robot publishes 6 DOF arm. Verify joint order:
```bash
# On robot
ros2 topic echo /hdas/feedback_arm_left --once

# Check joint names in msg.name
# Update self.arm_joint_names in node to match
```

### 2. Test Arm Position Control

```bash
# Publish test positions
ros2 topic pub /motion_target/target_joint_state_arm_left sensor_msgs/JointState "{
    name: ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
    position: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}" --once

# Robot should move to home position
```

### 3. Test Base Velocity Control

```bash
# Publish test velocities (move forward slowly)
ros2 topic pub /mobile_base/commands/velocity geometry_msgs/Twist "{
    linear: {x: 0.1, y: 0.0, z: 0.0},
    angular: {x: 0.0, y: 0.0, z: 0.0}
}" --rate 10

# Stop (Ctrl+C then publish zero)
ros2 topic pub /mobile_base/commands/velocity geometry_msgs/Twist "{
    linear: {x: 0.0, y: 0.0, z: 0.0},
    angular: {x: 0.0, y: 0.0, z: 0.0}
}" --once
```

### 4. Monitor Odometry

```bash
ros2 topic echo /odom_wheel
# Verify x, y update when base moves
# Verify orientation quaternion changes when rotating
```

---

## 🚀 Deployment Commands (Updated)

### Transfer Files
```bash
# Transfer updated files
scp deployment/ros2_policy_node_robot.py orin@orin-ip:~/cinebot_ws/src/cinebot_control/scripts/
scp deployment/policy_inference_robot.launch.py orin@orin-ip:~/cinebot_ws/src/cinebot_control/launch/
scp deployment/policy_demo.onnx orin@orin-ip:~/cinebot_ws/models/
scp deployment/normalization_stats.npz orin@orin-ip:~/cinebot_ws/models/
```

### Build and Launch
```bash
# On Orin
cd ~/cinebot_ws
colcon build --packages-select cinebot_control
source install/setup.bash

# Launch with robot interface
ros2 launch cinebot_control policy_inference_robot.launch.py \
    model_path:=~/cinebot_ws/models/policy_demo.onnx \
    stats_path:=~/cinebot_ws/models/normalization_stats.npz \
    control_frequency:=20.0
```

---

## ⚠️ Safety Considerations

### 1. Initial Testing with Conservative Scaling

```bash
# Launch with reduced action scaling (0.5x)
ros2 launch cinebot_control policy_inference_robot.launch.py \
    base_linear_scale:=0.75 \
    base_angular_scale:=1.0 \
    arm_velocity_scale:=0.5
```

### 2. Monitor Commands

```bash
# Terminal 1: Watch arm commands
ros2 topic echo /motion_target/target_joint_state_arm_left

# Terminal 2: Watch base commands
ros2 topic echo /mobile_base/commands/velocity

# Check for:
# - Reasonable position changes (arm)
# - Reasonable velocities (base)
# - No sudden jumps or oscillations
```

### 3. Emergency Stop

Ensure you have a way to stop the robot:
- Physical E-stop button
- ROS2 command to stop: `ros2 topic pub /mobile_base/commands/velocity geometry_msgs/Twist "{}" --once`
- Kill node: `Ctrl+C` in launch terminal

---

## 🐛 Troubleshooting

### Issue: Arm moves erratically

**Possible causes:**
1. Position integration drift
2. Action scaling too high
3. Observation mismatch with training

**Solutions:**
- Reset integrated positions periodically: `integrated_arm_pos = current_arm_pos.copy()`
- Reduce `arm_velocity_scale` to 0.25
- Add position limits and clipping

### Issue: Base doesn't move or moves unexpectedly

**Check:**
- Is robot in autonomous mode? (May need to enable via robot controller)
- Are velocities being published? `ros2 topic hz /mobile_base/commands/velocity`
- Are velocities reasonable? Not too small (<0.01) or too large (>2.0)

**Solution:**
- Check robot controller state/mode
- Verify base command topic name exactly matches
- Test with manual `ros2 topic pub` first

### Issue: Observation dimension mismatch

**Error:** `Observation dimension mismatch: XX != 74`

**Solution:**
- Review `build_observation()` function
- Compare with training observation structure in `src/task_spec.py`
- Print observation components and verify each dimension

---

## 📚 Next Steps

1. **Verify joint names and order** - Check `/hdas/feedback_arm_left` message
2. **Test manual control** - Publish to command topics directly
3. **Update observation builder** - Match exact training structure
4. **Test in WSL first** - Use updated `ros2_policy_node_robot.py`
5. **Deploy to Orin** - With conservative scaling
6. **Monitor and tune** - Adjust scaling based on robot response

---

**Status:** ✅ Robot interface documented and code updated  
**Files Created:**
- `ros2_policy_node_robot.py` - Updated inference node
- `policy_inference_robot.launch.py` - Updated launch file
- `ROBOT_INTERFACE.md` - This specification document
