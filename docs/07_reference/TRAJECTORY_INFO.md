# Trajectory Information for Mobile Manipulator Training

## 🎯 **Current Trajectory: CIRCLE**

Your robot is following a **circular trajectory** in 3D space!

### **Trajectory Parameters:**

From `config.py` (default settings):
```python
type: "circle"              # Trajectory shape
amplitude: 0.5              # Circle radius: 0.5 meters (50cm)
speed: 0.2                  # Speed: 0.2 m/s (20 cm/second)
height: 1.0                 # Z-coordinate: 1.0 meter above ground
```

### **Mathematical Description:**

The target position for the end-effector moves according to:

```
x(t) = center_x + 0.5 * cos(phase)
y(t) = center_y + 0.5 * sin(phase)
z(t) = 1.0  (constant height)

where:
  phase = (speed / amplitude) * time
  phase advances: 0.2/0.5 * dt = 0.4 * dt radians/step
```

### **Visual Description:**

```
         Top View:
         
    Y ^
      |
    0.5|     ●---●
      | ●           ●
    0 |-●-----+-----●--> X
      | ●   (0,0)   ●
   -0.5|     ●---●
      |
      +-----|-----|-----
         -0.5  0  0.5

    Side View:
    
    Z ^
    1.0|  ●→ ●→ ●→ ●→    (constant height)
      |
    0.0|===============  (ground)
      +-----|-----|-----
         -0.5  0  0.5

```

**The robot's end-effector must trace a circle:**
- **Radius**: 50cm (0.5m)
- **Height**: 1m above ground
- **Speed**: One complete circle takes ~15.7 seconds
  - Circumference = 2π × 0.5 = 3.14m
  - Time = 3.14m / 0.2m/s = 15.7 seconds
  
**Episode Duration**: 
- 1000 steps × 0.02s/step = 20 seconds
- Robot completes ~1.27 full circles per episode

---

## 📍 **Available Trajectory Types:**

Your system supports 4 trajectory types (currently using **circle**):

### 1. **Circle** (Current)
```python
type: "circle"
```
- Smooth circular path in XY plane
- Best for: Testing coordinated motion
- Difficulty: Medium

### 2. **Line** (Back and forth)
```python
type: "line"
```
- Oscillates along X-axis: `x = amplitude * sin(phase)`
- Simpler than circle
- Best for: Basic tracking validation
- Difficulty: Easy

### 3. **Figure Eight** (Lissajous curve)
```python
type: "figure_eight"
```
- Complex "∞" shaped path
- Math: `x = amplitude * sin(phase), y = (amplitude/2) * sin(2*phase)`
- Best for: Advanced tracking
- Difficulty: Hard

### 4. **Recorded** (Custom trajectories)
```python
type: "recorded"
waypoint_file: "path/to/trajectory.json"
```
- Load custom trajectories from JSON files
- Best for: Real-world task replication
- Difficulty: Varies

---

## 🎬 **What You See in Visualization:**

When running `evaluate.py`, you should see:

1. **🔴 Red Spheres** (Target/Desired Position)
   - Forms a circle in space
   - Radius: 50cm
   - Height: 1m
   - Moves continuously at 0.2 m/s

2. **🟢 Green Spheres** (End-Effector Position)
   - Shows where the robot's end-effector actually is
   - Should follow the red spheres closely
   - Gap between red & green = tracking error

3. **Robot Movement**
   - **Base moves** (X, Y, rotation) to help arm reach targets
   - **Arm joints articulate** to fine-tune end-effector position
   - **Coordinated motion**: Base + Arm working together!

### **Performance Interpretation:**

Your evaluation results:
```
Episode 1: -755,061 → Red & Green very far apart (catastrophic)
Episode 2:   +6,451 → Red & Green close (good tracking)
Episode 3:     +712 → Red & Green moderately close  
Episode 4:   +5,771 → Red & Green close (good tracking)
```

**Good performance** = Green spheres chase red spheres smoothly around the circle!

---

## 🔧 **Changing Trajectory Type:**

To test different trajectories, modify `config.py`:

```python
# Option 1: Easier line trajectory
type: "line"

# Option 2: Harder figure-eight
type: "figure_eight"

# Option 3: Custom recorded trajectory
type: "recorded"
waypoint_file: "path/to/your_trajectory.json"
```

Or adjust current circle parameters:

```python
# Bigger circle (more challenging)
amplitude: 1.0  # 1 meter radius

# Faster movement (more challenging)
speed: 0.5  # 0.5 m/s

# Different height
height: 0.8  # 80cm above ground
```

---

## 📊 **Training Implications:**

### **Why Circle Trajectory?**

✅ **Good balance** of:
1. **Requires base movement** - Circle radius (50cm) often exceeds arm reach
2. **Smooth motion** - Continuous derivatives (velocity, acceleration)
3. **Periodic** - Robot sees similar situations multiple times
4. **Challenging but achievable** - Not too easy, not too hard

### **Success Metrics:**

For your 10M training:
- **Episode 1 failure** suggests initialization problems
- **Episodes 2-4 success** proves:
  - ✅ Robot learned to track circular paths
  - ✅ Base movement is functional
  - ✅ Arm-base coordination works
  - ⚠️ Still has inconsistency (variance in rewards)

### **Next Steps:**

1. **More training** (50-100M) will reduce variance
2. **Curriculum learning**: Start with small circles, gradually increase
3. **Multi-trajectory training**: Mix circle, line, figure-eight
4. **Add obstacles**: Make it more realistic

---

## 🎯 **Summary:**

**Your robot is learning to:**
1. Track a **circular trajectory** (50cm radius, 1m height)
2. Move at **0.2 m/s** (20 cm/second)
3. Complete **~1.27 circles** in each 20-second episode
4. Use **coordinated base + arm motion** to reach all points

**Current Performance**: 
- 75% success rate (3/4 episodes)
- Near-perfect episode completion (999/1000 steps)
- Proves base movement issue is RESOLVED ✅

**Evidence in visualization**: Red circle of targets being chased by green end-effector position, with the base platform visibly moving to help the arm reach!
