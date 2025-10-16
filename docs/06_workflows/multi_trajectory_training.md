# Using Multiple Trajectories for Training

Your `trajectoryToLearn/world_json` folder contains **1000+ cinematic trajectories** across 10 categories! Here's how to leverage them for robust training.

## Trajectory Dataset 📁

```
trajectoryToLearn/world_json/cinematic_db/
├─ arc_left_push/      (100 trajectories)
├─ arc_right_pull/     (100 trajectories)
├─ crane_down/         (100 trajectories)
├─ crane_up/           (100 trajectories)
├─ dolly_pull_out/     (100 trajectories)
├─ dolly_push_in/      (100 trajectories)
├─ handheld_subtle/    (100 trajectories)
├─ orbit_left/         (100 trajectories)
├─ orbit_right/        (100 trajectories)
└─ tracking_zigzag/    (100 trajectories)

Total: 1000 cinematic camera movements!
```

## Training Modes 🎯

### Mode 1: Single Trajectory (Current Default)

**Use case**: Testing, debugging, or focusing on mastering one specific movement

```python
# In config or environment initialization
trajectory_config = {
    "type": "recorded",
    "waypoint_file": "trajectoryToLearn/1_pull_world_scaled.json"
}
```

**Training command:**
```bash
python scripts/reinforcement_learning/sb3/train.py \
    --task MobileMMTrackEE-v0 \
    --num_envs 256 \
    --trajectory_type recorded \
    --waypoint_file trajectoryToLearn/1_pull_world_scaled.json
```

**Pros**: 
- Fast convergence on specific trajectory
- Easy to verify learning

**Cons**:
- Overfits to single trajectory
- Poor generalization

---

### Mode 2: Multi-Trajectory (Recommended!) 🚀

**Use case**: Robust policy that generalizes across diverse cinematic movements

```python
# In config or environment initialization
trajectory_config = {
    "type": "multi_recorded",
    "trajectory_dir": "trajectoryToLearn/world_json/cinematic_db"
}
```

**Training command:**
```bash
python scripts/reinforcement_learning/sb3/train.py \
    --task MobileMMTrackEE-v0 \
    --num_envs 1024 \
    --trajectory_type multi_recorded \
    --trajectory_dir trajectoryToLearn/world_json/cinematic_db
```

**How it works**:
1. **At episode reset**: Each parallel environment samples a random trajectory
2. **Training diversity**: With 1024 envs, you're training on 1024 different trajectories simultaneously!
3. **Automatic resampling**: Every episode reset gives a new random trajectory

**Pros**:
- ✅ Learns generalizable policy
- ✅ Robust to trajectory variations
- ✅ No overfitting to single demo
- ✅ Naturally implements demonstration diversity

**Cons**:
- Slightly slower initial convergence
- Requires more training steps

---

### Mode 3: Category-Specific Training

**Use case**: Master specific cinematic techniques (e.g., only crane movements)

```python
trajectory_config = {
    "type": "multi_recorded",
    "trajectory_dir": "trajectoryToLearn/world_json/cinematic_db/crane_up"
}
```

**Training command:**
```bash
python scripts/reinforcement_learning/sb3/train.py \
    --task MobileMMTrackEE-v0 \
    --num_envs 512 \
    --trajectory_type multi_recorded \
    --trajectory_dir trajectoryToLearn/world_json/cinematic_db/crane_up
```

Train separate policies for different categories, then combine or switch between them.

---

## Curriculum Learning Strategy 📚

**Progressive difficulty**: Start simple, add complexity

### Phase 1: Single Trajectory Warm-up (100K steps)
```bash
python train.py --trajectory_type recorded \
    --waypoint_file trajectoryToLearn/1_pull_world_scaled.json \
    --total_timesteps 100000
```

### Phase 2: Same Category Variations (500K steps)
```bash
python train.py --trajectory_type multi_recorded \
    --trajectory_dir trajectoryToLearn/world_json/scene_1 \
    --total_timesteps 500000
```

### Phase 3: Full Dataset (5M+ steps)
```bash
python train.py --trajectory_type multi_recorded \
    --trajectory_dir trajectoryToLearn/world_json/cinematic_db \
    --total_timesteps 5000000
```

---

## Expected Training Metrics 📊

### Single Trajectory Mode
- **Convergence**: 50K-200K steps
- **Success rate**: >95% on trained trajectory
- **Generalization**: Poor (<50% on new trajectories)

### Multi-Trajectory Mode (1000 trajectories)
- **Convergence**: 500K-2M steps
- **Success rate**: 80-90% across all trajectory types
- **Generalization**: Excellent (>75% on unseen variations)

---

## Implementation Details 🔧

The multi-trajectory system:

1. **Pre-loads** all JSON files into memory at initialization
2. **Randomly samples** one trajectory per environment at each reset
3. **Pads** shorter trajectories to match longest in batch
4. **Converts** xyzw→wxyz quaternion format automatically
5. **Resamples** on episode termination for maximum diversity

Key advantage: **With 1024 environments, you get 1024 different trajectory contexts per policy update!**

---

## Configuration Update Needed 🛠️

To use multi-trajectory mode, update your environment config:

```python
# src/rl_platform/tasks/mobile_mm/config.py

@dataclass
class TrajectoryConfig:
    type: str = "multi_recorded"  # Changed from "circle"
    trajectory_dir: str = "trajectoryToLearn/world_json/cinematic_db"  # New
    amplitude: float = 0.5
    speed: float = 0.2
    height: float = 1.0
```

---

## Monitoring Training 📈

### Key metrics to watch:
- **Mean trajectory diversity**: Should see ~10 different categories
- **Success rate per category**: Track in TensorBoard
- **Generalization gap**: Test on held-out trajectories

### TensorBoard command:
```bash
tensorboard --logdir logs/sb3/MobileMMTrackEE-v0
```

### Weights & Biases integration:
```bash
python train.py --wandb_project cinebotRL --wandb_run_name multi_traj_1k
```

---

## Recommendations 💡

**For best results**:

1. **Start with multi-trajectory mode** from the beginning
2. **Use 1024+ environments** on your RTX 3090
3. **Train for 5-10M steps** for robust policy
4. **Log trajectory categories** to ensure balanced sampling
5. **Save checkpoints** every 500K steps

**Expected training time**:
- RTX 3090 with 1024 envs: ~100K steps/hour
- 5M steps = ~50 hours of training
- Use `--headless` for maximum speed

---

## Next Steps 🚀

1. **Test current setup**: `python scripts/test_mobile_mm_env.py`
2. **Start training**: Use multi-trajectory mode from day 1
3. **Monitor diversity**: Check that all 10 categories appear
4. **Evaluate generalization**: Test on held-out trajectories

Your 1000+ trajectory dataset is a goldmine for learning robust, generalizable policies! 🎬✨
