# Comprehensive Quantitative Evaluation System

This directory contains a complete evaluation system for assessing trained MobileMMTrackEE policies with detailed metrics and visualizations.

## 📊 What Gets Measured

### Tracking Accuracy
- **Position Error**: Mean, median, P95, P99, max (in cm)
- **Orientation Error**: Mean, median, P95 (in degrees)
- **Per-axis breakdown**: X, Y, Z position errors

### Robot State
- **Joint Angles**: Distribution, range, mean for all 6 DOF
- **Joint Velocities**: Max, P95 for all 6 DOF (rad/s)
- **Base Velocities**: Linear (vx, vy) and angular (ωz)

### Reward Analysis
- **Total Reward**: Mean, std, min, max per episode
- **Component Breakdown**: Individual reward terms
- **Temporal Trends**: Reward over episodes

### Success Metrics
- **Success Rate**: Percentage of successful episodes
- **Episode Length**: Mean, std episode duration
- **Failure Analysis**: Common failure modes

## 🚀 Quick Start

### 1. Run Quantitative Evaluation

```powershell
# Quick test (20 episodes, 4 envs, with GUI):
.\scripts\launch_evaluation_quantitative.ps1 -NumEpisodes 20 -NumEnvs 4

# Full evaluation (200 episodes, 64 envs, headless):
.\scripts\launch_evaluation_quantitative.ps1 -NumEpisodes 200 -NumEnvs 64 -Headless

# Evaluate on all 1,038 trajectories:
.\scripts\launch_evaluation_quantitative.ps1 -UseAllTrajectories -NumEpisodes 500 -Headless
```

### 2. Visualize Results

```powershell
# Generate plots and report:
python scripts/reinforcement_learning/sb3/visualize_eval_results.py --input_dir evaluation_results

# View plots:
explorer evaluation_plots\

# Read summary report:
notepad evaluation_plots\evaluation_report.txt
```

## 📁 Output Files

### evaluation_results/
- `eval_summary_YYYYMMDD_HHMMSS.json` - Complete statistics
- `episodes_YYYYMMDD_HHMMSS.csv` - Per-episode data
- `steps_YYYYMMDD_HHMMSS.csv` - Per-step data (sampled)
- `arrays_YYYYMMDD_HHMMSS.npz` - Raw numpy arrays

### evaluation_plots/
- `tracking_errors.png` - Position/orientation error distributions
- `joint_angles.png` - Joint usage analysis
- `joint_velocities.png` - Joint velocity profiles
- `base_velocities.png` - Base motion analysis
- `reward_components.png` - Reward breakdown
- `episode_statistics.png` - Episode-level statistics
- `evaluation_report.txt` - Text summary with assessment

## 🎯 Success Criteria

### Tracking Accuracy
- ✅ **Excellent**: Mean position error < 10 cm, orientation < 5°
- ✅ **Good**: Mean position error < 20 cm, orientation < 10°
- ⚠️ **Acceptable**: Mean position error < 50 cm, orientation < 20°
- ❌ **Needs Improvement**: Above thresholds

### Success Rate
- ✅ **Excellent**: > 90%
- ✅ **Good**: > 75%
- ⚠️ **Acceptable**: > 50%
- ❌ **Needs Improvement**: < 50%

## 📊 Metrics Glossary

### Position Error
- **Mean**: Average distance from target (cm)
- **Median**: Middle value (robust to outliers)
- **P95**: 95th percentile (captures worst-case)
- **P99**: 99th percentile (extreme cases)
- **Max**: Worst tracking error observed

### Orientation Error
- **Angular error**: Angle between actual and target orientation
- **Measured in**: Both radians and degrees
- **P95**: 95% of samples below this threshold

### Joint Metrics
- **Range**: Difference between max and min angle
- **Max velocity**: Peak joint speed (rad/s or deg/s)
- **P95 velocity**: 95th percentile speed

### Base Metrics
- **Linear velocity**: Forward (vx) and lateral (vy) motion
- **Angular velocity**: Rotation rate (ωz)
- **Max/P95**: Peak and typical speeds

## 🔧 Advanced Usage

### Custom Checkpoint

```powershell
.\scripts\launch_evaluation_quantitative.ps1 `
    -Checkpoint "logs/sb3/session_7d/final_model.zip" `
    -OutputDir "eval_session_7d" `
    -Headless
```

### High-Frequency Logging

```powershell
# Save every step (more data, slower):
.\scripts\launch_evaluation_quantitative.ps1 `
    -SaveEveryNSteps 1 `
    -NumEpisodes 50
```

### Trajectory-Specific Evaluation

```powershell
# Chassis-requiring trajectories only:
.\scripts\launch_evaluation_quantitative.ps1 `
    -UseChassisOnly `
    -NumEpisodes 200 `
    -Headless

# All trajectories (comprehensive):
.\scripts\launch_evaluation_quantitative.ps1 `
    -UseAllTrajectories `
    -NumEpisodes 500 `
    -Headless
```

## 📈 Interpreting Results

### Position Error Analysis
1. **Check mean and median**: Should be < 20 cm for good tracking
2. **Compare mean vs median**: Large difference indicates outliers
3. **Check P95**: Should be < 50 cm for reliable performance
4. **Inspect max**: Identifies worst-case scenarios

### Orientation Error Analysis
1. **Check mean**: Should be < 10° for good tracking
2. **Check P95**: Should be < 20° for reliable orientation
3. **Large errors**: May indicate gimbal lock or singularities

### Joint Usage Analysis
1. **Check range**: Should use full workspace effectively
2. **Check velocities**: Should not exceed hardware limits
3. **Symmetric usage**: Check if all joints are utilized

### Base Motion Analysis
1. **Lateral motion (vy)**: Should be non-zero for holonomic control
2. **Angular motion**: Should be smooth and purposeful
3. **Velocity limits**: Should respect robot constraints

### Reward Components
1. **Positive components**: Should dominate
2. **Negative penalties**: Should be small
3. **Balance**: Check if rewards align with task goals

## 🐛 Troubleshooting

### Evaluation is slow
- Increase `--num_envs` (64 recommended)
- Use `--headless` mode
- Reduce `--num_episodes` for quick tests
- Increase `--save_every_n_steps` (less frequent logging)

### Out of memory
- Reduce `--num_envs`
- Increase `--save_every_n_steps`
- Reduce `--num_episodes`

### Missing metrics
- Check if environment exposes required buffers
- Verify `extract_env_states()` function
- Enable debug mode to inspect data

### Plotting errors
```powershell
# Install dependencies:
pip install matplotlib seaborn pandas
```

## 📚 Related Documentation

- **Training**: `docs/TRAIN_ON_WINDOWS.md`
- **Deployment**: `deployment/DEPLOYMENT_CHECKLIST.md`
- **Architecture**: `docs/architecture/training_architecture.md`
- **Troubleshooting**: `docs/reference/troubleshooting.md`

## 🎓 Best Practices

### Evaluation Strategy
1. **Quick test** (20 episodes, 4 envs): Sanity check
2. **Medium test** (100 episodes, 32 envs): Initial assessment
3. **Full test** (200+ episodes, 64 envs): Comprehensive evaluation
4. **All trajectories** (500+ episodes): Final validation

### Data Management
- Save results with descriptive names
- Archive results before re-running
- Compare across training sessions
- Track improvements over time

### Analysis Workflow
1. Run evaluation → Save raw data
2. Generate plots → Visual inspection
3. Read report → Quantitative assessment
4. Compare with baselines → Track progress
5. Identify weaknesses → Guide improvements

## 🔬 Research Extensions

### Custom Metrics
Add new metrics in `EvaluationLogger`:
```python
# In log_step():
if 'custom_metric' in env_states:
    self.custom_metric_log.append(env_states['custom_metric'])
```

### Trajectory-Specific Analysis
Filter episodes by trajectory type:
```python
# In visualize_eval_results.py:
episodes_df[episodes_df['trajectory_type'] == 'dolly']
```

### Real-World Comparison
Compare sim vs real data:
```python
# Load both datasets and compare distributions
sim_data = load_evaluation_data('eval_sim.json')
real_data = load_evaluation_data('eval_real.json')
```

## 📞 Support

For questions or issues:
1. Check `docs/reference/troubleshooting.md`
2. Review evaluation logs
3. Inspect raw data files
4. Contact: See project README

---

**Last Updated**: October 29, 2025  
**Maintainer**: CinebotRL Team
