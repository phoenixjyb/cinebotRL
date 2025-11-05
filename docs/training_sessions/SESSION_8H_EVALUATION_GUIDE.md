# Session 8h Checkpoint Evaluation Guide

This guide explains how to evaluate Session 8h checkpoints and compare them against Session 8f/8g baselines.

## Quick Start

### Full Evaluation (Recommended)
Evaluate all 5 key checkpoints (20M, 40M, 60M, 80M, 100M) with 200 episodes each:

```powershell
.\scripts\launch_session_8h_evaluation.ps1 -Headless
```

**Expected duration**: ~2.5 hours (30 minutes per checkpoint)

### Quick Test
Test evaluation with fewer checkpoints and episodes:

```powershell
.\scripts\launch_session_8h_evaluation.ps1 -Quick
```

This evaluates 20M, 40M, 100M with 50 episodes each (~20 minutes total).

### Custom Evaluation
Evaluate specific checkpoints with custom settings:

```powershell
# Evaluate only 40M and 100M checkpoints
.\scripts\launch_session_8h_evaluation.ps1 `
    -Checkpoints "40M","100M" `
    -NumEpisodes 100 `
    -Headless

# Full evaluation with more episodes for better statistics
.\scripts\launch_session_8h_evaluation.ps1 `
    -NumEpisodes 300 `
    -Headless
```

## Evaluation Metrics

The script tracks and reports the following metrics:

### Primary Metrics (Comparison Focus)
- **Position Tracking Error** (cm): Mean distance between end-effector and target trajectory
  - Target: <300 cm
  - Baselines: Session 8f = 308 cm, Session 8g @ 40M = 301 cm

- **Orientation Tracking Error** (°): Mean angular error in end-effector orientation
  - Target: <60°
  - Baselines: Session 8f = 46.5°, Session 8g @ 40M = 130°

- **Workspace Violations** (%): Percentage of time robot violates workspace limits
  - Target: <5%

### Secondary Metrics
- Episode reward (mean, std, min, max)
- Episode length (mean, std)
- Position error statistics (median, p95, std)
- Orientation error statistics (median, p95, std)

## Checkpoints to Evaluate

Session 8h has 5 key checkpoints representing different training phases:

| Checkpoint | Steps | Purpose | Expected Performance |
|------------|-------|---------|---------------------|
| 20M | 20,054,016 | Early training baseline | Position: 400-500 cm, Orientation: 80-120° |
| 40M | 40,206,336 | Pre-curriculum transition | Position: 280-320 cm, Orientation: 50-90° |
| 60M | 60,063,744 | Post-curriculum transition | Position: 260-300 cm, Orientation: 45-70° |
| 80M | 80,019,456 | Late-stage convergence | Position: 250-290 cm, Orientation: 45-65° |
| 100M (final) | 100,000,000+ | Final performance | Position: 250-300 cm, Orientation: 45-80° |

### Why These Checkpoints?
- **20M**: Early baseline to show initial learning progress
- **40M**: Compare with Session 8g's last stable checkpoint before collapse
- **60M**: Evaluate post-curriculum-transition performance (gradual 45-55M transition)
- **80M**: Late-stage convergence, potential best performance
- **100M**: Final model after full training

## Comparison Baselines

### Session 8f (100M steps)
- Position error: **308 cm**
- Orientation error: **46.5°**
- Status: ✅ Completed, stable
- Configuration: Instant curriculum transition @ 50M

### Session 8g @ 40M
- Position error: **301 cm**
- Orientation error: **130°**
- Status: ⚠️ Last stable checkpoint before KL divergence collapse

### Session 8g @ 100M
- Status: ❌ Collapsed (KL divergence spike, unstable policy)
- Metrics: Invalid (policy degraded)

## Output Files

Results are saved to `evaluation_results/session_8h_comparison/`:

```
evaluation_results/session_8h_comparison/
├── session_8h_comparison_20251104_HHMMSS.json  # Raw evaluation data
└── summary_table.txt                            # Quick reference table
```

### JSON Output Structure
```json
{
  "checkpoint_name": "Session 8h @ 40M",
  "checkpoint_path": "logs/.../ppo_mobile_mm_40206336_steps.zip",
  "num_episodes": 200,
  "episode_reward": {
    "mean": 1234.5,
    "std": 123.4,
    "min": 900.0,
    "max": 1500.0
  },
  "position_error": {
    "mean_cm": 285.3,
    "median_cm": 275.1,
    "std_cm": 45.2,
    "p95_cm": 350.8
  },
  "orientation_error": {
    "mean_deg": 52.1,
    "median_deg": 48.3,
    "std_deg": 15.4,
    "p95_deg": 75.2
  },
  "workspace_violations": {
    "rate_percent": 2.5,
    "count": 5
  }
}
```

## Interpretation Guide

### Position Error Analysis
- **<250 cm**: Excellent tracking, competitive with Session 8f
- **250-300 cm**: Good tracking, meets target performance
- **300-350 cm**: Acceptable, similar to baseline
- **>350 cm**: Poor tracking, investigate issues

### Orientation Error Analysis
- **<45°**: Excellent orientation control
- **45-60°**: Good orientation, target range
- **60-90°**: Acceptable orientation
- **>90°**: Poor orientation control, likely chassis-angle dominance

### Curriculum Transition Impact
Compare 40M → 60M → 80M to evaluate gradual curriculum transition:
- Ideal: Smooth improvement or plateau (no performance drop)
- Session 8g (instant transition): Sharp degradation @ 50M
- Session 8h (gradual 45-55M): Expected smooth transition

### Best Checkpoint Selection
The best checkpoint is typically:
1. **80M**: Late-stage convergence often yields best balance
2. **60M**: Post-curriculum transition, if curriculum improved performance
3. **100M**: Final model, if training remained stable

Compare all checkpoints before selecting the best model for deployment.

## Troubleshooting

### Issue: "Session 8h directory not found"
**Solution**: Update `$Session8hDir` in launcher script to correct path.

```powershell
# Check actual path
Get-ChildItem logs\sb3\mobilemmtrackee_v0\

# Update in launch_session_8h_evaluation.ps1 if needed
```

### Issue: "No checkpoints found"
**Solution**: Verify checkpoint files exist:

```powershell
Get-ChildItem logs\sb3\mobilemmtrackee_v0\20251103_235918\checkpoints\*.zip | Select-Object Name, Length, LastWriteTime
```

### Issue: Evaluation crashes on specific checkpoint
**Solution**: Skip problematic checkpoint and continue:

```powershell
# Evaluate only working checkpoints
.\scripts\launch_session_8h_evaluation.ps1 `
    -Checkpoints "20M","40M","80M" `
    -Headless
```

### Issue: Metrics not in output
**Solution**: This likely means the environment's `info` dict doesn't contain tracking errors. The evaluation script will still report episode rewards and lengths.

## Next Steps After Evaluation

1. **Review Results**: Check comparison table and identify best checkpoint
2. **Analyze Trends**: Plot position/orientation errors over training (20M→100M)
3. **Curriculum Impact**: Compare 40M vs 60M to evaluate gradual transition
4. **Select Best Model**: Choose checkpoint with best position/orientation balance
5. **Export for Deployment**: Convert best checkpoint to ONNX for robot deployment

## Related Documentation

- [Session 8h Training Results](docs/training_sessions/SESSION_8H_RESULTS.md)
- [Session 8h Implementation](SESSION_8H_IMPLEMENTATION.md)
- [Deployment Guide](deployment/DEPLOYMENT_GUIDE.md)
- [Training Sessions Master Log](TRAINING_SESSIONS_MASTER_LOG.md)
