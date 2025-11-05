# Session 8b vs 8c: Configuration Comparison

## Quick Reference Table

| Parameter | Session 8b | Session 8c | Change | Rationale |
|-----------|-----------|-----------|---------|-----------|
| **REWARD WEIGHTS** |||||
| `position_tracking` | 150 | **200** | +33% | Ensure tracking remains dominant |
| `orientation_tracking` | 75 | **100** | +33% | Match position tracking boost |
| `reachability_maintenance_reward` | 50 | **100** | +100% | Make staying in workspace high priority |
| `base_progress_reward` | 400 | **450** | +12.5% | Encourage proactive movement ("carrot") |
| `excessive_base_movement_penalty` | 15 | **10** | -33% | Allow more base exploration |
| `velocity_limit_penalty` | 1.5 | **1.0** | -33% | Let base move faster when needed |
| `jerk_limit_penalty` | 0.01 | **0.005** | -50% | Allow agile movements |
| **REACHABILITY PENALTY** |||||
| Penalty curve | Linear | **Quadratic** | Sharpen | Make large distances prohibitively expensive |
| Formula | `-2(d-0.6)` | **`-2(d-0.6)²`** | Exponential | At 2.0m: -280 → -3,920 (14× harsher!) |
| **TRAINING SCHEDULE** |||||
| `num_envs` | 20,480 | **128-192** | -99% | Stronger per-env gradient, enable curriculum |
| `n_steps` | 128 | **96** | -25% | Better sample efficiency |
| `batch_size` | 4,096 | **2,048** | -50% | Match smaller rollout buffer |
| `n_epochs` | 10 | **4** | -60% | ~48 minibatches/update still sufficient |
| Training phases | 1 monolithic | **3 curriculum** | Phased | 40M easy + 60M medium + 100M full |
| Wall-clock time | ~12 hours | **~30-40 hours** | +3× | Trade speed for sample efficiency |
| **ENTROPY DECAY** |||||
| `decay_start_timestep` | 100M | **120M** | +20M | Delay to keep exploration high longer |
| `decay_duration_timesteps` | 100M | **80M** | -20M | Faster decay once started |
| Decay timing | 50% → 100% | **60% → 100%** | Later | Prevent premature exploitation |
| **KL DIVERGENCE** |||||
| `kl_warmup` | 0.25 | **0.15** | -40% | Tighter control from start |
| `kl_main` | 0.15 | **0.1** | -33% | Prevent policy oscillation |
| `kl_finetune` | 0.07 | **0.05** | -29% | More stable convergence |
| `target_kl` | 1.0 | **0.5** | -50% | Smaller policy updates |
| **PPO HYPERPARAMETERS** |||||
| `learning_rate` | 3e-4 | **3e-4** | Same | Proven good |
| `clip_range` | 0.2 | **0.2** | Same | Standard PPO |
| `clip_range_vf` | 1.0 | **0.3** | -70% | Stabilize critic during reward spikes |
| `normalize_advantage` | False (default) | **True** | Enabled | Normalize advantages for stable gradient signals |
| `gamma` | 0.99 | **0.99** | Same | Standard discount |
| `gae_lambda` | 0.95 | **0.95** | Same | Standard GAE |
| **CHECKPOINTING** |||||
| `save_freq` | 4M steps | **2M steps** | 2× more | Finer evaluation granularity |
| Total checkpoints | ~50 | **~100** | 2× more | Evaluate at 40M, 100M, 160M, 200M |
| **MONITORING** |||||
| Training callback | Basic | **Enhanced** | NEW | Log reachability stats, errors, velocities |
| Log frequency | Every iteration | **Every 5 iterations** | Detailed | Catch issues early |

## Performance Predictions

| Metric | Session 8b (Actual) | Session 8c (Target) | Expected Change |
|--------|---------------------|---------------------|-----------------|
| Mean position error | 238.5 cm | **100-150 cm** | -40% to -60% |
| Mean orientation error | 47.8° | **30-35°** | ~-30% |
| Reachability reward | **-135.21** ❌ | **+50 to +100** ✅ | Sign flip! |
| Base-target distance | 1.95m | **0.4-0.6m** | -69% to -77% |
| Episode reward (mean) | -11,081 | **+50,000** | +461% |
| Episode reward (median) | +56,199 | **+100,000** | +78% |
| Reward variance (std) | ±154,940 | **±50,000** | -68% (more consistent) |
| Base velocity | 0.34 m/s | **0.3-0.5 m/s** | Maintain mobility |
| Self-collisions | 0.0 | **0.0** | Maintain safety |

## Key Trade-offs

### Session 8b Advantages ✅
- **Fast wall-clock time**: 12.7 hours for 200M timesteps @ 4,427 FPS
- **High throughput**: 20,480 parallel envs
- **Simple workflow**: Single monolithic run, no checkpoints to manage

### Session 8b Disadvantages ❌
- **Poor reachability**: -135.21 reward (hugely negative)
- **Weak per-env signal**: Each env contributes 1/20,480 to gradient
- **No curriculum**: Can't adapt difficulty during training
- **Bimodal performance**: 50% excellent, 50% catastrophic

### Session 8c Advantages ✅
- **Proper reachability**: Expected positive reward (+50 to +100)
- **Strong per-env signal**: Each env contributes 1/128 to 1/192 to gradient
- **Curriculum learning**: Start easy, gradually increase difficulty
- **Better sample efficiency**: More learning per timestep
- **Tighter control**: Lower KL bounds prevent instability
- **Value clipping**: Stabilizes critic during reward spikes

### Session 8c Disadvantages ❌
- **Slow wall-clock time**: 30-40 hours for 200M timesteps
- **Lower throughput**: 128-192 envs vs 20,480
- **Complex workflow**: 3 phases, checkpoint management, evaluation between phases
- **Manual intervention**: Need to evaluate and decide whether to continue

## When to Use Each Approach

### Use Session 8b Style (Monolithic, High Throughput)
- ✅ Reward functions are well-tuned
- ✅ Policy is close to converged
- ✅ Just need more data
- ✅ Wall-clock time is critical
- ✅ GPU memory is abundant

### Use Session 8c Style (Curriculum, Low Throughput)
- ✅ Reward functions need tuning
- ✅ Policy needs to learn gradually
- ✅ Sample efficiency matters more than speed
- ✅ Want to evaluate at checkpoints
- ✅ GPU memory is limited

## Session 8c Usage Quick Reference

### Smoke Test (15-30 minutes)
```powershell
.\scripts\launch_session_8c.ps1 -Phase smoke
```

### Curriculum Training (30-40 hours total)
```powershell
# Phase 1: Easy (6-8 hours)
.\scripts\launch_session_8c.ps1 -Phase easy

# Evaluate at 40M, then:
# Phase 2: Medium (9-12 hours)
.\scripts\launch_session_8c.ps1 -Phase medium -Checkpoint <path>

# Evaluate at 100M, then:
# Phase 3: Full (15-20 hours)
.\scripts\launch_session_8c.ps1 -Phase full -Checkpoint <path>
```

### Complete Run (30-40 hours, no phases)
```powershell
.\scripts\launch_session_8c.ps1 -Phase complete
```

## Critical Success Factors

### For Session 8c to Succeed
1. ✅ **Reachability reward must go positive** (within first 50M timesteps)
2. ✅ **Base must stay within 0.3-0.6m** (TensorBoard: mean base-target distance)
3. ✅ **No catastrophic failures** (episode reward min > -50k, not -405k)
4. ✅ **Consistent performance** (reward std < ±50k, not ±155k)

### Red Flags to Watch For
1. ❌ **Reachability stays negative after 50M** → Increase weight or sharpen penalty
2. ❌ **Tracking accuracy degrades** → Increase tracking weights
3. ❌ **Base barely moves** → Increase base_progress_reward
4. ❌ **NaN values** → Lower learning rate, increase value clipping

## Next Steps

1. ✅ **Implementation complete** (all files modified)
2. 📋 **Run smoke test** (mandatory before full training)
3. 📋 **Launch Phase 1** (if smoke test validates config)
4. 📋 **Evaluate at checkpoints** (40M, 100M, 160M, 200M)
5. 📋 **Generate Session 8b vs 8c comparison** (after completion)
6. 📋 **Proceed to Session 9** (if metrics meet targets)

---

**Status**: Ready for smoke test  
**Estimated Time**: 15-30 min smoke test → 6-8 hours Phase 1 → Evaluate → Continue or adjust  
**Key Command**: `.\scripts\launch_session_8c.ps1 -Phase smoke`
