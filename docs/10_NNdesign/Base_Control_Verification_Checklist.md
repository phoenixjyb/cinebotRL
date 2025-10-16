# Base Control Fix - Verification Checklist

## Pre-Training Verification

### ✅ Code Changes Confirmed

- [x] **Fix 1**: Base joint lookup by name (safety improvement)
  - File: `src/rl_platform/tasks/mobile_mm/env.py`
  - Joint names: `["joint_x", "joint_y", "joint_theta"]`
  - Lookup method: Name-based (robust against URDF changes)

- [x] **Fix 2**: Velocity-to-position integration with differential drive
  - File: `src/rl_platform/tasks/mobile_mm/env.py`
  - Control method: `set_joint_position_target()` (CORRECT for PPR joints)
  - Integration: Reads current positions, applies differential drive kinematics
  - Formula: `dx = vx*cos(θ)*dt`, `dy = vx*sin(θ)*dt`, `dθ = wz*dt`

- [x] **Fix 3**: Correct observation indices for arm joints
  - File: `src/rl_platform/tasks/mobile_mm/observations.py`
  - Changed: `joint_pos` → `joint_pos[:, 3:9]` (only arm joints)
  - Changed: `joint_vel` → `joint_vel[:, 3:9]` (only arm joints)
  - Dimensions: 6 arm joints × 2 (pos+vel) = 12 dims ✅

---

## What to Check During Training

### Visual Verification (GUI Mode)

Run training with visualization:
```powershell
cd I:\isaaclab
.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py
```

**Expected Behavior**:
- ✅ Base moves smoothly in XY plane
- ✅ Robot follows trajectory without tipping
- ✅ Differential drive: forward/backward + rotation
- ✅ Green sphere (end-effector) tracks red spheres (targets)
- ✅ Stable, planar motion (no extreme tilting)

**Red Flags**:
- ❌ Base still not moving → Check base_vx/base_wz in actions
- ❌ Robot tipping over → Check control magnitude/limits
- ❌ Erratic motion → Check integration timestep (dt)

---

### Log Verification

**Check base joint targets are changing**:
```python
# Should see in logs:
[MobileMMTrackEE] Base joint IDs initialized: [0, 1, 2]
[MobileMMTrackEE] Base joint names: ['joint_x', 'joint_y', 'joint_theta']
```

**Monitor TensorBoard**:
```powershell
tensorboard --logdir=logs/sb3/mobile_mm
```

**Key Metrics to Watch**:
1. **Tracking Error** (should decrease over time):
   - Position error: Target ~0.05m or less
   - Orientation error: Should improve steadily

2. **Mean Reward** (should increase):
   - Initial: ~-100 to -50 (poor tracking)
   - After learning: Approaching 0 or positive (good tracking)

3. **Episode Length** (should increase):
   - Initial: May terminate early due to failures
   - After learning: Reaching max episode length (500 steps)

4. **Value Function Loss** (should stabilize):
   - Should decrease and stabilize after initial exploration

---

### Debugging Commands

**If base still not moving**:
```python
# Add debug prints in env.py _pre_physics_step():
print(f"Base vx: {base_vx.mean().item():.4f}, wz: {base_wz.mean().item():.4f}")
print(f"Position deltas: dx={dx.mean().item():.4f}, dy={dy.mean().item():.4f}, dtheta={dtheta.mean().item():.4f}")
print(f"New targets: {new_base_targets[0].cpu().numpy()}")
```

**If robot tips over**:
```python
# Check if actions are too large:
print(f"Action range: vx=[{base_vx.min():.2f}, {base_vx.max():.2f}], wz=[{base_wz.min():.2f}, {base_wz.max():.2f}]")
```

**If observations seem wrong**:
```python
# Check observation dimensions:
obs = env.get_observations()
print(f"Observation shape: {obs['policy'].shape}")  # Should be [2048, 70]
print(f"Arm joint pos range: {arm_joint_pos.min():.2f} to {arm_joint_pos.max():.2f}")
```

---

## Expected Training Improvements

### Before Fixes
- ❌ Base never moved (velocity targets ignored)
- ❌ Robot tipped over (no stabilization)
- ❌ Poor tracking (limited to arm-only reach)
- ❌ Low rewards (~-200 to -100)
- ❌ Network couldn't learn proper control

### After Fixes
- ✅ Base moves with differential drive
- ✅ Stable planar motion
- ✅ Full workspace accessibility
- ✅ Better tracking (base + arm coordination)
- ✅ Higher rewards (can reach targets)
- ✅ Faster convergence (correct state feedback)

---

## Training Command (With Visualization)

```powershell
# Full training with GUI
cd I:\isaaclab
.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py

# Headless training (faster)
.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py --headless

# Resume from checkpoint
.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py --checkpoint logs/sb3/mobile_mm/YYYY-MM-DD_HH-MM-SS/checkpoints/model_XXXXX_steps.zip
```

---

## Success Criteria

### Immediate (First 1000 Iterations)
- [ ] Base is moving (visually confirmed)
- [ ] Robot remains stable (no tipping)
- [ ] Tracking error starts decreasing
- [ ] No NaN values in logs

### Short-term (10K Iterations)
- [ ] Mean reward improving (trending upward)
- [ ] Tracking error < 0.1m consistently
- [ ] Robot completing full episodes
- [ ] Coordinated base + arm movements

### Long-term (100K+ Iterations)
- [ ] Tracking error < 0.05m
- [ ] Smooth trajectory following
- [ ] Mean reward > -20
- [ ] Consistent performance across environments

---

## Rollback Plan (If Needed)

If fixes cause unexpected issues:

```powershell
# Revert to previous commit
git log --oneline -5  # Find previous commit hash
git revert d814b27   # Revert base control fix
git revert HEAD      # Or revert last commit

# Or create hotfix branch
git checkout -b hotfix/base-control
# Make adjustments
git commit -am "hotfix: Adjust base control parameters"
```

---

## Next Steps After Verification

1. **If base moves correctly**:
   - ✅ Let training run for 1M steps
   - Monitor TensorBoard metrics
   - Save checkpoints every 50K steps
   - Document final performance

2. **If issues found**:
   - Add debug prints to identify problem
   - Check dt, velocity ranges, position limits
   - Verify joint_ids are correct
   - Test with single environment first

3. **After successful training**:
   - Document final tracking performance
   - Create evaluation script
   - Test on different trajectories
   - Consider deploying to real robot

---

**Status**: ✅ All fixes implemented and committed
**Commit**: `d814b27` - "fix: Align control signals with state feedback for base and arm"
**Ready**: To test training with proper base control
**Date**: 2025-10-16
