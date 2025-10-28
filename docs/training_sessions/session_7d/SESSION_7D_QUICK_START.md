# Session 7d Training - Quick Start Guide

## ⚡ TL;DR

**Accelerated training configuration: 8192 environments, ~11 hours (vs 22 hours)**

```powershell
# Start training:
.\scripts\launch_session_7d_accelerated.ps1
```

---

## 📋 Pre-Flight Checklist

Before starting the 11-hour training run:

### ✅ System Requirements
- [x] RTX 3090 with 24GB VRAM
- [x] Isaac Lab installed at `I:\isaaclab`
- [x] Project at `C:\Users\yanbo\wSpace\cinebotRL`
- [ ] No other GPU-intensive processes running
- [ ] Stable power supply (UPS recommended for 11-hour run)

### ✅ Code Status
- [x] Session 7d reward tuning implemented (`config.py`, `rewards.py`)
- [x] Lazy marker initialization (non-blocking issue)
- [x] Git changes committed
- [ ] Optional: Create git tag before training

### ✅ Training Configuration
- [x] 8192 environments (2x Session 7c)
- [x] 200M total timesteps
- [x] Hyperparameters validated
- [x] VRAM estimate: ~24-28GB (fits with management)

---

## 🚀 Start Training

### Option 1: Direct Launch (Recommended)
```powershell
cd C:\Users\yanbo\wSpace\cinebotRL
.\scripts\launch_session_7d_accelerated.ps1
```

The script will:
1. Verify paths (Isaac Lab, project)
2. Show configuration summary
3. Ask for confirmation
4. Start training (~11 hours)

### Option 2: Manual Command
```powershell
cd I:\isaaclab

$env:GYMNASIUM_DISABLE_PLUGIN_ENTRYPOINTS = "1"

.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py `
    --task MobileMMTrackEE-v0 `
    --num_envs 8192 `
    --headless `
    --total_timesteps 200000000 `
    --learning_rate 0.0003 `
    --n_steps 64 `
    --batch_size 1024 `
    --n_epochs 10 `
    --ent_coef 0.01 `
    --clip_range 0.2 `
    --gamma 0.99 `
    --gae_lambda 0.95 `
    --enable_entropy_decay `
    --final_ent_coef 0.0001 `
    --decay_start_timestep 100000000 `
    --decay_duration_timesteps 50000000 `
    --save_freq 2048000
```

---

## 📊 Monitoring During Training

### TensorBoard (Real-time Metrics)
```powershell
# In a separate PowerShell window:
cd C:\Users\yanbo\wSpace\cinebotRL

# Find the latest run:
$latest = Get-ChildItem logs\sb3\MobileMMTrackEE-v0 | Sort-Object LastWriteTime -Descending | Select-Object -First 1

# Launch TensorBoard:
tensorboard --logdir="logs\sb3\MobileMMTrackEE-v0\$($latest.Name)"

# Open browser to: http://localhost:6006
```

### Key Metrics to Watch

| Metric | Expected Behavior | Warning Signs |
|--------|-------------------|---------------|
| **train/loss** | Decreasing, smooth | Spikes, NaN, diverges |
| **train/entropy_loss** | Decreases after 100M steps (decay) | Stays constant (no decay) |
| **rollout/ep_rew_mean** | Increases from ~0 to 30-50 | Stays negative/flat |
| **rollout/reachability** | Increases from 6% to 30-50% | Stays <10% |
| **rollout/mean_ee_error** | Decreases from 1.01m to <0.70m | Stays >1.0m |
| **train/fps** | 40k-80k timesteps/sec | <20k (slow) |

### GPU Monitoring
```powershell
# Watch GPU usage:
nvidia-smi -l 5  # Update every 5 seconds

# Watch for:
# - GPU Util: >90% (good)
# - Memory: 20-23GB (good), >24GB (warning!)
# - Temperature: <85°C (good)
```

---

## ⏱️ Training Timeline

| Timestep | Duration | Event | Action |
|----------|----------|-------|--------|
| 0M | 0:00 | Start | Monitor VRAM for first 10 min |
| 25M | ~1.5h | Early progress | Check reachability improving |
| 50M | ~3h | Mid-early | Verify no NaN/divergence |
| 100M | ~5.5h | Midpoint | Entropy decay starts |
| 150M | ~8h | Late training | Should see plateau in rewards |
| 200M | ~11h | **Complete** | Evaluate results |

---

## 🎯 Success Criteria

Training is **successful** if:

✅ **Completes without crashes**
- No CUDA OOM errors
- No NaN/Inf in losses
- Checkpoints saved regularly

✅ **Reachability improves**
- Baseline (Session 7c): 6%
- Target: **30-50%**
- Acceptable: >15%

✅ **Tracking error decreases**
- Baseline (Session 7c): 1.01m mean error
- Target: **<0.70m**
- Acceptable: <0.85m

✅ **Base mobilization works**
- Base velocity rewards: 5-20 pts (was 0-2 in 7c)
- Base moves to reach targets beyond arm reach

---

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory
```
Solution: Reduce environments
# Edit scripts/launch_session_7d_accelerated.ps1
NumEnvs = 7168  # Or 6144 for more safety

# Restart training (will take longer)
```

### Issue: Training Unstable (Loss Spikes)
```
Solutions:
1. Reduce learning rate: --learning_rate 0.0001
2. Reduce batch size: --batch_size 768
3. Increase n_steps: --n_steps 96
```

### Issue: FPS Too Low (<20k)
```
Check:
1. Other GPU processes: tasklist | findstr cuda
2. CPU bottleneck: Task Manager → Performance
3. Disk I/O: Check if SSD is full
```

### Issue: Reachability Stays Low
```
This is expected early on (<50M steps).
If still <10% after 100M steps:
- Check base velocity rewards in logs
- May need to extend training beyond 200M
```

---

## 📁 Output Files

After training completes:

```
logs/sb3/MobileMMTrackEE-v0/<timestamp>/
├── events.out.tfevents.*  # TensorBoard logs
├── checkpoints/
│   ├── rl_model_2048000_steps.zip
│   ├── rl_model_4096000_steps.zip
│   └── ... (every ~2M steps)
└── config.json  # Hyperparameters used
```

**Final checkpoint**: `logs/sb3/MobileMMTrackEE-v0/<timestamp>/checkpoints/rl_model_200000000_steps.zip`

---

## 🔄 After Training

### 1. Evaluate Performance
```powershell
# Run evaluation script:
python scripts/evaluate_policy.py `
    --checkpoint logs/sb3/MobileMMTrackEE-v0/<timestamp>/checkpoints/rl_model_200000000_steps.zip `
    --num_episodes 100 `
    --save_results results/session_7d_eval.json
```

### 2. Compare with Session 7c
```powershell
# Generate comparison report:
python scripts/compare_sessions.py `
    --session_7c logs/sb3/MobileMMTrackEE-v0/<7c_timestamp>/ `
    --session_7d logs/sb3/MobileMMTrackEE-v0/<7d_timestamp>/ `
    --output docs/session_7d_results.md
```

### 3. Visualize (Optional)
```powershell
# If reachability improved significantly:
python scripts/visualize_policy.py `
    --checkpoint logs/sb3/MobileMMTrackEE-v0/<timestamp>/checkpoints/rl_model_200000000_steps.zip `
    --num_envs 1  # Single env with GUI
```

---

## 💾 Backup Recommendations

Before starting 11-hour training:

```powershell
# 1. Commit current state
git add -A
git commit -m "Session 7d: Pre-training checkpoint (8192 envs, reward tuning)"
git push

# 2. (Optional) Create checkpoint tag
git tag -a session-7d-start -m "Session 7d accelerated training start"
git push --tags
```

After training completes:

```powershell
# 1. Backup checkpoints (large files!)
robocopy logs\sb3\MobileMMTrackEE-v0\<timestamp>\checkpoints `
         H:\backups\session_7d_checkpoints\ /E

# 2. Commit results summary
git add docs/session_7d_results.md
git commit -m "Session 7d: Training complete, results documented"
git push
```

---

## 📞 Need Help?

**Common Questions:**

**Q: Can I stop training early?**  
A: Yes! Checkpoints are saved every ~2M steps. Latest checkpoint can be used for evaluation.

**Q: Can I resume if training crashes?**  
A: Yes! Use the latest checkpoint:
```powershell
.\scripts\launch_session_7d_accelerated.ps1 `
    --checkpoint logs/sb3/.../checkpoints/rl_model_XXXXXX_steps.zip
```

**Q: What if 8192 envs don't fit in VRAM?**  
A: Edit `NumEnvs = 6144` in the launch script. Training will take ~15 hours instead of 11.

---

**Ready to start?**

```powershell
.\scripts\launch_session_7d_accelerated.ps1
```

Good luck! 🚀
