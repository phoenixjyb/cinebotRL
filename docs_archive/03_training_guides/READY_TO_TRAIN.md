# 🚀 Ready to Train - Command Reference

## ✅ Changes Applied

**Code Changes:**
- ✅ clip_range_vf=1.0 (stabilize critic)
- ✅ ent_coef=0.01 (enable exploration)
- ✅ target_kl=0.01 (prevent large policy changes)
- ✅ Committed to git (commit f137fd9)

**Already Enabled:**
- ✅ Reward normalization (norm_reward=True)
- ✅ All 5 bug fixes from yesterday

---

## 🎯 Quick Start

### Option 1: Test Run First (RECOMMENDED - 5-10 mins)

```powershell
cd I:\isaaclab
.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py `
  --task MobileMMTrackEE-v0 `
  --num_envs 64 `
  --total_timesteps 500000 `
  --n_steps 4096 `
  --batch_size 256 `
  --headless
```

**What this tests:**
- New hyperparameters work
- No crashes
- Metrics improving

**Watch for:**
- Explained variance stays positive
- Clip fraction decreases
- No errors in first few iterations

---

### Option 2: Full Training Run (30-40 mins)

```powershell
cd I:\isaaclab
.\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py `
  --task MobileMMTrackEE-v0 `
  --num_envs 1024 `
  --total_timesteps 5000000 `
  --n_steps 4096 `
  --batch_size 256 `
  --headless
```

**When to use:**
- After test run succeeds
- Ready for full training
- Have 40 minutes

---

## 📊 Monitor Training (Optional - Open in 2nd Terminal)

```powershell
# Terminal 2:
cd I:\isaaclab
.\isaaclab.bat -p -m tensorboard --logdir H:\wSpace\cinebotRL\logs\sb3
```

Then open browser: **http://localhost:6006**

**Key metrics to watch:**
1. **rollout/ep_rew_mean** - Should increase
2. **train/explained_variance** - Should be 0.3-0.8 (not negative!)
3. **train/clip_fraction** - Should be 5-10%
4. **train/value_loss** - Should increase from 0.0005
5. **train/entropy_loss** - Should gradually decrease

---

## 📈 What Changed from Yesterday

| Parameter | Yesterday | Today | Why |
|-----------|-----------|-------|-----|
| **n_steps** | 2048 | 4096 | More data per update → stable critic |
| **batch_size** | 512 | 256 | More updates per epoch → faster learning |
| **ent_coef** | 0.0 | 0.01 | Enable exploration bonus |
| **clip_range_vf** | - | 1.0 | Prevent large critic jumps |
| **target_kl** | - | 0.01 | Early stop if policy changes too much |

---

## ✅ Success Criteria

### Test Run (500K steps):
- [ ] Completes without crashes
- [ ] Explained variance > 0
- [ ] Clip fraction < 15%
- [ ] Value loss increasing

### Full Run (5M steps):
- [ ] Explained variance: 0.5-0.8
- [ ] Clip fraction: 5-10%
- [ ] Episode reward: steady increase
- [ ] Episode length: >800 steps

---

## 🎬 Recommended: Start with Test Run

**Copy-paste this:**
```powershell
cd I:\isaaclab; .\isaaclab.bat -p C:\Users\yanbo\wSpace\cinebotRL\scripts\reinforcement_learning\sb3\train.py --task MobileMMTrackEE-v0 --num_envs 64 --total_timesteps 500000 --n_steps 4096 --batch_size 256 --headless
```

**Then check results after ~5 minutes and decide:**
- ✅ Good metrics → Scale up to full run
- ⚠️ Issues → Review TensorBoard and adjust

---

## 📝 Next Steps After Training

1. **Review TensorBoard metrics**
2. **Evaluate best policy:**
   ```powershell
   .\scripts\visualize_policy.ps1 -Checkpoint "path/to/best_model.zip"
   ```
3. **Commit results:**
   ```bash
   git add .
   git commit -m "results: Successful training with tuned hyperparameters"
   git push origin train-windows
   ```

---

## 🆘 If Something Goes Wrong

**Training crashes:**
- Check log files in `H:\wSpace\cinebotRL\logs\sb3\`
- Review error message
- May need to adjust batch_size lower (128)

**Metrics not improving:**
- Let it run for at least 100K steps first
- Check TensorBoard for trends
- May need to adjust reward structure

**System issues:**
- GPU memory: Reduce num_envs (512 instead of 1024)
- CUDA errors: Usually harmless warnings, ignore
- Isaac Sim crash: Restart and try again

---

## 💡 Pro Tips

1. **Start small:** Test run first to verify changes work
2. **Monitor early:** First 100K steps tell you a lot
3. **Be patient:** Critic learning with normalized rewards takes time
4. **Don't interrupt:** Let full run complete (checkpoints save progress)
5. **Check TensorBoard:** Visual metrics are easier to interpret

---

**Ready to train? Start with the test run! 🚀**

