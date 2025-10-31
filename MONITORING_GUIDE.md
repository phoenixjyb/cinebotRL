# Session 8c Monitoring Guide

Training status: Session 8c-v2 completed 200M steps on October 31, 2025. The updated reward shaping is active, but evaluation shows the policy still struggles with base positioning (mean EE error 3.28 m). Use this guide to launch new runs and track the critical signals that verify the fixes.

---

## Launch Checklist
- Start training from I:\isaaclab using the existing launcher:
  ```powershell
  .\scripts\launch_session_8c.ps1 -Phase complete
  ```
- In a separate terminal, attach the monitoring script to the latest log directory:
  ```powershell
  python scripts/monitoring/check_training_progress.py --log_dir logs/sb3/mobilemmtrackee_v0/<timestamp> --watch
  ```
- Start TensorBoard for deeper inspection:
  ```powershell
  tensorboard --logdir logs/sb3
  ```
  Open http://localhost:6006 and filter by the `monitoring/*`, `reward_components/*`, and `train/*` scalars.

---

## Monitoring Channels

### 1. TrainingMonitorCallback (train.py)
The callback prints an overview every five rollouts and mirrors the numbers to TensorBoard. Each block contains:
- `reward_components/*`: mean value per episode for the dominant terms
  - Position/orientation tracking should dominate the positive side.
  - `reachability_bonus` should trend upward as the base keeps targets inside the soft margin.
  - `reachability_distance_penalty`, `target_distance_penalty`, and `position_distance_penalty` should shrink once the chassis mobilises.
  - `base_overshoot_penalty` should remain below ~5 on average; sustained values above 10 indicate overshoot oscillations.
- `[Base-Target Distance]` summary with zone percentages:
  - Optimal (<0.40 m) should exceed 40% by mid-training.
  - Acceptable (0.40-0.60 m) should cover the remaining reachable cases.
  - Unreachable (>0.60 m) must stay below 15%; anything above 25% signals the chassis is not catching up.
- `[Workspace Distance]` report based on the voxel map:
  - `workspace_soft_exceed_pct`: fraction outside the soft margin (0.20 m). Aim for <40%.
  - `workspace_hard_exceed_pct`: fraction beyond the hard margin (0.60 m). Aim for <10%.
- `[Base Movement]` to track planar speed and yaw rate. Expect mean planar speed in the 0.25-0.35 m/s band once the curriculum reaches full amplitude.

If a run does not emit these blocks, confirm that `TrainingMonitorCallback` is active and that the Isaac environment exposes the new buffers (`_workspace_distance_buf`, `_last_reachability_stats`).

### 2. check_training_progress.py
This command-line tool provides a read-only dashboard from `progress.csv` without touching the training process. Key thresholds:
- Explained variance >= 0.75 => critic is trustworthy. 0.65-0.75 requires monitoring. <0.65 means pause and restart with curriculum.
- Approximate KL < 0.03 => PPO updates are stable. Higher values usually come from entropy decay being disabled or the KL schedule being overridden.
- Clip fraction < 0.20 => policy updates remain in range.
- Base/Workspace metrics (if present) replicate the callback percentages for offline inspection.
- The script flags iteration 58 (~120M steps) so you can verify the entropy decay callback actually walks `ent_coef` toward the final value (1e-4 by default).

### 3. TensorBoard Scalars
TensorBoard remains the fastest way to spot trends. Watch:
- `train/explained_variance`, `train/approx_kl`, `train/std`, `train/entropy_loss`
- `monitoring/base_target_dist_mean`, `monitoring/optimal_zone_pct`, `monitoring/unreachable_zone_pct`
- `monitoring/workspace_distance_mean`, `monitoring/workspace_hard_exceed_pct`
- `reward_components/reachability_bonus`, `reward_components/reachability_distance_penalty`, `reward_components/target_distance_penalty`
- `reward_components/base_mobilization`, `reward_components/base_target_alignment`

Overlay Session 8b vs new runs to confirm that the bonuses/penalties behave as expected.

---

## Reachability Shaping Reference
The reward now separates gentle incentives from hard penalties:
- **Soft margin bonus** (`reachability_bonus`):
  - Computed from the workspace distance buffer.
  - Full credit when the target lies inside the 0.20 m margin. Decays linearly toward zero as the target approaches the hard boundary.
- **Hard margin penalty** (`reachability_distance_penalty`):
  - Activates when the workspace distance exceeds 0.60 m. Scales with the squared excess distance, weighted by `reachability_distance_weight` (80 by default).
- **Fallback distance penalty** (`position_distance_penalty`):
  - Linear term based on EE error to keep gradients informative when the exponential tracking reward saturates at large distances.
- **Legacy target distance penalty** (`target_distance_penalty`):
  - Still active with a low weight (1.0) to penalise static bases that ignore unreachable targets. The 90% moving discount is now respected.
- **Base mobilisation incentives** (`base_mobilization`, `base_target_alignment`):
  - Reward improvements in base-to-target distance that are directly attributable to the chassis motion.
  - Cap progress to 0.35 m per step to prevent reward spikes.

Interpretation tips:
- A healthy run shows `reachability_bonus` moving toward +20 to +40 while `reachability_distance_penalty` settles near zero.
- If `reachability_distance_penalty` remains hundreds of points negative, inspect `monitoring/unreachable_zone_pct` and the mobilization rewards; low mobilisation (~0.1) means the base is not closing the gap.
- Use the workspace hard-margin exceed metric to pick trajectories that consistently break reachability; these may need curriculum adjustment.

---

## Session 8c-v2 Post-Mortem (200M steps)
Quantitative evaluation (`evaluation_results/session_8cv2_200M/eval_summary_20251031_080603.json`) reveals the current status:
- Mean position error 3.28 m (95th percentile 4.69 m) => base rarely reaches the trajectory.
- Mean orientation error 20.5 deg => significant improvement versus Session 8b (47.8 deg).
- `reachability_bonus` average -1207 => the policy spends most time outside the workspace.
- Mobilisation rewards remain tiny (0.11 for `base_mobilization`, 0.004 for `base_target_alignment`), while `base_overshoot_penalty` averages 8.4.
- Target distance penalty sits at 1.89, indicating the legacy term is active but not sufficient to recover reachability.

**Implications for the next run**:
1. Continue logging workspace and base distance metrics; the fixes are in place, but the policy has not yet exploited them.
2. For 8c-v3, start from scratch so VecNormalize stats and reward shaping begin aligned with the new design.
3. Analyse trajectories with high hard-margin exceedance. Consider filtering any that require simultaneous large base and arm moves beyond the map coverage.
4. Prepare comparison plots using `scripts/reinforcement_learning/sb3/compare_sessions.py` to measure improvements after each batch of fixes.

---

## Evaluation Workflow
- Quantitative check (10-20 episodes):
  ```powershell
  python scripts/reinforcement_learning/sb3/evaluate_quantitative.py --checkpoint <path> --num_episodes 20 --headless
  ```
- Plot full evaluation results for deeper insight:
  ```powershell
  python scripts/reinforcement_learning/sb3/visualize_eval_results.py --input <eval_summary.json> --output_dir evaluation_plots/session_8cv3_dryrun
  ```
- Compare multiple checkpoints or sessions:
  ```powershell
  python scripts/reinforcement_learning/sb3/compare_sessions.py --session evaluation_results/session_8b_200M/eval_summary_*.json --session evaluation_results/session_8cv2_200M/eval_summary_*.json
  ```

Keep this guide nearby during live training; it lists every scalar the new monitoring stack exposes so you can quickly diagnose regressions.
