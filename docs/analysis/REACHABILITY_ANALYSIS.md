# Reachability Shaping Analysis - Session 8c-v2

**Date:** October 31, 2025  
**Context:** Session 8c-v2 finished 200M steps with the updated reachability shaping. The fixes (soft/hard margins, base mobilisation cap, VecNormalize restore) are in place, yet evaluation shows the robot still fails to keep the trajectory inside the reachable workspace. This report restates the design intent, compares it with the data, and lists next actions for Session 8c-v3.

---

## 1. What the Reachability Metric Measures
- `workspace_distance` is the voxel distance between the current EE target and the arm reach map. It is zero when the point is well inside the workspace and grows as the target moves outward.
- Two thresholds define the behaviour:
  - **Soft margin (0.20 m)**: inside this band the robot receives a positive bonus.
  - **Hard margin (0.60 m)**: beyond this band the policy receives a quadratic penalty.
- The reward decomposition in `rewards.compute_combined_reward` mirrors this logic:
  - `reachability_bonus = reachability_maintenance_reward * bonus_factor`, where `bonus_factor = clamp(1 - distance / soft_margin, 0, 1)`.
  - `reachability_distance_penalty = reachability_distance_weight * max(distance - hard_margin, 0)^2`.
  - `position_distance_penalty` supplements the exponential tracking reward with a linear term so gradients survive even when the EE is metres away.
  - `target_distance_penalty` keeps the legacy base-to-target signal with a 90% discount while the base is moving.
- Mobilisation rewards now isolate chassis contribution:
  - `base_mobilization` compares the distance with and without the last base move, capped at 0.35 m/step and gated by arm reach.
  - `base_target_alignment` rewards velocity that points toward an unreachable target.

**Goal:** keep the target in the soft band whenever possible, move the base proactively when the target drifts toward the hard band, and stop the base once the arm can finish the job.

---

## 2. What the Data Shows (session_8cv2_200M)
Aggregated over 200 evaluation episodes:
- Mean EE position error **3.28 m** (median 3.58 m, 95th percentile 4.69 m).
- Orientation error improved to **20.5 deg** mean (down from 47.8 deg in Session 8b), so the wrist controller is doing its job.
- `reachability_bonus` mean **-1207** and `reachability_distance_penalty` dominates the reward table.
- `base_mobilization` mean **0.11** and `base_target_alignment` mean **0.004** => the chassis rarely makes meaningful progress toward the target.
- `base_overshoot_penalty` mean **8.41** => when the base finally moves it often swings past the waypoint.
- Legacy `target_distance_penalty` mean **1.89** => still active but too small to counteract chronic violations.
- Training monitor snapshots (iteration 89) reported `monitoring/unreachable_zone_pct` above 60% and `monitoring/workspace_hard_exceed_pct` above 50%, confirming the workspace buffer aligns with evaluation.

**Interpretation:** The shaping now exposes accurate diagnostics, but the policy has not internalised them. The arm tracks orientation while the base remains mostly stationary, so the trajectory lives in the penalty regime and the bonus never accumulates.

---

## 3. Likely Causes
1. **Trajectory curriculum still aggressive:** Many recorded paths require simultaneous camera dolly and pan. With the base idle, EE error explodes and the policy observes large penalties that may overwhelm the sparse mobilisation rewards.
2. **Mobilisation cap too conservative during exploration:** The 0.35 m/step cap prevents spikes but may also compress early progress signals when the base starts metres away.
3. **Observation normalisation mismatch early in run:** Even though VecNormalize now loads correctly, the first 5-10M steps still operate with cold statistics, reducing the signal-to-noise ratio for mobilisation terms.
4. **Logging confirms the right features but not the right balance:** `reachability_bonus` gives strong negative numbers, yet the policy does not reallocate effort from orientation/arm to base because the positive path (mobilisation rewards) is comparatively small.

---

## 4. Recommended Actions for Session 8c-v3
1. **Relaunch from scratch:** Old checkpoints are biased by the previous shaping. Start a clean run so the policy sees the new bonus/penalty structure from step zero.
2. **Tighten monitoring gates:** Treat `monitoring/unreachable_zone_pct > 25%` as a stop condition during the next run. If the metric plateaus above this value for five iterations, pause and inspect trajectories.
3. **Boost mobilisation credit during early curriculum:** Temporarily raise `base_progress_reward` or relax `mobilization_progress_cap` to 0.45 for stage 0-1 of the curriculum, then taper back once the chassis starts moving consistently.
4. **Sampling hygiene:** Use the new monitoring logs to filter trajectories that sit permanently beyond the hard margin. Either adjust their timing or exclude them from the early curriculum to avoid overwhelming penalties before the base learns to move.
5. **Evaluation cadence:** After each 40M-step checkpoint, run `evaluate_quantitative.py` and record `reachability_bonus`, `reachability_distance_penalty`, and mobilisation means. Add these to `comparison_all` to verify that bonuses improve over time.

---

## 5. Fast Diagnostic Checklist
- If `reachability_bonus` stays below -200 by iteration 20, look at the base distance histograms immediately.
- If `base_mobilization` plateaus below 0.3 while `target_distance_penalty` remains >1.0, the base is still under-reacting.
- When `monitoring/workspace_hard_exceed_pct` spikes, cross-reference with the logged trajectory IDs to identify problematic clips.
- Use `scripts/reinforcement_learning/sb3/visualize_eval_results.py` to plot workspace distance time series; flat lines above 0.6 m indicate the chassis never moved.

The fixes give us the instrumentation we needed; the next training cycle must focus on leveraging those signals to keep the target in reach and to coordinate chassis and arm motion instead of letting the arm shoulder the entire tracking job.
