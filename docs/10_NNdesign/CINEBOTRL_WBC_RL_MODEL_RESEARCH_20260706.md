# CineBotRL WBC RL Model Research - 2026-07-06

## Context

Current CineBotRL policy is a Stable-Baselines3 PPO `MlpPolicy` with a flat actor/critic MLP:

- observation: 85 dims in the active checkpoint
- action: 9 dims: `joint6_arm_yaw`, `joint5_arm_pitch`, `joint4_elbow_pitch`, `joint3_gimbal_yaw`, `joint2_gimbal_roll`, `joint1_gimbal_pitch`, `base_vx`, `base_vy`, `base_wz`
- actor/critic hidden dims: `[256, 256, 128]`
- default action contract: `sim_6joint_gimbal_v1`
- RS4-aware contract exists as `rs4_attitude_rate_v1`, but is not active/deployment-ready

Recent smoke evidence shows the asset/physics path is stable enough, but the policy quality is not: the raw-policy recovery smoke completed, yet produced roughly `1.14 m` mean EE position error and high unreachable-zone time. This points to learning architecture/data/curriculum, not just rendering or physics.

## Sources Reviewed

### HOVER / Neural WBC

- Repo: https://github.com/NVlabs/HOVER
- Project: https://hover-versatile-humanoid.github.io/
- Paper: https://arxiv.org/abs/2410.21229

Relevant ideas:

- Train a privileged teacher first, then distill into a deployable student.
- Use kinematic motion/reference commands as a common abstraction across modes.
- Use mode-specific and sparsity-based masks so one policy can operate under different command surfaces.
- Student policy remains small and deployable, typically an MLP, but the training structure is stronger than a direct flat PPO baseline.
- Repo includes IsaacLab integration, teacher policy training, student policy training, student history, command masks, masked metrics, sim-to-sim and deployment scaffolding.

Local inspection of HOVER code showed:

- Teacher policy config defaults to actor/critic hidden dims `[512, 256, 128]`, ELU, PPO with adaptive KL.
- Student policy defaults to `[512, 256, 128]`, ELU, very low action noise for imitation/distillation.
- Student observations explicitly include command/reference masks and history.
- Evaluation reports masked and unmasked motion metrics.

Fit for CineBotRL:

- High. This is the best architectural pattern to borrow.
- Do not copy humanoid body definitions; borrow the training pattern: teacher -> masked student -> per-group metrics.
- Our equivalents of HOVER modes should be: `base_only`, `base_plus_arm`, `rs4_attitude`, `full_sim_joint`, `obstacle_recovery`.

### RSL-RL / IsaacLab RL Stack

- Repo: https://github.com/leggedrobotics/rsl_rl
- IsaacLab RL docs: https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html
- IsaacLab wrapper docs: https://isaac-sim.github.io/IsaacLab/main/source/api/lab_rl/isaaclab_rl.html

Relevant ideas:

- RSL-RL is robotics-first and GPU-oriented.
- It is natively supported by IsaacLab alongside RL-Games, SKRL, and SB3.
- It includes PPO, student-teacher distillation, MLP/RNN model classes, distribution modules, rollout storage, normalizers, and export helpers.

Fit for CineBotRL:

- High as the next training backend if we want serious teacher/student training.
- Lower immediate cost path: keep SB3 for a short custom-policy experiment, but prepare an RSL-RL branch for the real WBC pipeline.
- RSL-RL is more aligned with IsaacLab robotics practice than SB3 for large vectorized simulation and distillation.

### Deep Whole-Body Control: Unified Manipulation + Locomotion

- Project: https://manipulation-locomotion.github.io/
- Code: https://github.com/MarkFzp/Deep-Whole-Body-Control

Relevant ideas:

- Do not over-decouple locomotion and manipulation if the task requires coordination.
- Unified policy can produce smoother whole-body behavior than separate hand-engineered base and arm controllers.
- The paper introduces action/reward-aware training ideas such as advantage mixing and online adaptation.

Fit for CineBotRL:

- Medium-high conceptually.
- The original code is older legged-manipulator/IsaacGym style, so it is not a direct drop-in.
- The useful bit is not the exact network. The useful bit is causal/action-group treatment of rewards and gradients: base rewards should mostly train base channels; arm/gimbal tracking terms should mostly train arm/gimbal channels, while still sharing an encoder.

### Causal MoMa / Causal Policy Gradient

- Paper: https://arxiv.org/abs/2305.04866
- RSS page: https://rss2023.github.io/rss2023-website/program/papers/049/

Relevant ideas:

- Mobile manipulation action spaces are large and multi-objective.
- Automatically or explicitly mapping reward terms to action subspaces can reduce gradient variance versus monolithic PPO.
- This is directly relevant to our base/arm/gimbal split.

Fit for CineBotRL:

- High as a diagnostic and loss-design idea.
- We do not need to implement the full paper first.
- Practical first version: maintain action groups and per-group auxiliary losses/metrics:
  - base actions: `base_vx`, `base_vy`, `base_wz`
  - arm actions: first 3 arm joints
  - gimbal/RS4 actions: last 3 wrist/attitude channels
  - reward groups: base reachability, EE position, EE orientation, obstacle safety, smoothness

### HumanoidVerse / ASAP

- HumanoidVerse repo: https://github.com/LeCAR-Lab/HumanoidVerse
- ASAP repo: https://github.com/LeCAR-Lab/ASAP

Relevant ideas:

- Multi-simulator and modular task/algorithm separation.
- Motion tracking training with history-heavy observations.
- ASAP introduces a delta-action model and sim-to-real correction/finetuning workflow.

Fit for CineBotRL:

- Medium.
- Useful for motion-tracking from GIK/ARCore trajectories and later sim-to-real residual correction.
- Too much framework migration for the immediate next step.

### Transformer-Based Humanoid Locomotion

- Project: https://learning-humanoid-locomotion.github.io/
- Paper: https://arxiv.org/abs/2303.03381

Relevant ideas:

- Causal Transformer over observation-action history can adapt from history.
- Strong for large-scale locomotion with massive simulation data.

Fit for CineBotRL:

- Low for immediate use.
- Our dataset and task are not yet mature enough to justify a Transformer controller.
- Keep as a later option only after teacher labels and MLP-headed baselines saturate.

## Recommended Architecture For CineBotRL

Use a HOVER/RSL-RL-inspired teacher-student WBC design, not a larger flat MLP.

### Stage A - Near-Term, Minimal Disruption

Implement a custom grouped actor-critic in the current SB3 path to prove the architecture change before backend migration.

Proposed network:

```text
obs[85]
  -> shared_encoder: 256/256 ELU or SiLU
  -> base_head:      128 -> 3  # vx, vy, wz
  -> arm_head:       128 -> 3  # arm yaw/pitch/elbow or first arm group
  -> gimbal_head:    128 -> 3  # sim wrist or RS4 attitude/rate group
  -> value_head:     256/128 -> 1
```

Keep 9D output ordering unchanged for compatibility.

Add explicit loss/diagnostic groups:

- `base_action_mse`, `base_vy_mse`, `base_reachability_success`
- `arm_action_mse`, `arm_limit_clip_pct`, `arm_envelope_valid_pct`
- `gimbal_action_mse`, `rs4_roll_mask_pct`, `orientation_error_deg`
- `obstacle_unsafe_pct`, `obstacle_collision_pct`
- `ee_pos_error_mean_m`, `ee_pos_error_p95_m`

This is the fastest way to test whether action-group specialization helps without rewriting the whole trainer.

### Stage B - Proper WBC Training Stack

Move the serious trainer to RSL-RL/IsaacLab:

- teacher policy: privileged sim state + GIK/trajectory progress + obstacle state
- student policy: deployable observations + command/mode mask + history
- distillation: student imitates teacher actions, with DAgger-style rollout mixing
- PPO finetune: after BC/distillation gate passes
- export: TorchScript/ONNX for deployment experiments

### Stage C - HOVER-Style Command Masks

Define command modes for our robot:

```text
base_only:
  enabled actions: base_vx, base_vy, base_wz
  use case: recovery/reachability and chassis imitation

base_plus_arm:
  enabled actions: base_vx, base_vy, base_wz, arm_yaw, arm_pitch, arm_elbow
  use case: EE position tracking with stable gimbal/home attitude

rs4_attitude:
  enabled actions: base_vx, base_vy, base_wz, arm group, rs4_yaw_rate, rs4_pitch_rate
  masked: rs4_roll_rate until real RS4 behavior is validated

full_sim_joint:
  enabled actions: all 9 sim joint/velocity channels
  use case: Isaac-only diagnostics, not direct deployment claim

obstacle_recovery:
  enabled actions: base group first, arm/gimbal stabilized
  use case: maneuver around 40 cm diameter / 50 cm height obstacle while tracking EE trajectory
```

### Stage D - Data Gate Before Training

Do not train from poor labels. Required gates:

- trajectory duration >= 5 s
- base labels valid for all samples
- arm/gimbal labels inside RL-safe envelope
- target reachable or marked as recovery curriculum, not direct imitation
- no world-frame/base-frame mismatch
- obstacle frame is world-fixed/root-fixed, not accidentally robot-attached

## Decision

Best open-source design to use: **HOVER-style teacher/student WBC with masks**, implemented on top of **RSL-RL/IsaacLab** for the serious path.

Immediate implementation should not be a full repo migration. First prove the architecture with a grouped-head SB3 policy and per-group metrics. If that improves the smoke gate, port the same contract to RSL-RL teacher/student.

## Concrete Next Work Items

1. Add a repo-backed `GroupedActorCriticPolicy` for SB3 with shared encoder plus base/arm/gimbal heads.
2. Add a compatibility test proving output shape is still 9 and action ordering is unchanged.
3. Add per-action-group eval metrics to `evaluate_recovery_candidate.py`.
4. Run BC warm-start on base-only and RS4-masked datasets.
5. Run a bounded 65k-step PPO gate; compare against the current flat MLP checkpoint.
6. If grouped-head gate improves, start RSL-RL branch with teacher/student/mask pipeline.

## Non-Recommendations

- Do not jump directly to a Transformer controller. It needs much more data and stronger baseline gates.
- Do not split base and arm into fully independent policies as the main path; use separate heads but keep shared state/context so coordination can still be learned.
- Do not call the RS4 path deployment-ready until the simulator adapter, dataset schema, axis signs, roll policy, and hardware command semantics are validated.
