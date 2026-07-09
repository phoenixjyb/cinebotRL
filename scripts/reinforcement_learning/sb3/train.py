"""Training script for MobileMMTrackEE task using Stable Baselines3.

This script trains a mobile manipulator to track end-effector trajectories
using the PPO algorithm from Stable Baselines3.

Designed for Windows with Isaac Lab. No WSL-specific workarounds needed!

Usage:
    # Recommended — use the PowerShell launcher (handles paths automatically):
    .\\scripts\\launch_training_windows.ps1 -Headless -NumEnvs 1024

    # Direct Isaac Lab call (set ISAAC_LAB_ROOT env var first):
    & "$env:ISAAC_LAB_ROOT\\isaaclab.bat" -p scripts/reinforcement_learning/sb3/train.py \\
        --task RecomoProto2TrackEE-v0 \\
        --num_envs 1024 \\
        --headless
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to Python path
# Get the script's actual location, then go up to project root
SCRIPT_DIR = Path(__file__).resolve().parent  # scripts/reinforcement_learning/sb3/
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent  # Up 3 levels to project root

# Add paths for module imports
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

print(f"[DEBUG] PROJECT_ROOT: {PROJECT_ROOT}")
print(f"[DEBUG] sys.path includes: {PROJECT_ROOT}, {PROJECT_ROOT / 'src'}")

# NOTE: Do NOT import task_spec here! Isaac Sim must be initialized first.
# We'll import and register tasks inside main() after AppLauncher runs.


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train mobile manipulator with Stable Baselines3"
    )
    
    # Environment settings
    parser.add_argument(
        "--task",
        type=str,
        default="RecomoProto2TrackEE-v0",
        help="Task ID to train on",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1024,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no GUI)",
    )
    parser.add_argument(
        "--enable_cameras",
        action="store_true",
        help="Enable Isaac rendering/camera extensions. Required for headless video recording.",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Record Isaac-rendered RGB training clips with Gymnasium RecordVideo.",
    )
    parser.add_argument(
        "--video_length",
        type=int,
        default=400,
        help="Length of each rendered training video clip in environment steps.",
    )
    parser.add_argument(
        "--video_interval",
        type=int,
        default=10_000,
        help="Record one rendered training clip every N environment steps.",
    )
    parser.add_argument(
        "--render_experience",
        type=str,
        default=None,
        help=(
            "Isaac/Kit experience file to use for rendering. Defaults to the local "
            "D3D12 headless rendering app when --video is enabled and it exists."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible short gates (Python, NumPy, Torch, env config).",
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=10_000_000,
        help="Total number of timesteps to train",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,  # Proto2 v2: lower for safer policy updates on fragile startup
        help="Learning rate for PPO",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=128,  # 128 steps x 4096 envs = 524K timesteps/iteration (better GAE estimation)
        help="Number of steps per rollout (128-512 recommended for stable GAE)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,  # Good balance for 65K rollout buffer
        help="Minibatch size for PPO updates",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=10,
        help="Number of epochs per PPO update",
    )
    
    # PPO hyperparameters
    parser.add_argument(
        "--ent_coef",
        type=float,
        default=0.001,
        help="Entropy coefficient for exploration (Proto2 v2 default: conservative 0.001)",
    )
    parser.add_argument(
        "--target_kl",
        type=float,
        default=None,
        help="Target KL divergence for early stopping (None = disabled)",
    )
    parser.add_argument(
        "--clip_range",
        type=float,
        default=0.2,
        help="PPO clipping range for policy updates",
    )
    parser.add_argument(
        "--clip_range_vf",
        type=float,
        default=None,
        help="Value function clipping range (None = no clipping, 0.3 recommended for Session 8c to stabilize critic)",
    )
    parser.add_argument(
        "--vf_coef",
        type=float,
        default=0.5,
        help="Value-function loss coefficient for PPO critic updates.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.5,
        help="Maximum gradient norm for PPO updates.",
    )
    parser.add_argument(
        "--reset_reward_stats",
        action="store_true",
        help="When resuming, keep observation normalization but reset VecNormalize reward-return statistics.",
    )
    parser.add_argument(
        "--disable_vec_normalize",
        action="store_true",
        help=(
            "Train PPO on raw observations/rewards. Intended for narrow Stage A "
            "BC warm-start gates where the BC actor was trained on raw observations."
        ),
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor for future rewards",
    )
    parser.add_argument(
        "--gae_lambda",
        type=float,
        default=0.95,
        help="Lambda parameter for Generalized Advantage Estimation",
    )
    
    # Entropy decay parameters (prevents late-stage policy divergence)
    parser.add_argument(
        "--enable_entropy_decay",
        action="store_true",
        help="Enable entropy coefficient decay to prevent policy divergence after convergence",
    )
    parser.add_argument(
        "--final_ent_coef",
        type=float,
        default=0.0001,
        help="Final entropy coefficient after decay (only used if --enable_entropy_decay)",
    )
    parser.add_argument(
        "--decay_start_timestep",
        type=int,
        default=150_000_000,
        help="Timestep to start entropy decay (default: 150M steps)",
    )
    parser.add_argument(
        "--decay_duration_timesteps",
        type=int,
        default=150_000_000,
        help="Duration of entropy decay in timesteps (default: 150M steps for slower decay)",
    )
    
    # KL divergence scheduling parameters (prevents instability and oscillations)
    parser.add_argument(
        "--enable_kl_schedule",
        action="store_true",
        help="Enable target_kl scheduling for stable policy updates across training phases",
    )
    parser.add_argument(
        "--kl_warmup",
        type=float,
        default=0.07,
        help="Target KL during warmup phase (0-10%% of training, default: 0.07)",
    )
    parser.add_argument(
        "--kl_main",
        type=float,
        default=0.02,
        help="Target KL during main phase (10-80%% of training, default: 0.02)",
    )
    parser.add_argument(
        "--kl_finetune",
        type=float,
        default=0.01,
        help="Target KL during fine-tune phase (80-100%% of training, default: 0.01)",
    )
    
    # Logging and checkpointing
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Directory for logs (default: logs/sb3/{task}/{timestamp})",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=100_000,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="cinebotrl",
        help="W&B project name",
    )
    
    # Resume from checkpoint
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--pretrained_policy",
        type=str,
        default=None,
        help="Path to BC-pretrained policy (.zip) to warm-start PPO. "
             "Transfers actor (policy) network weights only; critic is re-initialised.",
    )
    parser.add_argument(
        "--pretrained_action_indices",
        type=str,
        default=None,
        help=(
            "Optional comma-separated action indices whose action-head rows should be "
            "copied from --pretrained_policy. Use this for masked/base-only BC, e.g. "
            "'6,7,8'. If omitted, all action rows are copied."
        ),
    )
    parser.add_argument(
        "--copy_pretrained_log_std",
        action="store_true",
        help=(
            "Also copy log_std from --pretrained_policy. Default keeps the freshly "
            "created PPO log_std_init so masked BC policies do not inject high "
            "stochastic action noise."
        ),
    )
    parser.add_argument(
        "--pretrained_unselected_log_std",
        type=float,
        default=None,
        help=(
            "When --pretrained_action_indices is used, optionally set log_std for "
            "non-selected action dimensions to this value. Useful for base-only BC "
            "warm starts where arm/gimbal channels should begin near neutral."
        ),
    )
    parser.add_argument(
        "--policy_log_std_override",
        type=float,
        default=None,
        help="Set all policy log_std entries after checkpoint/pretrained loading.",
    )
    parser.add_argument(
        "--freeze_base_actions",
        action="store_true",
        help=(
            "Stage A gate: zero base action rows [6,7,8] before env dynamics "
            "while preserving the 9D policy output shape."
        ),
    )
    parser.add_argument(
        "--base_action_scale",
        type=float,
        default=1.0,
        help=(
            "Gradual base-unfreeze gate: multiply base action rows [6,7,8] by this "
            "factor before env dynamics. Ignored when --freeze_base_actions is set."
        ),
    )
    parser.add_argument(
        "--base_action_delta_limit",
        type=float,
        default=0.0,
        help=(
            "Optional PPO-side normalized per-step slew limit for base action rows [6,7,8]. "
            "0 disables the clamp."
        ),
    )
    parser.add_argument(
        "--base_action_slew_limit",
        type=float,
        default=0.0,
        help=(
            "Optional env-level normalized per-step slew limit for base action rows [6,7,8]. "
            "This is saved in the task config and should be reused for rendered gates."
        ),
    )
    parser.add_argument(
        "--freeze_base_actions_for_non_base_required",
        action="store_true",
        help=(
            "For mixed FK stages, zero base rows [6,7,8] on trajectories whose "
            "metadata filename does not contain 'base_required'. This keeps "
            "fixed-base samples aligned with the fixed-base regression gate."
        ),
    )
    parser.add_argument(
        "--policy_arch",
        type=str,
        default="flat",
        choices=["flat", "grouped"],
        help=(
            "Actor-critic architecture. 'flat' preserves the existing SB3 MlpPolicy. "
            "'grouped' uses a shared encoder with separate arm/gimbal/base action heads."
        ),
    )
    parser.add_argument(
        "--grouped_shared_hidden_dims",
        type=str,
        default="256,256",
        help="Comma-separated shared encoder dims for --policy_arch grouped.",
    )
    parser.add_argument(
        "--grouped_head_hidden_dim",
        type=int,
        default=128,
        help="Hidden dim for each grouped action head.",
    )
    parser.add_argument(
        "--grouped_value_hidden_dims",
        type=str,
        default="256,128",
        help="Comma-separated value-network dims for --policy_arch grouped.",
    )
    parser.add_argument(
        "--base_assist_aux_dataset",
        type=str,
        default=None,
        help=(
            "Optional masked action .npz dataset used for auxiliary supervised "
            "updates after each PPO rollout. Dataset observations must already "
            "match the policy observation normalization."
        ),
    )
    parser.add_argument(
        "--base_assist_aux_action_indices",
        type=str,
        default="6,7",
        help=(
            "Comma-separated action rows trained by --base_assist_aux_dataset, "
            "for example '6,7' or '6,7,8'."
        ),
    )
    parser.add_argument(
        "--base_assist_aux_gradient_steps",
        type=int,
        default=0,
        help="Number of masked supervised action-head minibatch updates after each PPO rollout.",
    )
    parser.add_argument(
        "--base_assist_aux_batch_size",
        type=int,
        default=2048,
        help="Minibatch size for auxiliary supervised action-head updates.",
    )
    parser.add_argument(
        "--base_assist_aux_lr",
        type=float,
        default=1e-4,
        help="Learning rate for auxiliary supervised action-head updates.",
    )
    parser.add_argument(
        "--base_assist_aux_max_grad_norm",
        type=float,
        default=0.5,
        help="Gradient clipping norm for auxiliary supervised action-head updates.",
    )
    parser.add_argument(
        "--base_assist_aux_ignore_sample_weight",
        action="store_true",
        help=(
            "Ignore optional sample_weight in the aux dataset. By default, datasets "
            "with sample_weight are sampled proportionally to those weights."
        ),
    )
    parser.add_argument(
        "--base_assist_aux_sample_weight_power",
        type=float,
        default=1.0,
        help=(
            "Exponent applied to optional aux dataset sample_weight before sampling. "
            "Use values below 1.0 to soften hard-state oversampling."
        ),
    )
    
    # Device selection
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for training (default: auto - use GPU if available)",
    )
    
    # Trajectory configuration
    parser.add_argument(
        "--trajectory_type",
        type=str,
        default="circle",
        choices=["line", "circle", "figure_eight", "recorded", "multi_recorded"],
        help="Type of trajectory to use (default: circle for basic training, multi_recorded for diverse real trajectories)",
    )
    parser.add_argument(
        "--trajectory_dir",
        type=str,
        default="trajectoryToLearn/world_json",
        help="Directory containing recorded trajectories (for multi_recorded mode)",
    )
    parser.add_argument(
        "--use_all_trajectories",
        action="store_true",
        help="Use ALL trajectories from directory (recommended for training), not just chassis-requiring ones",
    )
    parser.add_argument(
        "--use_chassis_only",
        action="store_true",
        help="Use only chassis-requiring trajectories (for testing base movement, not recommended for training)",
    )
    parser.add_argument(
        "--trajectory_stage",
        type=str,
        default=None,
        choices=[
            "stage0_policy_envelope_fk",
            "stage0_policy_envelope_fk_slow",
            "stage0_policy_envelope_fk_medium",
            "stage0_policy_envelope_fk_large08",
            "stage0_policy_envelope_fk_base025",
            "stage0_policy_envelope_fk_base040",
            "stage0_policy_envelope_fk_mix_large08_base025",
            "stage0_fixedbase_micro",
            "stage_gik_no_obstacle79_nominal",
            "stage_gik_one_obstacle63_accepted",
            "stage0_easy",
            "stage1_recovery",
            "stage2_moderate",
            "stage3_full",
        ],
        help=(
            "Use staged trajectory curriculum "
            "(stage0_policy_envelope_fk=FK targets generated from allowed actions, "
            "stage0_policy_envelope_fk_slow=smaller first-stage FK targets, "
            "stage0_policy_envelope_fk_medium=medium first-stage FK targets, "
            "stage0_policy_envelope_fk_large08=larger 0.08m FK targets, "
            "stage0_policy_envelope_fk_base025=small base-required FK targets, "
            "stage0_policy_envelope_fk_base040=larger base-required FK targets, "
            "stage0_policy_envelope_fk_mix_large08_base025=mixed fixed/base-required FK targets, "
            "stage0_fixedbase_micro=fixed-base reachable micro paths, "
            "stage_gik_no_obstacle79_nominal=accepted live GIK/ARCore nominal trajectories, "
            "stage_gik_one_obstacle63_accepted=accepted GIK one-obstacle trajectories, "
            "stage0_easy=short cinematic paths, stage1=recovery, stage2=moderate, "
            "stage3=full)"
        ),
    )
    parser.add_argument(
        "--max_trajectories",
        type=int,
        default=None,
        help="Limit number of trajectories to load (None = all, useful for debugging)",
    )
    parser.add_argument(
        "--min_trajectory_duration",
        type=float,
        default=5.0,
        help="Reject recorded trajectories shorter than this many seconds.",
    )
    parser.add_argument(
        "--episode_length_s",
        type=float,
        default=0.0,
        help="Optional env episode length override in seconds. Keep 0 to use the task default.",
    )
    parser.add_argument(
        "--random_start_waypoint",
        action="store_true",
        help="Start recorded trajectories from a random waypoint on reset.",
    )
    parser.add_argument(
        "--start_waypoint_min_fraction",
        type=float,
        default=0.25,
        help="Minimum random reset waypoint as a fraction of trajectory length.",
    )
    parser.add_argument(
        "--start_waypoint_max_fraction",
        type=float,
        default=0.70,
        help="Maximum random reset waypoint as a fraction of trajectory length.",
    )
    parser.add_argument(
        "--start_waypoint_max_fraction_initial",
        type=float,
        default=None,
        help=(
            "Optional curriculum start value for the maximum random reset waypoint fraction. "
            "When set, it ramps linearly to --start_waypoint_max_fraction."
        ),
    )
    parser.add_argument(
        "--start_waypoint_fraction_decay_steps",
        type=int,
        default=0,
        help="Timesteps over which start waypoint max fraction ramps from initial to final.",
    )
    parser.add_argument(
        "--reset_base_to_trajectory_start",
        action="store_true",
        help="Place the base near waypoint zero even when target playback starts later.",
    )
    parser.add_argument(
        "--reset_anchor_target_blend",
        type=float,
        default=0.0,
        help=(
            "Blend reset anchor from trajectory start toward the current target when "
            "--reset_base_to_trajectory_start is active. 0 preserves the current recovery baseline; 1 starts at the current target."
        ),
    )
    parser.add_argument(
        "--reset_anchor_target_blend_initial",
        type=float,
        default=None,
        help=(
            "Optional curriculum start value for reset anchor target blend. "
            "When set, it decays linearly to --reset_anchor_target_blend."
        ),
    )
    parser.add_argument(
        "--reset_anchor_target_blend_decay_steps",
        type=int,
        default=0,
        help="Timesteps over which reset anchor target blend decays from initial to final.",
    )
    parser.add_argument(
        "--reachability_distance_weight",
        type=float,
        default=None,
        help="Override reward weight for workspace distance beyond the hard reachability margin.",
    )
    parser.add_argument(
        "--base_target_far_penalty",
        type=float,
        default=None,
        help="Override penalty for base-target distance beyond the reachability hard margin.",
    )
    parser.add_argument(
        "--base_target_command_tracking_penalty",
        type=float,
        default=None,
        help="Override penalty for far-target base commands that fail to chase the target.",
    )
    parser.add_argument(
        "--base_target_regression_penalty",
        type=float,
        default=None,
        help="Override penalty for actual chassis motion that increases base-target distance.",
    )
    parser.add_argument(
        "--base_action_delta_penalty",
        type=float,
        default=None,
        help="Override reward penalty for rapid normalized base_vx/base_vy/base_wz changes.",
    )
    parser.add_argument(
        "--min_obstacle_distance_weight",
        type=float,
        default=None,
        help="Override obstacle clearance reward/penalty weight.",
    )
    parser.add_argument(
        "--safety_radius",
        type=float,
        default=None,
        help="Override desired signed clearance beyond robot footprint + obstacle radius.",
    )
    parser.add_argument(
        "--debug_resets",
        action="store_true",
        help="Print verbose reset pose and waypoint diagnostics.",
    )
    parser.add_argument(
        "--debug_trajectory_sampling",
        action="store_true",
        help="Print verbose trajectory sampling diagnostics.",
    )
    parser.add_argument(
        "--action_contract",
        type=str,
        default="sim_6joint_gimbal_v1",
        choices=["sim_6joint_gimbal_v1", "rs4_attitude_rate_v1"],
        help=(
            "Policy action contract. Default preserves the existing Isaac 6-joint gimbal semantics. "
            "rs4_attitude_rate_v1 is guarded until the simulator adapter is wired."
        ),
    )
    parser.add_argument(
        "--experimental_rs4_adapter",
        action="store_true",
        help=(
            "Explicit opt-in for the experimental RS4 adapter path. This currently still fails fast "
            "because env execution is not wired yet; it exists to prevent accidental old-semantics execution."
        ),
    )
    parser.add_argument(
        "--disable_recovery_stability_preset",
        action="store_true",
        help="Do not auto-apply lower LR / target_kl defaults for stage1_recovery.",
    )
    parser.add_argument(
        "--enable_base_assist",
        action="store_true",
        help="Blend an expert body-frame vx/vy command into executed base control.",
    )
    parser.add_argument(
        "--disable_auto_base_assist",
        action="store_true",
        help="Do not auto-enable base assist for stage1_recovery.",
    )
    parser.add_argument(
        "--base_assist_initial_blend",
        type=float,
        default=0.6,
        help="Initial executed-command blend toward expert base command.",
    )
    parser.add_argument(
        "--base_assist_mode",
        type=str,
        default="target_direction",
        choices=["target_direction", "target_velocity", "target_offset_follow"],
        help=(
            "Expert source for base assist. target_direction chases the target center; "
            "target_velocity follows the lookahead target displacement; "
            "target_offset_follow preserves the generated FK base-target offset."
        ),
    )
    parser.add_argument(
        "--base_assist_final_blend",
        type=float,
        default=0.0,
        help="Final executed-command blend after decay.",
    )
    parser.add_argument(
        "--base_assist_decay_steps",
        type=int,
        default=2_000_000,
        help="Vectorized env steps over which base assist decays.",
    )
    parser.add_argument(
        "--base_assist_activation_distance",
        type=float,
        default=0.7,
        help="Enable base assist only beyond this base-target XY distance.",
    )
    parser.add_argument(
        "--base_assist_full_speed_distance",
        type=float,
        default=1.4,
        help="Distance where expert base assist reaches max normalized action.",
    )
    parser.add_argument(
        "--base_assist_max_action",
        type=float,
        default=0.8,
        help="Normalized expert vx/vy action cap for base assist.",
    )
    parser.add_argument(
        "--base_assist_imitation_weight",
        type=float,
        default=25.0,
        help="Auxiliary penalty weight for raw policy vx/vy error vs expert assist command.",
    )
    parser.add_argument(
        "--base_assist_lookahead_steps",
        type=int,
        default=0,
        help="Use the final N-step lookahead target for base-assist expert direction; 0 uses current target.",
    )
    parser.add_argument(
        "--enable_base_assist_yaw",
        action="store_true",
        help="Also blend/imitate an expert base_wz command that points the chassis toward far targets.",
    )
    parser.add_argument(
        "--disable_auto_base_assist_yaw",
        action="store_true",
        help="Do not auto-enable yaw assist for stage1_recovery.",
    )
    parser.add_argument(
        "--base_assist_yaw_max_action",
        type=float,
        default=0.6,
        help="Normalized expert base_wz cap for yaw assist.",
    )
    parser.add_argument(
        "--base_assist_yaw_full_error",
        type=float,
        default=1.2,
        help="Heading error in radians that maps to base_assist_yaw_max_action.",
    )
    parser.add_argument(
        "--base_assist_yaw_imitation_weight",
        type=float,
        default=10.0,
        help="Auxiliary penalty weight for raw policy base_wz error vs yaw expert.",
    )
    parser.add_argument(
        "--enable_obstacles",
        action="store_true",
        help="Enable the ground-disc obstacle avoidance task.",
    )
    parser.add_argument(
        "--obstacles_from_trajectory_metadata",
        action="store_true",
        help=(
            "Place each env's obstacle from trajectory metadata.obstacle.center_xy "
            "during reset. Use with exported GIK one-obstacle stages."
        ),
    )
    parser.add_argument("--obstacle_x", type=float, default=0.0, help="Static obstacle disc X in each env local frame.")
    parser.add_argument("--obstacle_y", type=float, default=0.5, help="Static obstacle disc Y in each env local frame.")
    parser.add_argument(
        "--obstacle_radius",
        type=float,
        default=None,
        help="Override obstacle disc radius in meters. Defaults to the task config.",
    )
    parser.add_argument(
        "--obstacle_height",
        type=float,
        default=None,
        help="Override obstacle disc height in meters. Defaults to the task config.",
    )
    parser.add_argument(
        "--disable_obstacle_randomization",
        action="store_true",
        help="Keep the obstacle at --obstacle_x/--obstacle_y instead of randomizing per reset.",
    )
    parser.add_argument(
        "--obstacle_x_range",
        type=float,
        nargs=2,
        default=(-0.35, 0.35),
        metavar=("MIN", "MAX"),
        help="Randomized obstacle local X range when obstacles are enabled.",
    )
    parser.add_argument(
        "--obstacle_y_range",
        type=float,
        nargs=2,
        default=(0.45, 1.0),
        metavar=("MIN", "MAX"),
        help="Randomized obstacle local Y range when obstacles are enabled.",
    )
    parser.add_argument(
        "--min_obstacle_start_clearance",
        type=float,
        default=0.10,
        help="Minimum reset-time base footprint clearance from randomized obstacle.",
    )
    parser.add_argument(
        "--enable_obstacle_curriculum",
        action="store_true",
        help="Linearly ramp randomized obstacle difficulty during training.",
    )
    parser.add_argument(
        "--obstacle_curriculum_steps",
        type=int,
        default=500_000,
        help="Number of timesteps used to ramp from start obstacle settings to final settings.",
    )
    parser.add_argument(
        "--obstacle_curriculum_start_x_range",
        type=float,
        nargs=2,
        default=(-0.15, 0.15),
        metavar=("MIN", "MAX"),
        help="Initial obstacle local X range for curriculum.",
    )
    parser.add_argument(
        "--obstacle_curriculum_start_y_range",
        type=float,
        nargs=2,
        default=(1.30, 1.80),
        metavar=("MIN", "MAX"),
        help="Initial obstacle local Y range for curriculum.",
    )
    parser.add_argument(
        "--obstacle_curriculum_start_clearance",
        type=float,
        default=0.45,
        help="Initial minimum reset-time obstacle clearance for curriculum.",
    )
    parser.add_argument(
        "--obstacle_curriculum_start_weight",
        type=float,
        default=0.5,
        help="Initial obstacle reward weight for curriculum.",
    )
    parser.add_argument(
        "--obstacle_curriculum_final_weight",
        type=float,
        default=None,
        help="Final obstacle reward weight for curriculum (default: task config value).",
    )
    
    return parser.parse_args()


def _parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _policy_action_dim(policy) -> int:
    return int(policy.action_space.shape[0])


def _copy_pretrained_actor(model, bc_model, action_indices: list[int] | None) -> str:
    """Copy actor weights from BC model to PPO model for flat or grouped policies."""

    import torch

    from scripts.reinforcement_learning.sb3.grouped_policy import GroupedActionMlpExtractor

    target_extractor = model.policy.mlp_extractor
    source_extractor = bc_model.policy.mlp_extractor
    if isinstance(target_extractor, GroupedActionMlpExtractor):
        if not isinstance(source_extractor, GroupedActionMlpExtractor):
            raise TypeError("grouped PPO warm-start requires a grouped BC policy")
        target_extractor.copy_actor_from(
            source_extractor,
            action_indices=action_indices,
            zero_unselected=action_indices is not None,
        )
        if action_indices is None:
            return "grouped shared encoder and all grouped action heads"
        return f"grouped shared encoder and selected action rows {action_indices}; zeroed non-selected rows"

    if isinstance(source_extractor, GroupedActionMlpExtractor):
        raise TypeError("flat PPO warm-start cannot copy directly from a grouped BC policy")

    model.policy.mlp_extractor.policy_net.load_state_dict(
        bc_model.policy.mlp_extractor.policy_net.state_dict()
    )
    if action_indices is None:
        model.policy.action_net.load_state_dict(bc_model.policy.action_net.state_dict())
        return "flat actor feature weights and all action head rows"

    with torch.no_grad():
        selected = set(action_indices)
        for action_idx in range(_policy_action_dim(model.policy)):
            if action_idx not in selected:
                model.policy.action_net.weight.data[action_idx].zero_()
                model.policy.action_net.bias.data[action_idx].zero_()
        for action_idx in action_indices:
            model.policy.action_net.weight.data[action_idx].copy_(
                bc_model.policy.action_net.weight.data[action_idx]
            )
            model.policy.action_net.bias.data[action_idx].copy_(
                bc_model.policy.action_net.bias.data[action_idx]
            )
    return f"flat actor feature weights and action head rows {action_indices}; zeroed non-selected rows"


def _install_metadata_obstacle_randomizer(raw_env) -> None:
    """Use per-trajectory obstacle metadata for obstacle reset placement."""
    import torch

    if not getattr(raw_env, "obstacles_enabled", False):
        raise RuntimeError("--obstacles_from_trajectory_metadata requires an obstacle-enabled env")

    original_randomize_obstacles = raw_env._randomize_obstacles

    def metadata_randomize_obstacles(env_ids, base_xy_local):
        original_randomize_obstacles(env_ids, base_xy_local)
        metadata = getattr(raw_env.trajectory_manager, "current_trajectory_metadata", []) or []
        rows = []
        used = 0
        for env_id in env_ids.detach().cpu().tolist():
            row = raw_env.obstacle_disc_xy_local[env_id].clone()
            if env_id < len(metadata) and isinstance(metadata[env_id], dict):
                raw_meta = metadata[env_id].get("metadata", {})
                obstacle_meta = raw_meta.get("obstacle") if isinstance(raw_meta, dict) else None
                center_xy = obstacle_meta.get("center_xy") if isinstance(obstacle_meta, dict) else None
                if isinstance(center_xy, (list, tuple)) and len(center_xy) == 2:
                    candidate = torch.tensor(center_xy, dtype=row.dtype, device=row.device)
                    if torch.isfinite(candidate).all():
                        row = candidate
                        used += 1
            rows.append(row)
        if rows:
            raw_env.obstacle_disc_xy_local[env_ids] = torch.stack(rows, dim=0)
            raw_env._obstacle_xy_buf = raw_env.obstacle_disc_xy_local.clone()
            raw_env._write_obstacle_poses_to_sim(env_ids)
        if not hasattr(raw_env, "_metadata_obstacle_randomizer_logged"):
            print(
                "[train] metadata obstacle placement active: "
                f"applied {used}/{int(env_ids.numel())} env(s) on first reset",
                flush=True,
            )
            raw_env._metadata_obstacle_randomizer_logged = True

    raw_env._randomize_obstacles = metadata_randomize_obstacles


def resolve_render_experience(args) -> str | None:
    """Pick the Isaac rendering app that works on the Windows/Blackwell host."""
    if args.render_experience:
        return args.render_experience
    if not args.video:
        return None

    candidates = [
        Path(r"G:\isaaclab\apps\isaaclab.python.headless.rendering.d3d12.kit"),
        Path("/mnt/g/isaaclab/apps/isaaclab.python.headless.rendering.d3d12.kit"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def attach_record_video_vecenv_passthrough(video_env, raw_env):
    """Expose Isaac VecEnv attributes hidden by Gymnasium RecordVideo."""
    for name in ("num_envs", "device", "render_mode"):
        if name != "render_mode" and hasattr(video_env, name):
            continue
        if name == "render_mode" and getattr(video_env, name, None) is not None:
            continue
        for source in (raw_env, getattr(raw_env, "unwrapped", None)):
            if source is not None and hasattr(source, name):
                value = getattr(source, name)
                if name == "render_mode" and value is None:
                    continue
                setattr(video_env, name, value)
                break
        else:
            if name == "render_mode":
                setattr(video_env, name, "rgb_array")
    return video_env


def main():
    """Main training loop."""
    args = parse_args()
    explicit_args = {
        raw_arg.split("=", 1)[0]
        for raw_arg in sys.argv[1:]
        if raw_arg.startswith("--")
    }
    if args.trajectory_stage and "--trajectory_type" not in explicit_args:
        args.trajectory_type = "multi_recorded"
    if args.base_action_scale < 0.0 or args.base_action_scale > 1.0:
        raise ValueError("--base_action_scale must be in [0, 1]")
    if args.base_action_delta_limit < 0.0:
        raise ValueError("--base_action_delta_limit must be >= 0")
    if args.base_action_slew_limit < 0.0:
        raise ValueError("--base_action_slew_limit must be >= 0")
    if args.obstacles_from_trajectory_metadata and not args.enable_obstacles:
        raise ValueError("--obstacles_from_trajectory_metadata requires --enable_obstacles")

    auto_recovery_stability = (
        args.trajectory_stage == "stage1_recovery"
        and not args.disable_recovery_stability_preset
    )
    if auto_recovery_stability:
        if "--learning_rate" not in explicit_args:
            args.learning_rate = min(args.learning_rate, 5e-5)
        if "--target_kl" not in explicit_args and args.target_kl is None:
            args.target_kl = 0.04

    recovery_base_assist_preset_applied = []
    if args.trajectory_stage == "stage1_recovery" and not args.disable_auto_base_assist:
        recovery_assist_defaults = {
            "--base_assist_initial_blend": ("base_assist_initial_blend", 0.9),
            "--base_assist_final_blend": ("base_assist_final_blend", 0.5),
            "--base_assist_activation_distance": ("base_assist_activation_distance", 0.45),
            "--base_assist_full_speed_distance": ("base_assist_full_speed_distance", 0.90),
            "--base_assist_max_action": ("base_assist_max_action", 1.0),
            "--base_assist_imitation_weight": ("base_assist_imitation_weight", 50.0),
        }
        for flag, (attr, value) in recovery_assist_defaults.items():
            if flag not in explicit_args:
                setattr(args, attr, value)
                recovery_base_assist_preset_applied.append(flag)

    if args.seed is not None:
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        random.seed(args.seed)
        try:
            import numpy as np
            np.random.seed(args.seed)
        except ImportError:
            pass
    
    print("=" * 70)
    print("MobileMMTrackEE Training with Stable Baselines3")
    print("=" * 70)
    
    # Step 1: Initialize Isaac Sim via AppLauncher
    # This MUST happen before importing any Isaac Lab or task modules
    print("\n[1/6] Initializing Isaac Sim...")
    try:
        from isaaclab.app import AppLauncher
        import torch
        
        # Enable TF32 for Tensor Cores when the selected GPU supports it.
        # Provides ~8x speedup on matrix multiplications with minimal precision loss
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True  # Auto-tune kernels for your input sizes
            if args.seed is not None:
                torch.manual_seed(args.seed)
                torch.cuda.manual_seed_all(args.seed)
                torch.backends.cudnn.benchmark = False
            print("    [OK] TF32 + cuDNN benchmark enabled (8x matmul speedup + auto-tuned kernels)")
        
        # Auto-detect best GPU
        best_device = 0
        best_compute = 0.0
        for i in range(torch.cuda.device_count()):
            cap = torch.cuda.get_device_capability(i)
            cap_val = cap[0] + cap[1] * 0.1
            if cap_val >= 7.0 and cap_val > best_compute:
                best_compute = cap_val
                best_device = i
                gpu_name = torch.cuda.get_device_name(i)
                gpu_mem_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"    Selected GPU {i}: {gpu_name} (compute {cap[0]}.{cap[1]}, {gpu_mem_gb:.1f}GB)")
                
                # Warn if GPU is underutilized based on memory capacity
                # Rough estimate: ~3MB per environment for mobile manipulator
                recommended_envs = int((gpu_mem_gb - 4) / 0.003)  # Leave 4GB for overhead
                if args.num_envs < recommended_envs * 0.3:  # Less than 30% capacity
                    print(f"    [WARN]  GPU Memory Underutilized!")
                    print(f"       Current: {args.num_envs} envs (~{args.num_envs * 3 / 1024:.1f}GB)")
                    print(f"       Recommended: {recommended_envs // 2} envs (50% capacity)")
                    print(f"       Maximum: ~{recommended_envs} envs (80% capacity)")
                    print(f"       Hint: try --num_envs {recommended_envs // 2}")
        
        # Create AppLauncher to initialize Isaac Sim
        render_experience = resolve_render_experience(args)
        app_launcher_kwargs = {
            "headless": args.headless,
            "enable_cameras": bool(args.enable_cameras or args.video),
            "video": bool(args.video),
            "device": f"cuda:{best_device}",
        }
        if render_experience:
            app_launcher_kwargs["experience"] = render_experience
            print(f"    [OK] Using Isaac render experience: {render_experience}")
        app_launcher = AppLauncher(**app_launcher_kwargs)
        simulation_app = app_launcher.app
        print("    [OK] Isaac Sim initialized")
        
    except Exception as e:
        print(f"    [FAIL] Failed to initialize Isaac Sim: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 2: NOW we can safely import and register our custom tasks
    # Isaac Sim is running, so Isaac Lab imports will work
    print("\n[2/6] Registering custom tasks...")
    try:
        from task_spec import register_isaac_lab_tasks
        register_isaac_lab_tasks()
        print(f"    [OK] Registered task: {args.task}")
    except Exception as e:
        print(f"    [FAIL] Failed to register tasks: {e}")
        import traceback
        traceback.print_exc()
        simulation_app.close()
        return 1
    
    # Step 3: Import SB3 and other training dependencies
    print("\n[3/6] Importing training dependencies...")
    try:
        import gymnasium as gym
        from gymnasium import spaces
        import numpy as np
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
        from stable_baselines3.common.utils import constant_fn
        from stable_baselines3.common.vec_env import VecNormalize, VecEnv, VecEnvWrapper
        print("    [OK] Dependencies imported")
    except ImportError as e:
        print(f"    [FAIL] Failed to import dependencies: {e}")
        simulation_app.close()
        return
    
    # Entropy decay callback to prevent policy divergence
    class EntropyDecayCallback(BaseCallback):
        """
        Decays entropy coefficient during training to prevent late-stage divergence.
        
        Problem: Constant high ent_coef causes policy to maximize entropy instead of
        tracking performance after convergence.
        
        Solution: Start with high ent_coef for exploration, decay to low value for convergence.
        
        Args:
            initial_ent_coef: Starting entropy coefficient (default: 0.001 for exploration)
            final_ent_coef: Ending entropy coefficient (default: 0.0001 for convergence)
            decay_start_timestep: When to start decay (default: 100M steps)
            decay_duration_timesteps: How long decay takes (default: 100M steps)
        """
        def __init__(
            self, 
            initial_ent_coef: float = 0.001,
            final_ent_coef: float = 0.0001,
            decay_start_timestep: int = 100_000_000,
            decay_duration_timesteps: int = 100_000_000,
            verbose: int = 0
        ):
            super().__init__(verbose)
            self.initial_ent_coef = initial_ent_coef
            self.final_ent_coef = final_ent_coef
            self.decay_start = decay_start_timestep
            self.decay_duration = decay_duration_timesteps
            self.decay_end = decay_start_timestep + decay_duration_timesteps
            
        def _on_step(self) -> bool:
            """Update entropy coefficient based on current timestep."""
            current_timestep = self.num_timesteps
            
            if current_timestep < self.decay_start:
                # Before decay: use initial value
                new_ent_coef = self.initial_ent_coef
            elif current_timestep >= self.decay_end:
                # After decay: use final value
                new_ent_coef = self.final_ent_coef
            else:
                # During decay: linear interpolation
                progress = (current_timestep - self.decay_start) / self.decay_duration
                new_ent_coef = self.initial_ent_coef * (1 - progress) + self.final_ent_coef * progress
            
            # Update model's entropy coefficient
            # FIX (8c-v2): Update the schedule function that SB3 actually uses during training
            from stable_baselines3.common.utils import constant_fn
            self.model.ent_coef_schedule = constant_fn(new_ent_coef)
            self.model.ent_coef = new_ent_coef  # Also update attribute for logging
            
            # Log every 10M steps
            if self.verbose > 0 and current_timestep % 10_000_000 < 65536:  # Within one rollout
                print(f"[EntropyDecay] Step {current_timestep/1e6:.1f}M: ent_coef = {new_ent_coef:.6f}")
            
            return True  # Continue training
    
    # Adaptive KL divergence scheduling for proper policy updates
    class AdaptiveKLSchedule(BaseCallback):
        """
        Adaptive KL scheduling that prevents early stopping and allows proper learning.
        
        Problem: Current KL limits cause "Early stopping at step 0" -> no learning!
        - KL divergence spikes early due to new action space (base movement)
        - Immediate early stopping prevents any policy updates
        - Learning efficiency drops to ~1 step per iteration
        
        Solution: Stage-adaptive KL with recovery mechanism:
        - Very Early (0-5M): Extremely loose (1.0) - allow major exploration
        - Early (5-20M): Loose (0.5) - permit substantial learning
        - Learning (20-60M): Moderate (0.2) - balanced updates
        - Stable (60-80M): Normal (0.1) - standard learning
        - Fine-tune (80-100M): Tight (0.05) - precise convergence
        
        Adaptive features:
        - Recent early stopping detection -> temporary KL boost
        - Low explained variance -> increased exploration allowance
        - Training progress monitoring -> automatic adjustments
        """
        def __init__(
            self,
            total_timesteps: int,
            kl_very_early: float = 1.0,    # 0-5M: Allow major exploration
            kl_early: float = 0.5,         # 5-20M: Substantial learning
            kl_learning: float = 0.2,      # 20-60M: Balanced updates
            kl_stable: float = 0.1,        # 60-80M: Normal learning  
            kl_finetune: float = 0.05,     # 80-100M: Precise convergence
            verbose: int = 1
        ):
            super().__init__(verbose)
            self.total_timesteps = total_timesteps
            self.kl_very_early = kl_very_early
            self.kl_early = kl_early
            self.kl_learning = kl_learning
            self.kl_stable = kl_stable
            self.kl_finetune = kl_finetune
            
            # Adaptive tracking
            self.recent_early_stops = 0
            self.check_interval = 10  # Check every 10 rollouts
            self.last_check_timestep = 0
            
        def _get_stage_kl(self, current_timestep: int) -> tuple[float, str]:
            """Determine KL limit based on training stage."""
            progress = current_timestep / self.total_timesteps
            
            if progress < 0.05:  # 0-5M steps
                return self.kl_very_early, "very_early"
            elif progress < 0.20:  # 5-20M steps  
                return self.kl_early, "early"
            elif progress < 0.60:  # 20-60M steps
                return self.kl_learning, "learning"
            elif progress < 0.80:  # 60-80M steps
                return self.kl_stable, "stable"
            else:  # 80-100M steps
                return self.kl_finetune, "finetune"
        
        def _on_rollout_end(self) -> bool:
            """Adaptive KL update with early stopping detection."""
            current_timestep = self.num_timesteps
            
            # Get base KL for current stage
            base_target_kl, stage = self._get_stage_kl(current_timestep)
            
            # Adaptive boost if recent early stopping detected
            kl_boost = 1.0
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                # Check if we're getting early stops (indicates KL too tight)
                if current_timestep - self.last_check_timestep >= self.check_interval * 4096 * 128:
                    # Simple heuristic: if we're not learning much, boost KL
                    try:
                        recent_losses = getattr(self.model.logger, 'recent_losses', [])
                        if len(recent_losses) > 5:
                            if all(loss > 0.1 for loss in recent_losses[-5:]):  # High losses = poor learning
                                kl_boost = 2.0  # Double the KL allowance
                                if self.verbose > 0:
                                    print(f"[AdaptiveKL] Learning struggle detected - boosting KL by 2x")
                    except:
                        pass  # Ignore if logging not available
                    
                    self.last_check_timestep = current_timestep
            
            # Apply adaptive KL
            new_target_kl = base_target_kl * kl_boost
            
            # Update model's target_kl (handle None case)
            current_kl = self.model.target_kl if self.model.target_kl is not None else 0.0
            if abs(current_kl - new_target_kl) > 0.01:  # Significant change
                self.model.target_kl = new_target_kl
                if self.verbose > 0:
                    boost_str = f" (boosted {kl_boost}x)" if kl_boost > 1.0 else ""
                    print(f"[AdaptiveKL] Step {current_timestep/1e6:.1f}M: target_kl = {new_target_kl:.3f} ({stage}){boost_str}")
            
            return True
        
        def _on_step(self) -> bool:
            """Called at every step."""
            return True

    class MaskedActionAuxCallback(BaseCallback):
        """Run masked supervised updates on selected action-head rows between PPO rollouts."""

        def __init__(
            self,
            dataset_path: str,
            action_indices: list[int],
            gradient_steps: int,
            batch_size: int,
            lr: float,
            max_grad_norm: float,
            use_sample_weight: bool = True,
            sample_weight_power: float = 1.0,
            log_freq: int = 1,
            verbose: int = 1,
        ):
            super().__init__(verbose)
            self.dataset_path = Path(dataset_path)
            self.action_indices = action_indices
            self.gradient_steps = gradient_steps
            self.batch_size = batch_size
            self.lr = lr
            self.max_grad_norm = max_grad_norm
            self.use_sample_weight = use_sample_weight
            self.sample_weight_power = sample_weight_power
            self.log_freq = log_freq
            self.rollout_count = 0
            self.obs_tensor = None
            self.action_tensor = None
            self.mask_tensor = None
            self.sample_prob_tensor = None
            self.selected = None
            self.optimizer = None
            self._grouped_head_names = None
            self._preserve_flat_unselected = False

        def _on_training_start(self) -> None:
            import torch
            from scripts.reinforcement_learning.sb3.grouped_policy import GroupedActionMlpExtractor

            if self.gradient_steps <= 0:
                raise ValueError("--base_assist_aux_gradient_steps must be positive when aux dataset is enabled")
            if self.sample_weight_power <= 0.0:
                raise ValueError("--base_assist_aux_sample_weight_power must be positive")
            if not self.dataset_path.exists():
                raise FileNotFoundError(f"base assist aux dataset not found: {self.dataset_path}")

            data = np.load(self.dataset_path, allow_pickle=False)
            required = ["observations", "actions", "action_valid_mask"]
            missing = [key for key in required if key not in data]
            if missing:
                raise ValueError(f"{self.dataset_path} missing keys: {missing}")

            observations = data["observations"].astype(np.float32)
            actions = data["actions"].astype(np.float32)
            mask = data["action_valid_mask"].astype(np.float32)
            if observations.ndim != 2 or actions.ndim != 2 or mask.ndim != 2:
                raise ValueError("aux dataset observations/actions/action_valid_mask must be 2D arrays")
            if observations.shape[0] != actions.shape[0] or actions.shape != mask.shape:
                raise ValueError(
                    "aux dataset shape mismatch: "
                    f"observations={observations.shape}, actions={actions.shape}, mask={mask.shape}"
                )
            if not np.isfinite(observations).all() or not np.isfinite(actions).all() or not np.isfinite(mask).all():
                raise ValueError(f"{self.dataset_path} contains non-finite values")

            policy = self.model.policy
            action_dim = int(policy.action_space.shape[0])
            invalid = [idx for idx in self.action_indices if idx < 0 or idx >= action_dim]
            if invalid:
                raise ValueError(f"invalid aux action indices {invalid}; action_dim={action_dim}")

            selected_mask = mask[:, self.action_indices]
            keep = selected_mask.sum(axis=1) > 0
            if not np.any(keep):
                raise ValueError("aux dataset has no valid labels for requested action indices")

            device = self.model.device
            self.obs_tensor = torch.as_tensor(observations[keep], dtype=torch.float32, device=device)
            self.action_tensor = torch.as_tensor(actions[keep][:, self.action_indices], dtype=torch.float32, device=device)
            self.mask_tensor = torch.as_tensor(selected_mask[keep], dtype=torch.float32, device=device)
            if self.use_sample_weight and "sample_weight" in data:
                sample_weight = data["sample_weight"].astype(np.float32)
                if sample_weight.ndim != 1 or sample_weight.shape[0] != observations.shape[0]:
                    raise ValueError(
                        "sample_weight must be a 1D array matching observation rows: "
                        f"sample_weight={sample_weight.shape}, observations={observations.shape}"
                    )
                if not np.isfinite(sample_weight).all() or np.any(sample_weight < 0.0):
                    raise ValueError("sample_weight must contain finite non-negative values")
                kept_weight = sample_weight[keep]
                if float(kept_weight.sum()) > 0.0:
                    kept_weight = np.power(kept_weight, self.sample_weight_power).astype(np.float32)
                    kept_weight = kept_weight / kept_weight.sum()
                    self.sample_prob_tensor = torch.as_tensor(kept_weight, dtype=torch.float32, device=device)
            self.selected = torch.as_tensor(self.action_indices, dtype=torch.long, device=device)

            extractor = policy.mlp_extractor
            if isinstance(extractor, GroupedActionMlpExtractor):
                selected_set = set(int(index) for index in self.action_indices)
                grouped_head_names = [
                    name
                    for name, indices in extractor.action_groups.items()
                    if selected_set.intersection(indices)
                ]
                if not grouped_head_names:
                    raise ValueError(f"no grouped action heads selected for rows {self.action_indices}")
                params = []
                for name in grouped_head_names:
                    params.extend(extractor.action_heads[name].parameters())
                self._grouped_head_names = grouped_head_names
                self._preserve_flat_unselected = False
            else:
                if not hasattr(policy.action_net, "weight") or not hasattr(policy.action_net, "bias"):
                    raise TypeError(
                        "auxiliary action loss only supports flat action_net policies or "
                        "GroupedActionMlpExtractor policies"
                    )
                params = [policy.action_net.weight, policy.action_net.bias]
                self._grouped_head_names = None
                self._preserve_flat_unselected = True

            self.optimizer = torch.optim.Adam(params, lr=self.lr)

            if self.verbose > 0:
                valid_counts = mask.sum(axis=0).astype(float).tolist()
                print("[BaseAssistAux] enabled")
                print(f"  dataset: {self.dataset_path}")
                print(f"  samples: {self.obs_tensor.shape[0]:,}/{observations.shape[0]:,} usable")
                print(f"  action rows: {self.action_indices}")
                if self._grouped_head_names is not None:
                    print(f"  grouped heads: {self._grouped_head_names}")
                print(f"  valid counts: {valid_counts}")
                print(
                    "  per rollout: "
                    f"{self.gradient_steps} steps, batch={self.batch_size}, lr={self.lr:g}"
                )
                if self.sample_prob_tensor is not None:
                    print(f"  sampling: weighted by dataset sample_weight^{self.sample_weight_power:g}")
                elif "sample_weight" in data and not self.use_sample_weight:
                    print("  sampling: uniform (sample_weight ignored)")
                else:
                    print("  sampling: uniform")

        def _predict_selected(self, obs_batch):
            policy = self.model.policy
            features = policy.extract_features(obs_batch)
            latent_pi, _ = policy.mlp_extractor(features)
            mean_actions = policy.action_net(latent_pi)
            return mean_actions.index_select(dim=1, index=self.selected)

        @staticmethod
        def _masked_mse(pred, target, valid):
            sq = (pred - target).pow(2) * valid
            return sq.sum() / valid.sum().clamp_min(1.0)

        def _on_rollout_start(self) -> None:
            import torch

            self.rollout_count += 1
            policy = self.model.policy
            policy.set_training_mode(True)
            sample_count = self.obs_tensor.shape[0]
            last_loss = 0.0

            if self._preserve_flat_unselected:
                with torch.no_grad():
                    before_weight = policy.action_net.weight.detach().clone()
                    before_bias = policy.action_net.bias.detach().clone()
                    unselected = torch.ones(policy.action_space.shape[0], dtype=torch.bool, device=self.model.device)
                    unselected[self.selected] = False

            for _ in range(self.gradient_steps):
                if self.sample_prob_tensor is not None:
                    batch_idx = torch.multinomial(
                        self.sample_prob_tensor,
                        num_samples=self.batch_size,
                        replacement=True,
                    )
                else:
                    batch_idx = torch.randint(0, sample_count, (self.batch_size,), device=self.model.device)
                obs_b = self.obs_tensor.index_select(0, batch_idx)
                act_b = self.action_tensor.index_select(0, batch_idx)
                mask_b = self.mask_tensor.index_select(0, batch_idx)

                self.optimizer.zero_grad()
                loss = self._masked_mse(self._predict_selected(obs_b), act_b, mask_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]["params"], self.max_grad_norm)
                self.optimizer.step()
                last_loss = float(loss.detach().cpu().item())

                if self._preserve_flat_unselected:
                    with torch.no_grad():
                        policy.action_net.weight[unselected] = before_weight[unselected]
                        policy.action_net.bias[unselected] = before_bias[unselected]

            self.logger.record("train/base_assist_aux_loss", last_loss)
            if self.verbose > 0 and self.rollout_count % self.log_freq == 0:
                print(f"[BaseAssistAux] rollout={self.rollout_count} loss={last_loss:.6f}")

        def _on_step(self) -> bool:
            return True
    
    # Training monitoring callback for detailed metrics
    class TrainingMonitorCallback(BaseCallback):
        """
        Logs detailed training metrics including reward components, errors, and base movement.
        Provides real-time visibility into training progress beyond standard PPO metrics.
        """
        def __init__(self, log_freq: int = 5, verbose: int = 1):
            super().__init__(verbose)
            self.log_freq = log_freq  # Log every N iterations
            self.iteration_count = 0

        @staticmethod
        def _get_env_attr(env, name: str, default=None):
            """Find an attribute through the small wrapper stack used by SB3/IsaacLab."""
            visited = set()
            stack = [env]
            while stack:
                current = stack.pop()
                if current is None or id(current) in visited:
                    continue
                visited.add(id(current))
                if hasattr(current, name):
                    return getattr(current, name)
                for child_name in ("unwrapped", "venv", "env"):
                    child = getattr(current, child_name, None)
                    if child is not None:
                        stack.append(child)
            return default
            
        def _on_rollout_end(self) -> bool:
            """Log detailed metrics at end of each rollout."""
            import torch

            self.iteration_count += 1
            
            # Only log every log_freq iterations
            if self.iteration_count % self.log_freq != 0:
                return True
            
            try:
                # Access the underlying Isaac Lab environment
                env = self.training_env
                while hasattr(env, 'venv'):
                    env = env.venv
                
                # Get environment statistics if available
                if hasattr(env, 'unwrapped'):
                    isaac_env = env.unwrapped
                    
                    # Collect reward component statistics
                    if hasattr(isaac_env, 'episode_sums') and 'rewards' in isaac_env.episode_sums:
                        rewards_dict = isaac_env.episode_sums['rewards']
                        
                        print("\n" + "="*60)
                        print(f"[Training Monitor] Iteration {self.iteration_count} @ {self.num_timesteps/1e6:.1f}M steps")
                        print("="*60)
                        
                        # Reward components (show mean across all envs + log to TensorBoard)
                        print("\n[Reward Components] Mean per episode:")
                        important_rewards = [
                            'position_tracking',
                            'orientation_tracking',
                            'progress_bonus',
                            'base_target_alignment',
                            'base_target_away_penalty',
                            'base_target_command_reward',
                            'base_target_command_away_penalty',
                            'base_target_command_tracking_penalty',
                            'base_assist_imitation_penalty',
                            'reachability_bonus',
                            'reachability_distance_penalty',
                            'position_distance_penalty',
                            'base_overshoot_penalty',
                            'excessive_base_movement_penalty',
                            'velocity_limit_penalty',
                            'jerk_limit_penalty',
                            'obstacle_reward',
                        ]
                        for key in important_rewards:
                            if key in rewards_dict:
                                values = rewards_dict[key]
                                if hasattr(values, 'mean'):
                                    mean_val = values.mean().item()
                                    print(f"  {key:35s}: {mean_val:8.3f}")
                                    # Log to TensorBoard for monitoring
                                    self.logger.record(f"reward_components/{key}", mean_val)
                    
                    # Position and orientation errors
                    if hasattr(isaac_env, '_ee_position') and hasattr(isaac_env, '_target_positions'):
                        import torch
                        pos_error = torch.norm(isaac_env._ee_position - isaac_env._target_positions, dim=-1)
                        print(f"\n[Tracking Errors]")
                        print(f"  Position error (m):  mean={pos_error.mean().item():.4f}, std={pos_error.std().item():.4f}")
                        print(f"                       min={pos_error.min().item():.4f}, max={pos_error.max().item():.4f}")
                    
                    # CRITICAL: Base-target and workspace distance diagnostics
                    if hasattr(isaac_env, '_base_target_distance_buf') or (
                        hasattr(isaac_env, '_robot') and hasattr(isaac_env, '_target_positions')
                    ):
                        if hasattr(isaac_env, '_base_target_distance_buf'):
                            base_target_dist = isaac_env._base_target_distance_buf
                        else:
                            base_pos = isaac_env._robot.data.root_pos_w
                            target_pos = isaac_env._target_positions
                            base_target_dist = torch.norm(target_pos[:, :2] - base_pos[:, :2], dim=-1)
                            base_target_dist = torch.nan_to_num(
                                base_target_dist, nan=1.7, posinf=1.7, neginf=1.7
                            ).clamp(max=1.7)

                        optimal_margin = 0.4
                        hard_margin = isaac_env.reward_weights.get("reachability_hard_margin", 0.6)
                        acceptable_pct = (
                            (base_target_dist > optimal_margin) & (base_target_dist <= hard_margin)
                        ).float().mean().item() * 100
                        optimal_pct = (base_target_dist <= optimal_margin).float().mean().item() * 100
                        unreachable_pct = (base_target_dist > hard_margin).float().mean().item() * 100

                        print(f"\n[Base-Target Distance] critical for reachability shaping")
                        print(f"  Distance (m):        mean={base_target_dist.mean().item():.4f}, std={base_target_dist.std().item():.4f}")
                        print(f"                       min={base_target_dist.min().item():.4f}, max={base_target_dist.max().item():.4f}")
                        print(f"  Zone distribution:   Optimal (<{optimal_margin:.2f}m): {optimal_pct:.1f}%")
                        print(f"                       Acceptable ({optimal_margin:.2f}-{hard_margin:.2f}m): {acceptable_pct:.1f}%")
                        print(f"                       Unreachable (>{hard_margin:.2f}m): {unreachable_pct:.1f}%  <-- keep below 15%")

                        self.logger.record("monitoring/base_target_dist_mean", base_target_dist.mean().item())
                        self.logger.record("monitoring/base_target_dist_std", base_target_dist.std().item())
                        self.logger.record("monitoring/base_target_dist_max", base_target_dist.max().item())
                        self.logger.record("monitoring/optimal_zone_pct", optimal_pct)
                        self.logger.record("monitoring/acceptable_zone_pct", acceptable_pct)
                        self.logger.record("monitoring/unreachable_zone_pct", unreachable_pct)

                    nonfinite_mask = self._get_env_attr(isaac_env, '_reachability_nonfinite_mask')
                    if nonfinite_mask is not None:
                        nonfinite_pct = nonfinite_mask.float().mean().item() * 100
                        print(f"  Non-finite reachability targets: {nonfinite_pct:.1f}%")
                        self.logger.record("monitoring/nonfinite_reachability_pct", nonfinite_pct)

                    if hasattr(isaac_env, '_workspace_distance_buf'):
                        workspace_dist = isaac_env._workspace_distance_buf
                        hard_margin = isaac_env.reward_weights.get("reachability_hard_margin", 0.6)
                        soft_margin = isaac_env.reward_weights.get("reachability_soft_margin", 0.2)
                        beyond_soft_pct = (workspace_dist > soft_margin).float().mean().item() * 100
                        beyond_hard_pct = (workspace_dist > hard_margin).float().mean().item() * 100

                        print(f"\n[Workspace Distance]")
                        print(f"  Distance to workspace (m): mean={workspace_dist.mean().item():.4f}, std={workspace_dist.std().item():.4f}")
                        print(f"                              max={workspace_dist.max().item():.4f}")
                        print(f"  Soft margin ({soft_margin:.2f}m) exceedances: {beyond_soft_pct:.1f}%")
                        print(f"  Hard margin ({hard_margin:.2f}m) exceedances: {beyond_hard_pct:.1f}%")

                        self.logger.record("monitoring/workspace_distance_mean", workspace_dist.mean().item())
                        self.logger.record("monitoring/workspace_distance_std", workspace_dist.std().item())
                        self.logger.record("monitoring/workspace_distance_max", workspace_dist.max().item())
                        self.logger.record("monitoring/workspace_soft_exceed_pct", beyond_soft_pct)
                        self.logger.record("monitoring/workspace_hard_exceed_pct", beyond_hard_pct)
                    
                    if hasattr(isaac_env, '_obstacle_clearance_buf') and getattr(isaac_env, 'obstacles_enabled', False):
                        clearance = isaac_env._obstacle_clearance_buf
                        safety_radius = isaac_env.reward_weights.get("safety_radius", 0.2)
                        unsafe_pct = (clearance < safety_radius).float().mean().item() * 100
                        collision_pct = (clearance < 0.0).float().mean().item() * 100
                        print(f"\n[Obstacle Clearance]")
                        print(f"  Clearance (m):      mean={clearance.mean().item():.4f}, min={clearance.min().item():.4f}")
                        print(f"  Unsafe (<{safety_radius:.2f}m): {unsafe_pct:.1f}%")
                        print(f"  Collision (<0m):    {collision_pct:.1f}%")
                        if hasattr(isaac_env, '_obstacle_xy_buf'):
                            obstacle_xy = isaac_env._obstacle_xy_buf
                            print(
                                f"  Obstacle XY local:  x_mean={obstacle_xy[:, 0].mean().item():.3f}, "
                                f"x_std={obstacle_xy[:, 0].std().item():.3f}, "
                                f"y_mean={obstacle_xy[:, 1].mean().item():.3f}, "
                                f"y_std={obstacle_xy[:, 1].std().item():.3f}"
                            )
                            self.logger.record("monitoring/obstacle_x_mean", obstacle_xy[:, 0].mean().item())
                            self.logger.record("monitoring/obstacle_x_std", obstacle_xy[:, 0].std().item())
                            self.logger.record("monitoring/obstacle_y_mean", obstacle_xy[:, 1].mean().item())
                            self.logger.record("monitoring/obstacle_y_std", obstacle_xy[:, 1].std().item())
                        self.logger.record("monitoring/obstacle_clearance_mean", clearance.mean().item())
                        self.logger.record("monitoring/obstacle_clearance_min", clearance.min().item())
                        self.logger.record("monitoring/obstacle_unsafe_pct", unsafe_pct)
                        self.logger.record("monitoring/obstacle_collision_pct", collision_pct)

                    # Base movement statistics
                    if hasattr(isaac_env, '_robot') and hasattr(isaac_env._robot.data, 'root_lin_vel_w'):
                        base_vel = isaac_env._robot.data.root_lin_vel_w
                        base_speed = torch.norm(base_vel[:, :2], dim=-1)  # Planar speed
                        print(f"\n[Base Movement]")
                        print(f"  Linear speed (m/s): mean={base_speed.mean().item():.4f}, std={base_speed.std().item():.4f}")
                        print(f"                      min={base_speed.min().item():.4f}, max={base_speed.max().item():.4f}")
                        
                        # Angular velocity
                        if hasattr(isaac_env._robot.data, 'root_ang_vel_w'):
                            base_ang_vel = isaac_env._robot.data.root_ang_vel_w[:, 2]  # Yaw rate
                            print(f"  Yaw rate (rad/s):   mean={base_ang_vel.mean().item():.4f}, std={base_ang_vel.std().item():.4f}")

                    coeff = self._get_env_attr(isaac_env, '_base_assist_coeff')
                    if coeff is not None:
                        active_pct = (coeff > 0.0).float().mean().item() * 100
                        print(f"\n[Base Assist]")
                        print(f"  Coeff:              mean={coeff.mean().item():.4f}, active={active_pct:.1f}%")
                        self.logger.record("monitoring/base_assist_coeff_mean", coeff.mean().item())
                        self.logger.record("monitoring/base_assist_active_pct", active_pct)
                        expert = self._get_env_attr(isaac_env, '_base_assist_expert_action')
                        if expert is not None:
                            expert_mag = torch.norm(expert, dim=-1)
                            print(
                                f"  Expert action:      mag_mean={expert_mag.mean().item():.4f}, "
                                f"mag_max={expert_mag.max().item():.4f}"
                            )
                            self.logger.record("monitoring/base_assist_expert_mag_mean", expert_mag.mean().item())
                            self.logger.record("monitoring/base_assist_expert_mag_max", expert_mag.max().item())
                            prev_actions = self._get_env_attr(isaac_env, 'prev_actions')
                            if prev_actions is not None:
                                policy_base = prev_actions[:, -3:-1]
                                policy_error = torch.norm(policy_base - expert, dim=-1)
                                active_mask = coeff > 0.0
                                active_error = (
                                    policy_error[active_mask].mean().item()
                                    if torch.any(active_mask)
                                    else 0.0
                                )
                                print(
                                    f"  Policy error:       mean={policy_error.mean().item():.4f}, "
                                    f"active_mean={active_error:.4f}"
                                )
                                self.logger.record("monitoring/base_assist_policy_error_mean", policy_error.mean().item())
                                self.logger.record("monitoring/base_assist_policy_error_active_mean", active_error)
                    
                    # Reachability statistics (if available from recent logs)
                    if hasattr(isaac_env, '_last_reachability_stats'):
                        stats = isaac_env._last_reachability_stats
                        reachable = stats.get('reachable', 0)
                        total = stats.get('total', 1)
                        percentage = (reachable / total * 100) if total > 0 else 0
                        distance = stats.get('avg_distance', 0)
                        alignment = stats.get('avg_alignment', 0)
                        print(f"\n[Reachability] (from latest stats)")
                        print(f"  Reachable envs: {reachable}/{total} ({percentage:.1f}%)")
                        print(f"  Avg distance:   {distance:.3f} m")
                        print(f"  Avg alignment:  {alignment:.3f}")
                    
                    print("="*60 + "\n")
                    
            except Exception as e:
                # Silently skip if environment doesn't expose these metrics
                if self.verbose > 1:
                    print(f"[TrainingMonitor] Could not access detailed metrics: {e}")
            
            return True

        def _on_step(self) -> bool:
            """Called at every step. Required by BaseCallback."""
            return True

    class ObstacleCurriculumCallback(BaseCallback):
        """Ramp obstacle placement and reward weight from easy to target settings."""

        def __init__(
            self,
            start_x_range,
            final_x_range,
            start_y_range,
            final_y_range,
            start_clearance: float,
            final_clearance: float,
            start_weight: float,
            final_weight: float,
            transition_steps: int,
            verbose: int = 1,
        ):
            super().__init__(verbose)
            self.start_x_range = tuple(float(v) for v in start_x_range)
            self.final_x_range = tuple(float(v) for v in final_x_range)
            self.start_y_range = tuple(float(v) for v in start_y_range)
            self.final_y_range = tuple(float(v) for v in final_y_range)
            self.start_clearance = float(start_clearance)
            self.final_clearance = float(final_clearance)
            self.start_weight = float(start_weight)
            self.final_weight = float(final_weight)
            self.transition_steps = max(int(transition_steps), 1)
            self._last_logged_bucket = None

        @staticmethod
        def _lerp_pair(start, final, progress: float):
            return tuple(start[i] + (final[i] - start[i]) * progress for i in range(2))

        @staticmethod
        def _unwrap_isaac_env(env):
            while hasattr(env, "venv"):
                env = env.venv
            return env.unwrapped if hasattr(env, "unwrapped") else env

        def _on_training_start(self) -> None:
            self._apply_curriculum()

        def _on_rollout_end(self) -> bool:
            self._apply_curriculum()
            return True

        def _on_step(self) -> bool:
            return True

        def _apply_curriculum(self) -> None:
            isaac_env = self._unwrap_isaac_env(self.training_env)
            if not getattr(isaac_env, "obstacles_enabled", False):
                return

            progress = min(max(float(self.num_timesteps) / float(self.transition_steps), 0.0), 1.0)
            x_range = self._lerp_pair(self.start_x_range, self.final_x_range, progress)
            y_range = self._lerp_pair(self.start_y_range, self.final_y_range, progress)
            clearance = self.start_clearance + (self.final_clearance - self.start_clearance) * progress
            weight = self.start_weight + (self.final_weight - self.start_weight) * progress

            obstacle_cfg = isaac_env.task_cfg.obstacles
            obstacle_cfg.disc_position_x_range = x_range
            obstacle_cfg.disc_position_y_range = y_range
            obstacle_cfg.min_start_clearance = clearance
            isaac_env.reward_weights["min_obstacle_distance_weight"] = weight

            self.logger.record("curriculum/obstacle_progress", progress)
            self.logger.record("curriculum/obstacle_x_min", x_range[0])
            self.logger.record("curriculum/obstacle_x_max", x_range[1])
            self.logger.record("curriculum/obstacle_y_min", y_range[0])
            self.logger.record("curriculum/obstacle_y_max", y_range[1])
            self.logger.record("curriculum/obstacle_min_start_clearance", clearance)
            self.logger.record("curriculum/obstacle_weight", weight)

            bucket = int(progress * 10)
            bucket_changed = bucket != self._last_logged_bucket
            if bucket_changed and hasattr(isaac_env, "_randomize_obstacles"):
                try:
                    import torch

                    robot = getattr(isaac_env, "robot", None) or getattr(isaac_env, "_robot", None)
                    if robot is not None and hasattr(robot, "data"):
                        env_ids = torch.arange(isaac_env.num_envs, device=isaac_env.device)
                        base_xy_local = (
                            robot.data.root_pos_w[env_ids, :2]
                            - isaac_env.scene.env_origins[env_ids, :2]
                        )
                        isaac_env._randomize_obstacles(env_ids, base_xy_local)
                except Exception as exc:
                    if self.verbose > 0:
                        print(f"[ObstacleCurriculum] WARNING: obstacle resample failed: {exc}")

            if self.verbose > 0 and bucket != self._last_logged_bucket:
                print(
                    "[ObstacleCurriculum] "
                    f"step={self.num_timesteps:,}, progress={progress:.2f}, "
                    f"x=({x_range[0]:.2f},{x_range[1]:.2f}), "
                    f"y=({y_range[0]:.2f},{y_range[1]:.2f}), "
                    f"clearance={clearance:.2f}, weight={weight:.2f}"
                )
                self._last_logged_bucket = bucket
    
    # Session 8h: Auto-pause callback for training instability detection
    class AutoPauseCallback(BaseCallback):
        """
        Monitors training stability and automatically pauses if instability detected.
        
        Session 8g failed catastrophically due to no auto-pause mechanism.
        Session 8h implements proactive monitoring to prevent collapse.
        
        Monitors:
        - KL divergence: Triggers if exceeds threshold (default 0.1)
        - Explained variance: Triggers if drops below threshold (default 0.0)
        
        When triggered:
        - Saves emergency checkpoint
        - Prints rollback guidance
        - Stops training immediately
        """
        def __init__(
            self, 
            enable: bool = True,
            kl_threshold: float = 0.1,
            variance_threshold: float = 0.0,
            variance_patience: int = 3,
            variance_clip_threshold: float = 0.25,
            warmup_steps: int = 500_000,  # Skip monitoring in first 500K steps (early instability is normal)
            checkpoint_dir: str = None,
            verbose: int = 1
        ):
            super().__init__(verbose)
            self.enable = enable
            self.kl_threshold = kl_threshold
            self.variance_threshold = variance_threshold
            self.variance_patience = variance_patience
            self.variance_clip_threshold = variance_clip_threshold
            self.variance_violation_count = 0
            self.warmup_steps = warmup_steps
            self.checkpoint_dir = checkpoint_dir
            self.triggered = False
            
            if self.enable:
                print(f"[AutoPause] Monitoring enabled:")
                print(f"  KL threshold: {kl_threshold}")
                print(f"  Variance threshold: {variance_threshold} (patience={variance_patience})")
                print(f"  Variance requires clip_fraction > {variance_clip_threshold} to pause")
                print(f"  Warmup period: {warmup_steps:,} steps (monitoring starts after)")
        
        def _on_rollout_end(self) -> bool:
            """Check stability metrics after each rollout."""
            if not self.enable or self.triggered:
                return True
            
            # Skip monitoring during warmup period
            if self.num_timesteps < self.warmup_steps:
                return True
            
            try:
                # Get metrics from PPO logger
                if hasattr(self.logger, 'name_to_value'):
                    current_kl = self.logger.name_to_value.get('train/approx_kl', 0.0)
                    current_variance = self.logger.name_to_value.get('train/explained_variance', 1.0)
                    current_clip_fraction = self.logger.name_to_value.get('train/clip_fraction', 0.0)
                    
                    # Check KL divergence
                    if current_kl > self.kl_threshold:
                        self._trigger_pause(
                            reason=f"KL divergence exceeded threshold: {current_kl:.4f} > {self.kl_threshold}",
                            current_kl=current_kl,
                            current_variance=current_variance
                        )
                        return False
                    
                    # Check explained variance
                    if current_variance < self.variance_threshold:
                        update_is_aggressive = current_clip_fraction > self.variance_clip_threshold
                        if self.verbose > 0:
                            print(
                                f"[AutoPause] Explained variance below threshold "
                                f"({current_variance:.4f} < {self.variance_threshold}), "
                                f"clip_fraction={current_clip_fraction:.4f}"
                            )
                        if update_is_aggressive:
                            self.variance_violation_count += 1
                            if self.verbose > 0:
                                print(
                                    f"[AutoPause] Low-variance violation "
                                    f"{self.variance_violation_count}/{self.variance_patience}"
                                )
                        else:
                            if self.variance_violation_count and self.verbose > 0:
                                print("[AutoPause] Policy update is stable; resetting variance violation counter")
                            self.variance_violation_count = 0
                            return True
                        if self.variance_violation_count >= self.variance_patience:
                            self._trigger_pause(
                                reason=(
                                    "Explained variance remained below threshold for "
                                    f"{self.variance_violation_count} aggressive rollouts: "
                                    f"{current_variance:.4f} < {self.variance_threshold}, "
                                    f"clip_fraction={current_clip_fraction:.4f} > {self.variance_clip_threshold}"
                                ),
                                current_kl=current_kl,
                                current_variance=current_variance
                            )
                            return False
                    else:
                        if self.variance_violation_count and self.verbose > 0:
                            print("[AutoPause] Explained variance recovered; resetting violation counter")
                        self.variance_violation_count = 0
                
            except Exception as e:
                if self.verbose > 1:
                    print(f"[AutoPause] Error checking metrics: {e}")
            
            return True
        
        def _trigger_pause(self, reason: str, current_kl: float, current_variance: float):
            """Trigger auto-pause and save emergency checkpoint."""
            self.triggered = True
            
            print("\n" + "!"*80)
            print("!!! AUTO-PAUSE TRIGGERED !!!")
            print("!"*80)
            print(f"\nReason: {reason}")
            print(f"\nCurrent Metrics:")
            print(f"  KL divergence: {current_kl:.4f} (threshold: {self.kl_threshold})")
            print(f"  Explained variance: {current_variance:.4f} (threshold: {self.variance_threshold})")
            print(f"  Timesteps: {self.num_timesteps:,} ({self.num_timesteps/1e6:.1f}M)")
            
            # Save emergency checkpoint
            if self.checkpoint_dir:
                import os
                emergency_path = os.path.join(
                    self.checkpoint_dir, 
                    f"emergency_pause_{self.num_timesteps}_steps"
                )
                self.model.save(emergency_path)
                print(f"\n✅ Emergency checkpoint saved: {emergency_path}.zip")
            
            print("\n" + "="*80)
            print("ROLLBACK GUIDANCE")
            print("="*80)
            print("\n1. Identify last stable checkpoint (before instability):")
            print("   - Check logs for when metrics started degrading")
            print("   - Look for checkpoints saved every 2M steps")
            print("\n2. Resume from stable checkpoint:")
            print("   python train.py --checkpoint <stable_checkpoint.zip> --total_timesteps <remaining>")
            print("\n3. Consider adjustments:")
            print("   - Reduce learning rate: --learning_rate 2e-4 (was 3e-4)")
            print("   - Extend transition period: curriculum_transition_steps=20M (was 10M)")
            print("   - Increase orientation weight: curriculum_stage_2_orientation_weight=40.0 (was 30.0)")
            print("\n4. Session 8g lesson:")
            print("   - Variance dropped to -0.241 @ 36M but no auto-pause → collapsed @ 100M")
            print("   - Session 8h has auto-pause to prevent repeating this mistake")
            print("\n" + "!"*80)
            print("TRAINING PAUSED - Review metrics and rollback to stable checkpoint")
            print("!"*80 + "\n")
        
        def _on_step(self) -> bool:
            """Called at every step."""
            return not self.triggered  # Stop if triggered
    
    # Create wrapper to convert Isaac Lab observations to SB3 format
    class IsaacLabToSB3VecEnvWrapper(VecEnvWrapper):
        """VecEnv wrapper to convert Isaac Lab's dict observations with torch tensors to numpy arrays for SB3."""
        
        def __init__(
            self,
            venv,
            expected_obs_dim: int | None = None,
            freeze_base_actions: bool = False,
            base_action_scale: float = 1.0,
            base_action_delta_limit: float = 0.0,
            freeze_base_actions_for_non_base_required: bool = False,
        ):
            # Isaac Lab env is already a VecEnv
            VecEnvWrapper.__init__(self, venv)
            # We'll update observation space after first reset
            self._obs_space_updated = False
            self.expected_obs_dim = expected_obs_dim
            self._obs_adapter_logged = False
            self.freeze_base_actions = bool(freeze_base_actions)
            self.freeze_base_actions_for_non_base_required = bool(freeze_base_actions_for_non_base_required)
            self._freeze_base_logged = False
            self.base_action_scale = float(base_action_scale)
            self.base_action_delta_limit = float(base_action_delta_limit)
            self._prev_adapted_actions = None
            self._base_scale_logged = False
            
            # FIX: Isaac Lab's action_space includes batch dimension [num_envs, action_dim]
            # SB3 expects per-env action space [action_dim]
            if hasattr(venv.action_space, 'shape') and len(venv.action_space.shape) > 1:
                # Remove the num_envs dimension
                action_dim = venv.action_space.shape[-1]  # Last dimension is action_dim
                self.action_space = spaces.Box(
                    low=venv.action_space.low.flatten()[0],  # All actions have same limits
                    high=venv.action_space.high.flatten()[0],
                    shape=(action_dim,),
                    dtype=venv.action_space.dtype
                )
                print(f"[IsaacLabToSB3VecEnvWrapper] Fixed action_space: {venv.action_space.shape} -> {self.action_space.shape}")

        def _adapt_actions(self, actions):
            if actions.shape[-1] < 9:
                raise ValueError(
                    f"base action adapter requires at least 9 action dims, got {actions.shape[-1]}"
                )
            if (
                not self.freeze_base_actions
                and not self.freeze_base_actions_for_non_base_required
                and abs(self.base_action_scale - 1.0) < 1e-9
                and self.base_action_delta_limit <= 0.0
            ):
                return actions
            actions = actions.clone()
            if not self._freeze_base_logged:
                if self.freeze_base_actions:
                    print("[IsaacLabToSB3VecEnvWrapper] Freezing base action rows [6,7,8]")
                elif self.freeze_base_actions_for_non_base_required:
                    print(
                        "[IsaacLabToSB3VecEnvWrapper] Contract-aware mixed base adapter: "
                        "scale base_required rows and freeze non-base-required rows"
                    )
                else:
                    print(
                        "[IsaacLabToSB3VecEnvWrapper] Scaling base action rows [6,7,8] "
                        f"by {self.base_action_scale:.3f}"
                    )
                if self.base_action_delta_limit > 0.0:
                    print(
                        "[IsaacLabToSB3VecEnvWrapper] Limiting base action delta rows [6,7,8] "
                        f"to {self.base_action_delta_limit:.3f}/step"
                    )
                self._freeze_base_logged = True
            if self.freeze_base_actions:
                actions[..., 6:9] = 0.0
            elif self.freeze_base_actions_for_non_base_required:
                actions[..., 6:9] *= self.base_action_scale
                raw_env = self.venv.unwrapped if hasattr(self.venv, "unwrapped") else self.venv
                manager = getattr(raw_env, "trajectory_manager", None)
                metadata = getattr(manager, "current_trajectory_metadata", None) if manager is not None else None
                if metadata:
                    keep = [
                        "base_required" in str(item.get("file", ""))
                        if isinstance(item, dict)
                        else False
                        for item in metadata[: actions.shape[0]]
                    ]
                    if len(keep) < actions.shape[0]:
                        keep.extend([False] * (actions.shape[0] - len(keep)))
                    keep_tensor = torch.as_tensor(keep, dtype=torch.bool, device=actions.device)
                    actions[~keep_tensor, 6:9] = 0.0
                else:
                    actions[..., 6:9] = 0.0
            else:
                actions[..., 6:9] *= self.base_action_scale
            if self.base_action_delta_limit > 0.0:
                if self._prev_adapted_actions is None or self._prev_adapted_actions.shape != actions.shape:
                    self._prev_adapted_actions = torch.zeros_like(actions)
                base_delta = torch.clamp(
                    actions[..., 6:9] - self._prev_adapted_actions[..., 6:9],
                    min=-self.base_action_delta_limit,
                    max=self.base_action_delta_limit,
                )
                actions[..., 6:9] = self._prev_adapted_actions[..., 6:9] + base_delta
                self._prev_adapted_actions = actions.detach().clone()
            return actions

        def _trajectory_progress_column(self, obs: np.ndarray) -> np.ndarray:
            raw_env = self.venv.unwrapped if hasattr(self.venv, "unwrapped") else self.venv
            manager = getattr(raw_env, "trajectory_manager", None)
            num_envs = obs.shape[0] if obs.ndim == 2 else 1
            if manager is None:
                return np.zeros((num_envs, 1), dtype=np.float32)

            current_idx = getattr(manager, "current_waypoint_idx", None)
            lengths = getattr(manager, "recorded_lengths", None)
            if current_idx is None or lengths is None:
                return np.zeros((num_envs, 1), dtype=np.float32)

            if hasattr(current_idx, "detach"):
                current_idx = current_idx.detach()
            if hasattr(lengths, "detach"):
                lengths = lengths.detach()
            if hasattr(current_idx, "cpu"):
                current_idx = current_idx.cpu()
            if hasattr(lengths, "cpu"):
                lengths = lengths.cpu()

            current_np = np.asarray(current_idx, dtype=np.float32).reshape(-1)
            lengths_np = np.asarray(lengths, dtype=np.float32).reshape(-1)
            if current_np.shape[0] < num_envs or lengths_np.shape[0] < num_envs:
                return np.zeros((num_envs, 1), dtype=np.float32)
            denom = np.maximum(lengths_np[:num_envs] - 1.0, 1.0)
            progress = np.clip(current_np[:num_envs] / denom, 0.0, 1.0)
            return progress.reshape(num_envs, 1).astype(np.float32, copy=False)

        def _adapt_obs_dim(self, obs: np.ndarray) -> np.ndarray:
            if self.expected_obs_dim is None:
                return obs
            if obs.ndim == 2 and obs.shape[1] == self.expected_obs_dim:
                return obs
            if obs.ndim == 1 and obs.shape[0] == self.expected_obs_dim:
                return obs
            if obs.ndim == 2 and obs.shape[1] == self.expected_obs_dim - 1:
                if not self._obs_adapter_logged:
                    print(
                        "[IsaacLabToSB3VecEnvWrapper] Appending trajectory progress column "
                        f"for checkpoint compatibility: {obs.shape[1]} -> {self.expected_obs_dim}"
                    )
                    self._obs_adapter_logged = True
                return np.concatenate([obs, self._trajectory_progress_column(obs)], axis=1)
            if obs.ndim == 2 and obs.shape[1] == self.expected_obs_dim + 1:
                if not self._obs_adapter_logged:
                    print(
                        "[IsaacLabToSB3VecEnvWrapper] Dropping final observation column "
                        f"for checkpoint compatibility: {obs.shape[1]} -> {self.expected_obs_dim}"
                    )
                    self._obs_adapter_logged = True
                return obs[:, : self.expected_obs_dim]
            if obs.ndim == 1 and obs.shape[0] == self.expected_obs_dim - 1:
                return np.concatenate([obs, np.zeros((1,), dtype=np.float32)], axis=0)
            if obs.ndim == 1 and obs.shape[0] == self.expected_obs_dim + 1:
                return obs[: self.expected_obs_dim]
            return obs

        def reset(self):
            obs = self.venv.reset()
            self._prev_adapted_actions = None
            
            # Handle new Gymnasium API: reset() returns (obs, info)
            if isinstance(obs, tuple):
                obs, info = obs
            
            # Convert dict of torch tensors to numpy array
            if isinstance(obs, dict):
                obs_tensor = obs.get("policy", list(obs.values())[0])
                if hasattr(obs_tensor, 'cpu'):
                    obs = obs_tensor.cpu().numpy()
                else:
                    obs = np.array(obs_tensor)
                obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
                obs = self._adapt_obs_dim(obs)
                
                # Update observation space on first reset
                if not self._obs_space_updated:
                    # For the observation space, we want per-env shape without batch dim
                    obs_shape = obs.shape[1:] if len(obs.shape) > 1 else obs.shape
                    self.observation_space = spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=obs_shape,
                        dtype=np.float32
                    )
                    self._obs_space_updated = True
            return obs
        
        def step_async(self, actions):
            # Isaac Lab envs use sync step(), not async
            # Store actions for step_wait to process
            # Convert numpy actions to torch tensors for Isaac Lab and move to device
            import torch
            if isinstance(actions, np.ndarray):
                # Get the device from the underlying environment
                device = self.venv.unwrapped.device if hasattr(self.venv.unwrapped, 'device') else 'cuda:0'
                actions = torch.from_numpy(actions).float().to(device)
            actions = self._adapt_actions(actions)
            self._actions = actions
        
        def step_wait(self):
            # Call the synchronous step() method with stored actions
            result = self.venv.step(self._actions)
            
            # Handle both old (4 values) and new (5 values) Gymnasium API
            if len(result) == 5:
                # New Gymnasium API: (obs, reward, terminated, truncated, info)
                obs, rewards, terminated, truncated, infos = result
                # Combine terminated and truncated into done for old API
                dones = terminated | truncated
            else:
                # Old Gym API: (obs, reward, done, info)
                obs, rewards, dones, infos = result
            
            # Convert observations
            if isinstance(obs, dict):
                obs_tensor = obs.get("policy", list(obs.values())[0])
                if hasattr(obs_tensor, 'cpu'):
                    obs = obs_tensor.cpu().numpy()
                else:
                    obs = np.array(obs_tensor)
                obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
                obs = self._adapt_obs_dim(obs)
            
            # Convert rewards and dones to numpy
            if hasattr(rewards, 'cpu'):
                rewards = rewards.cpu().numpy()
            rewards = np.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
            if hasattr(dones, 'cpu'):
                dones = dones.cpu().numpy()
            done_arr = np.asarray(dones, dtype=bool).reshape(-1)
            if (
                self._prev_adapted_actions is not None
                and done_arr.size > 0
                and done_arr.size == self._prev_adapted_actions.shape[0]
            ):
                done_tensor = torch.as_tensor(
                    done_arr,
                    dtype=torch.bool,
                    device=self._prev_adapted_actions.device,
                )
                self._prev_adapted_actions[done_tensor] = 0.0
            
            # Ensure infos is a list of dicts (SB3 expects this format)
            # Isaac Lab sometimes returns infos as a dict or other format
            if isinstance(infos, dict):
                # If infos is a single dict, wrap it in a list for each env
                infos = [infos.copy() for _ in range(len(rewards))]
            elif not isinstance(infos, list):
                # If infos is something else, create empty dicts
                infos = [{} for _ in range(len(rewards))]
            else:
                # infos is already a list, but ensure each element is a dict
                infos = [info if isinstance(info, dict) else {} for info in infos]
            
            return obs, rewards, dones, infos
    
    print("    [OK] Isaac Lab to SB3 VecEnv wrapper created")
    
    # Step 4: Setup logging and W&B
    print("\n[4/6] Setting up logging...")
    if args.log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_name = args.task.replace("-", "_").lower()
        args.log_dir = str(PROJECT_ROOT / f"logs/sb3/{task_name}/{timestamp}")
    
    os.makedirs(args.log_dir, exist_ok=True)
    print(f"    Log directory: {args.log_dir}")
    
    # Setup W&B if requested
    if args.wandb:
        try:
            import wandb
            from wandb.integration.sb3 import WandbCallback
            
            wandb.init(
                project=args.wandb_project,
                name=f"{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=vars(args),
                sync_tensorboard=True,
            )
            wandb_callback = WandbCallback(
                model_save_freq=args.save_freq,
                model_save_path=args.log_dir,
                verbose=2,
            )
            print("    [OK] W&B logging enabled")
        except ImportError:
            print("    [WARN]  wandb not available, skipping W&B logging")
            args.wandb = False
    
    # Step 5: Create environment
    print(f"\n[5/6] Creating training environment...")
    print(f"    Task: {args.task}")
    print(f"    Num envs: {args.num_envs}")
    print(f"    Headless: {args.headless}")
    print(f"    Trajectory type: {args.trajectory_type}")
    
    # Configure trajectory settings based on command-line args
    trajectory_config = {}
    if args.trajectory_type == "multi_recorded":
        print(f"    Trajectory directory: {args.trajectory_dir}")
        
        # Session 8h: Trajectory curriculum stages
        stage_reward_overrides = {}
        if args.trajectory_stage:
            print(f"    [Session 8h] Using trajectory curriculum: {args.trajectory_stage}")
            stage_dir = PROJECT_ROOT / "trajectoryToLearn" / args.trajectory_stage
            if stage_dir.exists():
                manifest_file = stage_dir / "manifest.txt"
                reset_config_file = stage_dir / "reset_config.json"
                if reset_config_file.exists():
                    with reset_config_file.open("r", encoding="utf-8") as f:
                        reset_config = json.load(f)
                    if "reset_base_x_offset" in reset_config:
                        trajectory_config["reset_base_x_offset"] = float(reset_config["reset_base_x_offset"])
                    if "reset_base_y_offset" in reset_config:
                        trajectory_config["reset_base_y_offset"] = float(reset_config["reset_base_y_offset"])
                    if "reset_arm_to_trajectory_metadata" in reset_config:
                        trajectory_config["reset_arm_to_trajectory_metadata"] = bool(
                            reset_config["reset_arm_to_trajectory_metadata"]
                        )
                    raw_reward_overrides = reset_config.get("reward_overrides", {})
                    if isinstance(raw_reward_overrides, dict):
                        stage_reward_overrides = {
                            str(name): float(value)
                            for name, value in raw_reward_overrides.items()
                        }
                    if "reset_base_x_offset" in trajectory_config and "reset_base_y_offset" in trajectory_config:
                        print(
                            "    [OK] Stage reset offset: "
                            f"x={trajectory_config['reset_base_x_offset']:.4f}, "
                            f"y={trajectory_config['reset_base_y_offset']:.4f}"
                        )
                stage_files = [p for p in stage_dir.rglob("*.json") if "__MACOSX" not in str(p)]
                if manifest_file.exists():
                    trajectory_config['stage_dir'] = str(PROJECT_ROOT)
                    trajectory_config['manifest_file'] = str(manifest_file)
                    trajectory_config['filter_indices'] = None
                    print(f"    [OK] Using stage manifest: {manifest_file}")
                elif stage_files:
                    # Use trajectories from stage directory
                    trajectory_config['stage_dir'] = str(stage_dir)
                    trajectory_config['filter_indices'] = None
                    print(f"    [OK] Found {len(stage_files)} trajectories from {args.trajectory_stage}")
                    print(f"      Stage directory: {stage_dir}")
                else:
                    # Stage exists but empty - fall back to chassis-only as proxy
                    print(f"    [WARN] {args.trajectory_stage} exists but contains no trajectories")
                    print(f"    [INFO] Falling back to chassis-only as stage0 proxy")
                    args.use_chassis_only = True  # Enable fallback
            else:
                print(f"    [WARN] Stage directory {stage_dir} does not exist")
                print(f"    [INFO] Falling back to chassis-only as stage0 proxy")
                args.use_chassis_only = True  # Enable fallback
        
        # Determine which trajectories to use (original logic + stage fallback)
        if args.use_chassis_only:
            if args.trajectory_stage:
                print("    Using chassis-only trajectories as {args.trajectory_stage} proxy")
            else:
                print("    [WARN]  Using ONLY chassis-requiring trajectories (for testing, not recommended for training)")
            # Load chassis-required indices
            chassis_indices_file = "data/trajectory_filters/chassis_required_indices.txt"
            if Path(chassis_indices_file).exists():
                import re
                with open(chassis_indices_file, 'r') as f:
                    content = f.read()
                match = re.search(r'CHASSIS_REQUIRED_INDICES = \[(.*?)\]', content, re.DOTALL)
                if match:
                    indices_str = match.group(1)
                    chassis_indices = [int(x.strip()) for x in indices_str.replace('\n', ' ').split(',') if x.strip()]
                    trajectory_config['filter_indices'] = chassis_indices
                    if args.max_trajectories:
                        trajectory_config['filter_indices'] = chassis_indices[:args.max_trajectories]
                    print(f"    Loaded {len(trajectory_config['filter_indices'])} chassis-requiring trajectory indices")
                else:
                    print(f"    [WARN]  Could not parse {chassis_indices_file}, using all trajectories")
            else:
                print(f"    [WARN]  {chassis_indices_file} not found, using all trajectories")
        elif args.use_all_trajectories:
            print("    [OK] Using ALL trajectories (recommended for training diverse policy)")
            trajectory_config['filter_indices'] = None
            if args.max_trajectories:
                print(f"    Limited to first {args.max_trajectories} trajectories")
        else:
            # Default behavior - use all
            if not args.trajectory_stage:  # Don't print if stage already handled
                print("    Using all available trajectories (default)")
            trajectory_config['filter_indices'] = None
        
        trajectory_config['max_trajectories'] = args.max_trajectories
    
    try:
        # Import environment config to modify it
        from rl_platform.tasks.mobile_mm import MobileMMTrackEEEnvCfg
        from rl_platform.tasks.mobile_mm.config import TrajectoryConfig
        
        # Create custom environment configuration
        env_cfg = MobileMMTrackEEEnvCfg()
        env_cfg.num_envs = args.num_envs
        env_cfg.scene.num_envs = args.num_envs
        if args.episode_length_s > 0.0:
            env_cfg.episode_length_s = float(args.episode_length_s)
            print(f"    Episode length override: {env_cfg.episode_length_s:.2f}s")
        if args.base_action_slew_limit > 0.0:
            env_cfg.task_config.base_action_slew_limit = float(args.base_action_slew_limit)
            print(f"    Base action env slew limit: {env_cfg.task_config.base_action_slew_limit:.3f}/step")
        if args.seed is not None:
            env_cfg.seed = args.seed
            env_cfg.task_config.obstacles.seed = args.seed
            print(f"    Seed: {args.seed}")
        env_cfg.task_config.debug_resets = args.debug_resets
        env_cfg.task_config.action_contract_name = args.action_contract
        env_cfg.task_config.experimental_rs4_adapter = args.experimental_rs4_adapter
        print(
            "    Action contract: "
            f"{env_cfg.task_config.action_contract_name} "
            f"(experimental_rs4_adapter={env_cfg.task_config.experimental_rs4_adapter})"
        )
        
        # Convert trajectory_dir to absolute path if it's relative
        # Session 8h: Use stage_dir if trajectory curriculum is active
        if 'stage_dir' in trajectory_config:
            trajectory_dir = trajectory_config['stage_dir']
            print(f"    [Session 8h] Using stage directory: {trajectory_dir}")
        else:
            trajectory_dir = args.trajectory_dir
            if not Path(trajectory_dir).is_absolute():
                trajectory_dir = str(PROJECT_ROOT / trajectory_dir)
                print(f"    Resolved relative path to: {trajectory_dir}")
        
        env_cfg.task_config.obstacles.enable_obstacles = args.enable_obstacles
        env_cfg.task_config.obstacles.disc_position_xy = (args.obstacle_x, args.obstacle_y)
        if args.obstacle_radius is not None:
            env_cfg.task_config.obstacles.disc_radius = args.obstacle_radius
        if args.obstacle_height is not None:
            env_cfg.task_config.obstacles.disc_height = args.obstacle_height
        env_cfg.task_config.obstacles.randomize_per_reset = not args.disable_obstacle_randomization
        env_cfg.task_config.obstacles.disc_position_x_range = tuple(args.obstacle_x_range)
        env_cfg.task_config.obstacles.disc_position_y_range = tuple(args.obstacle_y_range)
        env_cfg.task_config.obstacles.min_start_clearance = args.min_obstacle_start_clearance
        if args.enable_obstacles:
            env_cfg.scene = env_cfg._create_scene_config()
            env_cfg.scene.num_envs = args.num_envs
            obstacle_cfg = env_cfg.task_config.obstacles
            print(
                f"    Obstacle disc: pos=({args.obstacle_x:.2f}, {args.obstacle_y:.2f}), "
                f"radius={obstacle_cfg.disc_radius:.2f}m, "
                f"height={obstacle_cfg.disc_height:.2f}m, "
                f"randomized={not args.disable_obstacle_randomization}, "
                f"x_range=({args.obstacle_x_range[0]:.2f}, {args.obstacle_x_range[1]:.2f}), "
                f"y_range=({args.obstacle_y_range[0]:.2f}, {args.obstacle_y_range[1]:.2f})"
            )

        auto_recovery_reset = args.trajectory_stage == "stage1_recovery"
        randomize_start_waypoint = args.random_start_waypoint or auto_recovery_reset
        reset_base_to_trajectory_start = args.reset_base_to_trajectory_start or auto_recovery_reset
        auto_base_assist = auto_recovery_reset and not args.disable_auto_base_assist
        enable_base_assist = args.enable_base_assist or auto_base_assist
        reward_overrides = {
            "reachability_distance_weight": args.reachability_distance_weight,
            "base_target_far_penalty": args.base_target_far_penalty,
            "base_target_command_tracking_penalty": args.base_target_command_tracking_penalty,
            "base_target_regression_penalty": args.base_target_regression_penalty,
            "base_action_delta": args.base_action_delta_penalty,
            "min_obstacle_distance_weight": args.min_obstacle_distance_weight,
            "safety_radius": args.safety_radius,
        }
        applied_reward_overrides = {
            name: value for name, value in reward_overrides.items() if value is not None
        }
        applied_reward_overrides.update(stage_reward_overrides)
        for name, value in applied_reward_overrides.items():
            setattr(env_cfg.task_config.rewards, name, float(value))
        if applied_reward_overrides:
            print(
                "    Reward overrides: "
                + ", ".join(f"{name}={value:g}" for name, value in applied_reward_overrides.items())
            )
        if auto_recovery_reset:
            reset_blend_desc = f"{args.reset_anchor_target_blend:.2f}"
            if args.reset_anchor_target_blend_initial is not None and args.reset_anchor_target_blend_decay_steps > 0:
                reset_blend_desc = (
                    f"{args.reset_anchor_target_blend_initial:.2f}->{args.reset_anchor_target_blend:.2f} "
                    f"over {args.reset_anchor_target_blend_decay_steps:,} steps"
                )
            start_max_desc = f"{args.start_waypoint_max_fraction:.2f}"
            if args.start_waypoint_max_fraction_initial is not None and args.start_waypoint_fraction_decay_steps > 0:
                start_max_desc = (
                    f"{args.start_waypoint_max_fraction_initial:.2f}->{args.start_waypoint_max_fraction:.2f} "
                    f"over {args.start_waypoint_fraction_decay_steps:,} steps"
                )
            print(
                "    [Session 8h] stage1_recovery reset: target starts at "
                f"{args.start_waypoint_min_fraction:.2f}-{start_max_desc} "
                "trajectory fraction; base starts near waypoint zero"
                f" with target-anchor blend {reset_blend_desc}"
            )
        if auto_recovery_stability:
            print(
                "    Recovery stability preset: "
                f"learning_rate={args.learning_rate:g}, target_kl={args.target_kl}"
            )
        if recovery_base_assist_preset_applied:
            print(
                "    Recovery base-assist preset: "
                "blend={:.2f}->{:.2f}, distance={:.2f}-{:.2f}m, "
                "max_action={:.2f}, imitation_weight={:.1f}, lookahead_steps={:d}, yaw={}".format(
                    args.base_assist_initial_blend,
                    args.base_assist_final_blend,
                    args.base_assist_activation_distance,
                    args.base_assist_full_speed_distance,
                    args.base_assist_max_action,
                    args.base_assist_imitation_weight,
                    args.base_assist_lookahead_steps,
                    args.enable_base_assist_yaw,
                )
            )
        env_cfg.task_config.base_assist.enable = enable_base_assist
        env_cfg.task_config.base_assist.mode = args.base_assist_mode
        env_cfg.task_config.base_assist.initial_blend = args.base_assist_initial_blend
        env_cfg.task_config.base_assist.final_blend = args.base_assist_final_blend
        env_cfg.task_config.base_assist.decay_steps = args.base_assist_decay_steps
        env_cfg.task_config.base_assist.activation_distance = args.base_assist_activation_distance
        env_cfg.task_config.base_assist.full_speed_distance = args.base_assist_full_speed_distance
        env_cfg.task_config.base_assist.max_action = args.base_assist_max_action
        env_cfg.task_config.base_assist.imitation_weight = args.base_assist_imitation_weight
        env_cfg.task_config.base_assist.lookahead_steps = max(int(args.base_assist_lookahead_steps), 0)
        env_cfg.task_config.base_assist.yaw_enable = bool(args.enable_base_assist_yaw)
        env_cfg.task_config.base_assist.yaw_max_action = args.base_assist_yaw_max_action
        env_cfg.task_config.base_assist.yaw_full_error = args.base_assist_yaw_full_error
        env_cfg.task_config.base_assist.yaw_imitation_weight = args.base_assist_yaw_imitation_weight
        if enable_base_assist:
            print(
                "    Base assist: enabled "
                f"mode={args.base_assist_mode}, "
                f"blend={args.base_assist_initial_blend:.2f}->{args.base_assist_final_blend:.2f} "
                f"over {args.base_assist_decay_steps:,} steps, "
                f"distance={args.base_assist_activation_distance:.2f}-{args.base_assist_full_speed_distance:.2f}m, "
                f"max_action={args.base_assist_max_action:.2f}, "
                f"imitation_weight={args.base_assist_imitation_weight:.1f}, "
                f"lookahead_steps={env_cfg.task_config.base_assist.lookahead_steps}, "
                f"yaw={env_cfg.task_config.base_assist.yaw_enable}"
            )

        # Configure trajectory
        env_cfg.task_config.trajectory = TrajectoryConfig(
            type=args.trajectory_type,
            trajectory_dir=trajectory_dir,
            trajectory_pattern="**/*.json",
            trajectory_manifest_file=trajectory_config.get('manifest_file'),
            trajectory_filter_indices=trajectory_config.get('filter_indices'),
            max_trajectories=trajectory_config.get('max_trajectories'),
            min_duration_seconds=args.min_trajectory_duration,
            randomize_start_waypoint=randomize_start_waypoint,
            start_waypoint_min_fraction=args.start_waypoint_min_fraction,
            start_waypoint_max_fraction=args.start_waypoint_max_fraction,
            start_waypoint_max_fraction_initial=args.start_waypoint_max_fraction_initial,
            start_waypoint_fraction_decay_steps=max(int(args.start_waypoint_fraction_decay_steps), 0),
            reset_base_to_trajectory_start=reset_base_to_trajectory_start,
            reset_anchor_target_blend=args.reset_anchor_target_blend,
            reset_anchor_target_blend_initial=args.reset_anchor_target_blend_initial,
            reset_anchor_target_blend_decay_steps=max(int(args.reset_anchor_target_blend_decay_steps), 0),
            reset_base_x_offset=trajectory_config.get("reset_base_x_offset", 0.4415),
            reset_base_y_offset=trajectory_config.get("reset_base_y_offset", 0.2405),
            reset_arm_to_trajectory_metadata=trajectory_config.get("reset_arm_to_trajectory_metadata", False),
            debug_sampling=args.debug_trajectory_sampling,
        )
        
        # Create environment directly with config
        from rl_platform.tasks.mobile_mm import MobileMMTrackEEEnv
        env = MobileMMTrackEEEnv(cfg=env_cfg, render_mode="rgb_array" if args.video else None)
        if args.obstacles_from_trajectory_metadata:
            raw_env = env.unwrapped if hasattr(env, "unwrapped") else env
            _install_metadata_obstacle_randomizer(raw_env)
            print("    [OK] Trajectory-metadata obstacle placement enabled")

        if args.video:
            video_dir = os.path.join(args.log_dir, "videos", "train")
            env = attach_record_video_vecenv_passthrough(gym.wrappers.RecordVideo(
                env,
                video_dir,
                step_trigger=lambda step: step % args.video_interval == 0,
                video_length=args.video_length,
                disable_logger=True,
            ), env)
            print(f"    [OK] Isaac rendered video recording enabled: {video_dir}")
        
        print(f"    [OK] Environment created")
        if args.trajectory_type == "multi_recorded":
            if trajectory_config.get('filter_indices') is not None:
                print(f"    [OK] Loaded {len(trajectory_config['filter_indices'])} filtered trajectories")
            else:
                print(f"    [OK] Loaded all available trajectories from {args.trajectory_dir}")
    
    except Exception as e:
        print(f"    [FAIL] Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        simulation_app.close()
        return 1
    
    try:
        
        # First wrap to convert Isaac Lab format to SB3 format (dict -> numpy)
        # Isaac Lab envs are already VecEnv, so use VecEnvWrapper
        expected_obs_dim = None
        if args.checkpoint:
            try:
                checkpoint_model = PPO.load(args.checkpoint, device="cpu", print_system_info=False)
                expected_obs_dim = int(np.prod(checkpoint_model.observation_space.shape))
                print(f"    [OK] Checkpoint observation dim detected: {expected_obs_dim}")
            except Exception as exc:
                print(f"    [WARN] Could not inspect checkpoint observation dim: {exc}")
        elif args.pretrained_policy:
            try:
                pretrained_model = PPO.load(args.pretrained_policy, device="cpu", print_system_info=False)
                expected_obs_dim = int(np.prod(pretrained_model.observation_space.shape))
                print(f"    [OK] Pretrained policy observation dim detected: {expected_obs_dim}")
            except Exception as exc:
                print(f"    [WARN] Could not inspect pretrained policy observation dim: {exc}")
        env = IsaacLabToSB3VecEnvWrapper(
            env,
            expected_obs_dim=expected_obs_dim,
            freeze_base_actions=args.freeze_base_actions,
            base_action_scale=args.base_action_scale,
            base_action_delta_limit=args.base_action_delta_limit,
            freeze_base_actions_for_non_base_required=args.freeze_base_actions_for_non_base_required,
        )
        
        # Do a dummy reset to let the wrapper discover the true observation shape
        # This updates the observation_space before VecNormalize reads it
        _ = env.reset()
        
        if args.disable_vec_normalize:
            print("    [OK] Environment created and wrapped without VecNormalize")
        else:
            # Then wrap with VecNormalize for better training stability
            # NOTE: VecNormalize will now see numpy arrays instead of dicts
            env = VecNormalize(
                env,
                norm_obs=True,
                norm_reward=True,
                clip_obs=10.0,
                clip_reward=10.0,
            )
            print("    [OK] Environment created and wrapped")
    except Exception as e:
        print(f"    [FAIL] Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        simulation_app.close()
        return 1
    
    # Step 6: Create and train PPO model
    print("\n[6/6] Setting up PPO training...")
    
    # Create callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq // args.num_envs,  # Per environment steps
        save_path=os.path.join(args.log_dir, "checkpoints"),
        name_prefix="ppo_mobile_mm",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Entropy decay callback (prevents policy divergence after convergence)
    if args.enable_entropy_decay:
        entropy_decay_callback = EntropyDecayCallback(
            initial_ent_coef=args.ent_coef,
            final_ent_coef=args.final_ent_coef,
            decay_start_timestep=args.decay_start_timestep,
            decay_duration_timesteps=args.decay_duration_timesteps,
            verbose=1,
        )
        callbacks.append(entropy_decay_callback)
        print(f"    [OK] Entropy decay enabled: {args.ent_coef} -> {args.final_ent_coef}")
        print(f"      Decay: {args.decay_start_timestep/1e6:.0f}M - {(args.decay_start_timestep + args.decay_duration_timesteps)/1e6:.0f}M steps")
    
    # Adaptive KL divergence schedule callback (prevents early stopping, enables learning)
    if args.enable_kl_schedule:
        kl_schedule_callback = AdaptiveKLSchedule(
            total_timesteps=args.total_timesteps,
            kl_very_early=max(args.kl_warmup * 4, 1.0),    # Much more aggressive early KL
            kl_early=max(args.kl_warmup * 2, 0.5),         # Still very loose  
            kl_learning=max(args.kl_main * 4, 0.2),        # More learning room
            kl_stable=max(args.kl_main * 2, 0.1),          # Reasonable updates
            kl_finetune=args.kl_finetune,                  # Normal end-game
            verbose=1
        )
        callbacks.append(kl_schedule_callback)
        print(f"    [OK] Adaptive KL schedule enabled: very_early={max(args.kl_warmup * 4, 1.0):.2f}, early={max(args.kl_warmup * 2, 0.5):.2f}")
        print(f"      Stages: 0-5M (explore), 5-20M (learn), 20-60M (balance), 60-80M (stable), 80-100M (finetune)")

    if args.base_assist_aux_dataset:
        aux_action_indices = [
            int(idx.strip())
            for idx in args.base_assist_aux_action_indices.split(",")
            if idx.strip()
        ]
        if not aux_action_indices:
            raise ValueError("--base_assist_aux_action_indices must contain at least one index")
        base_assist_aux_callback = MaskedActionAuxCallback(
            dataset_path=args.base_assist_aux_dataset,
            action_indices=aux_action_indices,
            gradient_steps=args.base_assist_aux_gradient_steps,
            batch_size=args.base_assist_aux_batch_size,
            lr=args.base_assist_aux_lr,
            max_grad_norm=args.base_assist_aux_max_grad_norm,
            use_sample_weight=not args.base_assist_aux_ignore_sample_weight,
            sample_weight_power=args.base_assist_aux_sample_weight_power,
            verbose=1,
        )
        callbacks.append(base_assist_aux_callback)
        print("    [OK] Base-assist auxiliary supervised callback enabled:")
        print(f"      dataset: {args.base_assist_aux_dataset}")
        print(
            f"      rows: {aux_action_indices}, steps/rollout: {args.base_assist_aux_gradient_steps}, "
            f"batch: {args.base_assist_aux_batch_size}, lr: {args.base_assist_aux_lr:g}"
        )

    if args.enable_obstacles and args.enable_obstacle_curriculum:
        final_obstacle_weight = (
            args.obstacle_curriculum_final_weight
            if args.obstacle_curriculum_final_weight is not None
            else env_cfg.task_config.rewards.min_obstacle_distance_weight
        )
        obstacle_curriculum_callback = ObstacleCurriculumCallback(
            start_x_range=args.obstacle_curriculum_start_x_range,
            final_x_range=args.obstacle_x_range,
            start_y_range=args.obstacle_curriculum_start_y_range,
            final_y_range=args.obstacle_y_range,
            start_clearance=args.obstacle_curriculum_start_clearance,
            final_clearance=args.min_obstacle_start_clearance,
            start_weight=args.obstacle_curriculum_start_weight,
            final_weight=final_obstacle_weight,
            transition_steps=args.obstacle_curriculum_steps,
            verbose=1,
        )
        callbacks.append(obstacle_curriculum_callback)
        print("    [OK] Obstacle curriculum enabled:")
        print(
            f"      x: {tuple(args.obstacle_curriculum_start_x_range)} -> {tuple(args.obstacle_x_range)}, "
            f"y: {tuple(args.obstacle_curriculum_start_y_range)} -> {tuple(args.obstacle_y_range)}"
        )
        print(
            f"      clearance: {args.obstacle_curriculum_start_clearance:.2f} -> {args.min_obstacle_start_clearance:.2f}, "
            f"weight: {args.obstacle_curriculum_start_weight:.2f} -> {final_obstacle_weight:.2f}, "
            f"steps: {args.obstacle_curriculum_steps:,}"
        )
    
    # Training monitor callback (detailed metrics logging)
    monitor_callback = TrainingMonitorCallback(
        log_freq=5,  # Log every 5 iterations
        verbose=1
    )
    callbacks.append(monitor_callback)
    print(f"    [OK] Training monitor enabled: logging detailed metrics every 5 iterations")
    
    # Session 8h: Auto-pause callback for instability detection
    # Read auto-pause settings from environment config
    try:
        # Access the unwrapped Isaac Lab environment to get config
        unwrapped_env = env
        while hasattr(unwrapped_env, 'venv'):
            unwrapped_env = unwrapped_env.venv
        if hasattr(unwrapped_env, 'unwrapped'):
            isaac_env = unwrapped_env.unwrapped
            if hasattr(isaac_env, 'task_cfg') and hasattr(isaac_env.task_cfg, 'rewards'):
                cfg = isaac_env.task_cfg.rewards
                enable_auto_pause = getattr(cfg, 'enable_auto_pause', False)
                kl_threshold = getattr(cfg, 'kl_threshold', 0.1)
                variance_threshold = getattr(cfg, 'variance_threshold', 0.0)
                
                if enable_auto_pause:
                    auto_pause_callback = AutoPauseCallback(
                        enable=True,
                        kl_threshold=kl_threshold,
                        variance_threshold=variance_threshold,
                        variance_patience=3,
                        variance_clip_threshold=0.25,
                        warmup_steps=500_000,  # Skip monitoring in first 500K steps
                        checkpoint_dir=os.path.join(args.log_dir, "checkpoints"),
                        verbose=1
                    )
                    callbacks.append(auto_pause_callback)
                    print(f"    [OK] Auto-pause enabled: KL>{kl_threshold}, variance<{variance_threshold}")
                    print(f"      Warmup: 500K steps (monitoring starts after)")
                    print(f"      Session 8g lesson: No auto-pause → collapse @ 100M")
                    print(f"      Session 8h fix: Proactive monitoring prevents catastrophic failure")
                else:
                    print(f"    [INFO] Auto-pause disabled in config")
    except Exception as e:
        print(f"    [WARN] Could not read auto-pause config from environment: {e}")
        print(f"    [INFO] Auto-pause callback not registered")
    
    if args.wandb:
        callbacks.append(wandb_callback)
    
    # Create or load model
    try:
        import torch
        
        # Device selection based on command-line argument
        if args.device == "cpu":
            device = "cpu"
            print("    [WARN]  CPU training forced via --device cpu")
        elif args.device == "cuda":
            if torch.cuda.is_available():
                device = f"cuda:{best_device}"
            else:
                print("    [WARN]  CUDA requested but not available, falling back to CPU")
                device = "cpu"
        else:  # auto
            device = f"cuda:{best_device}" if torch.cuda.is_available() else "cpu"
        
        if args.checkpoint:
            print(f"    Loading checkpoint: {args.checkpoint}")
            
            # FIX (8c-v2): Load VecNormalize stats before loading policy
            # CheckpointCallback saves as: ppo_mobile_mm_<steps>_steps.zip
            # and VecNormalize as: ppo_mobile_mm_vecnormalize_<steps>_steps.pkl
            checkpoint_path = Path(args.checkpoint)
            checkpoint_name = checkpoint_path.stem  # e.g., "ppo_mobile_mm_20000000_steps"
            
            # Replace "ppo_mobile_mm" with "ppo_mobile_mm_vecnormalize"
            vecnorm_name = checkpoint_name.replace("ppo_mobile_mm", "ppo_mobile_mm_vecnormalize", 1)
            vecnorm_path = checkpoint_path.parent / f"{vecnorm_name}.pkl"
            if not vecnorm_path.exists():
                sibling_vecnorm_path = checkpoint_path.parent / "vec_normalize.pkl"
                if sibling_vecnorm_path.exists():
                    print(f"    Derived VecNormalize stats not found at: {vecnorm_path.name}")
                    print(f"    Falling back to sibling VecNormalize stats: {sibling_vecnorm_path.name}")
                    vecnorm_path = sibling_vecnorm_path
            
            if vecnorm_path.exists():
                print(f"    Loading VecNormalize stats from: {vecnorm_path.name}")
                from stable_baselines3.common.vec_env import VecNormalize
                env = VecNormalize.load(str(vecnorm_path), env)
                print("    [OK] VecNormalize stats loaded successfully")
                if args.reset_reward_stats:
                    from stable_baselines3.common.running_mean_std import RunningMeanStd
                    import numpy as np

                    env.ret_rms = RunningMeanStd(shape=())
                    env.returns = np.zeros(env.num_envs)
                    env.old_reward = np.array([])
                    print("    [OK] VecNormalize reward-return stats reset; observation stats preserved")
            else:
                print(f"    [WARN]  VecNormalize stats not found at: {vecnorm_path}")
                print("    [WARN]  Continuing without normalization stats (may affect curriculum learning)")
            
            model = PPO.load(args.checkpoint, env=env, device=device)
            model.tensorboard_log = args.log_dir
            model.learning_rate = args.learning_rate
            model.lr_schedule = constant_fn(args.learning_rate)
            for param_group in model.policy.optimizer.param_groups:
                param_group["lr"] = args.learning_rate
            model.n_epochs = args.n_epochs
            model.batch_size = args.batch_size
            model.gamma = args.gamma
            model.gae_lambda = args.gae_lambda
            model.clip_range = constant_fn(args.clip_range)
            model.clip_range_vf = constant_fn(args.clip_range_vf) if args.clip_range_vf is not None else None
            model.target_kl = args.target_kl
            model.ent_coef = args.ent_coef
            model.ent_coef_schedule = constant_fn(args.ent_coef)
            model.vf_coef = args.vf_coef
            model.max_grad_norm = args.max_grad_norm
            print("    [OK] Resume hyperparameters applied to loaded PPO model")
        else:
            print("    Creating new PPO model...")
            
            # Enhanced network architecture for Proto2 trajectory tracking.
            obs_dim = int(np.prod(env.observation_space.shape))
            action_dim = int(np.prod(env.action_space.shape))
            policy_class = "MlpPolicy"
            if args.policy_arch == "grouped":
                from scripts.reinforcement_learning.sb3.grouped_policy import GroupedActorCriticPolicy

                shared_dims = _parse_int_list(args.grouped_shared_hidden_dims)
                value_dims = _parse_int_list(args.grouped_value_hidden_dims)
                policy_class = GroupedActorCriticPolicy
                policy_kwargs = dict(
                    shared_hidden_dims=shared_dims,
                    head_hidden_dim=args.grouped_head_hidden_dim,
                    value_hidden_dims=value_dims,
                    activation_fn=torch.nn.ELU,
                    ortho_init=True,
                    log_std_init=-2.0,
                )
            else:
                policy_kwargs = dict(
                    net_arch=dict(
                        pi=[256, 256, 128],
                        vf=[256, 256, 128],
                    ),
                    activation_fn=torch.nn.ReLU,
                    ortho_init=True,  # Orthogonal initialization (better for RL)
                    log_std_init=-2.0,  # Proto2 v2: std ~0.14 to avoid destabilizing random startup actions
                )
            
            print("    Network Architecture:")
            if args.policy_arch == "grouped":
                print(f"      Policy arch:     grouped shared/action-head actor")
                print(f"      Shared encoder:  [{obs_dim}] -> {shared_dims}")
                print(f"      Arm head:        [{shared_dims[-1]}] -> [{args.grouped_head_hidden_dim}] -> [3]")
                print(f"      Gimbal head:     [{shared_dims[-1]}] -> [{args.grouped_head_hidden_dim}] -> [3]")
                print(f"      Base head:       [{shared_dims[-1]}] -> [{args.grouped_head_hidden_dim}] -> [3]")
                print(f"      Action order:    [arm0..2, gimbal0..2, base_vx, base_vy, base_wz] -> [{action_dim}]")
                print(f"      Critic latent:   [{obs_dim}] -> {value_dims} -> [1]")
            else:
                print(f"      Policy arch:     flat SB3 MlpPolicy")
                print(f"      Actor (Policy):  [{obs_dim}] -> [256] -> [256] -> [128] -> [{action_dim}]")
                print(f"      Critic (Value):  [{obs_dim}] -> [256] -> [256] -> [128] -> [1]")
            print("    Action Distribution:")
            print("      Initial std:     ~0.14 (log_std_init=-2.0)")
            print("      Std control:     Entropy decay + KL schedule")
            
            model = PPO(
                policy_class,
                env,
                policy_kwargs=policy_kwargs,
                learning_rate=args.learning_rate,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                clip_range=args.clip_range,
                clip_range_vf=args.clip_range_vf if args.clip_range_vf is not None else None,  # Session 8c: 0.3 to stabilize critic
                normalize_advantage=True,  # Session 8c: Normalize advantages for stable gradient signals
                ent_coef=args.ent_coef,
                vf_coef=args.vf_coef,
                max_grad_norm=args.max_grad_norm,
                target_kl=args.target_kl,
                # Note: log_std_bounds not available in SB3 2.5.0
                # Std control via: entropy decay + KL schedule + log_std_init=-2.0
                tensorboard_log=args.log_dir,
                device=device,
                verbose=1,
            )
            print(f"    [OK] PPO model created on {device}")
            
            if args.pretrained_policy:
                print(f"    Loading BC pretrained policy from: {args.pretrained_policy}")
                try:
                    bc_model = PPO.load(args.pretrained_policy, device=device)
                    action_indices = None
                    if args.pretrained_action_indices:
                        action_indices = _parse_int_list(args.pretrained_action_indices)
                        invalid = [
                            idx
                            for idx in action_indices
                            if idx < 0 or idx >= _policy_action_dim(model.policy)
                        ]
                        if invalid:
                            raise ValueError(
                                f"invalid pretrained action indices {invalid}; "
                                f"action_dim={_policy_action_dim(model.policy)}"
                            )
                    copied_action_desc = _copy_pretrained_actor(model, bc_model, action_indices)
                    if args.copy_pretrained_log_std:
                        if action_indices is None:
                            model.policy.log_std.data.copy_(bc_model.policy.log_std.data)
                            copied_std_desc = "copied all pretrained log_std values"
                        else:
                            with torch.no_grad():
                                for action_idx in action_indices:
                                    model.policy.log_std.data[action_idx].copy_(
                                        bc_model.policy.log_std.data[action_idx]
                                    )
                            copied_std_desc = f"copied pretrained log_std for {action_indices}"
                    else:
                        copied_std_desc = "kept PPO log_std_init"
                    if action_indices is not None and args.pretrained_unselected_log_std is not None:
                        with torch.no_grad():
                            selected = set(action_indices)
                            for action_idx in range(_policy_action_dim(model.policy)):
                                if action_idx not in selected:
                                    model.policy.log_std.data[action_idx] = float(
                                        args.pretrained_unselected_log_std
                                    )
                        copied_std_desc += (
                            f"; set non-selected log_std={args.pretrained_unselected_log_std}"
                        )
                    print(f"    [OK] BC policy feature weights loaded")
                    print(f"    [OK] BC policy {copied_action_desc} loaded")
                    print(f"    [OK] {copied_std_desc}")
                    print(f"    [OK] Critic network randomly initialised (will learn from RL)")
                    del bc_model
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"    [WARN] Failed to load BC pretrained policy: {e}")
                    print(f"    [WARN] Continuing with randomly initialised policy")

            if args.policy_log_std_override is not None:
                with torch.no_grad():
                    model.policy.log_std.data.fill_(float(args.policy_log_std_override))
                print(f"    [OK] Policy log_std overridden to {args.policy_log_std_override}")
            
    except Exception as e:
        print(f"    [FAIL] Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        simulation_app.close()
        return 1
    
    # Print training configuration
    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Task:              {args.task}")
    print(f"Policy arch:       {args.policy_arch}")
    print(f"Num environments:  {args.num_envs}")
    print(f"Total timesteps:   {args.total_timesteps:,}")
    print(f"Learning rate:     {args.learning_rate}")
    print(f"Rollout steps:     {args.n_steps}")
    print(f"Batch size:        {args.batch_size}")
    print(f"PPO epochs:        {args.n_epochs}")
    print(f"Entropy coef:      {args.ent_coef}")
    if args.enable_entropy_decay:
        print(f"  -> Decay enabled: {args.ent_coef} -> {args.final_ent_coef}")
        print(f"  -> Decay period:  {args.decay_start_timestep/1e6:.0f}M - {(args.decay_start_timestep + args.decay_duration_timesteps)/1e6:.0f}M steps")
    print(f"Gamma:             {args.gamma}")
    print(f"GAE lambda:        {args.gae_lambda}")
    print(f"Clip range:        {args.clip_range}")
    print(f"Clip range vf:     {args.clip_range_vf if args.clip_range_vf is not None else 'None (disabled)'}")
    print(f"VF coef:           {args.vf_coef}")
    print(f"Max grad norm:     {args.max_grad_norm}")
    print(f"Reset reward RMS:  {args.reset_reward_stats}")
    print(f"Target KL:         {args.target_kl if args.target_kl else 'None (disabled)'}")
    if args.enable_kl_schedule:
        print(f"  -> KL schedule:   warmup={args.kl_warmup}, main={args.kl_main}, finetune={args.kl_finetune}")
        print(f"  -> Phase splits:  10% warmup, 70% main, 20% finetune")
    print(f"Save frequency:    {args.save_freq:,} steps")
    print(f"Log directory:     {args.log_dir}")
    print(f"Rendered video:    {'yes' if args.video else 'no'}")
    if args.video:
        print(f"  -> Length:       {args.video_length} steps")
        print(f"  -> Interval:     {args.video_interval} steps")
    print(f"Device:            {device}")
    print("Seed:              {}".format(args.seed if args.seed is not None else "None"))
    print(f"Freeze base acts:  {'yes (rows [6,7,8] zeroed before env step)' if args.freeze_base_actions else 'no'}")
    print(
        "Mixed FK base mask: {}".format(
            "yes (non-base-required trajectories freeze rows [6,7,8])"
            if args.freeze_base_actions_for_non_base_required
            else "no"
        )
    )
    if not args.freeze_base_actions:
        print(f"Base action scale: {args.base_action_scale}")
        if args.base_action_delta_limit > 0.0:
            print(f"Base delta limit:  {args.base_action_delta_limit}")
        if args.base_action_slew_limit > 0.0:
            print(f"Base env slew:     {args.base_action_slew_limit}")
    if args.base_action_delta_penalty is not None:
        print(f"Base delta penalty:{args.base_action_delta_penalty}")
    if args.pretrained_policy:
        print(f"Pretrained policy: {args.pretrained_policy}")
    if args.policy_log_std_override is not None:
        print(f"Policy log std:    {args.policy_log_std_override}")
    if args.base_assist_aux_dataset:
        print(f"Base assist aux:   {args.base_assist_aux_dataset}")
        print(f"  -> Rows:         {args.base_assist_aux_action_indices}")
        print(f"  -> Steps/rollout:{args.base_assist_aux_gradient_steps}")
    print("=" * 70 + "\n")
    
    # Train
    exit_code = 0
    try:
        print("Starting training...\n")
        reset_num_timesteps = args.checkpoint is None
        if args.checkpoint:
            print("Resuming from checkpoint with preserved timestep counters.")

        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            log_interval=1,
            tb_log_name="PPO",
            reset_num_timesteps=reset_num_timesteps,
        )
        
        # Save final model
        final_model_path = os.path.join(args.log_dir, "final_model")
        model.save(final_model_path)
        if hasattr(env, "save"):
            env.save(os.path.join(args.log_dir, "vec_normalize.pkl"))
        
        print(f"\n{'='*70}")
        print("[OK] Training complete!")
        print(f"{'='*70}")
        print(f"Final model saved to: {final_model_path}")
        print(f"Logs saved to: {args.log_dir}")
        
    except KeyboardInterrupt:
        print("\n\n[WARN]  Training interrupted by user")
        exit_code = 130
        
        # Save interrupt checkpoint
        interrupt_path = os.path.join(args.log_dir, "interrupt_checkpoint")
        model.save(interrupt_path)
        if hasattr(env, "save"):
            env.save(os.path.join(args.log_dir, "vec_normalize_interrupt.pkl"))
        
        print(f"Interrupt checkpoint saved to: {interrupt_path}")
    
    except Exception as e:
        print(f"\n[FAIL] Training failed: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1
    
    finally:
        print("\nCleaning up...")
        env.close()
        if args.wandb:
            wandb.finish()
        simulation_app.close()
        print("Done!")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
