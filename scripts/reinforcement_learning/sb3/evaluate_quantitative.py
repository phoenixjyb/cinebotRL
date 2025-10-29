"""Comprehensive Quantitative Evaluation Script for MobileMMTrackEE.

This script performs detailed evaluation with extensive logging of:
- Position and orientation tracking errors
- Joint angles and velocities
- Base velocities and odometry
- Reward components breakdown
- Reachability analysis
- Success/failure statistics

Usage:
    # Run full quantitative evaluation (headless, all trajectories):
    I:\isaaclab\isaaclab.bat -p scripts/reinforcement_learning/sb3/evaluate_quantitative.py \
        --checkpoint logs/sb3/MobileMMTrackEE-v0/Oct29_13-24-52_7d_200Mts_multi/final_model.zip \
        --num_envs 64 \
        --num_episodes 200 \
        --output_dir evaluation_results \
        --headless
    
    # Run quick evaluation with visualization (few trajectories):
    I:\isaaclab\isaaclab.bat -p scripts/reinforcement_learning/sb3/evaluate_quantitative.py \
        --checkpoint logs/.../final_model.zip \
        --num_envs 4 \
        --num_episodes 20 \
        --output_dir evaluation_results
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
from datetime import datetime
import json
from collections import defaultdict
from typing import Dict, List, Tuple

# Add project root to Python path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

print(f"[DEBUG] PROJECT_ROOT: {PROJECT_ROOT}")
print(f"[DEBUG] sys.path includes: {PROJECT_ROOT}, {PROJECT_ROOT / 'src'}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive quantitative evaluation of mobile manipulator policy"
    )
    
    # Required
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.zip file)",
    )
    
    # Environment settings
    parser.add_argument(
        "--task",
        type=str,
        default="MobileMMTrackEE-v0",
        help="Task ID to evaluate on",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=64,
        help="Number of parallel environments (more = faster evaluation)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no GUI, faster)",
    )
    
    # Evaluation settings
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=200,
        help="Number of episodes to evaluate (recommend 200+ for robust statistics)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic actions (no exploration noise)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to run on",
    )
    
    # Trajectory configuration
    parser.add_argument(
        "--trajectory_type",
        type=str,
        default="multi_recorded",
        choices=["line", "circle", "figure_eight", "recorded", "multi_recorded"],
        help="Type of trajectory to test against",
    )
    parser.add_argument(
        "--trajectory_dir",
        type=str,
        default="trajectoryToLearn/world_json",
        help="Directory containing recorded trajectories",
    )
    parser.add_argument(
        "--use_all_trajectories",
        action="store_true",
        help="Use ALL 1,038 trajectories from training dataset",
    )
    parser.add_argument(
        "--use_chassis_only",
        action="store_true",
        help="Use only chassis-requiring trajectories (519 trajectories)",
    )
    parser.add_argument(
        "--max_trajectories",
        type=int,
        default=None,
        help="Limit number of trajectories to load (None = all)",
    )
    
    # Output settings
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results and logs",
    )
    parser.add_argument(
        "--save_every_n_steps",
        type=int,
        default=10,
        help="Save detailed logs every N steps (lower = more data, slower)",
    )
    
    return parser.parse_args()


class EvaluationLogger:
    """Comprehensive logger for evaluation metrics."""
    
    def __init__(self, output_dir: Path, num_envs: int):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_envs = num_envs
        
        # Episode-level statistics
        self.episode_data = []
        
        # Step-level statistics (sampled)
        self.step_data = defaultdict(list)
        
        # Aggregated statistics
        self.tracking_errors_pos = []
        self.tracking_errors_ori = []
        self.joint_angles_log = []
        self.joint_velocities_log = []
        self.base_velocities_log = []
        self.reward_components_log = defaultdict(list)
        
        print(f"📊 Logger initialized. Output directory: {self.output_dir}")
    
    def log_step(self, step: int, env_states: Dict, save_detailed: bool = False):
        """Log step-level data."""
        if save_detailed:
            # Position tracking error
            if 'ee_pos_error' in env_states:
                error = env_states['ee_pos_error']  # [num_envs, 3]
                error_norm = np.linalg.norm(error, axis=1)  # [num_envs]
                self.tracking_errors_pos.extend(error_norm.tolist())
                self.step_data['ee_pos_error_x'].append(np.mean(error[:, 0]))
                self.step_data['ee_pos_error_y'].append(np.mean(error[:, 1]))
                self.step_data['ee_pos_error_z'].append(np.mean(error[:, 2]))
                self.step_data['ee_pos_error_norm'].append(np.mean(error_norm))
            
            # Orientation tracking error
            if 'ee_ori_error' in env_states:
                error = env_states['ee_ori_error']  # [num_envs] (angle in radians)
                self.tracking_errors_ori.extend(error.tolist())
                self.step_data['ee_ori_error_rad'].append(np.mean(error))
                self.step_data['ee_ori_error_deg'].append(np.mean(np.rad2deg(error)))
            
            # Joint angles
            if 'joint_pos' in env_states:
                joint_pos = env_states['joint_pos']  # [num_envs, 6]
                self.joint_angles_log.append(joint_pos.copy())
                for i in range(joint_pos.shape[1]):
                    self.step_data[f'joint_{i}_pos'].append(np.mean(joint_pos[:, i]))
            
            # Joint velocities
            if 'joint_vel' in env_states:
                joint_vel = env_states['joint_vel']  # [num_envs, 6]
                self.joint_velocities_log.append(joint_vel.copy())
                for i in range(joint_vel.shape[1]):
                    self.step_data[f'joint_{i}_vel'].append(np.mean(joint_vel[:, i]))
            
            # Base velocities
            if 'base_lin_vel' in env_states and 'base_ang_vel' in env_states:
                base_lin = env_states['base_lin_vel']  # [num_envs, 2]
                base_ang = env_states['base_ang_vel']  # [num_envs]
                self.base_velocities_log.append(np.concatenate([base_lin, base_ang[:, None]], axis=1))
                self.step_data['base_vel_x'].append(np.mean(base_lin[:, 0]))
                self.step_data['base_vel_y'].append(np.mean(base_lin[:, 1]))
                self.step_data['base_ang_vel'].append(np.mean(base_ang))
            
            # Reward components
            if 'reward_components' in env_states:
                for key, value in env_states['reward_components'].items():
                    self.reward_components_log[key].append(np.mean(value))
    
    def log_episode(self, episode_idx: int, env_idx: int, episode_stats: Dict):
        """Log episode-level data."""
        self.episode_data.append({
            'episode': episode_idx,
            'env': env_idx,
            **episode_stats
        })
    
    def compute_statistics(self) -> Dict:
        """Compute aggregate statistics from logged data."""
        stats = {}
        
        # Position tracking errors
        if self.tracking_errors_pos:
            pos_errors = np.array(self.tracking_errors_pos)
            stats['position_error'] = {
                'mean_m': float(np.mean(pos_errors)),
                'median_m': float(np.median(pos_errors)),
                'std_m': float(np.std(pos_errors)),
                'p95_m': float(np.percentile(pos_errors, 95)),
                'p99_m': float(np.percentile(pos_errors, 99)),
                'max_m': float(np.max(pos_errors)),
                'min_m': float(np.min(pos_errors)),
            }
            stats['position_error']['mean_cm'] = stats['position_error']['mean_m'] * 100
            stats['position_error']['median_cm'] = stats['position_error']['median_m'] * 100
            stats['position_error']['p95_cm'] = stats['position_error']['p95_m'] * 100
        
        # Orientation tracking errors
        if self.tracking_errors_ori:
            ori_errors = np.array(self.tracking_errors_ori)
            stats['orientation_error'] = {
                'mean_rad': float(np.mean(ori_errors)),
                'median_rad': float(np.median(ori_errors)),
                'std_rad': float(np.std(ori_errors)),
                'p95_rad': float(np.percentile(ori_errors, 95)),
                'p99_rad': float(np.percentile(ori_errors, 99)),
                'max_rad': float(np.max(ori_errors)),
                'mean_deg': float(np.rad2deg(np.mean(ori_errors))),
                'median_deg': float(np.rad2deg(np.median(ori_errors))),
                'p95_deg': float(np.rad2deg(np.percentile(ori_errors, 95))),
            }
        
        # Joint statistics
        if self.joint_angles_log:
            joint_angles = np.concatenate(self.joint_angles_log, axis=0)  # [N, 6]
            stats['joint_angles'] = {
                f'joint_{i}': {
                    'mean_rad': float(np.mean(joint_angles[:, i])),
                    'std_rad': float(np.std(joint_angles[:, i])),
                    'min_rad': float(np.min(joint_angles[:, i])),
                    'max_rad': float(np.max(joint_angles[:, i])),
                    'range_rad': float(np.ptp(joint_angles[:, i])),
                } for i in range(6)
            }
        
        if self.joint_velocities_log:
            joint_vels = np.concatenate(self.joint_velocities_log, axis=0)  # [N, 6]
            stats['joint_velocities'] = {
                f'joint_{i}': {
                    'mean_rad_s': float(np.mean(np.abs(joint_vels[:, i]))),
                    'max_rad_s': float(np.max(np.abs(joint_vels[:, i]))),
                    'p95_rad_s': float(np.percentile(np.abs(joint_vels[:, i]), 95)),
                } for i in range(6)
            }
        
        # Base statistics
        if self.base_velocities_log:
            base_vels = np.concatenate(self.base_velocities_log, axis=0)  # [N, 3]
            stats['base_velocities'] = {
                'linear_x': {
                    'mean_m_s': float(np.mean(base_vels[:, 0])),
                    'max_m_s': float(np.max(np.abs(base_vels[:, 0]))),
                    'p95_m_s': float(np.percentile(np.abs(base_vels[:, 0]), 95)),
                },
                'linear_y': {
                    'mean_m_s': float(np.mean(base_vels[:, 1])),
                    'max_m_s': float(np.max(np.abs(base_vels[:, 1]))),
                    'p95_m_s': float(np.percentile(np.abs(base_vels[:, 1]), 95)),
                },
                'angular_z': {
                    'mean_rad_s': float(np.mean(base_vels[:, 2])),
                    'max_rad_s': float(np.max(np.abs(base_vels[:, 2]))),
                    'p95_rad_s': float(np.percentile(np.abs(base_vels[:, 2]), 95)),
                },
            }
        
        # Reward components
        if self.reward_components_log:
            stats['reward_components'] = {
                key: {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                } for key, values in self.reward_components_log.items()
            }
        
        # Episode statistics
        if self.episode_data:
            episode_rewards = [ep['total_reward'] for ep in self.episode_data]
            episode_lengths = [ep['length'] for ep in self.episode_data]
            
            stats['episodes'] = {
                'count': len(self.episode_data),
                'mean_reward': float(np.mean(episode_rewards)),
                'std_reward': float(np.std(episode_rewards)),
                'median_reward': float(np.median(episode_rewards)),
                'min_reward': float(np.min(episode_rewards)),
                'max_reward': float(np.max(episode_rewards)),
                'mean_length': float(np.mean(episode_lengths)),
                'std_length': float(np.std(episode_lengths)),
            }
            
            # Success rate (if available)
            if 'success' in self.episode_data[0]:
                successes = [ep['success'] for ep in self.episode_data]
                stats['episodes']['success_rate'] = float(np.mean(successes))
                stats['episodes']['success_count'] = int(np.sum(successes))
        
        return stats
    
    def save_results(self, timestamp: str, args: argparse.Namespace):
        """Save all logged data and statistics to files."""
        # Compute statistics
        stats = self.compute_statistics()
        
        # Save summary statistics (JSON)
        summary_file = self.output_dir / f"eval_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'checkpoint': args.checkpoint,
                'num_episodes': args.num_episodes,
                'num_envs': args.num_envs,
                'trajectory_type': args.trajectory_type,
                'statistics': stats,
            }, f, indent=2)
        print(f"✓ Summary saved: {summary_file}")
        
        # Save episode data (CSV)
        if self.episode_data:
            import pandas as pd
            episode_df = pd.DataFrame(self.episode_data)
            episode_file = self.output_dir / f"episodes_{timestamp}.csv"
            episode_df.to_csv(episode_file, index=False)
            print(f"✓ Episode data saved: {episode_file}")
        
        # Save step-level data (CSV)
        if self.step_data:
            import pandas as pd
            step_df = pd.DataFrame(dict(self.step_data))
            step_file = self.output_dir / f"steps_{timestamp}.csv"
            step_df.to_csv(step_file, index=False)
            print(f"✓ Step data saved: {step_file}")
        
        # Save detailed numpy arrays
        arrays_file = self.output_dir / f"arrays_{timestamp}.npz"
        np.savez_compressed(
            arrays_file,
            tracking_errors_pos=np.array(self.tracking_errors_pos) if self.tracking_errors_pos else np.array([]),
            tracking_errors_ori=np.array(self.tracking_errors_ori) if self.tracking_errors_ori else np.array([]),
            joint_angles=np.concatenate(self.joint_angles_log, axis=0) if self.joint_angles_log else np.array([]),
            joint_velocities=np.concatenate(self.joint_velocities_log, axis=0) if self.joint_velocities_log else np.array([]),
            base_velocities=np.concatenate(self.base_velocities_log, axis=0) if self.base_velocities_log else np.array([]),
        )
        print(f"✓ Numpy arrays saved: {arrays_file}")
        
        return stats


def extract_env_states(env, obs, rewards, infos) -> Dict:
    """Extract detailed environment states for logging."""
    states = {}
    
    # Try to get raw environment (unwrap if needed)
    raw_env = env
    while hasattr(raw_env, 'venv'):
        raw_env = raw_env.venv
    if hasattr(raw_env, 'unwrapped'):
        raw_env = raw_env.unwrapped
    
    # Extract tracking errors
    if hasattr(raw_env, 'ee_pos_error_buf'):
        states['ee_pos_error'] = raw_env.ee_pos_error_buf.cpu().numpy()
    if hasattr(raw_env, 'ee_ori_error_buf'):
        states['ee_ori_error'] = raw_env.ee_ori_error_buf.cpu().numpy()
    
    # Extract joint states
    if hasattr(raw_env, 'robot') and hasattr(raw_env.robot, 'data'):
        robot_data = raw_env.robot.data
        if hasattr(robot_data, 'joint_pos'):
            states['joint_pos'] = robot_data.joint_pos.cpu().numpy()
        if hasattr(robot_data, 'joint_vel'):
            states['joint_vel'] = robot_data.joint_vel.cpu().numpy()
    
    # Extract base velocities
    if hasattr(raw_env, 'robot') and hasattr(raw_env.robot, 'data'):
        robot_data = raw_env.robot.data
        if hasattr(robot_data, 'root_lin_vel_w'):
            base_lin_vel_w = robot_data.root_lin_vel_w.cpu().numpy()  # [num_envs, 3]
            states['base_lin_vel'] = base_lin_vel_w[:, :2]  # Only X and Y (ignore Z)
        if hasattr(robot_data, 'root_ang_vel_w'):
            base_ang_vel_w = robot_data.root_ang_vel_w.cpu().numpy()  # [num_envs, 3]
            states['base_ang_vel'] = base_ang_vel_w[:, 2]  # Only Z rotation
    
    # Extract reward components
    if hasattr(raw_env, 'reward_components'):
        components = {}
        for key, value in raw_env.reward_components.items():
            if hasattr(value, 'cpu'):
                components[key] = value.cpu().numpy()
            else:
                components[key] = np.array(value)
        states['reward_components'] = components
    
    # Extract from infos if available
    for key in ['ee_pos_error', 'ee_ori_error', 'success']:
        if key in infos[0]:
            values = [info.get(key, 0) for info in infos]
            states[key] = np.array(values)
    
    return states


def main():
    """Main evaluation function."""
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Validate checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    print("=" * 80)
    print("🔬 COMPREHENSIVE QUANTITATIVE EVALUATION")
    print("=" * 80)
    print(f"Checkpoint:        {args.checkpoint}")
    print(f"Task:              {args.task}")
    print(f"Num envs:          {args.num_envs}")
    print(f"Num episodes:      {args.num_episodes}")
    print(f"Headless:          {args.headless}")
    print(f"Deterministic:     {args.deterministic}")
    print(f"Output directory:  {args.output_dir}")
    print(f"Timestamp:         {timestamp}")
    print()
    
    # Initialize Isaac Sim
    print("[1/6] Initializing Isaac Sim...")
    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError:
        print("    ✗ Could not import isaaclab.app. Make sure you're running with isaaclab.bat")
        sys.exit(1)
    
    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app
    print("    ✓ Isaac Sim initialized")
    
    # Now we can import Isaac Lab modules
    import torch
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecEnvWrapper
    
    # Create wrapper (same as evaluate.py)
    class IsaacLabToSB3VecEnvWrapper(VecEnvWrapper):
        """VecEnv wrapper to convert Isaac Lab's dict observations with torch tensors to numpy arrays for SB3."""
        
        def __init__(self, venv):
            VecEnvWrapper.__init__(self, venv)
            
            if hasattr(venv.action_space, 'shape') and len(venv.action_space.shape) > 1:
                action_dim = venv.action_space.shape[-1]
                self.action_space = spaces.Box(
                    low=venv.action_space.low.flatten()[0],
                    high=venv.action_space.high.flatten()[0],
                    shape=(action_dim,),
                    dtype=venv.action_space.dtype
                )
            
            dummy_obs = venv.reset()
            if isinstance(dummy_obs, tuple):
                dummy_obs, _ = dummy_obs
            if isinstance(dummy_obs, dict):
                obs_tensor = dummy_obs.get("policy", list(dummy_obs.values())[0])
                if hasattr(obs_tensor, 'cpu'):
                    dummy_obs = obs_tensor.cpu().numpy()
                else:
                    dummy_obs = np.array(obs_tensor)
            
            if len(dummy_obs.shape) > 1:
                obs_shape = (dummy_obs.shape[1],)
            else:
                obs_shape = dummy_obs.shape
            
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=obs_shape,
                dtype=np.float32
            )
            
        def reset(self):
            obs = self.venv.reset()
            if isinstance(obs, tuple):
                obs, info = obs
            if isinstance(obs, dict):
                obs_tensor = obs.get("policy", list(obs.values())[0])
                if hasattr(obs_tensor, 'cpu'):
                    obs = obs_tensor.cpu().numpy()
                else:
                    obs = np.array(obs_tensor)
            return obs
        
        def step_async(self, actions):
            if isinstance(actions, np.ndarray):
                device = self.venv.unwrapped.device if hasattr(self.venv.unwrapped, 'device') else 'cuda:0'
                actions = torch.from_numpy(actions).float().to(device)
            self._actions = actions
        
        def step_wait(self):
            result = self.venv.step(self._actions)
            
            if len(result) == 5:
                obs, rewards, terminated, truncated, infos = result
                dones = terminated | truncated
            else:
                obs, rewards, dones, infos = result
            
            if isinstance(obs, dict):
                obs_tensor = obs.get("policy", list(obs.values())[0])
                if hasattr(obs_tensor, 'cpu'):
                    obs = obs_tensor.cpu().numpy()
                else:
                    obs = np.array(obs_tensor)
            
            if hasattr(rewards, 'cpu'):
                rewards = rewards.cpu().numpy()
            if hasattr(dones, 'cpu'):
                dones = dones.cpu().numpy()
            
            if isinstance(infos, dict):
                infos = [infos.copy() for _ in range(len(rewards))]
            elif not isinstance(infos, list):
                infos = [{} for _ in range(len(rewards))]
            else:
                infos = [info if isinstance(info, dict) else {} for info in infos]
            
            return obs, rewards, dones, infos
    
    # Register custom tasks
    print("\n[2/6] Registering custom tasks...")
    try:
        from task_spec import register_isaac_lab_tasks
        register_isaac_lab_tasks()
        print(f"    ✓ Registered task: {args.task}")
    except Exception as e:
        print(f"    ✗ Failed to register tasks: {e}")
        simulation_app.close()
        sys.exit(1)
    
    # Set device
    print("\n[3/6] Setting up device...")
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"    ✓ Using device: {device}")
    
    # Create environment
    print(f"\n[4/6] Creating environment ({args.task})...")
    print(f"    Trajectory type: {args.trajectory_type}")
    if args.trajectory_type == "multi_recorded":
        print(f"    Trajectory directory: {args.trajectory_dir}")
        print(f"    Use all trajectories: {args.use_all_trajectories}")
        print(f"    Use chassis only: {args.use_chassis_only}")
        if args.max_trajectories:
            print(f"    Max trajectories: {args.max_trajectories}")
    
    try:
        from rl_platform.tasks.mobile_mm import MobileMMTrackEEEnvCfg, MobileMMTrackEEEnv
        from rl_platform.tasks.mobile_mm.config import TrajectoryConfig
        
        env_cfg = MobileMMTrackEEEnvCfg()
        env_cfg.scene.num_envs = args.num_envs
        
        # Prepare trajectory filter
        trajectory_filter_indices = None
        if args.trajectory_type == "multi_recorded":
            if args.use_chassis_only:
                import json
                analysis_file = Path("trajectoryToLearn/trajectory_analysis.json")
                if analysis_file.exists():
                    with open(analysis_file, 'r') as f:
                        analysis = json.load(f)
                    trajectory_filter_indices = analysis.get('chassis_requiring_indices', [])
                    print(f"    Using {len(trajectory_filter_indices)} chassis-requiring trajectories")
        
        env_cfg.task_config.trajectory = TrajectoryConfig(
            type=args.trajectory_type,
            trajectory_dir=args.trajectory_dir,
            trajectory_pattern="**/*.json",
            trajectory_filter_indices=trajectory_filter_indices,
            max_trajectories=args.max_trajectories,
        )
        
        base_env = MobileMMTrackEEEnv(cfg=env_cfg)
        
        # DISABLE termination for full episode evaluation
        base_env.task_cfg.terminate_on_tracking_error = False
        base_env.task_cfg.terminate_on_self_collision = False
        print(f"    ✓ Disabled early termination for full episode evaluation")
        
        env = IsaacLabToSB3VecEnvWrapper(base_env)
        
        print(f"    ✓ Environment created with {args.num_envs} parallel instances")
        print(f"    Observation space: {env.observation_space}")
        print(f"    Action space: {env.action_space}")
    except Exception as e:
        print(f"    ✗ Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
    
    # Load trained model
    print(f"\n[5/6] Loading trained model...")
    try:
        model = PPO.load(args.checkpoint, env=env, device=device)
        print(f"    ✓ Model loaded successfully")
    except Exception as e:
        print(f"    ✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        simulation_app.close()
        sys.exit(1)
    
    # Initialize logger with model-specific subdirectory
    print(f"\n[6/6] Initializing evaluation logger...")
    
    # Extract model folder name from checkpoint path
    # Example: logs/sb3/mobilemmtrackee_v0/20251028_200923/final_model.zip
    # -> 20251028_200923
    checkpoint_parent = Path(args.checkpoint).parent.name
    model_output_dir = Path(args.output_dir) / checkpoint_parent
    
    logger = EvaluationLogger(model_output_dir, args.num_envs)
    print(f"    ✓ Logger ready")
    print(f"    ✓ Results will be saved to: {model_output_dir}")
    
    print("\n" + "=" * 80)
    print("🚀 Starting Comprehensive Evaluation...")
    print("=" * 80)
    print(f"📊 Logging metrics:")
    print(f"   • Position tracking errors (x, y, z)")
    print(f"   • Orientation tracking errors (angle)")
    print(f"   • Joint angles and velocities (6 DOF)")
    print(f"   • Base velocities (vx, vy, ωz)")
    print(f"   • Reward components breakdown")
    print(f"   • Episode success/failure statistics")
    print()
    
    # Evaluation loop
    obs = env.reset()
    episode_count = 0
    current_episode_reward = np.zeros(args.num_envs)
    current_episode_length = np.zeros(args.num_envs)
    
    global_step = 0
    start_time = datetime.now()
    
    print(f"Running {args.num_episodes} episodes across {args.num_envs} environments...")
    print(f"Expected steps: ~{args.num_episodes * 500 // args.num_envs} (assuming 500 steps/episode)")
    print()
    
    while episode_count < args.num_episodes:
        # Get action from policy
        action, _states = model.predict(obs, deterministic=args.deterministic)
        
        # Step environment
        obs, rewards, dones, infos = env.step(action)
        
        # Extract environment states for logging
        env_states = extract_env_states(env, obs, rewards, infos)
        
        # Log step data (sample every N steps to reduce memory)
        save_detailed = (global_step % args.save_every_n_steps == 0)
        logger.log_step(global_step, env_states, save_detailed=save_detailed)
        
        # Track episode progress
        current_episode_reward += rewards
        current_episode_length += 1
        
        # Handle episode termination
        for i, done in enumerate(dones):
            if done:
                # Log episode statistics
                episode_stats = {
                    'total_reward': float(current_episode_reward[i]),
                    'length': int(current_episode_length[i]),
                }
                
                # Add success flag if available
                if 'success' in infos[i]:
                    episode_stats['success'] = bool(infos[i]['success'])
                
                logger.log_episode(episode_count, i, episode_stats)
                
                episode_count += 1
                
                # Print progress
                if episode_count % 10 == 0 or episode_count == args.num_episodes:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    rate = episode_count / elapsed if elapsed > 0 else 0
                    eta = (args.num_episodes - episode_count) / rate if rate > 0 else 0
                    print(f"  Progress: {episode_count}/{args.num_episodes} episodes "
                          f"({100*episode_count/args.num_episodes:.1f}%) | "
                          f"Rate: {rate:.1f} ep/s | ETA: {eta:.0f}s")
                
                # Reset counters
                current_episode_reward[i] = 0
                current_episode_length[i] = 0
                
                if episode_count >= args.num_episodes:
                    break
        
        global_step += 1
    
    total_time = (datetime.now() - start_time).total_seconds()
    
    # Save results and compute statistics
    print("\n" + "=" * 80)
    print("💾 Saving results...")
    print("=" * 80)
    stats = logger.save_results(timestamp, args)
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 EVALUATION SUMMARY")
    print("=" * 80)
    print(f"⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"📈 Episodes completed: {stats['episodes']['count']}")
    print()
    
    print("🎯 TRACKING ACCURACY:")
    if 'position_error' in stats:
        print(f"  Position Error:")
        print(f"    Mean:   {stats['position_error']['mean_cm']:.2f} cm")
        print(f"    Median: {stats['position_error']['median_cm']:.2f} cm")
        print(f"    P95:    {stats['position_error']['p95_cm']:.2f} cm")
        print(f"    Max:    {stats['position_error']['max_m']*100:.2f} cm")
    
    if 'orientation_error' in stats:
        print(f"  Orientation Error:")
        print(f"    Mean:   {stats['orientation_error']['mean_deg']:.2f}°")
        print(f"    Median: {stats['orientation_error']['median_deg']:.2f}°")
        print(f"    P95:    {stats['orientation_error']['p95_deg']:.2f}°")
    print()
    
    print("🎁 REWARDS:")
    print(f"  Mean:   {stats['episodes']['mean_reward']:.2f} ± {stats['episodes']['std_reward']:.2f}")
    print(f"  Median: {stats['episodes']['median_reward']:.2f}")
    print(f"  Range:  [{stats['episodes']['min_reward']:.2f}, {stats['episodes']['max_reward']:.2f}]")
    print()
    
    if 'success_rate' in stats['episodes']:
        print("✅ SUCCESS RATE:")
        print(f"  {stats['episodes']['success_rate']*100:.1f}% "
              f"({stats['episodes']['success_count']}/{stats['episodes']['count']} episodes)")
        print()
    
    print("📁 Output files:")
    print(f"  {args.output_dir}/eval_summary_{timestamp}.json")
    print(f"  {args.output_dir}/episodes_{timestamp}.csv")
    print(f"  {args.output_dir}/steps_{timestamp}.csv")
    print(f"  {args.output_dir}/arrays_{timestamp}.npz")
    print("=" * 80)
    
    # Cleanup
    print("\nCleaning up...")
    env.close()
    simulation_app.close()
    
    print("✅ Quantitative evaluation complete!")


if __name__ == "__main__":
    main()
