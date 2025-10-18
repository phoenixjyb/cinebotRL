"""Evaluation script for trained MobileMMTrackEE models.

This script loads a trained model and runs it in the environment with visualization
to see how well it performs on the end-effector tracking task.

Usage:
    # Visualize trained model with GUI:
    I:\isaaclab\isaaclab.bat -p scripts/reinforcement_learning/sb3/evaluate.py \\
        --checkpoint H:\wSpace\cinebotRL\logs\sb3\mobilemmtrackee_v0\20251016_184941\final_model.zip \\
        --num_envs 4
    
    # Run headless evaluation (get metrics only):
    I:\isaaclab\isaaclab.bat -p scripts/reinforcement_learning/sb3/evaluate.py \\
        --checkpoint H:\wSpace\cinebotRL\logs\sb3\mobilemmtrackee_v0\20251016_184941\final_model.zip \\
        --num_envs 16 \\
        --headless \\
        --num_episodes 100
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np

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
        description="Evaluate trained mobile manipulator policy"
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
        default=4,
        help="Number of parallel environments (use fewer for better visualization)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no GUI)",
    )
    
    # Evaluation settings
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions (no exploration noise)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to run on",
    )
    
    # Trajectory configuration (to test against recorded trajectories)
    parser.add_argument(
        "--trajectory_type",
        type=str,
        default="circle",
        choices=["line", "circle", "figure_eight", "recorded", "multi_recorded"],
        help="Type of trajectory to test against (use multi_recorded to test on your training trajectories)",
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
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Validate checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    print("=" * 70)
    print("MobileMMTrackEE Policy Evaluation")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Task: {args.task}")
    print(f"Num envs: {args.num_envs}")
    print(f"Headless: {args.headless}")
    print(f"Deterministic: {args.deterministic}")
    print()
    
    # Initialize Isaac Sim
    print("[1/5] Initializing Isaac Sim...")
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
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecEnvWrapper
    
    # Create wrapper to convert Isaac Lab observations to SB3 format
    class IsaacLabToSB3VecEnvWrapper(VecEnvWrapper):
        """VecEnv wrapper to convert Isaac Lab's dict observations with torch tensors to numpy arrays for SB3."""
        
        def __init__(self, venv):
            # Isaac Lab env is already a VecEnv
            VecEnvWrapper.__init__(self, venv)
            
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
            
            # FIX: Set observation space correctly by doing a reset to get actual obs shape
            # This is needed because PPO.load() checks spaces BEFORE first reset
            dummy_obs = venv.reset()
            if isinstance(dummy_obs, tuple):
                dummy_obs, _ = dummy_obs
            if isinstance(dummy_obs, dict):
                obs_tensor = dummy_obs.get("policy", list(dummy_obs.values())[0])
                if hasattr(obs_tensor, 'cpu'):
                    dummy_obs = obs_tensor.cpu().numpy()
                else:
                    dummy_obs = np.array(obs_tensor)
            
            # dummy_obs shape is [num_envs, obs_dim], we want [obs_dim] for the space
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
            
            print(f"✓ Wrapper initialized: obs_space={self.observation_space.shape}, action_space={self.action_space.shape}")
            
        def reset(self):
            obs = self.venv.reset()
            
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
            
            return obs
        
        def step_async(self, actions):
            # Convert numpy actions to torch tensors for Isaac Lab
            if isinstance(actions, np.ndarray):
                device = self.venv.unwrapped.device if hasattr(self.venv.unwrapped, 'device') else 'cuda:0'
                actions = torch.from_numpy(actions).float().to(device)
            self._actions = actions
        
        def step_wait(self):
            # Call the synchronous step() method with stored actions
            result = self.venv.step(self._actions)
            
            # Handle both old (4 values) and new (5 values) Gymnasium API
            if len(result) == 5:
                obs, rewards, terminated, truncated, infos = result
                dones = terminated | truncated
            else:
                obs, rewards, dones, infos = result
            
            # Convert observations
            if isinstance(obs, dict):
                obs_tensor = obs.get("policy", list(obs.values())[0])
                if hasattr(obs_tensor, 'cpu'):
                    obs = obs_tensor.cpu().numpy()
                else:
                    obs = np.array(obs_tensor)
            
            # Convert rewards and dones to numpy
            if hasattr(rewards, 'cpu'):
                rewards = rewards.cpu().numpy()
            if hasattr(dones, 'cpu'):
                dones = dones.cpu().numpy()
            
            # Ensure infos is a list of dicts
            if isinstance(infos, dict):
                infos = [infos.copy() for _ in range(len(rewards))]
            elif not isinstance(infos, list):
                infos = [{} for _ in range(len(rewards))]
            else:
                infos = [info if isinstance(info, dict) else {} for info in infos]
            
            return obs, rewards, dones, infos
    
    # Register custom tasks
    print("\n[2/5] Registering custom tasks...")
    try:
        from task_spec import register_isaac_lab_tasks
        register_isaac_lab_tasks()
        print(f"    ✓ Registered task: {args.task}")
    except Exception as e:
        print(f"    ✗ Failed to register tasks: {e}")
        simulation_app.close()
        sys.exit(1)
    
    # Set device
    print("\n[3/5] Setting up device...")
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"    ✓ Using device: {device}")
    
    # Create environment
    print(f"\n[4/5] Creating environment ({args.task})...")
    print(f"    Trajectory type: {args.trajectory_type}")
    if args.trajectory_type == "multi_recorded":
        print(f"    Trajectory directory: {args.trajectory_dir}")
        print(f"    Use all trajectories: {args.use_all_trajectories}")
        print(f"    Use chassis only: {args.use_chassis_only}")
        if args.max_trajectories:
            print(f"    Max trajectories: {args.max_trajectories}")
    
    try:
        # Import environment config to modify it (same as train.py)
        from rl_platform.tasks.mobile_mm import MobileMMTrackEEEnvCfg, MobileMMTrackEEEnv
        from rl_platform.tasks.mobile_mm.config import TrajectoryConfig
        
        # Create custom environment configuration
        env_cfg = MobileMMTrackEEEnvCfg()
        env_cfg.scene.num_envs = args.num_envs
        
        # Prepare trajectory filter (same logic as train.py)
        trajectory_filter_indices = None
        if args.trajectory_type == "multi_recorded":
            if args.use_chassis_only:
                # Load chassis-requiring trajectory indices
                import json
                from pathlib import Path
                analysis_file = Path("trajectoryToLearn/trajectory_analysis.json")
                if analysis_file.exists():
                    with open(analysis_file, 'r') as f:
                        analysis = json.load(f)
                    trajectory_filter_indices = analysis.get('chassis_requiring_indices', [])
                    print(f"    Using {len(trajectory_filter_indices)} chassis-requiring trajectories")
                else:
                    print(f"    WARNING: trajectory_analysis.json not found, using all trajectories")
        
        # Configure trajectory (same as train.py)
        env_cfg.task_config.trajectory = TrajectoryConfig(
            type=args.trajectory_type,
            trajectory_dir=args.trajectory_dir,
            trajectory_pattern="**/*.json",
            trajectory_filter_indices=trajectory_filter_indices,
            max_trajectories=args.max_trajectories,
        )
        
        # Create environment directly with config (same as train.py)
        base_env = MobileMMTrackEEEnv(cfg=env_cfg)
        
        # DISABLE termination conditions for evaluation (let episodes run full length)
        base_env.task_cfg.terminate_on_tracking_error = False
        base_env.task_cfg.terminate_on_self_collision = False
        print(f"    ✓ Disabled termination conditions for full episode visualization")
        
        # Wrap for SB3 compatibility using our wrapper class
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
    print(f"\n[5/5] Loading trained model...")
    try:
        # Load model with the environment (allows different num_envs)
        model = PPO.load(args.checkpoint, env=env, device=device)
        print(f"    ✓ Model loaded successfully")
        print(f"    Policy network: {model.policy}")
    except Exception as e:
        print(f"    ✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        simulation_app.close()
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("Starting Evaluation...")
    print("=" * 70)
    if not args.headless:
        print("🎨 VISUALIZATION ENABLED:")
        print("   🔴 Red spheres = Trajectory targets")
        print("   🟢 Green spheres = Robot end-effector")
        print("   Watch if green follows red!")
        print()
    
    # Evaluation loop
    episode_rewards = []
    episode_lengths = []
    tracking_errors = []
    
    obs = env.reset()
    episode_count = 0
    current_episode_reward = np.zeros(args.num_envs)
    current_episode_length = np.zeros(args.num_envs)
    
    print(f"Running {args.num_episodes} episodes...")
    
    global_step = 0
    while episode_count < args.num_episodes:
        # Get action from policy
        action, _states = model.predict(obs, deterministic=args.deterministic)
        
        # Step environment
        obs, rewards, dones, infos = env.step(action)
        
        # Debug: Print reward components every 100 steps (first episode only)
        if episode_count == 0 and global_step % 100 == 0:
            print(f"\n[DEBUG Step {global_step}] Total reward: {rewards[0]:.2f}")
            if hasattr(env.unwrapped, 'reward_components') and env.unwrapped.reward_components:
                for key, val in sorted(env.unwrapped.reward_components.items()):
                    v = val[0].item() if hasattr(val, 'item') else val[0]
                    sign = "+" if v >= 0 else ""
                    print(f"  {key:35s}: {sign}{v:+10.4f}")
        
        global_step += 1
        
        # Track metrics
        current_episode_reward += rewards
        current_episode_length += 1
        
        # Handle episode termination
        for i, done in enumerate(dones):
            if done:
                episode_rewards.append(current_episode_reward[i])
                episode_lengths.append(current_episode_length[i])
                
                episode_count += 1
                print(f"  Episode {episode_count}/{args.num_episodes}: "
                      f"Reward={current_episode_reward[i]:.2f}, "
                      f"Length={int(current_episode_length[i])}")
                
                current_episode_reward[i] = 0
                current_episode_length[i] = 0
                
                if episode_count >= args.num_episodes:
                    break
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("Evaluation Summary")
    print("=" * 70)
    print(f"Episodes completed: {len(episode_rewards)}")
    print(f"Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")
    print(f"Mean episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print("=" * 70)
    
    # Cleanup
    print("\nCleaning up...")
    env.close()
    simulation_app.close()
    
    print("✓ Evaluation complete!")


if __name__ == "__main__":
    main()
