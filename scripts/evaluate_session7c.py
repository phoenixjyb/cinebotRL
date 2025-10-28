"""
Session 7c Evaluation Script
Evaluates the trained model from Session 7c with base movement fix
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add project root and src to path (same as train.py)
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def main():
    parser = argparse.ArgumentParser(description="Evaluate Session 7c trained model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to evaluate (default: final_model.zip)"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Number of episodes to evaluate (default: 100)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run in headless mode"
    )
    parser.add_argument(
        "--save_stats",
        action="store_true",
        default=False,
        help="Save detailed statistics to file"
    )
    
    args = parser.parse_args()
    
    # Default to the final Session 7c model
    if args.checkpoint is None:
        log_dir = PROJECT_ROOT / "logs" / "sb3" / "mobilemmtrackee_v0" / "20251027_180246"
        checkpoint_path = log_dir / "final_model.zip"
    else:
        checkpoint_path = Path(args.checkpoint)
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"\n{'='*80}")
    print(f"SESSION 7C EVALUATION")
    print(f"{'='*80}")
    print(f"📦 Checkpoint: {checkpoint_path}")
    print(f"🎬 Episodes: {args.num_episodes}")
    print(f"👁️  Headless: {args.headless}")
    print(f"{'='*80}\n")
    
    # CRITICAL: Initialize Isaac Sim BEFORE importing task modules
    print("🔧 Initializing Isaac Sim...")
    try:
        from isaaclab.app import AppLauncher
        import torch
        
        # Create AppLauncher to initialize Isaac Sim
        app_launcher = AppLauncher(
            headless=args.headless,
            enable_cameras=False,
        )
        simulation_app = app_launcher.app
        print("    ✓ Isaac Sim initialized")
        
    except Exception as e:
        print(f"    ✗ Failed to initialize Isaac Sim: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Now we can import task-specific modules
    print("🔧 Loading SB3 and registering tasks...")
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize
    import gymnasium as gym
    
    # Register Isaac Lab tasks
    from src.task_spec import register_isaac_lab_tasks
    register_isaac_lab_tasks()
    print("    ✓ Tasks registered")
    
    print("🔧 Creating evaluation environment...")
    
    # Create environment (use same config as training)
    env_cfg_entry_point = "MobileMMTrackEE-v0"
    env = gym.make(env_cfg_entry_point, num_envs=1, headless=args.headless)
    
    # Wrap environment with IsaacLab to SB3 converter (same as train.py)
    print("🔧 Wrapping environment for SB3...")
    from stable_baselines3.common.vec_env import VecEnvWrapper
    from gymnasium import spaces
    
    class IsaacLabToSB3VecEnvWrapper(VecEnvWrapper):
        """VecEnv wrapper to convert Isaac Lab's dict observations with torch tensors to numpy arrays for SB3."""
        
        def __init__(self, venv):
            VecEnvWrapper.__init__(self, venv)
            self._obs_space_updated = False
            
            # FIX: Isaac Lab's action_space includes batch dimension [num_envs, action_dim]
            if hasattr(venv.action_space, 'shape') and len(venv.action_space.shape) > 1:
                action_dim = venv.action_space.shape[-1]
                self.action_space = spaces.Box(
                    low=venv.action_space.low.flatten()[0],
                    high=venv.action_space.high.flatten()[0],
                    shape=(action_dim,),
                    dtype=venv.action_space.dtype
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
                
                if not self._obs_space_updated:
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
    
    env = IsaacLabToSB3VecEnvWrapper(env)
    
    # Do a dummy reset to let the wrapper discover the observation shape
    print("🔧 Initializing observation space...")
    _ = env.reset()
    print(f"   Observation space: {env.observation_space.shape}")
    
    # Now load VecNormalize statistics
    vec_normalize_path = checkpoint_path.parent / "vec_normalize.pkl"
    if vec_normalize_path.exists():
        print(f"📊 Loading VecNormalize statistics from: {vec_normalize_path}")
        env = VecNormalize.load(str(vec_normalize_path), env)
        env.training = False  # Important: disable updates during evaluation
        env.norm_reward = False  # Don't normalize rewards during eval
    else:
        print(f"⚠️  VecNormalize statistics not found at: {vec_normalize_path}")
    
    print(f"🤖 Loading model from: {checkpoint_path}")
    model = PPO.load(str(checkpoint_path), env=env)
    
    print(f"\n{'='*80}")
    print(f"RUNNING EVALUATION")
    print(f"{'='*80}\n")
    
    # Statistics tracking
    episode_errors = []
    episode_rewards = []
    episode_lengths = []
    base_movements = []
    
    # Per-timestep tracking for detailed analysis
    all_ee_errors = []
    all_base_distances = []
    all_base_movements = []
    
    obs = env.reset()
    episode_count = 0
    current_episode_errors = []
    current_episode_rewards = []
    current_base_start_pos = None
    
    while episode_count < args.num_episodes:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        # Extract tracking statistics from info
        if isinstance(info, dict) and 'ee_error' in info:
            ee_error = info['ee_error']
            base_target_dist = info.get('base_target_distance', 0.0)
            
            current_episode_errors.append(ee_error)
            current_episode_rewards.append(reward[0])
            all_ee_errors.append(ee_error)
            all_base_distances.append(base_target_dist)
            
            # Track base movement from episode start
            if current_base_start_pos is None and 'base_pos' in info:
                current_base_start_pos = np.array(info['base_pos'])
            
            if current_base_start_pos is not None and 'base_pos' in info:
                base_pos = np.array(info['base_pos'])
                base_movement = np.linalg.norm(base_pos[:2] - current_base_start_pos[:2])
                all_base_movements.append(base_movement)
        
        if done[0]:
            episode_count += 1
            
            if len(current_episode_errors) > 0:
                mean_error = np.mean(current_episode_errors)
                total_reward = np.sum(current_episode_rewards)
                episode_length = len(current_episode_errors)
                final_base_movement = all_base_movements[-1] if all_base_movements else 0.0
                
                episode_errors.append(mean_error)
                episode_rewards.append(total_reward)
                episode_lengths.append(episode_length)
                base_movements.append(final_base_movement)
                
                print(f"Episode {episode_count:3d}: "
                      f"Error={mean_error:.4f}m, "
                      f"Reward={total_reward:.1f}, "
                      f"Length={episode_length:3d}, "
                      f"Base Movement={final_base_movement:.3f}m")
                
                current_episode_errors = []
                current_episode_rewards = []
                current_base_start_pos = None
            
            obs = env.reset()
    
    # Compute final statistics
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS - SESSION 7C")
    print(f"{'='*80}\n")
    
    if len(episode_errors) > 0:
        print(f"📊 **TRACKING PERFORMANCE**:")
        print(f"   Mean EE Error:     {np.mean(episode_errors):.4f} ± {np.std(episode_errors):.4f} m")
        print(f"   Median EE Error:   {np.median(episode_errors):.4f} m")
        print(f"   Min EE Error:      {np.min(episode_errors):.4f} m")
        print(f"   Max EE Error:      {np.max(episode_errors):.4f} m")
        print(f"   95th Percentile:   {np.percentile(episode_errors, 95):.4f} m\n")
        
        print(f"🚗 **BASE MOVEMENT**:")
        print(f"   Mean Movement:     {np.mean(base_movements):.4f} ± {np.std(base_movements):.4f} m")
        print(f"   Median Movement:   {np.median(base_movements):.4f} m")
        print(f"   Min Movement:      {np.min(base_movements):.4f} m")
        print(f"   Max Movement:      {np.max(base_movements):.4f} m\n")
        
        print(f"💰 **REWARDS**:")
        print(f"   Mean Episode Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"   Min Episode Reward:  {np.min(episode_rewards):.2f}")
        print(f"   Max Episode Reward:  {np.max(episode_rewards):.2f}\n")
        
        print(f"📏 **EPISODE LENGTHS**:")
        print(f"   Mean Length:       {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f} steps")
        print(f"   Min Length:        {np.min(episode_lengths)} steps")
        print(f"   Max Length:        {np.max(episode_lengths)} steps\n")
        
        # Performance classification
        excellent_pct = np.sum(np.array(episode_errors) < 0.1) / len(episode_errors) * 100
        good_pct = np.sum((np.array(episode_errors) >= 0.1) & (np.array(episode_errors) < 0.3)) / len(episode_errors) * 100
        poor_pct = np.sum((np.array(episode_errors) >= 0.3) & (np.array(episode_errors) < 2.0)) / len(episode_errors) * 100
        broken_pct = np.sum(np.array(episode_errors) >= 2.0) / len(episode_errors) * 100
        
        print(f"🎯 **PERFORMANCE CLASSIFICATION**:")
        print(f"   Excellent (<0.1m):  {excellent_pct:.1f}%")
        print(f"   Good (0.1-0.3m):    {good_pct:.1f}%")
        print(f"   Poor (0.3-2.0m):    {poor_pct:.1f}%")
        print(f"   Broken (>2.0m):     {broken_pct:.1f}%\n")
        
        print(f"{'='*80}")
        
        # Save detailed statistics if requested
        if args.save_stats:
            stats_file = checkpoint_path.parent / "evaluation_stats.npz"
            np.savez(
                stats_file,
                episode_errors=np.array(episode_errors),
                episode_rewards=np.array(episode_rewards),
                episode_lengths=np.array(episode_lengths),
                base_movements=np.array(base_movements),
                all_ee_errors=np.array(all_ee_errors),
                all_base_distances=np.array(all_base_distances),
                all_base_movements=np.array(all_base_movements),
            )
            print(f"\n💾 Statistics saved to: {stats_file}")
    
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
