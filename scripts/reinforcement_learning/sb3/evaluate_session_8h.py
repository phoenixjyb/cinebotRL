"""Session 8h Checkpoint Evaluation Script.

Evaluates multiple Session 8h checkpoints and compares them against Session 8f/8g baselines.

Usage:
    # Evaluate Session 8h checkpoints at key milestones
    I:\isaaclab\isaaclab.bat -p scripts/reinforcement_learning/sb3/evaluate_session_8h.py \
        --session_8h_dir logs/sb3/mobilemmtrackee_v0/20251103_235918 \
        --checkpoints 20M 40M 60M 80M 100M \
        --num_envs 64 \
        --num_episodes 200 \
        --headless
    
    # Quick test with fewer episodes
    I:\isaaclab\isaaclab.bat -p scripts/reinforcement_learning/sb3/evaluate_session_8h.py \
        --session_8h_dir logs/sb3/mobilemmtrackee_v0/20251103_235918 \
        --checkpoints 20M 40M 100M \
        --num_envs 16 \
        --num_episodes 50
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
from datetime import datetime
import json
from typing import Dict, List, Optional
import glob

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
        description="Evaluate Session 8h checkpoints and compare with baselines"
    )
    
    # Session 8h directory
    parser.add_argument(
        "--session_8h_dir",
        type=str,
        required=True,
        help="Path to Session 8h log directory (e.g., logs/sb3/mobilemmtrackee_v0/20251103_235918)",
    )
    
    # Checkpoints to evaluate
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=["20M", "40M", "60M", "80M", "100M"],
        help="Which checkpoints to evaluate (e.g., 20M 40M 100M or 'final')",
    )
    
    # Baseline models for comparison (optional)
    parser.add_argument(
        "--session_8f_checkpoint",
        type=str,
        default=None,
        help="Path to Session 8f final model for comparison (optional)",
    )
    parser.add_argument(
        "--session_8g_40m_checkpoint",
        type=str,
        default=None,
        help="Path to Session 8g 40M checkpoint for comparison (optional)",
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
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (faster)",
    )
    
    # Evaluation settings
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=200,
        help="Number of episodes per checkpoint (recommend 200+ for robust stats)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic actions",
    )
    
    # Trajectory configuration
    parser.add_argument(
        "--use_chassis_only",
        action="store_true",
        default=True,
        help="Use only chassis-requiring trajectories (matches training)",
    )
    
    # Output settings
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results/session_8h_comparison",
        help="Directory to save comparison results",
    )
    
    return parser.parse_args()


def find_checkpoint(base_dir: Path, milestone: str) -> Optional[Path]:
    """Find checkpoint file for a given milestone.
    
    Args:
        base_dir: Base directory containing checkpoints
        milestone: Milestone name (e.g., '20M', '40M', '100M', 'final')
    
    Returns:
        Path to checkpoint file, or None if not found
    """
    checkpoint_dir = base_dir / "checkpoints"
    
    if milestone.lower() == "final":
        # Check for final_model.zip in parent directory
        final_model = base_dir / "final_model.zip"
        if final_model.exists():
            return final_model
        return None
    
    # Convert milestone to steps (e.g., '20M' -> 20000000)
    if milestone.endswith('M'):
        target_steps = int(float(milestone[:-1]) * 1_000_000)
    elif milestone.endswith('K'):
        target_steps = int(float(milestone[:-1]) * 1_000)
    else:
        target_steps = int(milestone)
    
    # Find checkpoint closest to target steps
    pattern = str(checkpoint_dir / f"ppo_mobile_mm_*_steps.zip")
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        return None
    
    # Extract step counts and find closest
    closest_checkpoint = None
    min_diff = float('inf')
    
    for ckpt in checkpoints:
        try:
            # Extract steps from filename: ppo_mobile_mm_12345678_steps.zip
            steps_str = Path(ckpt).stem.split('_')[-2]
            steps = int(steps_str)
            diff = abs(steps - target_steps)
            
            if diff < min_diff:
                min_diff = diff
                closest_checkpoint = ckpt
        except (ValueError, IndexError):
            continue
    
    return Path(closest_checkpoint) if closest_checkpoint else None


def generate_comparison_report(
    all_results: List[Dict],
    output_dir: Path
):
    """Generate comparison report with tables and plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw results
    results_file = output_dir / f"session_8h_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS SUMMARY")
    print(f"{'='*80}\n")
    
    # Print comparison table
    print(f"{'Checkpoint':<15} | {'Pos Error (cm)':<18} | {'Ori Error (°)':<18} | {'Workspace Viol %':<18}")
    print(f"{'-'*15}-+-{'-'*18}-+-{'-'*18}-+-{'-'*18}")
    
    for result in all_results:
        name = result['checkpoint_name']
        pos_err = result.get('position_error', {}).get('mean_cm', 'N/A')
        ori_err = result.get('orientation_error', {}).get('mean_deg', 'N/A')
        ws_viol = result.get('workspace_violations', {}).get('rate_percent', 'N/A')
        
        if isinstance(pos_err, float):
            pos_str = f"{pos_err:.1f}"
        else:
            pos_str = str(pos_err)
        
        if isinstance(ori_err, float):
            ori_str = f"{ori_err:.1f}"
        else:
            ori_str = str(ori_err)
        
        if isinstance(ws_viol, float):
            ws_str = f"{ws_viol:.1f}"
        else:
            ws_str = str(ws_viol)
        
        print(f"{name:<15} | {pos_str:<18} | {ori_str:<18} | {ws_str:<18}")
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*80}\n")


def main():
    args = parse_args()
    
    session_8h_dir = Path(args.session_8h_dir)
    if not session_8h_dir.exists():
        raise FileNotFoundError(f"Session 8h directory not found: {session_8h_dir}")
    
    print(f"\n{'='*80}")
    print(f"SESSION 8H CHECKPOINT EVALUATION")
    print(f"{'='*80}")
    print(f"Session 8h directory: {session_8h_dir}")
    print(f"Checkpoints to evaluate: {', '.join(args.checkpoints)}")
    print(f"Episodes per checkpoint: {args.num_episodes}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*80}\n")
    
    # Find all checkpoints first
    checkpoints_to_eval = []
    
    for milestone in args.checkpoints:
        checkpoint_path = find_checkpoint(session_8h_dir, milestone)
        if checkpoint_path:
            checkpoints_to_eval.append((milestone, checkpoint_path))
            print(f"✅ Found {milestone}: {checkpoint_path.name}")
        else:
            print(f"❌ Could not find checkpoint for {milestone}")
    
    if not checkpoints_to_eval:
        print("\n❌ No checkpoints found to evaluate!")
        return
    
    # Step 1: Initialize Isaac Sim ONCE (for all checkpoints)
    print(f"\n{'='*80}")
    print("[1/4] Initializing Isaac Sim (one-time setup)...")
    print(f"{'='*80}\n")
    
    try:
        from isaaclab.app import AppLauncher
        
        app_launcher = AppLauncher(headless=args.headless)
        simulation_app = app_launcher.app
        print("    [OK] Isaac Sim initialized")
    except Exception as e:
        print(f"    [FAIL] Failed to initialize Isaac Sim: {e}")
        print("    Make sure you're running with isaaclab.bat!")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Register tasks
    print("\n[2/4] Registering tasks...")
    try:
        from task_spec import register_isaac_lab_tasks
        register_isaac_lab_tasks()
        print(f"    [OK] Task registered: {args.task}")
    except Exception as e:
        print(f"    [FAIL] Failed to register tasks: {e}")
        simulation_app.close()
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Import dependencies
    print("\n[3/4] Importing dependencies...")
    try:
        import gymnasium as gym
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecNormalize
        import torch
        print("    [OK] Dependencies imported")
    except Exception as e:
        print(f"    [FAIL] Failed to import dependencies: {e}")
        simulation_app.close()
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Create environment ONCE (reuse for all checkpoints)
    print("\n[4/6] Creating environment (one-time setup)...")
    try:
        env = gym.make(
            args.task,
            num_envs=args.num_envs,
            render_mode="rgb_array" if not args.headless else None
        )
        print(f"    [OK] Environment created: {args.task} with {args.num_envs} envs")
    except Exception as e:
        print(f"    [FAIL] Failed to create environment: {e}")
        simulation_app.close()
        import traceback
        traceback.print_exc()
        return
    
    # Step 5: Evaluate each checkpoint (reusing same environment)
    print(f"\n[5/6] Evaluating {len(checkpoints_to_eval)} checkpoints...")
    all_results = []
    
    for idx, (milestone, checkpoint_path) in enumerate(checkpoints_to_eval, 1):
        print(f"\n{'-'*80}")
        print(f"Checkpoint {idx}/{len(checkpoints_to_eval)}: {milestone}")
        print(f"Path: {checkpoint_path.name}")
        print(f"{'-'*80}")
        
        try:
            results = evaluate_single_checkpoint(
                checkpoint_path=checkpoint_path,
                checkpoint_name=f"Session 8h @ {milestone}",
                args=args,
                env=env,
                PPO=PPO,
                VecNormalize=VecNormalize,
                np=np
            )
            all_results.append(results)
            print(f"    [OK] {milestone} evaluation complete")
        except Exception as e:
            print(f"    [FAIL] Error evaluating {milestone}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Cleanup Isaac Sim
    print("\n[6/6] Cleaning up...")
    env.close()
    simulation_app.close()
    print("    [OK] Cleanup complete")
    
    # Generate comparison report
    if all_results:
        generate_comparison_report(all_results, Path(args.output_dir))
        print("\n✅ Evaluation complete!")
    else:
        print("\n❌ No successful evaluations!")


def evaluate_single_checkpoint(
    checkpoint_path: Path,
    checkpoint_name: str,
    args: argparse.Namespace,
    env,
    PPO,
    VecNormalize,
    np
) -> Dict:
    """Evaluate a single checkpoint using an existing environment.
    
    Args:
        checkpoint_path: Path to checkpoint file
        checkpoint_name: Display name for checkpoint
        args: Command line arguments
        env: Pre-created gymnasium environment (reused across checkpoints)
        PPO: Pre-imported PPO class
        VecNormalize: Pre-imported VecNormalize class
        np: Pre-imported numpy module
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Wrap environment with VecNormalize if stats exist
    vec_normalize_path = checkpoint_path.parent.parent / "vec_normalize.pkl"
    if vec_normalize_path.exists():
        print(f"    Loading normalization stats: {vec_normalize_path.name}")
        env_wrapped = VecNormalize.load(str(vec_normalize_path), env)
        env_wrapped.training = False
        env_wrapped.norm_reward = False
    else:
        print(f"    No normalization stats found, using raw environment")
        env_wrapped = env
    
    # Load model
    print(f"    Loading model: {checkpoint_path.name}")
    model = PPO.load(str(checkpoint_path), env=env_wrapped)
    
    # Evaluation loop
    print(f"    Running {args.num_episodes} episodes...")
    episode_count = 0
    episode_rewards = []
    episode_lengths = []
    
    # Tracking errors
    position_errors = []
    orientation_errors = []
    
    # Workspace violations
    workspace_violations = []
    
    obs = env_wrapped.reset()
    if isinstance(obs, tuple):
        obs, _ = obs

    while episode_count < args.num_episodes:
        # Get action from policy
        action, _states = model.predict(obs, deterministic=args.deterministic)
        
        # Step environment (handle both SB3 VecEnv 4-return and raw Gymnasium 5-return)
        step_result = env_wrapped.step(action)
        if len(step_result) == 5:
            obs, rewards, terminateds, truncateds, infos = step_result
            dones = terminateds | truncateds
        else:
            obs, rewards, dones, infos = step_result
        
        # Log metrics from info dict
        for i, done in enumerate(dones):
            if done:
                if 'episode' in infos[i]:
                    episode_info = infos[i]['episode']
                    episode_rewards.append(episode_info['r'])
                    episode_lengths.append(episode_info['l'])
                    
                    # Extract tracking errors if available
                    if 'position_error' in infos[i]:
                        position_errors.append(infos[i]['position_error'])
                    if 'orientation_error' in infos[i]:
                        orientation_errors.append(infos[i]['orientation_error'])
                    if 'workspace_violation' in infos[i]:
                        workspace_violations.append(infos[i]['workspace_violation'])
                    
                    episode_count += 1
                    
                    if episode_count % 10 == 0:
                        print(f"      Progress: {episode_count}/{args.num_episodes}")
                    
                    if episode_count >= args.num_episodes:
                        break
    
    # Compute statistics
    results = {
        'checkpoint_name': checkpoint_name,
        'checkpoint_path': str(checkpoint_path),
        'num_episodes': episode_count,
        'episode_reward': {
            'mean': float(np.mean(episode_rewards)),
            'std': float(np.std(episode_rewards)),
            'min': float(np.min(episode_rewards)),
            'max': float(np.max(episode_rewards)),
        },
        'episode_length': {
            'mean': float(np.mean(episode_lengths)),
            'std': float(np.std(episode_lengths)),
        },
    }
    
    # Position tracking errors (convert to cm)
    if position_errors:
        pos_errors_m = np.array(position_errors)
        results['position_error'] = {
            'mean_cm': float(np.mean(pos_errors_m) * 100),
            'median_cm': float(np.median(pos_errors_m) * 100),
            'std_cm': float(np.std(pos_errors_m) * 100),
            'p95_cm': float(np.percentile(pos_errors_m, 95) * 100),
        }
    
    # Orientation tracking errors (convert to degrees)
    if orientation_errors:
        ori_errors_rad = np.array(orientation_errors)
        results['orientation_error'] = {
            'mean_deg': float(np.rad2deg(np.mean(ori_errors_rad))),
            'median_deg': float(np.rad2deg(np.median(ori_errors_rad))),
            'std_deg': float(np.rad2deg(np.std(ori_errors_rad))),
            'p95_deg': float(np.rad2deg(np.percentile(ori_errors_rad, 95))),
        }
    
    # Workspace violations
    if workspace_violations:
        results['workspace_violations'] = {
            'rate_percent': float(np.mean(workspace_violations) * 100),
            'count': int(np.sum(workspace_violations)),
        }
    
    # Reset environment for next checkpoint (don't close it!)
    print(f"    Resetting environment for next checkpoint...")
    env_wrapped.reset()
    
    return results


if __name__ == "__main__":
    main()
