"""Quick checkpoint evaluation for Session 8c-v2 monitoring.

Fast evaluation on 10-20 episodes to check:
- Base-target distance distribution (critical for quadratic penalty assessment)
- Position/orientation error trends
- Reachability metrics

Usage:
    cd I:\isaaclab
    .\isaaclab.bat -p ..\cinebotRL\scripts\monitoring\quick_checkpoint_eval.py `
        --checkpoint ..\cinebotRL\logs\sb3\MobileMMTrackEE-v0\<timestamp>\checkpoints\ppo_mobile_mm_20000000_steps.zip `
        --num_episodes 10
"""

import argparse
from pathlib import Path
import sys

# Add project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    parser = argparse.ArgumentParser(description='Quick checkpoint evaluation for Session 8c-v2')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint .zip file')
    parser.add_argument('--num_episodes', type=int, default=10, help='Number of episodes (10-20 for quick check)')
    parser.add_argument('--num_envs', type=int, default=10, help='Number of parallel envs')
    parser.add_argument('--task', type=str, default='MobileMMTrackEE-v0', help='Task name')
    parser.add_argument('--headless', action='store_true', help='Run headless')
    
    args = parser.parse_args()
    
    # Validate checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ ERROR: Checkpoint not found: {checkpoint_path}")
        return
    
    print("="*80)
    print(f"🚀 QUICK EVALUATION: {checkpoint_path.name}")
    print(f"   Episodes: {args.num_episodes}")
    print(f"   Envs: {args.num_envs}")
    print("="*80)
    
    # Build command to delegate to main evaluation script
    eval_script = PROJECT_ROOT / 'scripts' / 'reinforcement_learning' / 'sb3' / 'evaluate_quantitative.py'
    trajectory_dir = PROJECT_ROOT / 'trajectoryToLearn' / 'world_json'
    
    import subprocess
    
    cmd = [
        sys.executable,  # Use current Python
        str(eval_script),
        '--checkpoint', str(checkpoint_path),
        '--task', args.task,
        '--num_episodes', str(args.num_episodes),
        '--num_envs', str(args.num_envs),
        '--trajectory_type', 'multi_recorded',
        '--trajectory_dir', str(trajectory_dir),
        '--use_all_trajectories'
    ]
    
    if args.headless:
        cmd.append('--headless')
    
    print(f"\n📝 Running evaluation...\n")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n✅ Evaluation complete!")
        print("\n💡 Check evaluation_results/ for detailed metrics")
        print("   Key metrics to review:")
        print("   - base_target_distance_mean: Should be 0.4-0.7m")
        print("   - position_error_mean: Target <1.0m")
        print("   - orientation_error_mean: Target <35°")
        print("   - reachability_reward: Should be positive")
    else:
        print("\n❌ Evaluation failed!")
    
    return result.returncode


if __name__ == '__main__':
    sys.exit(main())
