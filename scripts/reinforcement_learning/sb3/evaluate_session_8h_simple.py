"""Simple wrapper to evaluate multiple Session 8h checkpoints using evaluate_quantitative.py.

This script runs evaluate_quantitative.py multiple times (once per checkpoint) 
to avoid Isaac Sim re-initialization issues.

Usage:
    python scripts/reinforcement_learning/sb3/evaluate_session_8h_simple.py \
        --session_8h_dir logs/sb3/mobilemmtrackee_v0/20251103_235918 \
        --checkpoints 20M 40M 100M \
        --num_episodes 50
"""

import argparse
import subprocess
import sys
from pathlib import Path
import json
from datetime import datetime
import glob

# Add project root to Python path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Session 8h checkpoints by running evaluate_quantitative.py multiple times"
    )
    
    parser.add_argument(
        "--session_8h_dir",
        type=str,
        required=True,
        help="Path to Session 8h log directory",
    )
    
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=["20M", "40M", "60M", "80M", "100M"],
        help="Which checkpoints to evaluate (e.g., 20M 40M 100M)",
    )
    
    parser.add_argument(
        "--num_envs",
        type=int,
        default=64,
        help="Number of parallel environments",
    )
    
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=200,
        help="Number of episodes per checkpoint",
    )
    
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run in headless mode",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results/session_8h_comparison",
        help="Directory to save results",
    )
    
    return parser.parse_args()


def find_checkpoint(base_dir: Path, milestone: str):
    """Find checkpoint file for a given milestone."""
    checkpoint_dir = base_dir / "checkpoints"
    
    if milestone.lower() == "final":
        final_model = base_dir / "final_model.zip"
        if final_model.exists():
            return final_model
        return None
    
    # Convert milestone to steps
    if milestone.endswith('M'):
        target_steps = int(float(milestone[:-1]) * 1_000_000)
    elif milestone.endswith('K'):
        target_steps = int(float(milestone[:-1]) * 1_000)
    else:
        target_steps = int(milestone)
    
    # Find closest checkpoint
    pattern = str(checkpoint_dir / f"ppo_mobile_mm_*_steps.zip")
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        return None
    
    closest_checkpoint = None
    min_diff = float('inf')
    
    for ckpt in checkpoints:
        try:
            steps_str = Path(ckpt).stem.split('_')[-2]
            steps = int(steps_str)
            diff = abs(steps - target_steps)
            
            if diff < min_diff:
                min_diff = diff
                closest_checkpoint = ckpt
        except (ValueError, IndexError):
            continue
    
    return Path(closest_checkpoint) if closest_checkpoint else None


def run_evaluation(checkpoint_path: Path, checkpoint_name: str, args: argparse.Namespace):
    """Run evaluate_quantitative.py for a single checkpoint."""
    print(f"\n{'='*80}")
    print(f"Evaluating: {checkpoint_name}")
    print(f"Checkpoint: {checkpoint_path.name}")
    print(f"{'='*80}\n")
    
    # Build command
    isaac_lab_bat = Path("I:/isaaclab/isaaclab.bat")
    eval_script = SCRIPT_DIR / "evaluate_quantitative.py"
    
    output_subdir = Path(args.output_dir) / checkpoint_name.replace(" ", "_").replace("@", "at")
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        str(isaac_lab_bat),
        "-p",
        str(eval_script),
        "--checkpoint", str(checkpoint_path),
        "--num_envs", str(args.num_envs),
        "--num_episodes", str(args.num_episodes),
        "--output_dir", str(output_subdir),
    ]
    
    if args.headless:
        cmd.append("--headless")
    
    print(f"Running command:")
    print(f"  {' '.join(cmd)}\n")
    
    # Run evaluation
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n✅ {checkpoint_name} evaluation complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {checkpoint_name} evaluation failed: {e}")
        return False


def summarize_results(session_8h_dir: Path, checkpoints_evaluated: list, output_dir: Path):
    """Create a summary of all evaluations."""
    print(f"\n{'='*80}")
    print("GENERATING COMPARISON SUMMARY")
    print(f"{'='*80}\n")
    
    summary = {
        'session_8h_dir': str(session_8h_dir),
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'checkpoints': []
    }
    
    print(f"{'Checkpoint':<20} | {'Pos Error (cm)':<18} | {'Ori Error (°)':<18}")
    print(f"{'-'*20}-+-{'-'*18}-+-{'-'*18}")
    
    for name, result_dir in checkpoints_evaluated:
        # Find the summary JSON file
        json_files = list(result_dir.glob("eval_summary_*.json"))
        if not json_files:
            print(f"{name:<20} | {'N/A':<18} | {'N/A':<18}")
            continue
        
        # Load latest summary
        latest_summary = max(json_files, key=lambda p: p.stat().st_mtime)
        with open(latest_summary, 'r') as f:
            data = json.load(f)
        
        pos_err = data.get('position_error', {}).get('mean_cm', 'N/A')
        ori_err = data.get('orientation_error', {}).get('mean_deg', 'N/A')
        
        if isinstance(pos_err, (int, float)):
            pos_str = f"{pos_err:.1f}"
        else:
            pos_str = str(pos_err)
        
        if isinstance(ori_err, (int, float)):
            ori_str = f"{ori_err:.1f}"
        else:
            ori_str = str(ori_err)
        
        print(f"{name:<20} | {pos_str:<18} | {ori_str:<18}")
        
        summary['checkpoints'].append({
            'name': name,
            'position_error_cm': pos_err,
            'orientation_error_deg': ori_err,
            'result_dir': str(result_dir)
        })
    
    # Save combined summary
    summary_file = output_dir / f"session_8h_comparison_{summary['timestamp']}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Summary saved to: {summary_file}")
    print(f"{'='*80}\n")


def main():
    args = parse_args()
    
    session_8h_dir = Path(args.session_8h_dir)
    if not session_8h_dir.exists():
        print(f"❌ Session 8h directory not found: {session_8h_dir}")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"SESSION 8H CHECKPOINT EVALUATION")
    print(f"{'='*80}")
    print(f"Session 8h directory: {session_8h_dir}")
    print(f"Checkpoints to evaluate: {', '.join(args.checkpoints)}")
    print(f"Episodes per checkpoint: {args.num_episodes}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*80}\n")
    
    # Find all checkpoints
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
        sys.exit(1)
    
    # Evaluate each checkpoint (separate Isaac Sim sessions)
    checkpoints_evaluated = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for milestone, checkpoint_path in checkpoints_to_eval:
        checkpoint_name = f"Session_8h_at_{milestone}"
        success = run_evaluation(checkpoint_path, checkpoint_name, args)
        
        if success:
            result_dir = output_dir / checkpoint_name
            checkpoints_evaluated.append((checkpoint_name, result_dir))
    
    # Generate comparison summary
    if checkpoints_evaluated:
        summarize_results(session_8h_dir, checkpoints_evaluated, output_dir)
        print("\n✅ All evaluations complete!")
    else:
        print("\n❌ No successful evaluations!")
        sys.exit(1)


if __name__ == "__main__":
    main()
