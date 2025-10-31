"""Monitor Session 8c training progress in real-time.

Reads progress.csv and generates health metrics without interrupting training.

Usage:
    python scripts/monitoring/check_training_progress.py --log_dir logs/sb3/MobileMMTrackEE-v0/<timestamp>
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime


def analyze_training_health(progress_csv: Path) -> dict:
    """Analyze training metrics and return health assessment."""
    
    df = pd.read_csv(progress_csv)
    
    # Get latest metrics
    latest = df.iloc[-1]
    iteration = int(latest['time/iterations'])
    timesteps = int(latest['time/total_timesteps'])
    fps = float(latest['time/fps'])
    
    # Training metrics
    ev = float(latest['train/explained_variance'])
    approx_kl = float(latest['train/approx_kl'])
    clip_frac = float(latest['train/clip_fraction'])
    entropy = float(latest['train/entropy_loss'])
    std = float(latest['train/std'])
    value_loss = float(latest['train/value_loss'])
    
    # Calculate progress
    total_iterations = 96  # 200M / (16384 * 128)
    progress_pct = (iteration / total_iterations) * 100
    
    # Estimate completion time
    elapsed = float(latest['time/time_elapsed'])
    estimated_total = elapsed * (total_iterations / iteration) if iteration > 0 else 0
    remaining = estimated_total - elapsed
    
    # Health checks
    health = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'iteration': iteration,
        'total_iterations': total_iterations,
        'progress_pct': progress_pct,
        'timesteps': timesteps,
        'timesteps_target': 200_000_000,
        'fps': fps,
        'elapsed_hours': elapsed / 3600,
        'remaining_hours': remaining / 3600,
        'estimated_total_hours': estimated_total / 3600,
        
        # Training health
        'explained_variance': ev,
        'ev_status': 'GOOD' if ev >= 0.75 else 'WARNING' if ev >= 0.65 else 'CRITICAL',
        'approx_kl': approx_kl,
        'kl_status': 'GOOD' if approx_kl < 0.03 else 'WARNING',
        'clip_fraction': clip_frac,
        'clip_status': 'GOOD' if clip_frac < 0.2 else 'WARNING',
        'entropy_loss': entropy,
        'std': std,
        'value_loss': value_loss,
        
        # Trends (last 5 iterations)
        'ev_trend': None,
        'fps_trend': None,
    }

    if 'monitoring/unreachable_zone_pct' in df.columns:
        health['base_unreachable_pct'] = float(latest['monitoring/unreachable_zone_pct'])
    if 'monitoring/workspace_hard_exceed_pct' in df.columns:
        health['workspace_hard_exceed_pct'] = float(latest['monitoring/workspace_hard_exceed_pct'])
    
    # Calculate trends if enough data
    if len(df) >= 5:
        recent = df.tail(5)
        ev_trend = np.polyfit(range(5), recent['train/explained_variance'].values, 1)[0]
        fps_trend = np.polyfit(range(5), recent['time/fps'].values, 1)[0]
        
        health['ev_trend'] = 'IMPROVING' if ev_trend > 0.01 else 'STABLE' if ev_trend > -0.01 else 'DEGRADING'
        health['fps_trend'] = 'STABLE' if abs(fps_trend) < 500 else 'DECLINING' if fps_trend < 0 else 'IMPROVING'
    
    return health


def print_health_report(health: dict):
    """Print formatted health report."""
    
    print("\n" + "="*80)
    print(f"  SESSION 8C TRAINING PROGRESS - {health['timestamp']}")
    print("="*80)
    
    # Progress
    print("\nPROGRESS")
    print(f"  Iteration:  {health['iteration']}/{health['total_iterations']} ({health['progress_pct']:.1f}%)")
    print(f"  Timesteps:  {health['timesteps']:,} / {health['timesteps_target']:,}")
    print(f"  FPS:        {health['fps']:.0f} ({health.get('fps_trend', 'N/A')})")
    
    # Time estimates
    print("\nTIMING")
    print(f"  Elapsed:    {health['elapsed_hours']:.2f} hours")
    print(f"  Remaining:  {health['remaining_hours']:.2f} hours (est.)")
    print(f"  Total:      {health['estimated_total_hours']:.2f} hours (est.)")
    
    # Training health
    print("\nTRAINING HEALTH:")
    ev_icon = "[OK ]" if health['ev_status'] == 'GOOD' else "[WARN]" if health['ev_status'] == 'WARNING' else "[CRIT]"
    kl_icon = "[OK ]" if health['kl_status'] == 'GOOD' else "[WARN]"
    
    print(f"  {ev_icon} Explained Variance: {health['explained_variance']:.3f} [{health['ev_status']}]")
    if health['ev_trend']:
        print(f"     Trend: {health['ev_trend']}")
    
    print(f"  {kl_icon} Approx KL:          {health['approx_kl']:.4f} (target: <0.03)")
    print(f"  Clip Fraction:      {health['clip_fraction']:.3f} (target: <0.2)")
    print(f"  Entropy Loss:       {health['entropy_loss']:.2f}")
    print(f"  Std:                {health['std']:.3f}")
    print(f"  Value Loss:         {health['value_loss']:.3f}")
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    
    if health['ev_status'] == 'CRITICAL':
        print("  EV below 0.65. Consider switching to curriculum training.")
    elif health['ev_status'] == 'WARNING':
        print("  EV between 0.65 and 0.75. Monitor closely.")
    else:
        print("  EV healthy (>0.75). Continue training.")
    
    if health['approx_kl'] > 0.03:
        print("  KL above target. Policy taking large steps.")
    if 'base_unreachable_pct' in health:
        print(f"  Base unreachable zone: {health['base_unreachable_pct']:.1f}% of samples")
    if 'workspace_hard_exceed_pct' in health:
        print(f"  Workspace hard-margin exceedances: {health['workspace_hard_exceed_pct']:.1f}%")
    
    if health['iteration'] >= 10 and health['iteration'] % 10 == 0:
        checkpoint_path = f"checkpoints/ppo_mobile_mm_{health['timesteps']}_steps.zip"
        print("\nCHECKPOINT AVAILABLE:")
        print(f"  Path: {checkpoint_path}")
        print("  Run:  python scripts/reinforcement_learning/sb3/evaluate_quantitative.py \\ ")
        print("        --checkpoint <path> --num_episodes 10 --headless")
    
    if health['iteration'] == 58:
        print("\nENTROPY DECAY MILESTONE: 120M steps reached (iteration 58).")
        print("  Watch for [EntropyDecay] messages in training logs.")
    
    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Monitor Session 8c training progress')
    parser.add_argument('--log_dir', type=str, required=True,
                       help='Path to training log directory containing progress.csv')
    parser.add_argument('--watch', action='store_true',
                       help='Continuously monitor (refresh every 30s)')
    
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    progress_csv = log_dir / 'progress.csv'
    
    if not progress_csv.exists():
        print(f"ERROR: progress.csv not found at {progress_csv}")
        return
    
    if args.watch:
        import time
        print("Watching training progress (Ctrl+C to stop)...")
        try:
            while True:
                health = analyze_training_health(progress_csv)
                print_health_report(health)
                time.sleep(30)
        except KeyboardInterrupt:
            print("\nStopped watching progress monitor.")
    else:
        health = analyze_training_health(progress_csv)
        print_health_report(health)


if __name__ == '__main__':
    main()
