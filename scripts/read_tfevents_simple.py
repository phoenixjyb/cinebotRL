"""Quick TensorBoard event file reader."""

import struct
import sys
from pathlib import Path

def read_tfevents_simple(filepath):
    """Simple reader for TensorBoard event files."""
    print(f"\nReading: {filepath}")
    print("=" * 80)
    
    try:
        with open(filepath, 'rb') as f:
            data = f.read()
            
        print(f"File size: {len(data):,} bytes")
        
        # Try to find scalar summaries
        # TensorBoard events use protocol buffers, look for text patterns
        text = data.decode('utf-8', errors='ignore')
        
        # Look for common metric names
        metrics = [
            'rollout/ep_rew_mean',
            'rollout/ep_len_mean',
            'base_mobilization',
            'jerk_penalty',
            'base_vel',
            'position_tracking',
            'learning_rate',
            'approx_kl',
        ]
        
        print("\nSearching for metrics in file...")
        for metric in metrics:
            if metric in text:
                print(f"  ✅ Found: {metric}")
            else:
                print(f"  ❌ Not found: {metric}")
                
        # Count occurrences of "value" which indicates scalar values
        value_count = text.count('value')
        print(f"\nTotal 'value' occurrences: {value_count}")
        
    except Exception as e:
        print(f"Error reading file: {e}")

def main():
    event_file = Path("c:/Users/yanbo/wSpace/cinebotRL/logs/sb3/mobilemmtrackee_v0/20251022_230622/PPO_1/events.out.tfevents.1761145609.JiaFamily.52376.0")
    
    if not event_file.exists():
        print(f"File not found: {event_file}")
        return
    
    read_tfevents_simple(event_file)

if __name__ == "__main__":
    main()
