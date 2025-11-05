"""Quick script to check workspace distance from TensorBoard logs."""
from tensorboard.backend.event_processing import event_accumulator

log_file = r'C:\Users\yanbo\wSpace\cinebotRL\logs\sb3\mobilemmtrackee_v0\20251031_224729\PPO_1\events.out.tfevents.1761922124.JiaFamily.14236.0'

ea = event_accumulator.EventAccumulator(log_file)
ea.Reload()

ws_mean = ea.Scalars('monitoring/workspace_distance_mean')
ws_max = ea.Scalars('monitoring/workspace_distance_max')
soft_exceed = ea.Scalars('monitoring/workspace_soft_exceed_pct')
hard_exceed = ea.Scalars('monitoring/workspace_hard_exceed_pct')

print(f"\n{'='*60}")
print(f"Session 8e - Workspace Distance During Training")
print(f"{'='*60}")
print(f"\nLatest 10 checkpoints:")
print(f"{'Step':>12} {'Mean':>8} {'Max':>8} {'Soft%':>8} {'Hard%':>8}")
print(f"{'-'*60}")

for i in range(max(-10, -len(ws_mean)), 0):
    step = ws_mean[i].step
    mean_val = ws_mean[i].value
    max_val = ws_max[i].value if i < len(ws_max) else 0
    soft_pct = soft_exceed[i].value if i < len(soft_exceed) else 0
    hard_pct = hard_exceed[i].value if i < len(hard_exceed) else 0
    
    print(f"{step:12,} {mean_val:8.4f} {max_val:8.4f} {soft_pct:7.1f}% {hard_pct:7.1f}%")

print(f"\n{'='*60}")
print(f"Target: Mean ~0.5m, Soft margin at 0.7m, Hard margin at 0.9m")
print(f"{'='*60}\n")
