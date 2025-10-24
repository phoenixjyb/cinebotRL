"""Monitor reachability map building progress."""
import time
from pathlib import Path

matlab_dir = Path(__file__).parent.parent / "matlab"
map_file = matlab_dir / "reach_map_mobile_mm_arm_only.mat"

print("🔍 Monitoring reachability map building progress...")
print(f"   Expected output: {map_file}")
print(f"   Check every 30 seconds for file creation/growth")
print("-" * 80)

last_size = 0
start_time = time.time()

while True:
    if map_file.exists():
        size_mb = map_file.stat().st_size / (1024 * 1024)
        elapsed = time.time() - start_time
        
        if size_mb != last_size:
            print(f"⏱️  [{elapsed/60:5.1f} min] Map file: {size_mb:.2f} MB")
            last_size = size_mb
            
            # Estimate progress (expected final size: 50-100 MB)
            if size_mb > 1:
                est_progress = min(100, (size_mb / 75) * 100)  # Assume 75 MB target
                print(f"   Estimated progress: {est_progress:.1f}%")
        
        if size_mb > 40:  # Likely complete
            print(f"\n✅ Map file appears complete! ({size_mb:.2f} MB)")
            print(f"   Build time: {elapsed/60:.1f} minutes")
            print(f"\n📊 Next step: Visualize in MATLAB")
            print(f"   cd {matlab_dir}")
            print(f"   matlab")
            print(f"   >> visualize_reachability('reach_map_mobile_mm_arm_only.mat', 'mode', 5)")
            break
    else:
        elapsed = time.time() - start_time
        if elapsed % 60 < 30:  # Print every ~minute
            print(f"⏱️  [{elapsed/60:5.1f} min] Waiting for map file to be created...")
    
    time.sleep(30)
    
    # Safety timeout (2 hours)
    if time.time() - start_time > 7200:
        print("\n⚠️  Timeout after 2 hours. Check MATLAB terminal for errors.")
        break
