"""Simple trajectory file verification (no dependencies required).

This script checks:
1. Trajectory directory exists
2. JSON files can be found and parsed
3. File structure is correct
4. Expected number of trajectories present
"""

import json
from pathlib import Path


def main():
    print("\n" + "="*80)
    print("TRAJECTORY FILE VERIFICATION")
    print("="*80)
    
    # Find trajectory directory
    project_root = Path(__file__).parent.parent
    traj_dir = project_root / "trajectoryToLearn" / "world_json"
    
    print(f"\n[1/5] Checking directory...")
    print(f"  Path: {traj_dir}")
    
    if not traj_dir.exists():
        print(f"  ❌ FAIL: Directory does not exist!")
        return False
    
    print(f"  ✅ Directory exists")
    
    # Find JSON files
    print(f"\n[2/5] Discovering JSON files...")
    all_json = list(traj_dir.glob("**/*.json"))
    print(f"  Found {len(all_json)} total JSON files")
    
    # Filter out __MACOSX
    json_files = [f for f in all_json if "__MACOSX" not in str(f)]
    macosx_count = len(all_json) - len(json_files)
    
    print(f"  Filtered {macosx_count} __MACOSX files")
    print(f"  ✅ {len(json_files)} valid trajectory files")
    
    if len(json_files) < 1000:
        print(f"  ⚠️  Expected ~1038 files, found {len(json_files)}")
    
    # Test parsing
    print(f"\n[3/5] Testing JSON parsing...")
    test_files = json_files[:20]  # Test first 20
    
    valid_count = 0
    total_waypoints = 0
    
    for i, filepath in enumerate(test_files):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Validate structure
            if "poses" not in data:
                print(f"  ❌ File {i}: Missing 'poses' key")
                continue
            
            poses = data["poses"]
            if len(poses) == 0:
                print(f"  ❌ File {i}: Empty poses")
                continue
            
            # Check first pose
            pose = poses[0]
            if "position" not in pose or "orientation" not in pose:
                print(f"  ❌ File {i}: Invalid pose structure")
                continue
            
            # Check position is array of 3 numbers
            pos = pose["position"]
            if not (isinstance(pos, list) and len(pos) == 3):
                print(f"  ❌ File {i}: Invalid position format (expected list of 3)")
                continue
            
            # Check orientation is array of 4 numbers
            ori = pose["orientation"]
            if not (isinstance(ori, list) and len(ori) == 4):
                print(f"  ❌ File {i}: Invalid orientation format (expected list of 4)")
                continue
            
            valid_count += 1
            total_waypoints += len(poses)
            
            if i < 3:  # Show first 3
                print(f"  ✓ {filepath.name}: {len(poses)} waypoints")
        
        except json.JSONDecodeError as e:
            print(f"  ❌ File {i}: JSON error - {e}")
        except Exception as e:
            print(f"  ❌ File {i}: Error - {e}")
    
    print(f"  ✅ {valid_count}/{len(test_files)} files valid")
    print(f"  Average waypoints: {total_waypoints // valid_count if valid_count > 0 else 0}")
    
    # Check chassis indices file
    print(f"\n[4/5] Checking chassis indices file...")
    chassis_file = project_root / "chassis_required_indices.txt"
    
    if chassis_file.exists():
        import re
        with open(chassis_file, 'r') as f:
            content = f.read()
        
        match = re.search(r'CHASSIS_REQUIRED_INDICES = \[(.*?)\]', content, re.DOTALL)
        if match:
            indices_str = match.group(1)
            chassis_indices = [int(x.strip()) for x in indices_str.replace('\n', ' ').split(',') if x.strip()]
            print(f"  ✅ Found {len(chassis_indices)} chassis-requiring indices")
            
            # Show a few
            print(f"  First 10: {chassis_indices[:10]}")
        else:
            print(f"  ⚠️  File exists but couldn't parse indices")
    else:
        print(f"  ℹ️  chassis_required_indices.txt not found (run analyze_trajectories.py)")
    
    # Summary
    print(f"\n[5/5] Summary")
    print(f"  Total trajectories: {len(json_files)}")
    print(f"  Validation rate: {valid_count}/{len(test_files)} ({100*valid_count//len(test_files)}%)")
    
    if len(json_files) >= 1000 and valid_count == len(test_files):
        print(f"\n✅ ALL CHECKS PASSED!")
        print(f"\n  Your trajectory files are ready for training!")
        print(f"  Expected to load ~{len(json_files)} trajectories")
        return True
    elif valid_count == len(test_files):
        print(f"\n⚠️  Files are valid but count is low ({len(json_files)} < 1000)")
        return True
    else:
        print(f"\n❌ Some files failed validation")
        return False


if __name__ == "__main__":
    import sys
    success = main()
    
    print("\n" + "="*80)
    if success:
        print("✅ READY FOR TRAINING")
        print("\nRun training with:")
        print("  & \"I:\\isaaclab\\isaaclab.bat\" -p scripts/reinforcement_learning/sb3/train.py `")
        print("      --task MobileMMTrackEE-v0 `")
        print("      --trajectory_type multi_recorded `")
        print("      --use_all_trajectories `")
        print("      --headless")
    else:
        print("❌ ISSUES FOUND - Fix before training")
    print("="*80)
    
    sys.exit(0 if success else 1)
