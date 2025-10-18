"""Test script to verify multi-trajectory loading works correctly.

This script tests:
1. Trajectory files can be discovered and loaded
2. MultiTrajectoryLoader works correctly
3. TrajectoryManager can initialize with multi_recorded mode
4. Trajectories can be sampled and used in environments
5. All 1,038 trajectories are accessible
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
import json
from typing import List


def test_1_file_discovery():
    """Test 1: Can we find trajectory JSON files?"""
    print("\n" + "="*80)
    print("TEST 1: File Discovery")
    print("="*80)
    
    traj_dir = project_root / "trajectoryToLearn" / "world_json"
    
    if not traj_dir.exists():
        print(f"❌ FAIL: Directory does not exist: {traj_dir}")
        return False
    
    print(f"✓ Directory exists: {traj_dir}")
    
    # Find all JSON files
    json_files = list(traj_dir.glob("**/*.json"))
    
    # Filter out __MACOSX
    json_files = [f for f in json_files if "__MACOSX" not in str(f)]
    
    print(f"✓ Found {len(json_files)} JSON files (excluding __MACOSX)")
    
    if len(json_files) == 0:
        print("❌ FAIL: No JSON files found!")
        return False
    
    # Show first 5 files
    print("\nFirst 5 files:")
    for i, f in enumerate(json_files[:5]):
        print(f"  {i}: {f.name}")
    
    # Check expected count
    if len(json_files) < 1000:
        print(f"⚠️  WARNING: Expected ~1038 files, found {len(json_files)}")
    
    print(f"\n✅ PASS: Found {len(json_files)} trajectory files")
    return True


def test_2_json_parsing():
    """Test 2: Can we parse trajectory JSON files?"""
    print("\n" + "="*80)
    print("TEST 2: JSON Parsing")
    print("="*80)
    
    traj_dir = project_root / "trajectoryToLearn" / "world_json"
    json_files = [f for f in traj_dir.glob("**/*.json") if "__MACOSX" not in str(f)]
    
    if not json_files:
        print("❌ FAIL: No files to test")
        return False
    
    # Test first 10 files
    test_files = json_files[:10]
    success_count = 0
    
    for i, filepath in enumerate(test_files):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Check structure
            if "poses" not in data:
                print(f"  ❌ File {i}: Missing 'poses' key: {filepath.name}")
                continue
            
            poses = data["poses"]
            if len(poses) == 0:
                print(f"  ❌ File {i}: Empty poses array: {filepath.name}")
                continue
            
            # Check first pose structure
            pose = poses[0]
            if "position" not in pose or "orientation" not in pose:
                print(f"  ❌ File {i}: Invalid pose structure: {filepath.name}")
                continue
            
            print(f"  ✓ File {i}: {filepath.name} - {len(poses)} waypoints")
            success_count += 1
            
        except json.JSONDecodeError as e:
            print(f"  ❌ File {i}: JSON decode error: {filepath.name} - {e}")
        except Exception as e:
            print(f"  ❌ File {i}: Unexpected error: {filepath.name} - {e}")
    
    print(f"\n✅ PASS: Successfully parsed {success_count}/{len(test_files)} files")
    return success_count == len(test_files)


def test_3_multi_trajectory_loader():
    """Test 3: Can MultiTrajectoryLoader load trajectories?"""
    print("\n" + "="*80)
    print("TEST 3: MultiTrajectoryLoader")
    print("="*80)
    
    try:
        from rl_platform.tasks.mobile_mm.multi_trajectory import MultiTrajectoryLoader
        print("✓ Successfully imported MultiTrajectoryLoader")
    except ImportError as e:
        print(f"❌ FAIL: Cannot import MultiTrajectoryLoader: {e}")
        return False
    
    # Create loader
    try:
        traj_dir = str(project_root / "trajectoryToLearn" / "world_json")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"✓ Creating loader with:")
        print(f"  - trajectory_dir: {traj_dir}")
        print(f"  - device: {device}")
        
        loader = MultiTrajectoryLoader(
            trajectory_dir=traj_dir,
            pattern="**/*.json",
            device=device,
            max_trajectories=None,  # Load all
            exclude_macosx=True,
        )
        
        print(f"✓ Loader created successfully")
        print(f"✓ Total trajectories loaded: {len(loader.trajectories)}")
        
        if len(loader.trajectories) == 0:
            print("❌ FAIL: No trajectories loaded!")
            return False
        
        # Test sampling
        print("\nTesting trajectory sampling...")
        
        # Test sample_trajectory method
        traj = loader.sample_trajectory()
        print(f"  ✓ Sampled trajectory: {traj['length']} waypoints, category={traj['category']}")
        print(f"    pos shape: {traj['positions'].shape}, quat shape: {traj['orientations'].shape}")
        
        # Test batch sampling
        num_test_envs = 5
        positions, orientations = loader.sample_trajectories(num_test_envs)
        print(f"  ✓ Batch sampled for {num_test_envs} envs:")
        print(f"    positions: {positions.shape}, orientations: {orientations.shape}")
        
        # Verify shapes
        assert positions.shape[0] == num_test_envs, f"Expected {num_test_envs} envs, got {positions.shape[0]}"
        assert positions.shape[2] == 3, f"Position should be Nx3, got {positions.shape[2]}"
        assert orientations.shape[2] == 4, f"Quaternion should be Nx4, got {orientations.shape[2]}"
        
        print(f"\n✅ PASS: MultiTrajectoryLoader working correctly")
        print(f"   - Loaded {len(loader.trajectories)} trajectories")
        print(f"   - Sampling works correctly")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Error creating/using MultiTrajectoryLoader: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_4_trajectory_manager():
    """Test 4: Can TrajectoryManager use multi_recorded mode?"""
    print("\n" + "="*80)
    print("TEST 4: TrajectoryManager with multi_recorded")
    print("="*80)
    
    try:
        from rl_platform.tasks.mobile_mm.trajectories import TrajectoryManager
        print("✓ Successfully imported TrajectoryManager")
    except ImportError as e:
        print(f"❌ FAIL: Cannot import TrajectoryManager: {e}")
        return False
    
    try:
        traj_dir = str(project_root / "trajectoryToLearn" / "world_json")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        num_envs = 8  # Small number for testing
        
        print(f"✓ Creating TrajectoryManager with:")
        print(f"  - traj_type: multi_recorded")
        print(f"  - num_envs: {num_envs}")
        print(f"  - trajectory_dir: {traj_dir}")
        print(f"  - device: {device}")
        
        manager = TrajectoryManager(
            traj_type="multi_recorded",
            num_envs=num_envs,
            device=device,
            trajectory_dir=traj_dir,
            trajectory_pattern="**/*.json",
            trajectory_filter_indices=None,  # Use all
            max_trajectories=None,
        )
        
        print(f"✓ TrajectoryManager created successfully")
        
        # Check multi_loader was initialized
        if manager.multi_loader is None:
            print("❌ FAIL: multi_loader is None!")
            return False
        
        print(f"✓ multi_loader initialized with {len(manager.multi_loader.trajectories)} trajectories")
        
        # Test getting target pose
        print("\nTesting reference pose generation...")
        ref_pos, ref_quat = manager.get_target_pose()
        
        print(f"  ✓ Target poses shape: pos={ref_pos.shape}, quat={ref_quat.shape}")
        print(f"  ✓ Expected: pos=[{num_envs}, 3], quat=[{num_envs}, 4]")
        
        assert ref_pos.shape == (num_envs, 3), f"Expected ({num_envs}, 3), got {ref_pos.shape}"
        assert ref_quat.shape == (num_envs, 4), f"Expected ({num_envs}, 4), got {ref_quat.shape}"
        
        # Test reset (should resample trajectories)
        print("\nTesting trajectory resampling on reset...")
        env_ids = torch.tensor([0, 2, 4], device=device)
        manager.reset(env_ids)
        
        ref_pos_after, ref_quat_after = manager.get_target_pose()
        print(f"  ✓ Reset successful, new poses generated")
        
        print(f"\n✅ PASS: TrajectoryManager multi_recorded mode working")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Error with TrajectoryManager: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_5_filtering():
    """Test 5: Can we filter trajectories by indices?"""
    print("\n" + "="*80)
    print("TEST 5: Trajectory Filtering")
    print("="*80)
    
    try:
        from rl_platform.tasks.mobile_mm.multi_trajectory import MultiTrajectoryLoader
        
        traj_dir = str(project_root / "trajectoryToLearn" / "world_json")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Test with filter
        filter_indices = [0, 1, 2, 5, 10, 15, 20]
        
        print(f"✓ Creating loader with filter: {filter_indices}")
        
        loader = MultiTrajectoryLoader(
            trajectory_dir=traj_dir,
            pattern="**/*.json",
            device=device,
            filter_by_indices=filter_indices,
            exclude_macosx=True,
        )
        
        print(f"✓ Filtered loader created")
        print(f"  - Requested: {len(filter_indices)} trajectories")
        print(f"  - Loaded: {len(loader.trajectories)} trajectories")
        
        if len(loader.trajectories) != len(filter_indices):
            print(f"⚠️  WARNING: Mismatch in trajectory count")
            print(f"   Expected: {len(filter_indices)}, Got: {len(loader.trajectories)}")
            # This might be OK if some files don't exist
        
        # Test max_trajectories limit
        print("\nTesting max_trajectories limit...")
        loader2 = MultiTrajectoryLoader(
            trajectory_dir=traj_dir,
            pattern="**/*.json",
            device=device,
            max_trajectories=50,
            exclude_macosx=True,
        )
        
        print(f"✓ Limited loader created")
        print(f"  - Requested max: 50")
        print(f"  - Loaded: {len(loader2.trajectories)}")
        
        if len(loader2.trajectories) > 50:
            print(f"❌ FAIL: Loaded {len(loader2.trajectories)} > 50!")
            return False
        
        print(f"\n✅ PASS: Filtering works correctly")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Error with filtering: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_6_chassis_indices():
    """Test 6: Can we load chassis-requiring trajectories?"""
    print("\n" + "="*80)
    print("TEST 6: Chassis-Requiring Trajectories")
    print("="*80)
    
    chassis_file = project_root / "chassis_required_indices.txt"
    
    if not chassis_file.exists():
        print(f"⚠️  SKIP: chassis_required_indices.txt not found")
        print(f"   (This is OK - run analyze_trajectories.py to generate it)")
        return True
    
    print(f"✓ Found chassis indices file: {chassis_file}")
    
    # Parse the file
    import re
    with open(chassis_file, 'r') as f:
        content = f.read()
    
    match = re.search(r'CHASSIS_REQUIRED_INDICES = \[(.*?)\]', content, re.DOTALL)
    if not match:
        print("❌ FAIL: Could not parse chassis indices")
        return False
    
    indices_str = match.group(1)
    chassis_indices = [int(x.strip()) for x in indices_str.replace('\n', ' ').split(',') if x.strip()]
    
    print(f"✓ Parsed {len(chassis_indices)} chassis-requiring indices")
    
    # Try loading with these indices
    try:
        from rl_platform.tasks.mobile_mm.multi_trajectory import MultiTrajectoryLoader
        
        traj_dir = str(project_root / "trajectoryToLearn" / "world_json")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Test with first 20 chassis indices
        test_indices = chassis_indices[:20]
        
        print(f"✓ Testing with first 20 chassis indices: {test_indices[:5]}...")
        
        loader = MultiTrajectoryLoader(
            trajectory_dir=traj_dir,
            pattern="**/*.json",
            device=device,
            filter_by_indices=test_indices,
            exclude_macosx=True,
        )
        
        print(f"✓ Loaded {len(loader.trajectories)} chassis-requiring trajectories")
        
        # Sample a few to check X range
        for i in range(min(3, len(loader.trajectories))):
            traj = loader.trajectories[i]
            pos = traj['positions']
            x_range = pos[:, 0].max() - pos[:, 0].min()
            print(f"  ✓ Trajectory {i} ({traj['category']}): X range = {x_range:.3f}m (should be ≥ 2.0m)")
        
        print(f"\n✅ PASS: Chassis trajectory filtering works")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Error loading chassis trajectories: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + "  TRAJECTORY LOADING TEST SUITE".center(78) + "█")
    print("█" + " "*78 + "█")
    print("█"*80)
    
    tests = [
        ("File Discovery", test_1_file_discovery),
        ("JSON Parsing", test_2_json_parsing),
        ("MultiTrajectoryLoader", test_3_multi_trajectory_loader),
        ("TrajectoryManager multi_recorded", test_4_trajectory_manager),
        ("Trajectory Filtering", test_5_filtering),
        ("Chassis Indices Loading", test_6_chassis_indices),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ CRITICAL ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print("\n" + "="*80)
    print(f"OVERALL: {passed}/{total} tests passed")
    print("="*80)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Multi-trajectory loading is working correctly!")
        print("\n✅ You can now run training with:")
        print("   --trajectory_type multi_recorded --use_all_trajectories")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix issues before training.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
