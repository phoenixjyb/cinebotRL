import h5py
import sys

mat_file = sys.argv[1] if len(sys.argv) > 1 else 'matlab/reach_map_mobile_mm_arm_only.mat'

print(f"Inspecting: {mat_file}")
print("=" * 80)

with h5py.File(mat_file, 'r') as f:
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"Group: {name}")
    
    print("Top-level keys:")
    for key in f.keys():
        print(f"  - {key}")
    
    print("\nFull structure:")
    f.visititems(print_structure)
    
    print("\nConfig structure:")
    if 'config' in f:
        config = f['config']
        print(f"  Type: {type(config)}")
        if isinstance(config, h5py.Group):
            for key in config.keys():
                print(f"    - {key}: shape={config[key].shape}, dtype={config[key].dtype}")
