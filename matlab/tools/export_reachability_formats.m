function export_reachability_formats(map_file, output_dir)
% EXPORT_REACHABILITY_FORMATS
% Export reachability map to multiple formats for easy Python/C++/ROS access
%
% Usage:
%   export_reachability_formats('reach_map_mobile_mm.mat', 'matlab/exports')
%
% Exports:
%   1. .mat (MATLAB native, Python scipy.io.loadmat)
%   2. .h5 (HDF5, universal, best for large data)
%   3. .npz (NumPy compressed, Python native)
%   4. .json (metadata only, human-readable)

if nargin < 1
    map_file = 'reach_map_mobile_mm.mat';
end
if nargin < 2
    output_dir = 'matlab/exports';
end

% Create output directory
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

fprintf('╔════════════════════════════════════════════════════════╗\n');
fprintf('║  REACHABILITY MAP MULTI-FORMAT EXPORTER               ║\n');
fprintf('╚════════════════════════════════════════════════════════╝\n\n');

% Load map
fprintf('Loading: %s\n', map_file);
S = load(map_file, 'map');
map = S.map;

[~, base_name, ~] = fileparts(map_file);

%% ========================================================================
%% FORMAT 1: HDF5 (.h5) - RECOMMENDED FOR PYTHON
%% ========================================================================
% Best for: Large datasets, universal access (Python, C++, Julia, R)
% Python: h5py.File('map.h5', 'r')
fprintf('\n1️⃣  Exporting to HDF5 (.h5)...\n');
h5_file = fullfile(output_dir, [base_name, '.h5']);

% Delete if exists (HDF5 can't overwrite)
if exist(h5_file, 'file')
    delete(h5_file);
end

% Write metadata
h5create(h5_file, '/meta/urdfPath', [1 1], 'Datatype', 'string');
h5write(h5_file, '/meta/urdfPath', string(map.meta.urdfPath));

h5create(h5_file, '/meta/eeLink', [1 1], 'Datatype', 'string');
h5write(h5_file, '/meta/eeLink', string(map.meta.eeLink));

h5create(h5_file, '/meta/timestamp', [1 1], 'Datatype', 'string');
h5write(h5_file, '/meta/timestamp', string(map.meta.time));

h5create(h5_file, '/meta/payload_kg', [1 1]);
h5write(h5_file, '/meta/payload_kg', map.meta.payload_kg);

% Write grid parameters
h5create(h5_file, '/grid/origin', size(map.grid.origin));
h5write(h5_file, '/grid/origin', map.grid.origin);

h5create(h5_file, '/grid/voxel', size(map.grid.voxel));
h5write(h5_file, '/grid/voxel', map.grid.voxel);

h5create(h5_file, '/grid/size', size(map.grid.size));
h5write(h5_file, '/grid/size', map.grid.size);

h5create(h5_file, '/grid/shape', size(map.grid.shape));
h5write(h5_file, '/grid/shape', int32(map.grid.shape));

% Write orientation parameters
h5create(h5_file, '/orient/n_bins', [1 1]);
h5write(h5_file, '/orient/n_bins', int32(map.orient.n_bins));

h5create(h5_file, '/orient/cone_deg', [1 1]);
h5write(h5_file, '/orient/cone_deg', map.orient.cone_deg);

% Write data arrays (main payload)
h5create(h5_file, '/data/reachScore', size(map.data.reachScore), 'Datatype', 'single', 'ChunkSize', [1 1 1]);
h5write(h5_file, '/data/reachScore', map.data.reachScore);

h5create(h5_file, '/data/manipMax', size(map.data.manipMax), 'Datatype', 'single', 'ChunkSize', [1 1 1]);
h5write(h5_file, '/data/manipMax', map.data.manipMax);

h5create(h5_file, '/data/hasExampleQ', size(map.data.hasExampleQ), 'Datatype', 'uint8');
h5write(h5_file, '/data/hasExampleQ', uint8(map.data.hasExampleQ));

% Optional: IK seeds (can be large)
if ~isempty(map.data.exampleQ)
    h5create(h5_file, '/data/exampleQ', size(map.data.exampleQ), 'Datatype', 'single', 'ChunkSize', [1 1 1 1]);
    h5write(h5_file, '/data/exampleQ', map.data.exampleQ);
end

fprintf('   ✓ Saved: %s (%.1f MB)\n', h5_file, dir(h5_file).bytes/1e6);
fprintf('   Python: import h5py; f = h5py.File("%s", "r")\n', h5_file);

%% ========================================================================
%% FORMAT 2: NumPy Compressed (.npz) - PYTHON NATIVE
%% ========================================================================
% Best for: Pure Python workflows, no dependencies
% Python: np.load('map.npz')
fprintf('\n2️⃣  Exporting to NumPy (.npz)...\n');
npz_file = fullfile(output_dir, [base_name, '.npz']);

% Create Python script to do the conversion (MATLAB can't write .npz directly)
py_script = fullfile(output_dir, 'convert_mat_to_npz.py');
fid = fopen(py_script, 'w');
fprintf(fid, '#!/usr/bin/env python3\n');
fprintf(fid, '"""Auto-generated script to convert .mat to .npz"""\n');
fprintf(fid, 'import numpy as np\n');
fprintf(fid, 'from scipy.io import loadmat\n\n');
fprintf(fid, 'mat = loadmat("%s", struct_as_record=False, squeeze_me=True)\n', strrep(map_file, '\', '\\'));
fprintf(fid, 'm = mat["map"]\n\n');
fprintf(fid, '# Extract all fields\n');
fprintf(fid, 'np.savez_compressed(\n');
fprintf(fid, '    "%s",\n', strrep(npz_file, '\', '\\'));
fprintf(fid, '    # Metadata\n');
fprintf(fid, '    urdfPath=str(m.meta.urdfPath),\n');
fprintf(fid, '    eeLink=str(m.meta.eeLink),\n');
fprintf(fid, '    timestamp=str(m.meta.time),\n');
fprintf(fid, '    payload_kg=float(m.meta.payload_kg),\n');
fprintf(fid, '    # Grid\n');
fprintf(fid, '    grid_origin=m.grid.origin.astype(np.float32),\n');
fprintf(fid, '    grid_voxel=m.grid.voxel.astype(np.float32),\n');
fprintf(fid, '    grid_size=m.grid.size.astype(np.float32),\n');
fprintf(fid, '    grid_shape=m.grid.shape.astype(np.int32),\n');
fprintf(fid, '    # Orientation\n');
fprintf(fid, '    orient_n_bins=int(m.orient.n_bins),\n');
fprintf(fid, '    orient_cone_deg=float(m.orient.cone_deg),\n');
fprintf(fid, '    # Data\n');
fprintf(fid, '    reachScore=m.data.reachScore.astype(np.float32),\n');
fprintf(fid, '    manipMax=m.data.manipMax.astype(np.float32),\n');
fprintf(fid, '    hasExampleQ=m.data.hasExampleQ.astype(bool),\n');
fprintf(fid, '    exampleQ=m.data.exampleQ.astype(np.float32) if hasattr(m.data, "exampleQ") else np.array([])\n');
fprintf(fid, ')\n');
fprintf(fid, 'print(f"✓ Saved: %s ({np.load(\\"%s\\").get(\\"reachScore\\").nbytes/1e6:.1f} MB)")\n', npz_file, strrep(npz_file, '\', '\\'));
fclose(fid);

fprintf('   Created conversion script: %s\n', py_script);
fprintf('   Run: python %s\n', py_script);

%% ========================================================================
%% FORMAT 3: JSON (metadata only) - HUMAN READABLE
%% ========================================================================
% Best for: Configuration files, documentation, version control
fprintf('\n3️⃣  Exporting metadata to JSON...\n');
json_file = fullfile(output_dir, [base_name, '_metadata.json']);

meta_struct = struct();
meta_struct.urdfPath = char(map.meta.urdfPath);
meta_struct.eeLink = char(map.meta.eeLink);
meta_struct.timestamp = char(map.meta.time);
meta_struct.payload_kg = map.meta.payload_kg;
meta_struct.grid = struct();
meta_struct.grid.origin = map.grid.origin;
meta_struct.grid.voxel = map.grid.voxel;
meta_struct.grid.size = map.grid.size;
meta_struct.grid.shape = double(map.grid.shape);
meta_struct.orient = struct();
meta_struct.orient.n_bins = double(map.orient.n_bins);
meta_struct.orient.cone_deg = map.orient.cone_deg;
meta_struct.stats = struct();
meta_struct.stats.total_voxels = numel(map.data.reachScore);
meta_struct.stats.reachable_voxels = sum(map.data.reachScore(:) > 0);
meta_struct.stats.reachable_fraction = sum(map.data.reachScore(:) > 0) / numel(map.data.reachScore);
meta_struct.stats.mean_reach_score = mean(map.data.reachScore(map.data.reachScore > 0));

json_text = jsonencode(meta_struct, 'PrettyPrint', true);
fid = fopen(json_file, 'w');
fprintf(fid, '%s', json_text);
fclose(fid);

fprintf('   ✓ Saved: %s (%.1f KB)\n', json_file, dir(json_file).bytes/1e3);
fprintf('   Human-readable metadata for documentation\n');

%% ========================================================================
%% FORMAT 4: Binary Grid Files (raw, for C++/ROS)
%% ========================================================================
fprintf('\n4️⃣  Exporting raw binary grids...\n');

% Reach score grid
reach_bin = fullfile(output_dir, [base_name, '_reach_score.bin']);
fid = fopen(reach_bin, 'w');
fwrite(fid, map.data.reachScore(:), 'single');
fclose(fid);
fprintf('   ✓ Reach score: %s (%.1f MB)\n', reach_bin, dir(reach_bin).bytes/1e6);

% Manipulability grid
manip_bin = fullfile(output_dir, [base_name, '_manipulability.bin']);
fid = fopen(manip_bin, 'w');
fwrite(fid, map.data.manipMax(:), 'single');
fclose(fid);
fprintf('   ✓ Manipulability: %s (%.1f MB)\n', manip_bin, dir(manip_bin).bytes/1e6);

% Grid parameters (for loading binary files)
params_txt = fullfile(output_dir, [base_name, '_grid_params.txt']);
fid = fopen(params_txt, 'w');
fprintf(fid, '# Reachability Map Grid Parameters\n');
fprintf(fid, '# Binary files are single-precision float (4 bytes), C-order (row-major)\n\n');
fprintf(fid, 'shape: %d %d %d\n', map.grid.shape);
fprintf(fid, 'origin: %.6f %.6f %.6f\n', map.grid.origin);
fprintf(fid, 'voxel: %.6f %.6f %.6f\n', map.grid.voxel);
fprintf(fid, 'size: %.6f %.6f %.6f\n', map.grid.size);
fprintf(fid, '\n# To load in Python:\n');
fprintf(fid, '# data = np.fromfile("reach_score.bin", dtype=np.float32)\n');
fprintf(fid, '# grid = data.reshape(%d, %d, %d)\n', map.grid.shape);
fclose(fid);
fprintf('   ✓ Parameters: %s\n', params_txt);

%% ========================================================================
%% SUMMARY
%% ========================================================================
fprintf('\n╔════════════════════════════════════════════════════════╗\n');
fprintf('║  EXPORT COMPLETE                                      ║\n');
fprintf('╚════════════════════════════════════════════════════════╝\n\n');
fprintf('Output directory: %s\n\n', output_dir);

fprintf('📦 FORMATS:\n');
fprintf('   1. HDF5 (.h5)     - Best for Python/C++, universal\n');
fprintf('   2. NumPy (.npz)   - Python native (run conversion script)\n');
fprintf('   3. JSON           - Metadata only, human-readable\n');
fprintf('   4. Binary (.bin)  - Raw grids for C++/ROS\n\n');

fprintf('🐍 PYTHON USAGE:\n');
fprintf('   # Option A: HDF5 (recommended)\n');
fprintf('   import h5py\n');
fprintf('   f = h5py.File("%s", "r")\n', h5_file);
fprintf('   reach = f["data/reachScore"][:]\n\n');

fprintf('   # Option B: NumPy (after running conversion)\n');
fprintf('   import numpy as np\n');
fprintf('   data = np.load("%s")\n', npz_file);
fprintf('   reach = data["reachScore"]\n\n');

fprintf('   # Option C: SciPy (original .mat)\n');
fprintf('   from scipy.io import loadmat\n');
fprintf('   mat = loadmat("%s")\n', map_file);
fprintf('   reach = mat["map"]["data"]["reachScore"]\n\n');

end % export_reachability_formats
