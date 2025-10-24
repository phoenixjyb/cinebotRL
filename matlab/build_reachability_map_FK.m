function build_reachability_map_FK()
% BUILD_REACHABILITY_MAP_FK - Fast reachability map using Forward Kinematics
%
% This version samples joint space uniformly and uses FK to find reachable
% voxels. Much faster than IK-based approach (1-2 min vs 10-15 min).
%
% Algorithm:
%   1. Sample N random joint configurations within joint limits
%   2. Compute FK for each → get EE position
%   3. Find which voxel the EE lands in
%   4. Mark voxel as reachable, store best configuration
%
% Output: reach_map_mobile_mm_arm_only.mat
%   - reachScore: (nx,ny,nz) - 1.0 if reachable, 0.0 otherwise
%   - manipMax: (nx,ny,nz) - best manipulability found
%   - qExample: (nx,ny,nz,ndof) - joint config for each reachable voxel
%   - config: grid parameters
%   - metadata: build info

fprintf('=== Building Reachability Map (FK-based) ===\n\n');

%% ===== CONFIGURATION =====

% Robot URDF (use MATLAB-compatible version with relative mesh paths)
URDF_PATH = fullfile('..', 'assets_own', 'mobile_manipulator_PPR_matlab.urdf');
if ~isfile(URDF_PATH)
    % Fallback to original if MATLAB version doesn't exist
    URDF_PATH = fullfile('..', 'assets_own', 'mobile_manipulator_PPR_base_corrected.urdf');
    if ~isfile(URDF_PATH)
        error('URDF not found');
    end
end

% Which link to use as base (arm shoulder, NOT mobile base)
BASE_LINK = "left_arm_base_link";
EE_LINK = "left_gripper_link";

% Workspace grid in ARM BASE LINK frame
% From URDF: arm_mount_joint places left_arm_base_link at [0.16, 0, 0.9465] relative to chassis
% Grid origin [-0.8, -1.0, -0.6] is RELATIVE TO arm base link, meaning:
%   - Workspace extends ±0.8m in X, ±1.0m in Y, -0.6 to +0.8m in Z around shoulder
%   - In world frame, this is centered at [0.16, 0, 0.9465] (shoulder height, not ground!)
% Expanded by 40cm on each axis to check if FK covers more space
GRID_ORIGIN = [-0.8, -1.0, -0.6];  % [x, y, z] min corner (was [-0.6, -0.8, -0.4])
GRID_SIZE = [1.6, 2.0, 1.4];        % [dx, dy, dz] extent (was [1.2, 1.6, 1.0])
VOXEL_SIZE = 0.05;                  % 5cm resolution

% FK Sampling
N_SAMPLES = 100000;  % Number of random joint configs to try (more = better coverage)
USE_PARFOR = true;   % Parallel processing (recommended)

% Self-collision check
CHECK_COLLISIONS = true;
COLLISION_PAIRS = {
    % Format: {'link1', 'link2'}
    % NOTE: Link names from URDF (left_arm_link1 NOT left_arm_link_1!)
    
    % Arm links hitting chassis
    {'left_arm_link1', 'abstract_chassis_link'},
    {'left_arm_link2', 'abstract_chassis_link'},
    {'left_arm_link3', 'abstract_chassis_link'},
    
    % Arm self-collisions (non-adjacent links only)
    % Adjacent links (e.g., link1-link2) always touch - skip them!
    {'left_arm_link1', 'left_arm_link3'},
    {'left_arm_link1', 'left_arm_link4'},
    {'left_arm_link1', 'left_arm_link5'},
    {'left_arm_link1', 'left_arm_link6'},
    {'left_arm_link2', 'left_arm_link4'},
    {'left_arm_link2', 'left_arm_link5'},
    {'left_arm_link2', 'left_arm_link6'},
    {'left_arm_link3', 'left_arm_link5'},
    {'left_arm_link3', 'left_arm_link6'},
    {'left_arm_link4', 'left_arm_link6'},
    
    % Gripper collisions (end effector = left_gripper_link)
    {'left_gripper_link', 'abstract_chassis_link'},
    {'left_gripper_link', 'left_arm_link1'},
    {'left_gripper_link', 'left_arm_link2'},
    {'left_gripper_link', 'left_arm_link3'}
};

% Output
MAP_FILE = 'reach_map_mobile_mm_arm_only.mat';

%% ===== LOAD ROBOT =====

fprintf('Loading robot from: %s\n', URDF_PATH);
robot = importrobot(URDF_PATH, 'DataFormat', 'column');

% Build configuration mapping - iterate through bodies to get non-fixed joints
cfg = homeConfiguration(robot);               % column vector of joint positions
num_joints = numel(cfg);

% Extract joint names and limits by iterating through Bodies
joint_names = cell(num_joints, 1);
joint_limits_lower_all = nan(num_joints, 1);
joint_limits_upper_all = nan(num_joints, 1);

joint_idx = 0;
for b = 1:numel(robot.Bodies)
    body = robot.Bodies{b};
    jnt = body.Joint;
    
    % Skip fixed joints (they don't appear in homeConfiguration)
    if strcmp(jnt.Type, 'fixed')
        continue;
    end
    
    joint_idx = joint_idx + 1;
    joint_names{joint_idx} = jnt.Name;
    
    if ~isempty(jnt.PositionLimits)
        joint_limits_lower_all(joint_idx) = jnt.PositionLimits(1);
        joint_limits_upper_all(joint_idx) = jnt.PositionLimits(2);
    end
end

% Select arm joints from the configuration vector (joint names containing 'left_arm' or 'left_gripper')
arm_joint_config_mask = contains(joint_names, 'left_arm') | contains(joint_names, 'left_gripper');
arm_joint_config_indices = find(arm_joint_config_mask);

if isempty(arm_joint_config_indices)
    error('No arm joints found in robot configuration. Check joint naming in URDF.');
end

% Extract joint limits for arm joints
joint_limits_lower = joint_limits_lower_all(arm_joint_config_indices)';
joint_limits_upper = joint_limits_upper_all(arm_joint_config_indices)';

ndof = length(joint_limits_lower);
fprintf('  Total non-fixed joints: %d\n', num_joints);
fprintf('  DOF (arm only): %d\n', ndof);
fprintf('  Joint limits:\n');
for i = 1:ndof
    fprintf('    Joint %d: [%.3f, %.3f] rad\n', i, joint_limits_lower(i), joint_limits_upper(i));
end

%% ===== SETUP GRID =====

nx = round(GRID_SIZE(1) / VOXEL_SIZE);
ny = round(GRID_SIZE(2) / VOXEL_SIZE);
nz = round(GRID_SIZE(3) / VOXEL_SIZE);
Nvox = nx * ny * nz;

fprintf('\nGrid setup:\n');
fprintf('  Origin: [%.2f, %.2f, %.2f] m\n', GRID_ORIGIN);
fprintf('  Size: [%.2f, %.2f, %.2f] m\n', GRID_SIZE);
fprintf('  Voxel size: %.3f m\n', VOXEL_SIZE);
fprintf('  Grid dimensions: %d × %d × %d = %d voxels\n', nx, ny, nz, Nvox);

% Initialize output arrays
reachScore = zeros(nx, ny, nz, 'single');
manipMax = zeros(nx, ny, nz, 'single');
qExample = zeros(nx, ny, nz, ndof, 'single');

%% ===== PARALLEL SETUP =====

if USE_PARFOR
    pool = gcp('nocreate');
    if isempty(pool)
        fprintf('\nStarting parallel pool...\n');
        pool = parpool('local');
    end
    fprintf('  Using %d workers\n', pool.NumWorkers);
else
    fprintf('\nRunning in serial mode (set USE_PARFOR=true for parallel)\n');
end

%% ===== SAMPLE AND BUILD =====

fprintf('\n=== Sampling %d random configurations ===\n', N_SAMPLES);
tic;

% Pre-allocate arrays for parallel results
voxel_ids = zeros(N_SAMPLES, 1, 'int32');
valid_mask = false(N_SAMPLES, 1);
manip_values = zeros(N_SAMPLES, 1, 'single');
q_samples = zeros(N_SAMPLES, ndof, 'single');

% Ensure joint limits have valid numbers (fallback to [-pi,pi])
nan_mask = isnan(joint_limits_lower) | isnan(joint_limits_upper);
if any(nan_mask)
    warning('Some joint limits unknown; falling back to [-pi,pi] for those joints');
    joint_limits_lower(nan_mask) = -pi;
    joint_limits_upper(nan_mask) = pi;
end

% Generate all random samples first
rng(42);  % Reproducible
q_rand = joint_limits_lower + (joint_limits_upper - joint_limits_lower) .* rand(N_SAMPLES, ndof);

fprintf('Computing FK and checking collisions...\n');
progress_interval = max(1, floor(N_SAMPLES / 20));  % Show 20 updates

if USE_PARFOR
    parfor i = 1:N_SAMPLES
        [voxel_ids(i), valid_mask(i), manip_values(i), q_samples(i,:)] = ...
            eval_sample(robot, q_rand(i,:)', arm_joint_config_indices, ...
                       BASE_LINK, EE_LINK, ...
                       GRID_ORIGIN, VOXEL_SIZE, nx, ny, nz, ...
                       CHECK_COLLISIONS, COLLISION_PAIRS);
        
        if mod(i, progress_interval) == 0
            fprintf('  Processed %d/%d samples (%.1f%%)\n', i, N_SAMPLES, 100*i/N_SAMPLES);
        end
    end
else
    for i = 1:N_SAMPLES
        [voxel_ids(i), valid_mask(i), manip_values(i), q_samples(i,:)] = ...
            eval_sample(robot, q_rand(i,:)', arm_joint_config_indices, ...
                       BASE_LINK, EE_LINK, ...
                       GRID_ORIGIN, VOXEL_SIZE, nx, ny, nz, ...
                       CHECK_COLLISIONS, COLLISION_PAIRS);
        
        if mod(i, progress_interval) == 0
            fprintf('  Processed %d/%d samples (%.1f%%)\n', i, N_SAMPLES, 100*i/N_SAMPLES);
        end
    end
end

fprintf('FK computation complete in %.1f sec\n', toc);

%% ===== AGGREGATE RESULTS =====

fprintf('\nAggregating results into voxel grid...\n');
tic;

n_valid = sum(valid_mask);
fprintf('  Valid samples (no collision, in bounds): %d / %d (%.1f%%)\n', ...
        n_valid, N_SAMPLES, 100*n_valid/N_SAMPLES);

% DEBUG: Check why samples are being rejected
if n_valid == 0
    fprintf('\n⚠️  WARNING: NO VALID SAMPLES FOUND!\n');
    fprintf('   This usually means:\n');
    fprintf('   1. Collision checking is too strict\n');
    fprintf('   2. Grid bounds are wrong\n');
    fprintf('   3. FK is failing\n');
    fprintf('\n   Checking first 10 samples in detail...\n\n');
    
    for i = 1:min(10, N_SAMPLES)
        fprintf('Sample %d:\n', i);
        fprintf('  q = [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]\n', q_rand(i,:));
        
    % Try FK manually
    cfg_dbg = homeConfiguration(robot);
    q_full = zeros(numel(cfg_dbg), 1);
    q_full(arm_joint_config_indices) = q_rand(i,:);
        
        try
            T_ee = getTransform(robot, q_full, EE_LINK, BASE_LINK);
            ee_pos = T_ee(1:3, 4);
            fprintf('  EE pos: [%.3f, %.3f, %.3f]\n', ee_pos);
            
            % Check grid bounds
            rel_pos = ee_pos' - GRID_ORIGIN;
            fprintf('  Relative pos: [%.3f, %.3f, %.3f]\n', rel_pos);
            fprintf('  Grid max: [%.3f, %.3f, %.3f]\n', [nx, ny, nz] * VOXEL_SIZE);
            
            if any(rel_pos < 0)
                fprintf('  ❌ REJECTED: Below grid origin\n');
            elseif any(rel_pos >= [nx, ny, nz] * VOXEL_SIZE)
                fprintf('  ❌ REJECTED: Above grid bounds\n');
            else
                fprintf('  ✓ Inside grid\n');
            end
        catch ME
            fprintf('  ❌ FK FAILED: %s\n', ME.message);
        end
        fprintf('\n');
    end
    
    error('Build failed: no valid samples. See debug output above.');
end

% For each valid sample, update the voxel grid
for i = 1:N_SAMPLES
    if ~valid_mask(i)
        continue;
    end
    
    vox_id = voxel_ids(i);
    
    % Convert linear voxel ID to 3D indices
    [ix, iy, iz] = ind2sub([nx, ny, nz], vox_id);
    
    % Update if this is the first sample or has better manipulability
    if reachScore(ix, iy, iz) == 0 || manip_values(i) > manipMax(ix, iy, iz)
        reachScore(ix, iy, iz) = 1.0;
        manipMax(ix, iy, iz) = manip_values(i);
        qExample(ix, iy, iz, :) = q_samples(i, :);
    end
end

n_reachable = sum(reachScore(:) > 0);
fprintf('  Reachable voxels: %d / %d (%.1f%%)\n', ...
        n_reachable, Nvox, 100*n_reachable/Nvox);
fprintf('Aggregation complete in %.1f sec\n', toc);

%% ===== SAVE =====

config = struct();
config.grid_origin = GRID_ORIGIN;
config.grid_size = GRID_SIZE;
config.voxel_size = VOXEL_SIZE;
config.grid_dims = [nx, ny, nz];
config.base_link = char(BASE_LINK);
config.ee_link = char(EE_LINK);
config.urdf_path = URDF_PATH;

metadata = struct();
metadata.build_date = datetime('now');
metadata.n_samples = N_SAMPLES;
metadata.n_valid_samples = n_valid;
metadata.n_reachable_voxels = n_reachable;
metadata.check_collisions = CHECK_COLLISIONS;
metadata.use_parfor = USE_PARFOR;

fprintf('\nSaving map to: %s\n', MAP_FILE);
save(MAP_FILE, 'reachScore', 'manipMax', 'qExample', 'config', 'metadata', '-v7.3');

info = dir(MAP_FILE);
fprintf('  File size: %.1f MB\n', info.bytes / 1e6);

fprintf('\n=== Build Complete ===\n');
fprintf('Total time: %.1f sec\n', toc);
fprintf('Next: run quick_viz() to visualize\n\n');

end


%% ===== HELPER FUNCTION =====

function [voxel_id, is_valid, manip, q] = eval_sample(robot, q, arm_cfg_indices, ...
                                                       base_link, ee_link, ...
                                                       grid_origin, voxel_size, nx, ny, nz, ...
                                                       check_collisions, collision_pairs)
% Evaluate a single joint configuration
%
% Returns:
%   voxel_id - linear index of voxel (1 to Nvox)
%   is_valid - true if no collision and inside grid
%   manip - manipulability measure
%   q - joint configuration

% Initialize outputs
voxel_id = 0;
is_valid = false;
manip = 0;

% Build full configuration vector sized to non-fixed joints (homeConfiguration)
cfg = homeConfiguration(robot);
num_cfg_joints = numel(cfg);
q_full = zeros(num_cfg_joints, 1);
% arm_cfg_indices maps into this configuration vector
q_full(arm_cfg_indices) = q;

% Check self-collision first (fast rejection)
if check_collisions && check_self_collision(robot, q_full, collision_pairs)
    return;  % In collision, reject
end

% Compute FK to get EE position
try
    T_ee = getTransform(robot, q_full, ee_link, base_link);
    ee_pos = T_ee(1:3, 4);
catch
    return;  % FK failed
end

% Check if inside grid bounds
rel_pos = ee_pos' - grid_origin;
if any(rel_pos < 0) || any(rel_pos >= [nx, ny, nz] * voxel_size)
    return;  % Outside grid
end

% Convert position to voxel indices
ix = floor(rel_pos(1) / voxel_size) + 1;
iy = floor(rel_pos(2) / voxel_size) + 1;
iz = floor(rel_pos(3) / voxel_size) + 1;

% Clamp to valid range (defensive)
ix = max(1, min(nx, ix));
iy = max(1, min(ny, iy));
iz = max(1, min(nz, iz));

% Convert to linear index
voxel_id = sub2ind([nx, ny, nz], ix, iy, iz);

% Compute manipulability (simplified: use joint range utilization)
% Better configs are closer to middle of joint ranges
% Scale 0 (at limits) to 1 (at center)
joint_limits_lower = [-2.9671, -2.0944, -2.9671, -2.0944, -2.9671, -2.0944];  % Hardcoded for speed
joint_limits_upper = [2.9671, 2.0944, 2.9671, 2.0944, 2.9671, 2.0944];

q_normalized = (q - joint_limits_lower') ./ (joint_limits_upper' - joint_limits_lower');
q_centered = abs(q_normalized - 0.5);  % Distance from center (0 = center, 0.5 = limit)
manip = 1 - mean(2 * q_centered);  % Higher = more centered = better manipulability

is_valid = true;

% Return q as row vector to match q_samples(i,:)
q = q(:)';

end


function in_collision = check_self_collision(robot, q_full, collision_pairs)
% Check if configuration has self-collision
%
% Uses distance threshold for sphere-based collision detection

in_collision = false;
COLLISION_THRESHOLD = 0.05;  % 5cm safety margin

for i = 1:length(collision_pairs)
    link1_name = collision_pairs{i}{1};
    link2_name = collision_pairs{i}{2};
    
    try
        % Get transforms for both links
        T1 = getTransform(robot, q_full, link1_name);
        T2 = getTransform(robot, q_full, link2_name);
        
        % Extract positions
        p1 = T1(1:3, 4);
        p2 = T2(1:3, 4);
        
        % Check distance
        dist = norm(p1 - p2);
        
        if dist < COLLISION_THRESHOLD
            in_collision = true;
            return;
        end
    catch
        % If transform fails, assume no collision
        continue;
    end
end

end
