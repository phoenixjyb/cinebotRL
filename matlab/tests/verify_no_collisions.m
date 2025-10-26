%% Verify Reachability Map Contains No Collisions
% This script loads the reachability map and tests every stored configuration
% to verify none of them are in collision

clear; close all;

fprintf('=== Verifying Collision-Free Reachability Map ===\n\n');

%% Load map
MAP_FILE = 'reach_map_mobile_mm_arm_only.mat';
fprintf('Loading map: %s\n', MAP_FILE);
load(MAP_FILE);

fprintf('  Grid: %dx%dx%d voxels\n', config.grid_dims);
fprintf('  Reachable voxels: %d\n', sum(reachScore(:) > 0));

% Display build metadata
if exist('metadata', 'var')
    fprintf('\n  Build metadata:\n');
    fprintf('    Build date: %s\n', metadata.build_date);
    fprintf('    Samples tested: %d\n', metadata.n_samples);
    fprintf('    Valid samples: %d (%.1f%%)\n', metadata.n_valid_samples, ...
            100 * metadata.n_valid_samples / metadata.n_samples);
    if isfield(metadata, 'n_rejected_collision')
        fprintf('    Rejected (collision/OOB): %d (%.1f%%)\n', ...
                metadata.n_rejected_collision, metadata.collision_rejection_rate);
    end
    fprintf('    Collision checking: %s\n', string(metadata.check_collisions));
    if isfield(metadata, 'collision_mode')
        fprintf('    Collision mode: %s\n', metadata.collision_mode);
    end
else
    fprintf('\n  ⚠️  No metadata found in file (old build?)\n');
end

%% Load robot
fprintf('\nLoading robot...\n');
urdf_file = 'mobile_manipulator_PPR_matlab.urdf';
if ~exist(urdf_file, 'file')
    urdf_file = '../assets_own/mobile_manipulator_PPR_matlab.urdf';
end
robot = importrobot(urdf_file);
robot.DataFormat = 'column';

%% Find arm joint indices
cfg = homeConfiguration(robot);
num_joints = numel(cfg);

joint_names = cell(num_joints, 1);
joint_idx = 0;
for b = 1:numel(robot.Bodies)
    body = robot.Bodies{b};
    jnt = body.Joint;
    if strcmp(jnt.Type, 'fixed')
        continue;
    end
    joint_idx = joint_idx + 1;
    joint_names{joint_idx} = jnt.Name;
end

arm_joint_mask = contains(joint_names, 'left_arm');
arm_joint_indices = find(arm_joint_mask);

fprintf('  Configuration has %d joints\n', num_joints);
fprintf('  Arm joints at indices: [%s]\n', num2str(arm_joint_indices'));

%% Debug: Check stored configurations
fprintf('\nInspecting first 10 stored configurations...\n');
[ix_sample, iy_sample, iz_sample] = ind2sub([config.grid_dims(1), config.grid_dims(2), config.grid_dims(3)], ...
                                              find(reachScore > 0, 10));
for i = 1:min(10, length(ix_sample))
    q_test = squeeze(qExample(ix_sample(i), iy_sample(i), iz_sample(i), :));
    fprintf('  Voxel [%d,%d,%d]: q = [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]\n', ...
            ix_sample(i), iy_sample(i), iz_sample(i), q_test(1), q_test(2), q_test(3), ...
            q_test(4), q_test(5), q_test(6));
end

% Check if all zeros (bug indicator)
all_q = reshape(qExample, [], 6);
nonzero_configs = sum(any(all_q ~= 0, 2));
fprintf('\nNon-zero configurations: %d / %d\n', nonzero_configs, size(all_q, 1));
if nonzero_configs == 0
    error('❌ BUG: All stored configurations are zero! Build script did not store joint values.');
end

%% Test each stored configuration
[nx, ny, nz] = deal(config.grid_dims(1), config.grid_dims(2), config.grid_dims(3));

fprintf('\nTesting stored configurations for collisions...\n');
collision_count = 0;
total_configs = 0;

% Find all reachable voxels
[ix_all, iy_all, iz_all] = ind2sub([nx, ny, nz], find(reachScore > 0));
n_reachable = length(ix_all);

fprintf('  Checking %d reachable voxels...\n', n_reachable);
progress_interval = max(1, floor(n_reachable / 10));

for i = 1:n_reachable
    ix = ix_all(i);
    iy = iy_all(i);
    iz = iz_all(i);
    
    % Get stored configuration
    q_arm = squeeze(qExample(ix, iy, iz, :));
    
    % Build full configuration
    q_full = zeros(num_joints, 1);
    q_full(arm_joint_indices) = q_arm;
    
    % Debug first few configurations
    if i <= 3
        fprintf('\n  Testing config %d at voxel [%d,%d,%d]:\n', i, ix, iy, iz);
        fprintf('    q_arm = [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]\n', q_arm);
        fprintf('    q_full = [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f]\n', q_full);
    end
    
    % Check collision (using 'adjacent' mode like in build)
    [isColliding, sepDist] = checkCollision(robot, q_full, ...
        'Exhaustive', 'on', 'SkippedSelfCollisions', 'adjacent');
    
    if isColliding
        collision_count = collision_count + 1;
        
        % Print details of first few collisions
        if collision_count <= 10
            fprintf('\n  ⚠️  COLLISION FOUND at voxel [%d, %d, %d]:\n', ix, iy, iz);
            fprintf('     q_arm = [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]\n', q_arm);
            
            % Find colliding pairs
            [body1Idx, body2Idx] = find(isnan(sepDist));
            if ~isempty(body1Idx)
                collidingPairs = unique(sort([body1Idx, body2Idx], 2), 'rows');
                fprintf('     Colliding pairs:\n');
                for j = 1:min(3, size(collidingPairs, 1))
                    b1 = collidingPairs(j, 1);
                    b2 = collidingPairs(j, 2);
                    if b1 <= numel(robot.Bodies) && b2 <= numel(robot.Bodies)
                        fprintf('       %s <-> %s\n', robot.Bodies{b1}.Name, robot.Bodies{b2}.Name);
                    end
                end
            end
        end
    end
    
    total_configs = total_configs + 1;
    
    if mod(i, progress_interval) == 0
        fprintf('  Checked %d/%d (%.1f%%)\n', i, n_reachable, 100*i/n_reachable);
    end
end

%% Report results
fprintf('\n=== Verification Results ===\n');
fprintf('Total configurations tested: %d\n', total_configs);
fprintf('Collisions found: %d\n', collision_count);

if collision_count == 0
    fprintf('\n✅ SUCCESS: All stored configurations are collision-free!\n');
else
    fprintf('\n❌ FAILURE: %d configurations have collisions (%.1f%%)\n', ...
            collision_count, 100*collision_count/total_configs);
    fprintf('   This indicates a bug in the collision checking during build.\n');
end

fprintf('\n');
