%% Visualize Random Stored Configurations from Reachability Map
% This loads the map and visualizes some stored configurations
% to manually verify if they collide with chassis

clear; close all;

fprintf('=== Visualizing Stored Configurations ===\n\n');

%% Load map
MAP_FILE = 'reach_map_mobile_mm_arm_only.mat';
fprintf('Loading map: %s\n', MAP_FILE);
map_data = load(MAP_FILE);

% Display what's in the file
fprintf('Variables in map file:\n');
disp(fieldnames(map_data));

% Extract variables
if isfield(map_data, 'reachScore')
    reachScore = map_data.reachScore;
end
if isfield(map_data, 'config')
    config = map_data.config;
end
if isfield(map_data, 'metadata')
    metadata = map_data.metadata;
end

% Check for stored configurations
if isfield(map_data, 'qExample')
    qExample = map_data.qExample;
    fprintf('✅ Loaded qExample (best config per voxel): %s\n', mat2str(size(qExample)));
else
    error('Map file does not contain qExample (stored configurations)!');
end

%% Load robot
fprintf('Loading robot...\n');
urdf_file = 'mobile_manipulator_PPR_theta_before_x.urdf';
if ~exist(urdf_file, 'file')
    urdf_file = '../assets_own/mobile_manipulator_PPR_theta_before_x.urdf';
end
if ~exist(urdf_file, 'file')
    % Fallback to old URDF
    urdf_file = '../assets_own/mobile_manipulator_PPR_base_corrected.urdf';
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

fprintf('  Arm joints at indices: [%s]\n', num2str(arm_joint_indices'));

%% Get home configuration
q_home = homeConfiguration(robot);

%% Select random stored configurations to visualize
[nx, ny, nz] = deal(config.grid_dims(1), config.grid_dims(2), config.grid_dims(3));
[ix_all, iy_all, iz_all] = ind2sub([nx, ny, nz], find(reachScore > 0));
n_reachable = length(ix_all);

fprintf('\nTotal reachable voxels: %d\n', n_reachable);

%% Check ALL configurations with negative X in WORLD frame
fprintf('Checking ALL voxels with negative X in WORLD frame...\n');
fprintf('(Behind chassis front - most critical for collision)\n');

% Arm base transform from URDF
ARM_TRANSLATION = [0.16; 0; 0.9465];
ARM_ROTATION = [0, 1, 0; -1, 0, 0; 0, 0, 1];

% Chassis bounds in world frame (from URDF)
CHASSIS_X_MIN = -0.4;
CHASSIS_X_MAX = 0.4;
CHASSIS_Y_MIN = -0.3;
CHASSIS_Y_MAX = 0.3;
CHASSIS_Z_MIN = 0.0;
CHASSIS_Z_MAX = 0.55;

critical_voxels = [];
voxel_scores = [];
voxel_positions_world = [];
inside_chassis_count = 0;
collision_count = 0;

grid_origin = config.grid_origin;
voxel_size = config.voxel_size;

fprintf('Checking all %d voxels...\n', n_reachable);

for i = 1:n_reachable
    ix = ix_all(i);
    iy = iy_all(i);
    iz = iz_all(i);
    
    % Calculate voxel center position in arm base frame
    voxel_pos_arm = [grid_origin(1) + (ix - 0.5) * voxel_size;
                     grid_origin(2) + (iy - 0.5) * voxel_size;
                     grid_origin(3) + (iz - 0.5) * voxel_size];
    
    % Transform to world frame: p_world = R * p_arm + t
    voxel_pos_world = ARM_ROTATION * voxel_pos_arm + ARM_TRANSLATION;
    
    % Check if X in world frame is negative (behind chassis front)
    if voxel_pos_world(1) < 0
        % Calculate collision risk score:
        % - More negative X = deeper behind chassis = higher risk
        % - Smaller |Y| = closer to centerline = higher risk
        risk_score = -voxel_pos_world(1) + 0.5 * abs(voxel_pos_world(2));
        
        critical_voxels = [critical_voxels; i];
        voxel_scores = [voxel_scores; risk_score];
        voxel_positions_world = [voxel_positions_world; voxel_pos_world'];
        
        % Check if inside chassis volume
        inside_x = (voxel_pos_world(1) >= CHASSIS_X_MIN) && (voxel_pos_world(1) <= CHASSIS_X_MAX);
        inside_y = (voxel_pos_world(2) >= CHASSIS_Y_MIN) && (voxel_pos_world(2) <= CHASSIS_Y_MAX);
        inside_z = (voxel_pos_world(3) >= CHASSIS_Z_MIN) && (voxel_pos_world(3) <= CHASSIS_Z_MAX);
        
        if inside_x && inside_y && inside_z
            inside_chassis_count = inside_chassis_count + 1;
        end
        
        % Check actual collision
        q_stored = squeeze(qExample(ix, iy, iz, :));  % qExample is (nx, ny, nz, ndof)
        q_full = q_home;
        q_full(arm_joint_indices) = q_stored;
        has_collision = checkCollision(robot, q_full, 'SkippedSelfCollisions', 'adjacent');
        
        if has_collision
            collision_count = collision_count + 1;
        end
    end
end

fprintf('\n=== COMPREHENSIVE CHECK RESULTS ===\n');
fprintf('Total voxels with negative X (behind chassis): %d (%.1f%%)\n', ...
        length(critical_voxels), 100*length(critical_voxels)/n_reachable);
fprintf('Voxels inside chassis volume: %d\n', inside_chassis_count);
fprintf('Voxels with actual collisions: %d\n', collision_count);

if inside_chassis_count > 0
    fprintf('\n⚠️  WARNING: %d voxels are inside chassis volume!\n', inside_chassis_count);
    fprintf('   Collision detection may have gaps!\n');
end

if collision_count > 0
    fprintf('\n🚨 CRITICAL: %d voxels have collisions!\n', collision_count);
    fprintf('   Build script has bugs!\n');
else
    fprintf('\n✅ PASS: No collisions detected in negative X voxels\n');
end

% Statistics on Y distribution for negative X voxels
if ~isempty(voxel_positions_world)
    y_values = abs(voxel_positions_world(:, 2));
    fprintf('\nY-axis distribution (|Y|) for negative X voxels:\n');
    fprintf('  Min |Y|: %.3f m (closest to centerline)\n', min(y_values));
    fprintf('  Max |Y|: %.3f m (furthest from centerline)\n', max(y_values));
    fprintf('  Mean |Y|: %.3f m\n', mean(y_values));
    fprintf('  Voxels with |Y| < 0.1m: %d (%.1f%%)\n', ...
            sum(y_values < 0.1), 100*sum(y_values < 0.1)/length(y_values));
    fprintf('  Voxels with |Y| < 0.05m: %d (%.1f%%)\n', ...
            sum(y_values < 0.05), 100*sum(y_values < 0.05)/length(y_values));
end

if isempty(critical_voxels)
    fprintf('\n⚠️  WARNING: No negative X voxels in world frame found!\n');
    fprintf('   The workspace might not extend behind the chassis.\n');
    fprintf('   Falling back to voxels closest to X=0 in world frame...\n\n');
    
    % Find voxels closest to X=0 in world frame
    all_x_world_values = zeros(n_reachable, 1);
    for i = 1:n_reachable
        ix = ix_all(i);
        iy = iy_all(i);
        iz = iz_all(i);
        
        voxel_pos_arm = [grid_origin(1) + (ix - 0.5) * voxel_size;
                         grid_origin(2) + (iy - 0.5) * voxel_size;
                         grid_origin(3) + (iz - 0.5) * voxel_size];
        voxel_pos_world = ARM_ROTATION * voxel_pos_arm + ARM_TRANSLATION;
        all_x_world_values(i) = voxel_pos_world(1);
    end
    [~, sort_idx] = sort(all_x_world_values);
    n_viz = min(9, n_reachable);
    random_indices = sort_idx(1:n_viz);
    
    fprintf('  Visualizing %d voxels with smallest X in world frame...\n', n_viz);
    fprintf('  World X range: [%.3f, %.3f] m\n\n', all_x_world_values(sort_idx(1)), all_x_world_values(sort_idx(n_viz)));
else
    % Sort by highest collision risk (most negative X + smallest |Y|)
    [~, sort_idx] = sort(voxel_scores, 'descend');
    
    % Pick top 9 most critical
    n_viz = min(9, length(critical_voxels));
    random_indices = critical_voxels(sort_idx(1:n_viz));
    
    % Get world positions for display
    fprintf('\n=== VISUALIZING TOP 9 MOST CRITICAL VOXELS ===\n');
    fprintf('  Config | World X (m) | World Y (m) | |Y| (m) | Risk Score\n');
    fprintf('  -------|-------------|-------------|---------|------------\n');
    for k = 1:n_viz
        idx = sort_idx(k);
        fprintf('  %6d | %11.3f | %11.3f | %7.3f | %10.4f\n', ...
                k, voxel_positions_world(idx, 1), voxel_positions_world(idx, 2), ...
                abs(voxel_positions_world(idx, 2)), voxel_scores(idx));
    end
    fprintf('\n');
end

fprintf('Visualizing %d configurations...\n\n', length(random_indices));

figure('Position', [100, 100, 1200, 900]);

for i = 1:length(random_indices)
    idx = random_indices(i);
    ix = ix_all(idx);
    iy = iy_all(idx);
    iz = iz_all(idx);
    
    % Get stored configuration
    q_arm = squeeze(qExample(ix, iy, iz, :));
    
    % Build full configuration
    q_full = zeros(num_joints, 1);
    q_full(arm_joint_indices) = q_arm;
    
    % Check collision
    [isColliding, sepDist] = checkCollision(robot, q_full, ...
        'Exhaustive', 'on', 'SkippedSelfCollisions', 'adjacent');
    
    % Get EE position
    T_ee = getTransform(robot, q_full, 'left_gripper_link', 'left_arm_base_link');
    ee_pos = T_ee(1:3, 4);
    
    % Get voxel position in arm base frame
    voxel_pos_arm = [config.grid_origin(1) + (ix - 0.5) * config.voxel_size, ...
                     config.grid_origin(2) + (iy - 0.5) * config.voxel_size, ...
                     config.grid_origin(3) + (iz - 0.5) * config.voxel_size];
    
    % Transform to world frame
    % Arm base transform from URDF: xyz="0.16 0 0.9465" rpy="0 0 -1.5708"
    % Translation
    ARM_TRANSLATION = [0.16, 0, 0.9465]';
    % Rotation: -90° around Z-axis
    % Rz(-90°) = [0  1  0]
    %            [-1 0  0]
    %            [0  0  1]
    ARM_ROTATION = [0, 1, 0; 
                    -1, 0, 0; 
                    0, 0, 1];
    
    % Transform: p_world = R * p_arm + t
    voxel_pos_world = ARM_ROTATION * voxel_pos_arm' + ARM_TRANSLATION;
    voxel_pos_world = voxel_pos_world';
    
    % Chassis bounds in world frame: approximately
    % X: [-0.4, 0.4], Y: [-0.3, 0.3], Z: [0.0, 0.55]
    in_chassis_x = (voxel_pos_world(1) >= -0.4 && voxel_pos_world(1) <= 0.4);
    in_chassis_y = (voxel_pos_world(2) >= -0.3 && voxel_pos_world(2) <= 0.3);
    in_chassis_z = (voxel_pos_world(3) >= 0.0 && voxel_pos_world(3) <= 0.55);
    in_chassis_volume = in_chassis_x && in_chassis_y && in_chassis_z;
    
    % Plot
    subplot(3, 3, i);
    show(robot, q_full, 'Visuals', 'on', 'Collisions', 'off');
    
    % Title with details
    if isColliding
        % Find colliding pairs
        [body1Idx, body2Idx] = find(isnan(sepDist));
        n_collisions = size(unique(sort([body1Idx, body2Idx], 2), 'rows'), 1);
        title(sprintf('Config %d: ❌ COLLISION (%d pairs)', i, n_collisions), ...
              'Color', 'red', 'FontWeight', 'bold');
    elseif in_chassis_volume
        title(sprintf('Config %d: ⚠️ IN CHASSIS VOLUME!', i), ...
              'Color', [1, 0.5, 0], 'FontWeight', 'bold');  % Orange
    else
        title(sprintf('Config %d: ✓ No collision', i), 'Color', 'green');
    end
    
    xlabel(sprintf('World: [%.2f,%.2f,%.2f]', voxel_pos_world));
    view(45, 30);
    axis equal;
    grid on;
    
    % Print details
    fprintf('Config %d at voxel [%d,%d,%d]:\n', i, ix, iy, iz);
    fprintf('  Voxel pos (arm frame): [%.3f, %.3f, %.3f]\n', voxel_pos_arm);
    fprintf('  Voxel pos (world frame): [%.3f, %.3f, %.3f]\n', voxel_pos_world);
    fprintf('  Inside chassis volume: %d\n', in_chassis_volume);
    fprintf('  q_arm = [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]\n', q_arm);
    fprintf('  EE pos (arm frame): [%.3f, %.3f, %.3f]\n', ee_pos);
    fprintf('  Collision: %d\n', isColliding);
    
    if isColliding
        [body1Idx, body2Idx] = find(isnan(sepDist));
        collidingPairs = unique(sort([body1Idx, body2Idx], 2), 'rows');
        fprintf('  Colliding pairs:\n');
        for j = 1:min(3, size(collidingPairs, 1))
            b1 = collidingPairs(j, 1);
            b2 = collidingPairs(j, 2);
            if b1 <= numel(robot.Bodies) && b2 <= numel(robot.Bodies)
                fprintf('    %s <-> %s\n', robot.Bodies{b1}.Name, robot.Bodies{b2}.Name);
            end
        end
    end
    fprintf('\n');
end

sgtitle('Voxels with Most Negative X (Closest to/Behind Arm Base)', ...
        'FontSize', 14, 'FontWeight', 'bold');

fprintf('✓ Visualization complete!\n');
fprintf('  These are voxels with negative/smallest X (closest to arm base/chassis)\n');
fprintf('  Look for any RED titles (collisions) - there should be NONE!\n');
fprintf('  Visually inspect if arm penetrates chassis mesh\n\n');
