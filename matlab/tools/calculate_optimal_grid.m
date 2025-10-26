function [optimal_origin, optimal_size] = calculate_optimal_grid(urdf_path, base_link, ee_link)
% CALCULATE_OPTIMAL_GRID - Compute tight bounding box for arm workspace
%
% This function samples the arm's configuration space and finds the actual
% minimum bounding box for the reachable workspace. Use this to optimize
% GRID_ORIGIN and GRID_SIZE in build_reachability_map.m
%
% Usage:
%   [origin, size] = calculate_optimal_grid( ...
%       'C:\...\mobile_manipulator_PPR_base_corrected.urdf', ...
%       'left_arm_base_link', ...
%       'left_gripper_link')
%
% Returns:
%   optimal_origin - [x_min, y_min, z_min] in arm base frame
%   optimal_size   - [dx, dy, dz] workspace dimensions

fprintf('\n');
fprintf('╔═══════════════════════════════════════════════════════════╗\n');
fprintf('║  OPTIMAL GRID CALCULATOR FOR ARM WORKSPACE               ║\n');
fprintf('╚═══════════════════════════════════════════════════════════╝\n');
fprintf('\n');

%% Load robot
fprintf('Loading URDF: %s\n', urdf_path);
robot_full = importrobot(urdf_path, 'DataFormat', 'row');

% Extract arm subtree (exclude mobile base virtual joints)
fprintf('  Extracting arm subtree from "%s"\n', base_link);
robot = robotics.RigidBodyTree('DataFormat', 'row');
robot.Gravity = [0 0 -9.81];

% Copy arm chain from full robot
body_names = {};
current_body = robot_full.getBody(ee_link);
while ~strcmp(current_body.Name, base_link)
    body_names = [{current_body.Name}, body_names];
    if isempty(current_body.Parent)
        break;
    end
    current_body = robot_full.getBody(current_body.Parent.Name);
end

% Build arm-only tree
for i = 1:numel(body_names)
    body = copy(robot_full.getBody(body_names{i}));
    if i == 1
        robot.addBody(body, base_link);
    else
        robot.addBody(body, body_names{i-1});
    end
end

fprintf('  Arm DOF: %d\n', robot.NumBodies);
fprintf('  Joint limits:\n');
for i = 1:robot.NumBodies
    jnt = robot.Bodies{i}.Joint;
    if ~strcmp(jnt.Type, 'fixed')
        fprintf('    %s: [%.2f, %.2f] rad\n', ...
            jnt.Name, jnt.PositionLimits(1), jnt.PositionLimits(2));
    end
end

%% Sample configuration space
fprintf('\nSampling configuration space...\n');

% Number of samples (more = better coverage, but slower)
N_samples = 10000;  % 10K samples should give good coverage

% Collect joint limits
q_limits = zeros(robot.NumBodies, 2);
for i = 1:robot.NumBodies
    jnt = robot.Bodies{i}.Joint;
    if ~strcmp(jnt.Type, 'fixed')
        q_limits(i, :) = jnt.PositionLimits;
    end
end

% Random sampling with collision check
ee_positions = zeros(N_samples, 3);
valid_count = 0;

fprintf('  Generating %d random configurations...\n', N_samples);
for i = 1:N_samples
    % Random joint configuration within limits
    q = q_limits(:, 1) + rand(robot.NumBodies, 1) .* diff(q_limits, 1, 2);
    
    % Get EE position
    config = robot.homeConfiguration;
    for j = 1:robot.NumBodies
        config(j).JointPosition = q(j);
    end
    
    % Check self-collision (basic)
    try
        tform = getTransform(robot, config, ee_link, base_link);
        ee_pos = tform(1:3, 4)';
        
        % Simple validity check (not too close to base)
        if norm(ee_pos) > 0.1  % At least 10cm from shoulder
            valid_count = valid_count + 1;
            ee_positions(valid_count, :) = ee_pos;
        end
    catch
        % Invalid configuration, skip
        continue;
    end
    
    if mod(i, 1000) == 0
        fprintf('    Progress: %d/%d samples (%.1f%% valid)\n', ...
            i, N_samples, 100*valid_count/i);
    end
end

% Trim to valid samples
ee_positions = ee_positions(1:valid_count, :);
fprintf('  ✅ Generated %d valid EE positions\n', valid_count);

%% Compute bounding box
fprintf('\nComputing workspace bounding box...\n');

x_min = min(ee_positions(:, 1));
x_max = max(ee_positions(:, 1));
y_min = min(ee_positions(:, 2));
y_max = max(ee_positions(:, 2));
z_min = min(ee_positions(:, 3));
z_max = max(ee_positions(:, 3));

fprintf('  Raw workspace bounds:\n');
fprintf('    X: [%.3f, %.3f] m (range: %.3f m)\n', x_min, x_max, x_max - x_min);
fprintf('    Y: [%.3f, %.3f] m (range: %.3f m)\n', y_min, y_max, y_max - y_min);
fprintf('    Z: [%.3f, %.3f] m (range: %.3f m)\n', z_min, z_max, z_max - z_min);

% Add margin for safety (5% on each side)
margin_factor = 0.05;
x_margin = (x_max - x_min) * margin_factor;
y_margin = (y_max - y_min) * margin_factor;
z_margin = (z_max - z_min) * margin_factor;

optimal_origin = [x_min - x_margin, y_min - y_margin, z_min - z_margin];
optimal_size = [(x_max - x_min) + 2*x_margin, ...
                (y_max - y_min) + 2*y_margin, ...
                (z_max - z_min) + 2*z_margin];

fprintf('\n');
fprintf('╔═══════════════════════════════════════════════════════════╗\n');
fprintf('║  RECOMMENDED GRID PARAMETERS (with 5%% margin)            ║\n');
fprintf('╚═══════════════════════════════════════════════════════════╝\n');
fprintf('\n');
fprintf('GRID_ORIGIN = [%.3f, %.3f, %.3f];  %% [x_min, y_min, z_min]\n', ...
    optimal_origin(1), optimal_origin(2), optimal_origin(3));
fprintf('GRID_SIZE   = [%.3f, %.3f, %.3f];  %% [dx, dy, dz]\n', ...
    optimal_size(1), optimal_size(2), optimal_size(3));
fprintf('\n');

% Compare with current values
current_origin = [-0.6, -0.8, -0.4];
current_size = [1.2, 1.6, 1.0];

fprintf('Comparison with current settings:\n');
fprintf('  Current GRID_ORIGIN = [%.2f, %.2f, %.2f]\n', current_origin);
fprintf('  Current GRID_SIZE   = [%.2f, %.2f, %.2f]\n', current_size);
fprintf('\n');

volume_current = prod(current_size);
volume_optimal = prod(optimal_size);
fprintf('  Current volume: %.3f m³\n', volume_current);
fprintf('  Optimal volume: %.3f m³\n', volume_optimal);
fprintf('  Reduction: %.1f%%\n', 100*(1 - volume_optimal/volume_current));
fprintf('\n');

% Estimate voxel counts
voxel_size = 0.05;  % 5cm
n_current = prod(round(current_size / voxel_size));
n_optimal = prod(round(optimal_size / voxel_size));

fprintf('  Voxel count @ 5cm resolution:\n');
fprintf('    Current: %d voxels\n', n_current);
fprintf('    Optimal: %d voxels\n', n_optimal);
fprintf('    Speedup: %.1fx faster build!\n', n_current / n_optimal);
fprintf('\n');

%% Visualize
fprintf('Visualizing workspace...\n');
figure('Name', 'Arm Workspace Analysis', 'Position', [100, 100, 1200, 800]);

subplot(2, 2, 1);
scatter3(ee_positions(:, 1), ee_positions(:, 2), ee_positions(:, 3), 1, 'b', 'filled');
hold on;
% Current grid
plot_box(current_origin, current_size, 'r--', 2, 'Current Grid');
% Optimal grid
plot_box(optimal_origin, optimal_size, 'g-', 2, 'Optimal Grid');
scatter3(0, 0, 0, 100, 'r', 'p', 'filled');  % Shoulder
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('3D Workspace');
legend('EE Samples', 'Current Grid', 'Optimal Grid', 'Shoulder');
grid on; axis equal; view(45, 30);

subplot(2, 2, 2);
scatter(ee_positions(:, 1), ee_positions(:, 2), 1, 'b', 'filled');
hold on;
rectangle('Position', [current_origin(1), current_origin(2), current_size(1), current_size(2)], ...
          'EdgeColor', 'r', 'LineStyle', '--', 'LineWidth', 2);
rectangle('Position', [optimal_origin(1), optimal_origin(2), optimal_size(1), optimal_size(2)], ...
          'EdgeColor', 'g', 'LineWidth', 2);
scatter(0, 0, 100, 'r', 'p', 'filled');
xlabel('X (m)'); ylabel('Y (m)');
title('Top View (XY)');
grid on; axis equal;

subplot(2, 2, 3);
scatter(ee_positions(:, 1), ee_positions(:, 3), 1, 'b', 'filled');
hold on;
rectangle('Position', [current_origin(1), current_origin(3), current_size(1), current_size(3)], ...
          'EdgeColor', 'r', 'LineStyle', '--', 'LineWidth', 2);
rectangle('Position', [optimal_origin(1), optimal_origin(3), optimal_size(1), optimal_size(3)], ...
          'EdgeColor', 'g', 'LineWidth', 2);
scatter(0, 0, 100, 'r', 'p', 'filled');
xlabel('X (m)'); ylabel('Z (m)');
title('Side View (XZ)');
grid on; axis equal;

subplot(2, 2, 4);
scatter(ee_positions(:, 2), ee_positions(:, 3), 1, 'b', 'filled');
hold on;
rectangle('Position', [current_origin(2), current_origin(3), current_size(2), current_size(3)], ...
          'EdgeColor', 'r', 'LineStyle', '--', 'LineWidth', 2);
rectangle('Position', [optimal_origin(2), optimal_origin(3), optimal_size(2), optimal_size(3)], ...
          'EdgeColor', 'g', 'LineWidth', 2);
scatter(0, 0, 100, 'r', 'p', 'filled');
xlabel('Y (m)'); ylabel('Z (m)');
title('Front View (YZ)');
grid on; axis equal;

fprintf('\n✅ Analysis complete! Update build_reachability_map.m with recommended values.\n');
fprintf('\n');

end

function plot_box(origin, size, style, width, label)
    % Helper to plot 3D bounding box
    x = origin(1) + [0 size(1) size(1) 0 0 0; size(1) size(1) 0 0 size(1) size(1); ...
                     size(1) size(1) 0 0 size(1) size(1); 0 size(1) size(1) 0 0 0];
    y = origin(2) + [0 0 0 0 0 0; 0 size(2) size(2) 0 0 0; 0 size(2) size(2) 0 size(2) size(2); ...
                     0 0 size(2) size(2) size(2) size(2)];
    z = origin(3) + [0 0 0 0 0 size(3); 0 0 0 0 0 size(3); size(3) size(3) size(3) size(3) size(3) size(3); ...
                     size(3) size(3) size(3) size(3) 0 0];
    
    for i = 1:4
        plot3(x(i, :), y(i, :), z(i, :), style, 'LineWidth', width, 'DisplayName', label);
        if i == 1, hold on; end
    end
end
