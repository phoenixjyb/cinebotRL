function visualize_fk_map(map_file)
% VISUALIZE_FK_MAP - Simple visualization for FK-generated reachability map
%
% Usage:
%   visualize_fk_map()  % Uses default file
%   visualize_fk_map('reach_map_mobile_mm_arm_only.mat')
%
% Controls:
%   - Mouse: Rotate view
%   - Scroll: Zoom

if nargin < 1
    map_file = 'reach_map_mobile_mm_arm_only.mat';
end

fprintf('╔════════════════════════════════════════════════════════╗\n');
fprintf('║  FK REACHABILITY MAP VISUALIZER                        ║\n');
fprintf('╚════════════════════════════════════════════════════════╝\n\n');

% Arm base offset from mobile base (from URDF)
ARM_OFFSET = [0.16, 0.0, 0.9465];  % [x, y, z] in meters

%% Load map
fprintf('Loading map: %s\n', map_file);
if ~exist(map_file, 'file')
    error('Map file not found: %s', map_file);
end

load(map_file, 'reachScore', 'manipMax', 'qExample', 'config', 'metadata');

%% Load robot URDF
% Use MATLAB-compatible URDF with relative mesh paths (not package://)
urdf_matlab = strrep(config.urdf_path, 'base_corrected.urdf', 'matlab.urdf');
if exist(urdf_matlab, 'file')
    urdf_to_load = urdf_matlab;
    fprintf('Loading robot URDF (MATLAB version): %s\n', urdf_matlab);
else
    urdf_to_load = config.urdf_path;
    fprintf('Loading robot URDF: %s\n', config.urdf_path);
end

if ~exist(urdf_to_load, 'file')
    warning('URDF file not found: %s. Skipping robot visualization.', urdf_to_load);
    robot = [];
else
    % Get URDF directory
    urdf_dir = fileparts(urdf_to_load);
    addpath(urdf_dir);
    orig_dir = pwd;
    cd(urdf_dir);
    
    try
        [~, urdf_name, urdf_ext] = fileparts(urdf_to_load);
        robot = importrobot([urdf_name, urdf_ext]);
        robot.DataFormat = 'column';
        
        % Count loaded meshes
        mesh_count = 0;
        for i = 1:numel(robot.Bodies)
            if ~isempty(robot.Bodies{i}.Visuals)
                mesh_count = mesh_count + numel(robot.Bodies{i}.Visuals);
            end
        end
        
        fprintf('  ✓ Robot loaded: %d bodies, %d meshes\n', numel(robot.Bodies), mesh_count);
    catch ME
        warning('visualize:robotLoadFailed', 'Failed to load robot: %s', ME.message);
        robot = [];
    end
    
    cd(orig_dir);
end

fprintf('  Grid: %dx%dx%d voxels\n', config.grid_dims);
fprintf('  Origin: [%.2f, %.2f, %.2f] m\n', config.grid_origin);
fprintf('  Voxel size: %.3f m\n', config.voxel_size);
fprintf('  Reachable voxels: %d / %d (%.1f%%)\n', ...
        metadata.n_reachable_voxels, prod(config.grid_dims), ...
        100*metadata.n_reachable_voxels/prod(config.grid_dims));

%% Create figure
fig = figure('Name', 'FK Reachability Map', 'NumberTitle', 'off');
fig.Position = [100 100 1200 800];

%% Plot 1: Reachability voxel cloud with robot
subplot(1, 2, 1);
plot_reach_cloud(reachScore, config, ARM_OFFSET, robot, qExample);
title('Reachable Workspace (World Frame)');

%% Plot 2: Manipulability heatmap with robot
subplot(1, 2, 2);
plot_manip_cloud(reachScore, manipMax, config, ARM_OFFSET, robot, qExample);
title('Manipulability Index (World Frame)');

fprintf('\n✓ Visualization complete!\n');
fprintf('  - Left: Binary reachability (red = reachable)\n');
fprintf('  - Right: Manipulability (blue=low, red=high)\n');
fprintf('  - Robot shown at home configuration\n');
fprintf('  - Frame: World frame (chassis at ground, cloud centered at arm shoulder)\n\n');

end


function plot_reach_cloud(reachScore, config, arm_offset, robot, qExample)
% Plot reachability cloud in ARM BASE FRAME with robot in WORLD FRAME
% Cloud: relative to arm shoulder (0,0,0 = shoulder)
% Robot: shows full mobile base + arm (chassis at ground)

% Get grid parameters
nx = config.grid_dims(1);
ny = config.grid_dims(2);
nz = config.grid_dims(3);
origin = config.grid_origin;
voxel = config.voxel_size;

% Find reachable voxels
[ix, iy, iz] = ind2sub([nx, ny, nz], find(reachScore > 0));

% Convert voxel indices to ARM BASE FRAME coordinates
% This is the workspace relative to the shoulder
x_arm = origin(1) + (ix - 0.5) * voxel;
y_arm = origin(2) + (iy - 0.5) * voxel;
z_arm = origin(3) + (iz - 0.5) * voxel;

% Show robot in WORLD FRAME (chassis at ground)
if ~isempty(robot)
    q_home = homeConfiguration(robot);
    show(robot, q_home, 'Visuals', 'on', 'Collisions', 'off', 'Frames', 'off');
    hold on;
    fprintf('  Robot mesh rendered in world frame\n');
end

% Transform point cloud from ARM BASE to WORLD FRAME for visualization
% Arm base transform from URDF: xyz="0.16 0 0.9465" rpy="0 0 -1.5708"
% Translation
ARM_TRANSLATION = [0.16; 0; 0.9465];
% Rotation: -90° around Z-axis
% Rz(-90°) = [0  1  0]
%            [-1 0  0]
%            [0  0  1]
ARM_ROTATION = [0, 1, 0; 
                -1, 0, 0; 
                0, 0, 1];

% Transform each point: p_world = R * p_arm + t
points_arm = [x_arm(:), y_arm(:), z_arm(:)]';  % 3xN matrix
points_world = ARM_ROTATION * points_arm + ARM_TRANSLATION;
x_world = points_world(1, :)';
y_world = points_world(2, :)';
z_world = points_world(3, :)';

% Plot reachability cloud in world frame to match robot
scatter3(x_world, y_world, z_world, 10, 'r', 'filled', 'MarkerFaceAlpha', 0.3);

% Mark mobile base (chassis at ground, world origin)
scatter3(0, 0, 0, 150, 'k', 'filled', 'p', 'MarkerEdgeColor', 'w', 'LineWidth', 2);
text(0, 0, -0.1, 'Mobile Base', 'FontSize', 10, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

% Mark arm base (shoulder in world frame)
arm_base_world = ARM_TRANSLATION;  % Just the translation part for marker
scatter3(arm_base_world(1), arm_base_world(2), arm_base_world(3), 100, 'b', 'filled', ...
         'MarkerEdgeColor', 'w', 'LineWidth', 2);
text(arm_base_world(1), arm_base_world(2), arm_base_world(3)+0.1, 'Arm Base', ...
     'FontSize', 10, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

% Grid box in WORLD FRAME
% Define 8 corners in arm base frame then transform to world
corners_arm = [
    origin(1), origin(2), origin(3);
    origin(1) + nx*voxel, origin(2), origin(3);
    origin(1) + nx*voxel, origin(2) + ny*voxel, origin(3);
    origin(1), origin(2) + ny*voxel, origin(3);
    origin(1), origin(2), origin(3) + nz*voxel;
    origin(1) + nx*voxel, origin(2), origin(3) + nz*voxel;
    origin(1) + nx*voxel, origin(2) + ny*voxel, origin(3) + nz*voxel;
    origin(1), origin(2) + ny*voxel, origin(3) + nz*voxel;
]';  % 3x8 matrix

% Transform to world frame
corners_world = ARM_ROTATION * corners_arm + ARM_TRANSLATION;
corners = corners_world';  % 8x3 matrix
plot_box(corners);

hold off;
axis equal;
grid on;
xlabel('X (m)');
ylabel('Y (m)');
zlabel('Z (m)');
view(45, 30);

end


function plot_manip_cloud(reachScore, manipMax, config, arm_offset, robot, ~)
% Plot manipulability cloud in ARM BASE FRAME with robot in WORLD FRAME

% Get grid parameters
nx = config.grid_dims(1);
ny = config.grid_dims(2);
nz = config.grid_dims(3);
origin = config.grid_origin;
voxel = config.voxel_size;

% Find reachable voxels
reachable_idx = find(reachScore > 0);
[ix, iy, iz] = ind2sub([nx, ny, nz], reachable_idx);

% Get manipulability values
manip_vals = manipMax(reachable_idx);

% Convert to ARM BASE FRAME coordinates
x_arm = origin(1) + (ix - 0.5) * voxel;
y_arm = origin(2) + (iy - 0.5) * voxel;
z_arm = origin(3) + (iz - 0.5) * voxel;

% Show robot in WORLD FRAME (chassis at ground)
if ~isempty(robot)
    q_home = homeConfiguration(robot);
    show(robot, q_home, 'Visuals', 'on', 'Collisions', 'off', 'Frames', 'on');
    hold on;
end

% Transform point cloud to WORLD FRAME for visualization
% Arm base transform from URDF: xyz="0.16 0 0.9465" rpy="0 0 -1.5708"
ARM_TRANSLATION = [0.16; 0; 0.9465];
ARM_ROTATION = [0, 1, 0; -1, 0, 0; 0, 0, 1];

% Transform: p_world = R * p_arm + t
points_arm = [x_arm(:), y_arm(:), z_arm(:)]';
points_world = ARM_ROTATION * points_arm + ARM_TRANSLATION;
x_world = points_world(1, :)';
y_world = points_world(2, :)';
z_world = points_world(3, :)';

% Plot manipulability cloud in world frame
scatter3(x_world, y_world, z_world, 10, manip_vals, 'filled', 'MarkerFaceAlpha', 0.5);

colormap('jet');
cb = colorbar;
cb.Label.String = 'Manipulability Index';

% Mark mobile base (chassis at ground)
scatter3(0, 0, 0, 150, 'k', 'filled', 'p', 'MarkerEdgeColor', 'w', 'LineWidth', 2);
text(0, 0, -0.1, 'Mobile Base', 'FontSize', 10, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

% Mark arm base (shoulder in world frame)
arm_base_world = ARM_TRANSLATION;
scatter3(arm_base_world(1), arm_base_world(2), arm_base_world(3), 100, 'b', 'filled', ...
         'MarkerEdgeColor', 'w', 'LineWidth', 2);
text(arm_base_world(1), arm_base_world(2), arm_base_world(3)+0.1, 'Arm Base', ...
     'FontSize', 10, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

% Grid box in WORLD FRAME
% Define 8 corners in arm base frame then transform to world
corners_arm = [
    origin(1), origin(2), origin(3);
    origin(1) + nx*voxel, origin(2), origin(3);
    origin(1) + nx*voxel, origin(2) + ny*voxel, origin(3);
    origin(1), origin(2) + ny*voxel, origin(3);
    origin(1), origin(2), origin(3) + nz*voxel;
    origin(1) + nx*voxel, origin(2), origin(3) + nz*voxel;
    origin(1) + nx*voxel, origin(2) + ny*voxel, origin(3) + nz*voxel;
    origin(1), origin(2) + ny*voxel, origin(3) + nz*voxel;
]';  % 3x8 matrix

% Transform to world frame
corners_world = ARM_ROTATION * corners_arm + ARM_TRANSLATION;
corners = corners_world';  % 8x3 matrix
plot_box(corners);

hold off;
axis equal;
grid on;
xlabel('X (m)');
ylabel('Y (m)');
zlabel('Z (m)');
view(45, 30);

end


function plot_box(corners)
% Plot wireframe box from 8 corners

% Bottom face
plot3([corners(1,1) corners(2,1)], [corners(1,2) corners(2,2)], [corners(1,3) corners(2,3)], 'k--', 'LineWidth', 1);
plot3([corners(2,1) corners(3,1)], [corners(2,2) corners(3,2)], [corners(2,3) corners(3,3)], 'k--', 'LineWidth', 1);
plot3([corners(3,1) corners(4,1)], [corners(3,2) corners(4,2)], [corners(3,3) corners(4,3)], 'k--', 'LineWidth', 1);
plot3([corners(4,1) corners(1,1)], [corners(4,2) corners(1,2)], [corners(4,3) corners(1,3)], 'k--', 'LineWidth', 1);

% Top face
plot3([corners(5,1) corners(6,1)], [corners(5,2) corners(6,2)], [corners(5,3) corners(6,3)], 'k--', 'LineWidth', 1);
plot3([corners(6,1) corners(7,1)], [corners(6,2) corners(7,2)], [corners(6,3) corners(7,3)], 'k--', 'LineWidth', 1);
plot3([corners(7,1) corners(8,1)], [corners(7,2) corners(8,2)], [corners(7,3) corners(8,3)], 'k--', 'LineWidth', 1);
plot3([corners(8,1) corners(5,1)], [corners(8,2) corners(5,2)], [corners(8,3) corners(5,3)], 'k--', 'LineWidth', 1);

% Vertical edges
plot3([corners(1,1) corners(5,1)], [corners(1,2) corners(5,2)], [corners(1,3) corners(5,3)], 'k--', 'LineWidth', 1);
plot3([corners(2,1) corners(6,1)], [corners(2,2) corners(6,2)], [corners(2,3) corners(6,3)], 'k--', 'LineWidth', 1);
plot3([corners(3,1) corners(7,1)], [corners(3,2) corners(7,2)], [corners(3,3) corners(7,3)], 'k--', 'LineWidth', 1);
plot3([corners(4,1) corners(8,1)], [corners(4,2) corners(8,2)], [corners(4,3) corners(8,3)], 'k--', 'LineWidth', 1);

end
