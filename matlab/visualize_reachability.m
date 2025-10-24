function visualize_reachability(map_file, urdf_path)
% VISUALIZE_REACHABILITY - Interactive 3D visualization of reachability map
%
% Displays the robot model and reachability map with multiple visualization modes:
%   - Voxel clouds (colored by reach score or manipulability)
%   - Slice planes (horizontal/vertical cuts through workspace)
%   - Interactive query mode (click to test reachability)
%   - Robot configuration display
%
% Usage:
%   visualize_reachability()  % Uses defaults
%   visualize_reachability('reach_map.mat', 'robot.urdf')
%
% Controls:
%   - Mouse: Rotate view
%   - Scroll: Zoom
%   - Arrow keys: Change slice plane
%   - Number keys 1-5: Switch visualization modes

if nargin < 1
    map_file = 'reach_map_arm.mat';
end
if nargin < 2
    urdf_path = 'C:\Users\yanbo\wSpace\cinebotRL\assets_own\mobile_manipulator_PPR_base_corrected.urdf';
end

% Arm base offset (from URDF: left_arm_base_link relative to abstract_chassis_link)
ARM_OFFSET = [0.16, 0.0, 0.9465];  % [x, y, z] in meters

fprintf('╔════════════════════════════════════════════════════════╗\n');
fprintf('║  REACHABILITY MAP VISUALIZER                          ║\n');
fprintf('╚════════════════════════════════════════════════════════╝\n\n');

%% Load map
fprintf('Loading map: %s\n', map_file);
if ~exist(map_file, 'file')
    error('Map file not found: %s\nRun build_reachability_map() first!', map_file);
end
S = load(map_file, 'map');
map = S.map;

fprintf('  Grid: %dx%dx%d voxels\n', map.grid.shape);
fprintf('  Origin: [%.2f, %.2f, %.2f] m\n', map.grid.origin);
fprintf('  Voxel size: %.0f mm\n', 1000*map.grid.voxel(1));

% Statistics
reachable = map.data.reachScore > 0;
fprintf('  Reachable: %d / %d (%.1f%%)\n', sum(reachable(:)), numel(reachable), 100*mean(reachable(:)));

%% Load robot
fprintf('\nLoading robot: %s\n', urdf_path);
if ~exist(urdf_path, 'file')
    warning('URDF not found: %s\nSkipping robot visualization', urdf_path);
    robot = [];
else
    try
        % Load FULL robot (with mobile base) for visualization
        robot = importrobot(urdf_path, DataFormat='row');
        fprintf('  DOF: %d (including 3 virtual mobile base joints)\n', length(robot.homeConfiguration));
        fprintf('  End-effector: %s\n', map.meta.eeLink);
    catch ME
        warning('CinebotRL:VisualizeFailed', 'Failed to load robot: %s', ME.message);
        robot = [];
    end
end

%% Create figure
fig = figure('Name', 'Reachability Map Visualizer', ...
             'NumberTitle', 'off', ...
             'Position', [100 100 1400 800], ...
             'Color', 'w');

% Create UI panel
uipanel('Parent', fig, 'Position', [0.02 0.02 0.15 0.96], ...
        'Title', 'Controls', 'FontSize', 10, 'FontWeight', 'bold');

% Add buttons
btn_y = 0.92;
btn_h = 0.06;
btn_gap = 0.01;

% Visualization mode buttons
modes = {'Voxel Cloud', 'Reach Score Slice', 'Manipulability Slice', ...
         'Top View', 'Robot + Reach'};
btn_handles = zeros(length(modes), 1);
for i = 1:length(modes)
    btn_handles(i) = uicontrol('Parent', fig, 'Style', 'pushbutton', ...
        'String', modes{i}, ...
        'Units', 'normalized', ...
        'Position', [0.03 btn_y-(i-1)*(btn_h+btn_gap) 0.13 btn_h], ...
        'FontSize', 9, ...
        'Callback', @(src,evt) update_viz(i));
end

% Slice height slider (for slice modes)
uicontrol('Parent', fig, 'Style', 'text', ...
    'String', 'Slice Height (m):', ...
    'Units', 'normalized', ...
    'Position', [0.03 btn_y-6*(btn_h+btn_gap) 0.13 0.04], ...
    'FontSize', 9, 'HorizontalAlignment', 'left');

slice_slider = uicontrol('Parent', fig, 'Style', 'slider', ...
    'Min', map.grid.origin(3), ...
    'Max', map.grid.origin(3) + map.grid.size(3), ...
    'Value', map.grid.origin(3) + map.grid.size(3)/2, ...
    'Units', 'normalized', ...
    'Position', [0.03 btn_y-7*(btn_h+btn_gap) 0.13 0.03], ...
    'Callback', @(src,evt) update_slice());

slice_text = uicontrol('Parent', fig, 'Style', 'text', ...
    'String', sprintf('%.2f m', slice_slider.Value), ...
    'Units', 'normalized', ...
    'Position', [0.03 btn_y-7.5*(btn_h+btn_gap) 0.13 0.03], ...
    'FontSize', 9);

% Threshold slider (for filtering low scores)
uicontrol('Parent', fig, 'Style', 'text', ...
    'String', 'Min Reach Score:', ...
    'Units', 'normalized', ...
    'Position', [0.03 btn_y-9*(btn_h+btn_gap) 0.13 0.04], ...
    'FontSize', 9, 'HorizontalAlignment', 'left');

thresh_slider = uicontrol('Parent', fig, 'Style', 'slider', ...
    'Min', 0, 'Max', 1, 'Value', 0.3, ...
    'Units', 'normalized', ...
    'Position', [0.03 btn_y-10*(btn_h+btn_gap) 0.13 0.03], ...
    'Callback', @(src,evt) update_viz(current_mode));

thresh_text = uicontrol('Parent', fig, 'Style', 'text', ...
    'String', sprintf('%.2f', thresh_slider.Value), ...
    'Units', 'normalized', ...
    'Position', [0.03 btn_y-10.5*(btn_h+btn_gap) 0.13 0.03], ...
    'FontSize', 9);

% Statistics text
stats_text = uicontrol('Parent', fig, 'Style', 'text', ...
    'String', '', ...
    'Units', 'normalized', ...
    'Position', [0.03 0.05 0.13 0.15], ...
    'FontSize', 8, 'HorizontalAlignment', 'left', ...
    'BackgroundColor', 'w');

% Main axes
ax = axes('Parent', fig, 'Position', [0.20 0.10 0.75 0.85]);
hold(ax, 'on');
grid(ax, 'on');
axis(ax, 'equal');
xlabel(ax, 'X (m)');
ylabel(ax, 'Y (m)');
zlabel(ax, 'Z (m)');
view(ax, 3);

% State
current_mode = 5;  % Default: Robot + Reach
update_viz(current_mode);

%% Nested callback functions
    function update_viz(mode)
        current_mode = mode;
        cla(ax);
        hold(ax, 'on');
        
        threshold = thresh_slider.Value;
        
        % Arm offset for transforming map coords to world coords
        arm_offset = ARM_OFFSET;
        
        switch mode
            case 1  % Voxel Cloud
                plot_voxel_cloud(ax, map, threshold, 'reach', arm_offset);
                title(ax, sprintf('Reachability Voxel Cloud (threshold=%.2f)', threshold));
                
            case 2  % Reach Score Slice
                slice_height = slice_slider.Value;
                plot_slice(ax, map, slice_height, 'reach', threshold, arm_offset);
                title(ax, sprintf('Reach Score at Z=%.2fm (world)', slice_height));
                
            case 3  % Manipulability Slice
                slice_height = slice_slider.Value;
                plot_slice(ax, map, slice_height, 'manip', threshold, arm_offset);
                title(ax, sprintf('Manipulability at Z=%.2fm (world)', slice_height));
                
            case 4  % Top View
                plot_top_view(ax, map, threshold, arm_offset);
                title(ax, 'Workspace Top View (max over height)');
                view(ax, 0, 90);  % Top-down
                
            case 5  % Robot + Reach
                if ~isempty(robot)
                    plot_robot_with_reach(ax, robot, map, threshold, arm_offset);
                    title(ax, 'Robot Model with Reachable Workspace');
                else
                    plot_voxel_cloud(ax, map, threshold, 'reach', arm_offset);
                    title(ax, 'Reachable Workspace (robot model not available)');
                end
        end
        
        % Update axes limits (in world frame)
        margin = 0.2;
        % Map bounds in arm frame, transformed to world
        x_min_world = map.grid.origin(1) + arm_offset(1) - margin;
        x_max_world = map.grid.origin(1) + map.grid.size(1) + arm_offset(1) + margin;
        y_min_world = map.grid.origin(2) + arm_offset(2) - margin;
        y_max_world = map.grid.origin(2) + map.grid.size(2) + arm_offset(2) + margin;
        z_min_world = min(0, map.grid.origin(3) + arm_offset(3)) - margin;  % Include ground (z=0)
        z_max_world = map.grid.origin(3) + map.grid.size(3) + arm_offset(3) + margin;
        
        xlim(ax, [x_min_world, x_max_world]);
        ylim(ax, [y_min_world, y_max_world]);
        zlim(ax, [z_min_world, z_max_world]);
        
        % Update statistics
        update_stats(threshold);
        
        drawnow;
    end

    function update_slice()
        slice_text.String = sprintf('%.2f m', slice_slider.Value);
        if current_mode == 2 || current_mode == 3
            update_viz(current_mode);
        end
    end

    function update_stats(threshold)
        valid = map.data.reachScore >= threshold;
        n_valid = sum(valid(:));
        pct = 100 * n_valid / numel(valid);
        
        if n_valid > 0
            mean_reach = mean(map.data.reachScore(valid));
            mean_manip = mean(map.data.manipMax(valid));
        else
            mean_reach = 0;
            mean_manip = 0;
        end
        
        stats_str = sprintf(['Statistics:\n\n' ...
                             'Threshold: %.2f\n' ...
                             'Valid voxels: %d\n' ...
                             'Percentage: %.1f%%\n\n' ...
                             'Mean reach: %.3f\n' ...
                             'Mean manip: %.3f'], ...
                             threshold, n_valid, pct, mean_reach, mean_manip);
        stats_text.String = stats_str;
    end

end % main function

%% Visualization helper functions
function plot_voxel_cloud(ax, map, threshold, mode, arm_offset)
% Plot 3D scatter of reachable voxels (transformed to world frame)

% Get voxel centers in ARM FRAME
[nx, ny, nz] = deal(double(map.grid.shape(1)), double(map.grid.shape(2)), double(map.grid.shape(3)));
[X, Y, Z] = ndgrid( ...
    map.grid.origin(1) + (0:nx-1)*map.grid.voxel(1) + map.grid.voxel(1)/2, ...
    map.grid.origin(2) + (0:ny-1)*map.grid.voxel(2) + map.grid.voxel(2)/2, ...
    map.grid.origin(3) + (0:nz-1)*map.grid.voxel(3) + map.grid.voxel(3)/2);

% Transform to WORLD FRAME (add arm offset)
X = X + arm_offset(1);
Y = Y + arm_offset(2);
Z = Z + arm_offset(3);

% Filter by threshold
if strcmp(mode, 'reach')
    values = map.data.reachScore;
else
    values = map.data.manipMax;
end
valid = values >= threshold;

X = X(valid);
Y = Y(valid);
Z = Z(valid);
C = values(valid);

if isempty(X)
    text(ax, 0, 0, 0, 'No voxels above threshold!', 'FontSize', 14, 'Color', 'r');
    return;
end

% Plot as scatter
scatter3(ax, X(:), Y(:), Z(:), 20, C(:), 'filled', 'MarkerFaceAlpha', 0.6);
colormap(ax, 'jet');
colorbar(ax);
if strcmp(mode, 'reach')
    caxis(ax, [0 1]);
    colorbar(ax, 'Label', 'Reach Score');
else
    colorbar(ax, 'Label', 'Manipulability');
end

% Mark arm base origin in WORLD FRAME (shoulder position)
shoulder_world = arm_offset;
plot3(ax, shoulder_world(1), shoulder_world(2), shoulder_world(3), 'r*', 'MarkerSize', 25, 'LineWidth', 3);
plot3(ax, shoulder_world(1), shoulder_world(2), shoulder_world(3), 'ro', 'MarkerSize', 15, 'LineWidth', 2);
text(ax, shoulder_world(1), shoulder_world(2), shoulder_world(3)+0.08, ...
     'ARM BASE (Shoulder)', 'FontSize', 11, 'Color', 'r', 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

% Draw coordinate axes at arm base (in world frame)
axis_len = 0.15;
quiver3(ax, shoulder_world(1), shoulder_world(2), shoulder_world(3), axis_len, 0, 0, 'r', 'LineWidth', 2, 'MaxHeadSize', 0.5);
quiver3(ax, shoulder_world(1), shoulder_world(2), shoulder_world(3), 0, axis_len, 0, 'g', 'LineWidth', 2, 'MaxHeadSize', 0.5);
quiver3(ax, shoulder_world(1), shoulder_world(2), shoulder_world(3), 0, 0, axis_len, 'b', 'LineWidth', 2, 'MaxHeadSize', 0.5);
text(ax, shoulder_world(1)+axis_len+0.02, shoulder_world(2), shoulder_world(3), 'X', 'Color', 'r', 'FontSize', 10, 'FontWeight', 'bold');
text(ax, shoulder_world(1), shoulder_world(2)+axis_len+0.02, shoulder_world(3), 'Y', 'Color', 'g', 'FontSize', 10, 'FontWeight', 'bold');
text(ax, shoulder_world(1), shoulder_world(2), shoulder_world(3)+axis_len+0.02, 'Z', 'Color', 'b', 'FontSize', 10, 'FontWeight', 'bold');

% Mark mobile base origin (ground level)
plot3(ax, 0, 0, 0, 'ko', 'MarkerSize', 12, 'LineWidth', 2);
plot3(ax, 0, 0, 0, 'k+', 'MarkerSize', 15, 'LineWidth', 2);
text(ax, 0, 0, -0.05, 'Mobile Base', 'FontSize', 9, 'Color', 'k', 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
end

function plot_slice(ax, map, slice_height, mode, threshold, arm_offset)
% Plot 2D slice at given height (in world frame)

% Convert slice height from world to arm frame
slice_height_arm = slice_height - arm_offset(3);

% Find closest Z index
z_idx = round((slice_height_arm - map.grid.origin(3)) / map.grid.voxel(3)) + 1;
z_idx = max(1, min(double(map.grid.shape(3)), z_idx));

% Extract slice
if strcmp(mode, 'reach')
    slice_data = map.data.reachScore(:, :, z_idx);
    clabel = 'Reach Score';
    clim = [0 1];
else
    slice_data = map.data.manipMax(:, :, z_idx);
    clabel = 'Manipulability';
    clim = [0 max(map.data.manipMax(:))];
end

% Apply threshold mask
slice_data(slice_data < threshold) = NaN;

% Create grid for plotting in ARM FRAME
[nx, ny] = deal(double(map.grid.shape(1)), double(map.grid.shape(2)));
x = map.grid.origin(1) + (0:nx-1)*map.grid.voxel(1) + map.grid.voxel(1)/2;
y = map.grid.origin(2) + (0:ny-1)*map.grid.voxel(2) + map.grid.voxel(2)/2;
[X, Y] = meshgrid(y, x);

% Transform to WORLD FRAME
X = X + arm_offset(1);
Y = Y + arm_offset(2);

% Plot as surface (at world height)
surf(ax, X, Y, ones(size(X))*slice_height, slice_data, 'EdgeColor', 'none');
colormap(ax, 'jet');
colorbar(ax, 'Label', clabel);
caxis(ax, clim);
alpha(ax, 0.8);

% Mark arm base origin (in world frame)
shoulder_world = arm_offset;
plot3(ax, shoulder_world(1), shoulder_world(2), slice_height, 'r*', 'MarkerSize', 20, 'LineWidth', 2);
end

function plot_top_view(ax, map, threshold, arm_offset)
% Plot maximum reachability over height (top-down view, in world frame)

% Max over Z dimension
reach_max = max(map.data.reachScore, [], 3);
reach_max(reach_max < threshold) = NaN;

% Create grid in ARM FRAME
[nx, ny] = deal(double(map.grid.shape(1)), double(map.grid.shape(2)));
x = map.grid.origin(1) + (0:nx-1)*map.grid.voxel(1) + map.grid.voxel(1)/2;
y = map.grid.origin(2) + (0:ny-1)*map.grid.voxel(2) + map.grid.voxel(2)/2;

% Transform to WORLD FRAME
x = x + arm_offset(1);
y = y + arm_offset(2);

% Plot as image
imagesc(ax, y, x, reach_max);
axis(ax, 'equal', 'xy');
colormap(ax, 'jet');
colorbar(ax, 'Label', 'Max Reach Score');
caxis(ax, [0 1]);

% Mark arm base origin (world frame)
hold(ax, 'on');
shoulder_world = arm_offset;
plot(ax, shoulder_world(2), shoulder_world(1), 'r*', 'MarkerSize', 20, 'LineWidth', 2);
plot(ax, shoulder_world(2), shoulder_world(1), 'ro', 'MarkerSize', 40, 'LineWidth', 2);
text(ax, shoulder_world(2)+0.05, shoulder_world(1), 'Arm Base', 'FontSize', 10, 'Color', 'r', 'FontWeight', 'bold');

% Mark mobile base origin
plot(ax, 0, 0, 'ko', 'MarkerSize', 15, 'LineWidth', 2);
plot(ax, 0, 0, 'k+', 'MarkerSize', 20, 'LineWidth', 2);
text(ax, 0.05, 0, 'Mobile Base', 'FontSize', 10, 'Color', 'k', 'FontWeight', 'bold');

% Add workspace radius circles (arm reach from shoulder, in world frame)
theta = linspace(0, 2*pi, 100);
r_reach = 0.75;  % Arm reach from shoulder (calculated from URDF)
plot(ax, shoulder_world(2) + r_reach*cos(theta), shoulder_world(1) + r_reach*sin(theta), 'r--', 'LineWidth', 1.5);
legend(ax, {'', '', 'Arm Base', '', 'Mobile Base', 'Max Reach (0.75m)'}, 'Location', 'best');
end

function plot_robot_with_reach(ax, robot, map, threshold, arm_offset)
% Plot robot model with reachable workspace overlay (both in world frame)

% Plot robot at home configuration
q_home = homeConfiguration(robot);
try
    show(robot, q_home, 'Parent', ax, 'Frames', 'off', 'PreservePlot', false);
catch
    % Fallback for older MATLAB
    show(robot, q_home, 'Parent', ax);
end

% Overlay semi-transparent reachable voxels
hold(ax, 'on');
[nx, ny, nz] = deal(double(map.grid.shape(1)), double(map.grid.shape(2)), double(map.grid.shape(3)));
[X, Y, Z] = ndgrid( ...
    map.grid.origin(1) + (0:nx-1)*map.grid.voxel(1) + map.grid.voxel(1)/2, ...
    map.grid.origin(2) + (0:ny-1)*map.grid.voxel(2) + map.grid.voxel(2)/2, ...
    map.grid.origin(3) + (0:nz-1)*map.grid.voxel(3) + map.grid.voxel(3)/2);

% Transform to WORLD FRAME
X = X + arm_offset(1);
Y = Y + arm_offset(2);
Z = Z + arm_offset(3);

valid = map.data.reachScore >= threshold;
X = X(valid);
Y = Y(valid);
Z = Z(valid);
C = map.data.reachScore(valid);

if ~isempty(X)
    scatter3(ax, X(:), Y(:), Z(:), 15, C(:), 'filled', 'MarkerFaceAlpha', 0.3);
    colormap(ax, 'jet');
    colorbar(ax, 'Label', 'Reach Score');
    caxis(ax, [0 1]);
end

% Mark arm base (shoulder) in world frame
shoulder_world = arm_offset;
plot3(ax, shoulder_world(1), shoulder_world(2), shoulder_world(3), 'r*', 'MarkerSize', 20, 'LineWidth', 3);
plot3(ax, shoulder_world(1), shoulder_world(2), shoulder_world(3), 'ro', 'MarkerSize', 12, 'LineWidth', 2);

% Mark end-effector
try
    ee_name = char(map.meta.eeLink);
    T_ee = getTransform(robot, q_home, ee_name);
    ee_pos = T_ee(1:3, 4);
    plot3(ax, ee_pos(1), ee_pos(2), ee_pos(3), 'go', 'MarkerSize', 15, 'LineWidth', 3);
    text(ax, ee_pos(1), ee_pos(2), ee_pos(3)+0.05, 'EE', 'FontSize', 10, 'Color', 'g', 'FontWeight', 'bold');
catch
    warning('Could not display end-effector position');
end
end
