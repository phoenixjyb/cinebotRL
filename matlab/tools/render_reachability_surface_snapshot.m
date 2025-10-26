function render_reachability_surface_snapshot(map_file, image_path, options)
% RENDER_REACHABILITY_SURFACE_SNAPSHOT  Save a rendered image of the reachability surface.
%
%   render_reachability_surface_snapshot(map_file, image_path) extracts the
%   reachability surface from MAP_FILE and saves a PNG (or other supported
%   formats) to IMAGE_PATH.
%
%   Optional arguments (name-value in struct `options`):
%       options.view_az_el   - [az el] camera angles (deg). Default [135 30].
%       options.face_color   - 1x3 RGB face colour. Default [0.2 0.55 0.9].
%       options.face_alpha   - Face transparency. Default 0.85.
%       options.edge_color   - RGB for mesh edges. Default 'none'.
%       options.bg_color     - Figure background colour. Default [1 1 1].
%       options.image_size   - [width height] in pixels. Default [1200 900].
%       options.reduce_fraction - Passed through to extractor (default 0.35).
%       options.min_cluster  - Passed through to extractor (default 32).
%
%   Example:
%       render_reachability_surface_snapshot( ...
%           'reach_map_mobile_mm_arm_only.mat', ...
%           'exports/reach_surface.png');

arguments
    map_file (1,1) string
    image_path (1,1) string
    options.view_az_el (1,2) double = [135, 30]
    options.face_color (1,3) double = [0.2, 0.55, 0.9]
    options.face_alpha (1,1) double = 0.85
    options.edge_color = 'none'
    options.bg_color (1,3) double = [1, 1, 1]
    options.image_size (1,2) double = [1200, 900]
    options.reduce_fraction (1,1) double = 0.35
    options.min_cluster (1,1) double = 32
end

addpath('tools'); %#ok<ADDDEP> ensure extractor is visible

surface = extract_reachability_surface(map_file, ...
    'output_mat', "", ...
    'output_ply', "", ...
    'reduce_fraction', options.reduce_fraction, ...
    'min_cluster', options.min_cluster, ...
    'verbose', false);

ensure_parent_folder_local(image_path);

fig = figure('Visible', 'off', ...
    'Color', options.bg_color, ...
    'Units', 'pixels', ...
    'Position', [100 100 options.image_size]);

patch('Parent', axes('Parent', fig), ...
    'Faces', surface.faces, ...
    'Vertices', surface.vertices, ...
    'FaceColor', options.face_color, ...
    'EdgeColor', options.edge_color, ...
    'FaceAlpha', options.face_alpha);

ax = fig.CurrentAxes;
hold(ax, 'on');

% Draw coordinate axes for reference
origin = surface.meta.grid_origin;
axis_len = norm(surface.meta.grid_size) * 0.2;
quiver3(ax, origin(1), origin(2), origin(3), axis_len, 0, 0, 'Color', [0.9 0.2 0.2], 'LineWidth', 1.5, 'MaxHeadSize', 0.6);
quiver3(ax, origin(1), origin(2), origin(3), 0, axis_len, 0, 'Color', [0.2 0.8 0.2], 'LineWidth', 1.5, 'MaxHeadSize', 0.6);
quiver3(ax, origin(1), origin(2), origin(3), 0, 0, axis_len, 'Color', [0.2 0.4 0.9], 'LineWidth', 1.5, 'MaxHeadSize', 0.6);

scatter3(ax, surface.boundary_voxels(:,1), ...
            surface.boundary_voxels(:,2), ...
            surface.boundary_voxels(:,3), ...
            2, [0.15 0.15 0.15], 'filled', 'MarkerFaceAlpha', 0.1);

axis(ax, 'equal');
grid(ax, 'on');
ax.Box = 'on';
xlabel(ax, 'X (m)');
ylabel(ax, 'Y (m)');
zlabel(ax, 'Z (m)');
view(ax, options.view_az_el);

camlight(ax, 'headlight');
camlight(ax, 'right');
lighting(ax, 'gouraud');

title(ax, sprintf('Reachability Surface (%s)', map_file), 'Interpreter', 'none');

exportgraphics(fig, image_path, 'Resolution', 200);
close(fig);

end

function ensure_parent_folder_local(pathname)
[folder, ~, ~] = fileparts(pathname);
folder = string(folder);
if folder ~= "" && ~exist(folder, 'dir')
    mkdir(folder);
end
end
