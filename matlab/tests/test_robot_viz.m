% Test script to verify robot visualization works
% Run this to debug why robot isn't showing in main visualization

clear; close all;

%% Load robot
urdf_path = '..\assets_own\mobile_manipulator_PPR_base_corrected.urdf';
fprintf('Loading URDF: %s\n', urdf_path);

if ~exist(urdf_path, 'file')
    error('URDF not found at: %s', urdf_path);
end

% Change to URDF directory for mesh resolution
urdf_dir = fileparts(urdf_path);

% Add to MATLAB path
if ~contains(path, urdf_dir)
    addpath(urdf_dir);
    fprintf('  Added to path: %s\n', urdf_dir);
end

orig_dir = pwd;
cd(urdf_dir);

fprintf('  Working directory: %s\n', pwd);
fprintf('  Checking for meshes/ directory...\n');
if exist('meshes', 'dir')
    fprintf('  ✓ meshes/ directory found\n');
    mesh_files = dir('meshes/*.STL');
    fprintf('  ✓ Found %d STL files\n', length(mesh_files));
    % List first few files
    for i = 1:min(3, length(mesh_files))
        fprintf('    - %s\n', mesh_files(i).name);
    end
else
    fprintf('  ✗ meshes/ directory NOT found!\n');
end

robot = importrobot(urdf_path);
robot.DataFormat = 'column';

cd(orig_dir);

fprintf('✓ Robot loaded successfully\n');
fprintf('  Bodies: %d\n', numel(robot.Bodies));
fprintf('  Base: %s\n', robot.BaseName);

%% Test 1: Show robot alone
figure('Name', 'Test 1: Robot Only');
q_home = homeConfiguration(robot);
show(robot, q_home, 'Visuals', 'on', 'Collisions', 'off', 'Frames', 'on');
title('Robot at Home Configuration');
axis equal;
grid on;
view(45, 30);

fprintf('\nTest 1: Does the robot appear in the figure?\n');
fprintf('  If YES: Robot model is working\n');
fprintf('  If NO: Check mesh file paths in URDF\n');

%% Test 2: Robot + scatter plot
figure('Name', 'Test 2: Robot + Scatter');

% Create some dummy reachability points (hemisphere)
[theta, phi] = meshgrid(linspace(0, 2*pi, 20), linspace(0, pi/2, 10));
r = 0.6;
x = 0.16 + r * sin(phi(:)) .* cos(theta(:));
y = r * sin(phi(:)) .* sin(theta(:));
z = 0.9465 + r * cos(phi(:));

% Plot points first
scatter3(x, y, z, 20, 'r', 'filled', 'MarkerFaceAlpha', 0.3);
hold on;

% Add robot
ax = gca;
show(robot, q_home, 'Parent', ax, 'Visuals', 'on', 'Collisions', 'off', ...
     'Frames', 'off', 'PreservePlot', true);

% Add markers
scatter3(0, 0, 0, 150, 'k', 'filled', 'p');  % Mobile base
scatter3(0.16, 0, 0.9465, 100, 'b', 'filled');  % Arm shoulder

hold off;
axis equal;
grid on;
xlabel('X (m)');
ylabel('Y (m)');
zlabel('Z (m)');
title('Robot + Reachability Cloud');
view(45, 30);

fprintf('\nTest 2: Does the robot appear WITH the scatter plot?\n');
fprintf('  If YES: Integration is working\n');
fprintf('  If NO: PreservePlot issue or axes conflict\n');

%% Test 3: Check mesh paths
fprintf('\n=== Checking Mesh File Paths ===\n');
for i = 1:numel(robot.Bodies)
    body = robot.Bodies{i};
    if ~isempty(body.Visuals)
        for j = 1:numel(body.Visuals)
            mesh = body.Visuals{j}.Geometry;
            if isa(mesh, 'robotics.Mesh')
                fprintf('Body %s: %s\n', body.Name, mesh.Filename);
                % Try to resolve path
                if startsWith(mesh.Filename, 'package://')
                    fprintf('  WARNING: package:// path - may need resolution\n');
                end
            end
        end
    end
end

fprintf('\n=== DIAGNOSIS ===\n');
fprintf('If robot doesn''t show:\n');
fprintf('1. Check mesh files exist in assets_own/meshes/\n');
fprintf('2. URDF uses package:// which may need path resolution\n');
fprintf('3. Try absolute paths in URDF instead\n');
fprintf('\nSolution: Update URDF mesh paths to relative paths like:\n');
fprintf('  <mesh filename="../meshes/base_link.STL"/>\n');
