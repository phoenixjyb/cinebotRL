% Debug script to test robot visualization step by step

clear; close all;

%% Step 1: Load robot
fprintf('=== STEP 1: Loading Robot ===\n');
urdf_path = fullfile('..', 'assets_own', 'mobile_manipulator_PPR_base_corrected.urdf');
urdf_dir = fileparts(urdf_path);

% Add to path and change directory
addpath(urdf_dir);
orig_dir = pwd;
cd(urdf_dir);

fprintf('Current dir: %s\n', pwd);
fprintf('URDF path: %s\n', urdf_path);

robot = importrobot('mobile_manipulator_PPR_base_corrected.urdf');
robot.DataFormat = 'column';
q_home = homeConfiguration(robot);

fprintf('✓ Robot loaded with %d bodies\n', numel(robot.Bodies));

% Check if visuals exist
mesh_count = 0;
for i = 1:numel(robot.Bodies)
    if ~isempty(robot.Bodies{i}.Visuals)
        mesh_count = mesh_count + numel(robot.Bodies{i}.Visuals);
        fprintf('  Body %s: %d visuals\n', robot.Bodies{i}.Name, numel(robot.Bodies{i}.Visuals));
    end
end
fprintf('Total visual geometries: %d\n', mesh_count);

cd(orig_dir);

%% Step 2: Test show() alone
fprintf('\n=== STEP 2: Testing show() Alone ===\n');
figure('Name', 'Test: show() alone');
show(robot, q_home);
title('Robot with show() alone');
view(45, 30);
axis equal;
grid on;

fprintf('Check Figure 1: Do you see the robot?\n');
pause(1);

%% Step 3: Test scatter + show with hold
fprintf('\n=== STEP 3: Testing scatter + show with hold ===\n');
figure('Name', 'Test: scatter then show with hold');

% Create dummy points
x = 0.16 + 0.3*randn(100,1);
y = 0.3*randn(100,1);
z = 0.9465 + 0.3*abs(randn(100,1));

scatter3(x, y, z, 20, 'r', 'filled', 'MarkerFaceAlpha', 0.3);
hold on;
show(robot, q_home);
hold off;

title('Scatter + show() with hold');
view(45, 30);
axis equal;
grid on;

fprintf('Check Figure 2: Do you see robot AND red points?\n');
pause(1);

%% Step 4: Test with Parent and PreservePlot
fprintf('\n=== STEP 4: Testing with Parent and PreservePlot ===\n');
figure('Name', 'Test: Parent + PreservePlot');

scatter3(x, y, z, 20, 'r', 'filled', 'MarkerFaceAlpha', 0.3);
hold on;

ax = gca;
show(robot, q_home, 'Parent', ax, 'PreservePlot', true);

hold off;
title('Parent + PreservePlot');
view(45, 30);
axis equal;
grid on;

fprintf('Check Figure 3: Do you see robot AND red points?\n');
pause(1);

%% Step 5: Test without PreservePlot
fprintf('\n=== STEP 5: Testing without PreservePlot ===\n');
figure('Name', 'Test: Parent only (no PreservePlot)');

scatter3(x, y, z, 20, 'r', 'filled', 'MarkerFaceAlpha', 0.3);
hold on;

ax = gca;
show(robot, q_home, 'Parent', ax);

hold off;
title('Parent only (no PreservePlot)');
view(45, 30);
axis equal;
grid on;

fprintf('Check Figure 4: Do you see robot AND red points?\n');

%% Summary
fprintf('\n=== DIAGNOSIS ===\n');
fprintf('If robot shows in Figure 1 but NOT in 2-4:\n');
fprintf('  → show() is clearing/conflicting with scatter plot\n');
fprintf('  → Need different approach (manual mesh rendering)\n\n');
fprintf('If robot shows in Figure 2:\n');
fprintf('  → Simple hold works, use that!\n\n');
fprintf('If robot shows in Figure 3:\n');
fprintf('  → PreservePlot works, already using it\n\n');
fprintf('If robot shows in Figure 4:\n');
fprintf('  → PreservePlot not needed\n\n');
fprintf('If robot shows NOWHERE:\n');
fprintf('  → Mesh files not loading properly\n');
fprintf('  → Check: Are STL files in assets_own/meshes/?\n');
