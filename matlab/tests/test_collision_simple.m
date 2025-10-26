%% Test simple collision checking
% This tests if checkCollision works with our URDF

clear; close all;

fprintf('Loading URDF...\n');
urdf_file = 'mobile_manipulator_PPR_matlab.urdf';
if ~exist(urdf_file, 'file')
    urdf_file = '../assets_own/mobile_manipulator_PPR_matlab.urdf';
end

robot = importrobot(urdf_file);
robot.DataFormat = 'column';

fprintf('Robot loaded: %d bodies\n', numel(robot.Bodies));

% Get home configuration
q = homeConfiguration(robot);

fprintf('\nListing all body names:\n');
for i = 1:numel(robot.Bodies)
    fprintf('  %d: %s\n', i, robot.Bodies{i}.Name);
end

fprintf('\nTesting different checkCollision syntaxes...\n');

% Syntax 1: Just configuration (checks all collisions)
try
    fprintf('  Syntax 1: checkCollision(robot, q) - check ALL collisions\n');
    isCol = checkCollision(robot, q);
    fprintf('    Result: %d (1=any collision, 0=no collision)\n', isCol);
catch ME
    fprintf('    ERROR: %s\n', ME.message);
end

% Syntax 2: Try with body indices instead of names
try
    fprintf('\n  Syntax 2: checkCollision(robot, q, bodyIdx1, bodyIdx2)\n');
    % Body 1 = abstract_chassis_link, Body 2 = left_arm_base_link
    isCol = checkCollision(robot, q, 1, 2);
    fprintf('    Bodies 1-2 collision: %d\n', isCol);
catch ME
    fprintf('    ERROR: %s\n', ME.message);
end

% Syntax 3: Try string names
try
    fprintf('\n  Syntax 3: Try with string() wrapper\n');
    isCol = checkCollision(robot, q, string('left_arm_link1'), string('abstract_chassis_link'));
    fprintf('    Result: %d\n', isCol);
catch ME
    fprintf('    ERROR: %s\n', ME.message);
end

fprintf('\nDone!\n');
