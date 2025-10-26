%% Test Chassis Collision Detection
% This tests if checkCollision properly detects arm-chassis penetration

clear; close all;

fprintf('Loading URDF...\n');
urdf_file = 'mobile_manipulator_PPR_matlab.urdf';
if ~exist(urdf_file, 'file')
    urdf_file = '../assets_own/mobile_manipulator_PPR_matlab.urdf';
end

robot = importrobot(urdf_file);
robot.DataFormat = 'column';

fprintf('Robot loaded: %d bodies\n', numel(robot.Bodies));

% Debug: Print configuration structure
fprintf('\n=== Configuration Structure ===\n');
q_home = homeConfiguration(robot);
fprintf('Configuration vector length: %d\n', length(q_home));
fprintf('Number of bodies: %d\n', numel(robot.Bodies));

% Print all joint names and their home values
fprintf('\nJoint mapping:\n');
for i = 1:numel(robot.Bodies)
    body = robot.Bodies{i};
    if ~isempty(body.Joint)
        fprintf('  Body %d: %-30s Joint: %-25s Type: %s\n', ...
            i, body.Name, body.Joint.Name, body.Joint.Type);
    end
end

fprintf('\nHome configuration values:\n');
for i = 1:length(q_home)
    fprintf('  q(%d) = %.4f\n', i, q_home(i));
end
fprintf('==============================\n\n');

%% Test 1: Home configuration (should be collision-free)
fprintf('\nTest 1: Home configuration\n');
q_home = homeConfiguration(robot);

% Test with 'adjacent' mode
[isCol_adj, ~] = checkCollision(robot, q_home, ...
    'Exhaustive', 'on', 'SkippedSelfCollisions', 'adjacent');
fprintf('  Collision (adjacent mode): %d\n', isCol_adj);

% Test with 'parent' mode
[isCol_par, ~] = checkCollision(robot, q_home, ...
    'Exhaustive', 'on', 'SkippedSelfCollisions', 'parent');
fprintf('  Collision (parent mode):   %d\n', isCol_par);

% Visualize Test 1
figure('Name', 'Test 1: Home Configuration');
show(robot, q_home, 'Visuals', 'on', 'Collisions', 'off');
title('Test 1: Home Configuration (should be collision-free)');
view(45, 30);
axis equal;
grid on;

%% Test 2: Arm extended down (might hit chassis)
fprintf('\nTest 2: Arm reaching down toward chassis\n');
q_down = homeConfiguration(robot);

% Configuration vector structure (9 elements):
% q(1-3): base x, y, theta
% q(4-9): arm joints 1-6
% Joint 2 is at index 5, Joint 3 is at index 6
q_down(5) = -1.5;  % left_arm_joint_2 (shoulder pitch down)
q_down(6) = 1.0;   % left_arm_joint_3 (elbow bend)
fprintf('  Set q(5) [joint_2] = -1.5 rad\n');
fprintf('  Set q(6) [joint_3] = 1.0 rad\n');

% Test with 'adjacent' mode
[isCol_adj, sepDist_adj] = checkCollision(robot, q_down, ...
    'Exhaustive', 'on', 'SkippedSelfCollisions', 'adjacent');
fprintf('  Collision (adjacent mode): %d\n', isCol_adj);

% Test with 'parent' mode
[isCol_par, sepDist_par] = checkCollision(robot, q_down, ...
    'Exhaustive', 'on', 'SkippedSelfCollisions', 'parent');
fprintf('  Collision (parent mode):   %d\n', isCol_par);

% Visualize Test 2
figure('Name', 'Test 2: Arm Reaching Down');
show(robot, q_down, 'Visuals', 'on', 'Collisions', 'off');
title('Test 2: Arm Reaching Down (joint2=-1.5, joint3=1.0)');
view(45, 30);
axis equal;
grid on;

if isCol_adj
    % Find which bodies are in collision (adjacent mode)
    [body1Idx, body2Idx] = find(isnan(sepDist_adj));
    collidingPairs = unique(sort([body1Idx, body2Idx], 2), 'rows');
    fprintf('  Colliding pairs (adjacent mode):\n');
    for i = 1:size(collidingPairs, 1)
        b1 = collidingPairs(i, 1);
        b2 = collidingPairs(i, 2);
        if b1 <= numel(robot.Bodies) && b2 <= numel(robot.Bodies)
            name1 = robot.Bodies{b1}.Name;
            name2 = robot.Bodies{b2}.Name;
            fprintf('    %s <-> %s\n', name1, name2);
        end
    end
end

if isCol_par
    % Find which bodies are in collision (parent mode)
    [body1Idx, body2Idx] = find(isnan(sepDist_par));
    collidingPairs = unique(sort([body1Idx, body2Idx], 2), 'rows');
    fprintf('  Colliding pairs (parent mode):\n');
    for i = 1:size(collidingPairs, 1)
        b1 = collidingPairs(i, 1);
        b2 = collidingPairs(i, 2);
        if b1 <= numel(robot.Bodies) && b2 <= numel(robot.Bodies)
            name1 = robot.Bodies{b1}.Name;
            name2 = robot.Bodies{b2}.Name;
            fprintf('    %s <-> %s\n', name1, name2);
        end
    end
end

%% Test 3: Extreme configuration (definitely should collide)
fprintf('\nTest 3: Arm folded back into chassis\n');
q_fold = homeConfiguration(robot);

% Configuration vector structure:
% q(4) = left_arm_joint_1 (shoulder rotation)
% q(5) = left_arm_joint_2 (shoulder pitch)
q_fold(4) = 3.0;   % left_arm_joint_1 (rotate around)
q_fold(5) = -2.0;  % left_arm_joint_2 (fold down hard)
fprintf('  Set q(4) [joint_1] = 3.0 rad\n');
fprintf('  Set q(5) [joint_2] = -2.0 rad\n');

% Test with 'adjacent' mode
[isCol_adj, sepDist_adj] = checkCollision(robot, q_fold, ...
    'Exhaustive', 'on', 'SkippedSelfCollisions', 'adjacent');
fprintf('  Collision (adjacent mode): %d\n', isCol_adj);

% Test with 'parent' mode
[isCol_par, sepDist_par] = checkCollision(robot, q_fold, ...
    'Exhaustive', 'on', 'SkippedSelfCollisions', 'parent');
fprintf('  Collision (parent mode):   %d\n', isCol_par);

% Visualize Test 3
figure('Name', 'Test 3: Arm Folded Back');
show(robot, q_fold, 'Visuals', 'on', 'Collisions', 'off');
title('Test 3: Arm Folded Back (joint1=3.0, joint2=-2.0)');
view(45, 30);
axis equal;
grid on;

if isCol_adj
    [body1Idx, body2Idx] = find(isnan(sepDist_adj));
    collidingPairs = unique(sort([body1Idx, body2Idx], 2), 'rows');
    fprintf('  Colliding pairs (adjacent mode):\n');
    for i = 1:size(collidingPairs, 1)
        b1 = collidingPairs(i, 1);
        b2 = collidingPairs(i, 2);
        if b1 <= numel(robot.Bodies) && b2 <= numel(robot.Bodies)
            name1 = robot.Bodies{b1}.Name;
            name2 = robot.Bodies{b2}.Name;
            fprintf('    %s <-> %s\n', name1, name2);
        end
    end
end

if isCol_par
    [body1Idx, body2Idx] = find(isnan(sepDist_par));
    collidingPairs = unique(sort([body1Idx, body2Idx], 2), 'rows');
    fprintf('  Colliding pairs (parent mode):\n');
    for i = 1:size(collidingPairs, 1)
        b1 = collidingPairs(i, 1);
        b2 = collidingPairs(i, 2);
        if b1 <= numel(robot.Bodies) && b2 <= numel(robot.Bodies)
            name1 = robot.Bodies{b1}.Name;
            name2 = robot.Bodies{b2}.Name;
            fprintf('    %s <-> %s\n', name1, name2);
        end
    end
end

fprintf('\n✓ Tests complete! Check the 3 figures to visually verify robot poses.\n');
fprintf('\nInterpretation:\n');
fprintf('  - If Test 1 shows collision: URDF has overlapping geometry at home\n');
fprintf('  - If Test 2/3 show NO collision: Chassis collision mesh might be missing/incorrect\n');
fprintf('  - If Test 2/3 show collision: Collision detection is working!\n');
