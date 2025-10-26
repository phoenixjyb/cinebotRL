%% Check Robot Body Order
clear; close all;

fprintf('Loading URDF...\n');
urdf_file = 'mobile_manipulator_PPR_matlab.urdf';
if ~exist(urdf_file, 'file')
    urdf_file = '../assets_own/mobile_manipulator_PPR_matlab.urdf';
end

robot = importrobot(urdf_file);
robot.DataFormat = 'column';

fprintf('\nRobot Bodies Order:\n');
fprintf('%-5s %-30s %-20s\n', 'Index', 'Body Name', 'Parent');
fprintf('%s\n', repmat('-', 1, 60));

for i = 1:numel(robot.Bodies)
    body = robot.Bodies{i};
    if ~isempty(body.Joint) && ~isempty(body.Joint.Name)
        parent_idx = findBodyIndexByName(robot, robot.Bodies{i}.Parent.Name);
        fprintf('%-5d %-30s %-20s (idx=%d)\n', i, body.Name, robot.Bodies{i}.Parent.Name, parent_idx);
    else
        fprintf('%-5d %-30s %-20s\n', i, body.Name, '(root)');
    end
end

fprintf('\n');
fprintf('Analysis:\n');
fprintf('  "adjacent" mode skips: body[i] <-> body[i+1]\n');
fprintf('  For chassis-arm collision to be detected, they must NOT be adjacent indices\n');

function idx = findBodyIndexByName(robot, name)
    idx = -1;
    for i = 1:numel(robot.Bodies)
        if strcmp(robot.Bodies{i}.Name, name)
            idx = i;
            return;
        end
    end
end
