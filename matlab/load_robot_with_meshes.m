function robot = load_robot_with_meshes(urdf_path)
% Load robot URDF and manually fix mesh paths if needed
%
% MATLAB's importrobot doesn't resolve package:// paths automatically
% This function works around that by manually loading meshes

fprintf('Loading robot from: %s\n', urdf_path);

% Get directory info
[urdf_dir, ~, ~] = fileparts(urdf_path);
mesh_dir = fullfile(urdf_dir, 'meshes');

% Check if meshes exist
if ~exist(mesh_dir, 'dir')
    error('Meshes directory not found: %s', mesh_dir);
end

% Change to URDF directory and add to path
orig_dir = pwd;
addpath(urdf_dir);
cd(urdf_dir);

% Load robot
try
    robot = importrobot(urdf_path);
    robot.DataFormat = 'column';
catch ME
    cd(orig_dir);
    rethrow(ME);
end

% Restore directory
cd(orig_dir);

% Check if visuals loaded
mesh_count = 0;
for i = 1:numel(robot.Bodies)
    if ~isempty(robot.Bodies{i}.Visuals)
        mesh_count = mesh_count + numel(robot.Bodies{i}.Visuals);
    end
end

if mesh_count == 0
    warning('No visual meshes loaded! URDF may use package:// paths that MATLAB cannot resolve.');
    fprintf('Attempting manual mesh loading...\n');
    
    % Try to manually attach meshes
    robot = attach_meshes_manually(robot, mesh_dir);
    
    % Recount
    mesh_count = 0;
    for i = 1:numel(robot.Bodies)
        if ~isempty(robot.Bodies{i}.Visuals)
            mesh_count = mesh_count + numel(robot.Bodies{i}.Visuals);
        end
    end
end

fprintf('✓ Robot loaded: %d bodies, %d visual geometries\n', numel(robot.Bodies), mesh_count);

end


function robot = attach_meshes_manually(robot, mesh_dir)
% Manually attach mesh files to robot bodies
% This is a workaround for when package:// paths don't resolve

% Mapping of body names to mesh files (from URDF inspection)
mesh_map = containers.Map();
mesh_map('abstract_chassis_link') = 'base_link.STL';
mesh_map('left_arm_base_link') = 'left_arm_base_link.STL';
mesh_map('left_arm_link1') = 'left_arm_link1.STL';
mesh_map('left_arm_link2') = 'left_arm_link2.STL';
mesh_map('left_arm_link3') = 'left_arm_link3.STL';
mesh_map('left_arm_link4') = 'left_arm_link4.STL';
mesh_map('left_arm_link5') = 'left_arm_link5.STL';
mesh_map('left_arm_link6') = 'left_arm_link6.STL';
mesh_map('left_gripper_link') = 'end_effector.STL';

% Attach meshes to bodies
for i = 1:numel(robot.Bodies)
    body_name = robot.Bodies{i}.Name;
    
    if mesh_map.isKey(body_name)
        mesh_file = mesh_map(body_name);
        mesh_path = fullfile(mesh_dir, mesh_file);
        
        if exist(mesh_path, 'file')
            try
                % Create a mesh geometry
                mesh_geom = robotics.Mesh(mesh_path);
                
                % Create visual element
                visual = robotics.Visual();
                visual.Geometry = mesh_geom;
                
                % Add to body (clear existing empty visuals first)
                robot.Bodies{i}.Visuals = {visual};
                
                fprintf('  Attached mesh to %s: %s\n', body_name, mesh_file);
            catch ME
                warning('Failed to attach mesh to %s: %s', body_name, ME.message);
            end
        else
            warning('Mesh file not found: %s', mesh_path);
        end
    end
end

end
