% Quick script to calculate optimal grid parameters for your robot
% Run this to get scientifically-derived GRID_ORIGIN and GRID_SIZE values

cd('C:\Users\yanbo\wSpace\cinebotRL\matlab');

% Your robot parameters
URDF_PATH = "C:\Users\yanbo\wSpace\cinebotRL\assets_own\mobile_manipulator_PPR_base_corrected.urdf";
BASE_LINK = "left_arm_base_link";
EE_LINK = "left_gripper_link";

% Calculate optimal grid
[optimal_origin, optimal_size] = calculate_optimal_grid(URDF_PATH, BASE_LINK, EE_LINK);

% Save results
results.optimal_origin = optimal_origin;
results.optimal_size = optimal_size;
results.current_origin = [-0.6, -0.8, -0.4];
results.current_size = [1.2, 1.6, 1.0];
results.timestamp = datetime('now');

save('optimal_grid_params.mat', 'results');

fprintf('\n📁 Results saved to: optimal_grid_params.mat\n');
fprintf('\n');
fprintf('🔧 To use these values:\n');
fprintf('   1. Open build_reachability_map.m\n');
fprintf('   2. Replace lines 49-50 with the recommended values above\n');
fprintf('   3. Re-run the build script\n');
fprintf('\n');
