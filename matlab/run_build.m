% LAUNCHER SCRIPT - Run this in MATLAB directly
% 
% Usage in MATLAB:
%   >> cd C:\Users\yanbo\wSpace\cinebotRL\matlab
%   >> run_build
%
% This will:
% 1. Start parallel pool (if available)
% 2. Build reachability map with progress tracking
% 3. Save results to reach_map_mobile_mm_arm_only.mat
%
% Expected time with parallel (8 cores): 10-15 minutes
% Expected time without parallel: 45-90 minutes

clear;
clc;

fprintf('\n');
fprintf('╔═══════════════════════════════════════════════════════════╗\n');
fprintf('║    REACHABILITY MAP BUILD LAUNCHER (FK-based)            ║\n');
fprintf('╚═══════════════════════════════════════════════════════════╝\n');
fprintf('\n');

% Check parallel computing toolbox
has_parallel = license('test', 'Distrib_Computing_Toolbox');
if has_parallel
    fprintf('✅ Parallel Computing Toolbox detected\n');
    poolobj = gcp('nocreate');
    if isempty(poolobj)
        fprintf('   Starting parallel pool...\n');
        poolobj = parpool;
    else
        fprintf('   Parallel pool already running\n');
    end
    fprintf('   Workers: %d\n', poolobj.NumWorkers);
else
    fprintf('⚠️  Parallel Computing Toolbox not available\n');
    fprintf('   Will use serial processing (slower)\n');
end

fprintf('\n');
fprintf('Starting FK-based build (100K samples)...\n');
fprintf('Expected time: 1-2 minutes\n');
fprintf('════════════════════════════════════════════════════════════\n');
fprintf('\n');

% Run the main build script
try
    build_reachability_map_FK();
    fprintf('\n');
    fprintf('╔═══════════════════════════════════════════════════════════╗\n');
    fprintf('║                  BUILD COMPLETED! ✅                      ║\n');
    fprintf('╚═══════════════════════════════════════════════════════════╝\n');
    fprintf('\n');
    fprintf('Next steps:\n');
    fprintf('  1. Visualize: visualize_reachability(''reach_map_mobile_mm_arm_only.mat'', ''mode'', 5)\n');
    fprintf('  2. Test Python: python -c "from scripts.reachability_utils import ReachabilityMap; ..."\n');
    fprintf('\n');
catch err
    fprintf('\n');
    fprintf('╔═══════════════════════════════════════════════════════════╗\n');
    fprintf('║                    BUILD FAILED ❌                        ║\n');
    fprintf('╚═══════════════════════════════════════════════════════════╝\n');
    fprintf('\n');
    fprintf('Error message:\n');
    fprintf('  %s\n', err.message);
    fprintf('\n');
    fprintf('Stack trace:\n');
    for i = 1:length(err.stack)
        fprintf('  File: %s\n', err.stack(i).file);
        fprintf('  Function: %s (line %d)\n', err.stack(i).name, err.stack(i).line);
    end
    fprintf('\n');
    fprintf('Troubleshooting:\n');
    fprintf('  1. Check URDF path is correct\n');
    fprintf('  2. Verify Robotics System Toolbox is installed: ver(''robotics'')\n');
    fprintf('  3. Try disabling parallel: Edit build_reachability_map.m, set USE_PARFOR=false\n');
    fprintf('\n');
    rethrow(err);
end
