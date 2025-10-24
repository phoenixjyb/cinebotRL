% Quick visualization launcher after map is built
% Run this in MATLAB once reach_map_mobile_mm_arm_only.mat is created

% Change to the matlab directory
cd('C:\Users\yanbo\wSpace\cinebotRL\matlab');

% Check if map exists
if ~isfile('reach_map_mobile_mm_arm_only.mat')
    error('Map file not found! Build still in progress or failed.');
end

fprintf('\n');
fprintf('╔═══════════════════════════════════════════════════════════╗\n');
fprintf('║  REACHABILITY MAP VISUALIZATION - QUICK START            ║\n');
fprintf('╚═══════════════════════════════════════════════════════════╝\n');
fprintf('\n');

% Load map to check size
info = dir('reach_map_mobile_mm_arm_only.mat');
fprintf('✅ Map file found: %.1f MB\n', info.bytes/(1024*1024));
fprintf('\n');

% Show available modes
fprintf('📊 Available Visualization Modes:\n');
fprintf('   1. 3D Voxel Cloud (color = reachability)\n');
fprintf('   2. Horizontal Slice (at specific height)\n');
fprintf('   3. Manipulability Slice (dexterity map)\n');
fprintf('   4. Top View (bird''s eye)\n');
fprintf('   5. Robot + Reachability (RECOMMENDED!) 🌟\n');
fprintf('\n');

% Prompt user
fprintf('Quick launch Mode 5 (robot + reach overlay)?\n');
fprintf('This is the best visualization to verify everything is correct.\n');
fprintf('\n');
response = input('Visualize now? (y/n): ', 's');

if strcmpi(response, 'y')
    fprintf('\n🚀 Launching visualization...\n');
    fprintf('   (This shows reachability map in arm base frame)\n');
    fprintf('\n');
    
    % Launch FK map visualizer
    visualize_fk_map('reach_map_mobile_mm_arm_only.mat');
    
    fprintf('\n✅ Visualization opened!\n');
    fprintf('\n');
    fprintf('🔍 What to check:\n');
    fprintf('   • Black pentagon at ground (0,0,0) = mobile base\n');
    fprintf('   • Blue dot at (0.16, 0, 0.95) = arm shoulder\n');
    fprintf('   • Reachability cloud surrounds arm shoulder (NOT ground!)\n');
    fprintf('   • Cloud is ~0.6-0.8m radius hemisphere around shoulder\n');
    fprintf('   • Grid box shows workspace bounds in world frame\n');
    fprintf('   • Left plot: Binary reachability (red = reachable)\n');
    fprintf('   • Right plot: Manipulability heatmap (blue=low, red=high)\n');
    fprintf('\n');
    fprintf('🎮 Interactive controls:\n');
    fprintf('   • Rotate: Left-click drag\n');
    fprintf('   • Pan: Right-click drag\n');
    fprintf('   • Zoom: Scroll wheel\n');
    fprintf('\n');
else
    fprintf('\n👉 To visualize later, run:\n');
    fprintf('   visualize_reachability(''reach_map_mobile_mm_arm_only.mat'', ''mode'', 5)\n');
    fprintf('\n');
    fprintf('📖 For other modes and options:\n');
    fprintf('   help visualize_reachability\n');
    fprintf('\n');
end

fprintf('─────────────────────────────────────────────────────────────\n');
fprintf('Next steps after visualization:\n');
fprintf('1. ✅ Verify workspace looks correct\n');
fprintf('2. 🐍 Test Python loader: python -c "from scripts.reachability_utils import ReachabilityMap; m = ReachabilityMap(''matlab/reach_map_mobile_mm_arm_only.mat''); print(m)"\n');
fprintf('3. 🧪 Test with trajectory data (see REACHABILITY_BUILD_RUNNING.md)\n');
fprintf('4. 🚀 Integrate into Session 8 training\n');
fprintf('─────────────────────────────────────────────────────────────\n');
fprintf('\n');
