% Monitor reachability map build progress (run in separate MATLAB window)
%
% Usage:
%   >> monitor_progress
%
% This will show:
% - File size growth
% - Estimated progress
% - ETA
%
% Keep this running while build_reachability_map executes in another window

clear;
clc;

fprintf('\n');
fprintf('╔═══════════════════════════════════════════════════════════╗\n');
fprintf('║        REACHABILITY BUILD PROGRESS MONITOR               ║\n');
fprintf('╚═══════════════════════════════════════════════════════════╝\n');
fprintf('\n');

map_file = 'reach_map_mobile_mm_arm_only.mat';
expected_size_mb = 75; % Expected final size
update_interval = 10; % Check every 10 seconds

fprintf('Monitoring: %s\n', map_file);
fprintf('Expected final size: ~%.0f MB\n', expected_size_mb);
fprintf('Update interval: %d seconds\n', update_interval);
fprintf('\n');
fprintf('Press Ctrl+C to stop monitoring\n');
fprintf('═══════════════════════════════════════════════════════════\n');
fprintf('\n');

last_size = 0;
start_time = tic;
file_appeared = false;

while true
    if isfile(map_file)
        if ~file_appeared
            fprintf('✅ Map file created!\n\n');
            file_appeared = true;
        end
        
        info = dir(map_file);
        size_mb = info.bytes / (1024^2);
        elapsed_min = toc(start_time) / 60;
        
        % Estimate progress
        progress_pct = min(100, (size_mb / expected_size_mb) * 100);
        
        % Only print if size changed
        if size_mb ~= last_size
            % Estimate time remaining
            if progress_pct > 5
                rate_mb_per_min = size_mb / elapsed_min;
                remaining_mb = expected_size_mb - size_mb;
                eta_min = remaining_mb / rate_mb_per_min;
            else
                eta_min = NaN;
            end
            
            fprintf('[%s] Size: %6.1f MB | Progress: %5.1f%% | Elapsed: %5.1f min', ...
                datestr(now, 'HH:MM:SS'), size_mb, progress_pct, elapsed_min);
            
            if ~isnan(eta_min) && eta_min > 0
                fprintf(' | ETA: %5.1f min\n', eta_min);
            else
                fprintf(' | ETA: calculating...\n');
            end
            
            last_size = size_mb;
            
            % Check if complete
            if size_mb >= expected_size_mb * 0.9
                fprintf('\n');
                fprintf('═══════════════════════════════════════════════════════════\n');
                fprintf('Build appears complete! (%.1f MB)\n', size_mb);
                fprintf('Check main MATLAB window for final statistics.\n');
                fprintf('═══════════════════════════════════════════════════════════\n');
                break;
            end
        end
    else
        if ~file_appeared
            elapsed_min = toc(start_time) / 60;
            if mod(floor(elapsed_min), 1) == 0  % Print every minute
                fprintf('[%s] Waiting for map file to be created... (%.1f min elapsed)\n', ...
                    datestr(now, 'HH:MM:SS'), elapsed_min);
            end
        end
    end
    
    pause(update_interval);
    
    % Safety timeout
    if toc(start_time) > 7200  % 2 hours
        fprintf('\n⚠️  Timeout after 2 hours. Check if build is still running.\n');
        break;
    end
end

fprintf('\n');
fprintf('Monitoring stopped.\n');
fprintf('\n');
