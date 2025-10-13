
%{
% examplary:

% Read STL file
[faces, vertices] = stlread('your_model.stl');
% Reduce to 30% of original faces
[reducedFaces, reducedVertices] = reducepatch(faces, vertices, ratioLeft;
% Create triangulation object
TR = triangulation(faces, vertices);
%}

%%
% Parameters
inputFolder = pwd;                          % current folder where script lives
outputFolder = fullfile(pwd, 'stl_output');% output subfolder
reductionRatio = 0.3;                       % keep 30% of faces
skipIfExists = true;                        % skip when reduced file already exists

if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

fileList = dir(fullfile(inputFolder, '*.stl'));

for k = 1:numel(fileList)
    inputFile = fullfile(fileList(k).folder, fileList(k).name);
    [~, name, ext] = fileparts(fileList(k).name);
    outputFile = fullfile(outputFolder, [name '_reduced' ext]);

    if skipIfExists && exist(outputFile, 'file')
        fprintf('Skipping existing: %s\n', outputFile);
        continue;
    end

    try
        % Read STL -> triangulation TR, fileformat, binary attributes (if any)
        [TR, fileformat, attributes] = stlread(inputFile); %#ok<ASGLU>
    catch ME
        warning('Read failed for %s: %s', inputFile, ME.message);
        continue;
    end

    % Convert TR to faces (F) and vertices (V)
    F = TR.ConnectivityList;
    V = TR.Points;

    % Basic sanity check
    if max(F(:)) > size(V,1)
        warning('Face indices exceed vertex count for %s. Attempting cleanup.', inputFile);
        % remove unused/duplicate vertices and remap indices
        [V_unique, ~, idxMap] = unique(V, 'rows', 'stable');
        F = idxMap(F);
        V = V_unique;
        if max(F(:)) > size(V,1)
            warning('Cleanup failed for %s. Skipping file.', inputFile);
            continue;
        end
    end

    try
        % Reduce mesh -> returns [FR, VR] arrays
        [FR, VR] = reducepatch(F, V, reductionRatio);
    catch ME
        warning('reducepatch failed for %s: %s', inputFile, ME.message);
        continue;
    end

    % Report before/after counts
    fprintf('%s: faces %d -> %d, verts %d -> %d\n', ...
        name, size(F,1), size(FR,1), size(V,1), size(VR,1));

    % Convert to triangulation for stlwrite
    try
        TR_out = triangulation(FR, VR);
    catch ME
        warning('Failed to create triangulation for %s: %s', name, ME.message);
        continue;
    end

    % Note about attributes: binary per-triangle attributes cannot be
    % preserved reliably after reduction because triangle indices changed.
    % We discard them here. Implement remapping if you have a mapping rule.

    try
        stlwrite(TR_out, outputFile);
        fprintf('Saved: %s\n', outputFile);
    catch ME
        warning('Failed to write %s: %s', outputFile, ME.message);
    end
end

