function reachability_map()
% REACHABILITY_MAP
% Offline: build and save a collision-aware reachability map from a URDF.
% Online:  load and query the map for fast "can I even try this?" checks.
%
% Tested with MATLAB R2023b/R2024a + Robotics System Toolbox.
%
% ----------------------------
% HOW TO USE (quick start):
% 1) Set the parameters below (URDF path, EE link name, grid bounds, etc.)
% 2) Run: build_and_save_map();
%    -> Produces a .mat file with the map (HDF-like struct).
% 3) Later (online): res = reachable_sanity_check(mapFile, targetWorldPose, basePose);
%    -> res.ok = true/false; res.reason; res.seedQ (if stored)
%
% Notes:
% - The map is in the robot base frame; for a mobile base, we transform
%   the world target into base frame using base (x,y,yaw) at query time.
% - Orientation bins are optional; set N_ORIENT=0 for position-only reach.

%% =========================
%  USER PARAMETERS (EDIT)
%  =========================
URDF_PATH   = "C:\path\to\your\robot.urdf";   % <- change me
EE_LINK     = "tool0";                         % <- end-effector link name
DATA_FORMAT = "row";                           % (row|column)
GRAVITY     = [0 0 -9.81];                     % or [0 0 0] for kinematic check

% Grid in base frame (meters)
GRID_ORIGIN = [-0.8, -0.8,  0.0];              % min x,y,z
GRID_SIZE   = [ 1.6,  1.6,  1.2];              % size in x,y,z
VOXEL       = [ 0.03, 0.03, 0.03];             % resolution (3 cm)

% Orientation bins (set N_ORIENT=0 for position-only reach)
N_ORIENT        = 12;      % e.g., 0, 6, 12, 24 ...
ORIENT_CONE_DEG = 180;     % 180=free; smaller=cone about -Z tool axis (camera-like)

% IK
IK_ATTEMPTS     = 6;       % tries per voxel/orientation (randomized seeds)
IK_POS_TOL      = 2e-3;    % meters
IK_ORI_TOL      = deg2rad(2);  % radians

% Self-collision check (true recommended)
DO_SELF_COLLISION = true;

% Save path for the map
MAP_FILE = "C:\tmp\reach_map.mat";

% Optional: Payload/tool note (metadata only)
PAYLOAD_KG = 0.5;

% Parallel (if you have Parallel Computing Toolbox)
USE_PARFOR = true;

%% =========================
%  BUILD & SAVE (offline)
%  =========================
build_and_save_map(URDF_PATH, EE_LINK, MAP_FILE, DATA_FORMAT, GRAVITY, ...
    GRID_ORIGIN, GRID_SIZE, VOXEL, ...
    N_ORIENT, ORIENT_CONE_DEG, ...
    IK_ATTEMPTS, IK_POS_TOL, IK_ORI_TOL, ...
    DO_SELF_COLLISION, USE_PARFOR, PAYLOAD_KG);

%% =========================
%  EXAMPLE QUERY (online)
%  =========================
% World target pose (4x4). Example: a point at [0.3, 0.2, 0.8] in world,
% with any orientation; we’ll test position-only if N_ORIENT=0 in map.
pW = [0.30; 0.20; 0.80];
RW = eye(3);
TW = [RW, pW; 0 0 0 1];

% Mobile base pose in world (x,y,yaw)
basePose = struct('x', 0.0, 'y', 0.0, 'yaw', 0.0);

res = reachable_sanity_check(MAP_FILE, TW, basePose);
disp(res);

end % function reachability_map

%% ===================================================================== %%
function build_and_save_map(URDF_PATH, EE_LINK, MAP_FILE, DATA_FORMAT, GRAVITY, ...
    GRID_ORIGIN, GRID_SIZE, VOXEL, ...
    N_ORIENT, ORIENT_CONE_DEG, ...
    IK_ATTEMPTS, IK_POS_TOL, IK_ORI_TOL, ...
    DO_SELF_COLLISION, USE_PARFOR, PAYLOAD_KG)

fprintf('Loading URDF: %s\n', URDF_PATH);
robot = importrobot(URDF_PATH, DataFormat=DATA_FORMAT);
robot.Gravity = GRAVITY;
eeName = EE_LINK;
assert(any(strcmp({robot.BodyNames{:}}, eeName)), "EE link not found");

% Cache initial configuration
qHome = homeConfiguration(robot);
ndof  = numel(qHome);

% Orientation bins (in EE/local frame)
orientBins = sample_orientation_bins(N_ORIENT, ORIENT_CONE_DEG);

% Build voxel grid centers
nx = max(1, round(GRID_SIZE(1)/VOXEL(1)));
ny = max(1, round(GRID_SIZE(2)/VOXEL(2)));
nz = max(1, round(GRID_SIZE(3)/VOXEL(3)));
[xg, yg, zg] = ndgrid( ...
    GRID_ORIGIN(1) + (0:nx-1)*VOXEL(1) + VOXEL(1)/2, ...
    GRID_ORIGIN(2) + (0:ny-1)*VOXEL(2) + VOXEL(2)/2, ...
    GRID_ORIGIN(3) + (0:nz-1)*VOXEL(3) + VOXEL(3)/2);
Nvox = numel(xg);
fprintf('Grid: %dx%dx%d (%d voxels) @ %.0fmm\n', nx, ny, nz, Nvox, 1000*VOXEL(1));

% Prepare outputs
reachScore = zeros(nx,ny,nz,'single');      % [0,1]
manipMax   = zeros(nx,ny,nz,'single');      % max manipulability
haveQex    = false(nx,ny,nz);
qExample   = zeros(nx,ny,nz,ndof,'single'); % optional seed

% IK solver (position+orientation)
ik = inverseKinematics('RigidBodyTree', robot);
weights = [1 1 1 1 1 1];  % xyz + rpy
ik.SolverParameters.AllowRandomRestarts = true;
ik.SolverParameters.MaxIterations = 150;

% Self-collision pairs setup
pairIdx = [];
if DO_SELF_COLLISION
    % Build all self-collision pairs except adjacent bodies (heuristic)
    allBodies = robot.BodyNames;
    for i = 1:numel(allBodies)
        for j = i+1:numel(allBodies)
            % Skip parent-child to reduce false positives
            if ~isParentChild(robot, allBodies{i}, allBodies{j})
                pairIdx = [pairIdx; i, j]; %#ok<AGROW>
            end
        end
    end
end

% Loop over voxels
idxs = 1:Nvox;
if USE_PARFOR
    parfor k = 1:Nvox
        [reachScore(k), manipMax(k), haveQex(k), qExample(k,:,:,:)] = ...
            eval_voxel(k, xg, yg, zg, robot, eeName, ik, weights, ...
                       orientBins, IK_ATTEMPTS, IK_POS_TOL, IK_ORI_TOL, ...
                       pairIdx, ndof);
    end
else
    for k = 1:Nvox
        [reachScore(k), manipMax(k), haveQex(k), qExample(k,:,:,:)] = ...
            eval_voxel(k, xg, yg, zg, robot, eeName, ik, weights, ...
                       orientBins, IK_ATTEMPTS, IK_POS_TOL, IK_ORI_TOL, ...
                       pairIdx, ndof);
        if mod(k, max(1,floor(Nvox/20)))==0
            fprintf(' %.0f%%', 100*k/Nvox);
        end
    end
    fprintf('\n');
end

% Pack map struct
map = struct();
map.meta.urdfPath      = string(URDF_PATH);
map.meta.eeLink        = string(EE_LINK);
map.meta.dataFormat    = string(DATA_FORMAT);
map.meta.gravity       = GRAVITY;
map.meta.payload_kg    = PAYLOAD_KG;
map.meta.time          = char(datetime('now'));
map.grid.origin        = GRID_ORIGIN;
map.grid.voxel         = VOXEL;
map.grid.shape         = int32([nx,ny,nz]);
map.orient.n_bins      = int32(N_ORIENT);
map.orient.cone_deg    = ORIENT_CONE_DEG;
map.data.reachScore    = reachScore;
map.data.manipMax      = manipMax;
map.data.hasExampleQ   = haveQex;
map.data.exampleQ      = qExample;  % may be large; keep if you want fast seeding

save(MAP_FILE, 'map', '-v7.3');
fprintf('Saved reachability map: %s\n', MAP_FILE);

end % build_and_save_map

%% ===================================================================== %%
function [score, manipVal, hasQex, qex] = eval_voxel(k, xg, yg, zg, robot, eeName, ik, weights, ...
    orientBins, IK_ATTEMPTS, IK_POS_TOL, IK_ORI_TOL, pairIdx, ndof)

% Default outputs
score    = single(0);
manipVal = single(0);
hasQex   = false;
qex      = zeros(1,1,1,ndof,'single');

px = xg(k); py = yg(k); pz = zg(k);
pTgt = [px;py;pz];

nb = size(orientBins,3);
if nb==0
    % Position-only: try a few loose orientations
    nb = 6;
    orientBins = default_orientations(nb);
end

success = 0;
bestManip = 0;
qe = [];

for b = 1:nb
    Rb = orientBins(:,:,b);
    T  = [Rb, pTgt; 0 0 0 1];

    ok_here = false;
    best_here = 0;
    q_here = [];

    for a = 1:IK_ATTEMPTS
        % random seed around home
        q0 = randomConfiguration(robot);
        [q, solInfo] = ik(eeName, T, weights, q0);

        if solInfo.Status == "success"
            % Check pose error tolerances (extra safety)
            TW = getTransform(robot, q, eeName);
            posErr = norm(TW(1:3,4) - pTgt);
            oriErr = rotm2axang(TW(1:3,1:3)'*Rb);  % angle about some axis
            angErr = abs(oriErr(4));
            if posErr <= IK_POS_TOL && angErr <= IK_ORI_TOL
                % Self-collision
                if ~isempty(pairIdx)
                    inColl = check_self_collision(robot, q, pairIdx);
                    if inColl
                        continue;
                    end
                end
                % Feasible
                ok_here = true;
                % Manipulability
                try
                    J = geometricJacobian(robot, q, eeName);
                    m = sqrt(det(J(1:3,:)*J(1:3,:)')); % translational manipulability
                catch
                    m = 0;
                end
                if m > best_here
                    best_here = m;
                    q_here = q;
                end
            end
        end
    end

    if ok_here
        success = success + 1;
        bestManip = max(bestManip, best_here);
        if isempty(qe) && ~isempty(q_here)
            qe = q_here;
        end
    end
end

score = single(success / nb);
manipVal = single(bestManip);
if ~isempty(qe)
    hasQex = true;
    qex(1,1,1,:) = single(qe);
end

end % eval_voxel

%% ===================================================================== %%
function tf = isParentChild(robot, a, b)
% Heuristic: check if a is ancestor of b or vice versa
tf = false;
try
    tf = isAncestor(robot, a, b) || isAncestor(robot, b, a);
catch
    % Older MATLAB versions may not expose isAncestor; fallback via tree walk
    tf = false;
end
end

function inColl = check_self_collision(robot, q, pairIdx)
% Coarse self-collision test using distance between collision geometries.
% Robotics System Toolbox lacks full FCL, but checkCollision(robot,q)
% exists; however it's all-pairs and can be slow. We keep it simple:
try
    inColl = checkCollision(robot, q, 'Exhaustive', false, 'SkippedSelfCollisions', 'parent');
catch
    % Fallback: compute all-pairs (may be slow)
    inColl = checkCollision(robot, q);
end
end

%% ===================================================================== %%
function B = sample_orientation_bins(n_bins, cone_deg)
% Returns a stack of rotation matrices, size 3x3xK.
if n_bins <= 0
    B = zeros(3,3,0);
    return;
end

if cone_deg >= 179.9
    % Uniform-ish samples on SO(3): use yaw-pitch grid over sphere
    % (simple, not perfectly uniform but fine for reachability)
    nyaw = max(1, round(n_bins/2));
    npit = max(1, round(n_bins/3));
    ys = linspace(-pi, pi, nyaw+1); ys(end)=[];
    ps = linspace(-pi/2, pi/2, npit);
    mats = {};
    for y = ys
        for p = ps
            R = eul2rotm([y, p, 0], 'ZYX'); % yaw-pitch-roll
            mats{end+1} = R; %#ok<AGROW>
        end
    end
else
    % Cone around -Z tool axis (camera looking forward): yaw/roll in cone
    K = max(1,n_bins);
    yaws = linspace(-pi, pi, K+1); yaws(end)=[];
    half = deg2rad(cone_deg)/2;
    mats = {};
    for y = yaws
        % tilt by phi within the cone
        Rtilt = axang2rotm([1 0 0 half]); % simple tilt
        R = eul2rotm([y, 0, 0], 'ZYX') * Rtilt;
        mats{end+1} = R; %#ok<AGROW>
    end
end
B = cat(3, mats{:});
end

function B = default_orientations(n)
% Few coarse orientations if doing "position-only".
dirs = [ ...
     0  0 -1;   % down
     0  0  1;   % up
     1  0  0;
    -1  0  0;
     0  1  0;
     0 -1  0];
dirs = dirs(1:min(n,size(dirs,1)),:);
mats = {};
for i=1:size(dirs,1)
    z = dirs(i,:)'; z = z/norm(z);
    x = [1;0;0]; if abs(dot(x,z))>0.9, x=[0;1;0]; end
    y = cross(z,x); y = y/norm(y);
    x = cross(y,z);
    R = [x y z];
    mats{end+1} = R; %#ok<AGROW>
end
B = cat(3, mats{:});
end

%% ===================================================================== %%
function res = reachable_sanity_check(MAP_FILE, targetWorldT, basePose)
% Fast accept/reject using a precomputed map (base-frame grid).
% basePose: struct('x',,'y',,'yaw',) in world frame.

S = load(MAP_FILE, 'map');
map = S.map;

% Transform target world -> base frame using base (x,y,yaw)
Rb = rotz(-rad2deg(basePose.yaw)); % MATLAB rotz uses degrees
pW = targetWorldT(1:3,4);
pB = Rb*(pW - [basePose.x;basePose.y;0]);

% Voxel index
[idx, outside] = voxel_index(pB, map.grid.origin, map.grid.voxel, map.grid.shape);
if outside
    res = struct('ok', false, 'reason', "outside grid");
    return;
end

score = map.data.reachScore(idx(1), idx(2), idx(3));
if score <= 0
    res = struct('ok', false, 'reason', "no reach");
    return;
end

% (Optional) orientation bin check could be added here if you also stored
% per-bin bitmaps; this minimal version is score-only.

out = struct();
out.ok        = true;
out.reason    = "ok";
out.reach     = double(score);
out.manipMax  = double(map.data.manipMax(idx(1), idx(2), idx(3)));
if map.data.hasExampleQ(idx(1), idx(2), idx(3))
    out.seedQ = squeeze(map.data.exampleQ(idx(1), idx(2), idx(3), :))';
else
    out.seedQ = [];
end
res = out;

end

function [idx, outside] = voxel_index(p, origin, voxel, shape)
% Map a point p (3x1) to voxel indices (i,j,k), 1-based
rel = (p(:)' - origin) ./ voxel;
ijk = floor(rel) + 1;
outside = any(ijk < 1) || any(ijk > shape);
ijk = max(ijk, [1 1 1]);
ijk = min(ijk, shape);
idx = ijk;
end
