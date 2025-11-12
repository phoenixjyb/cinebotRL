#!/usr/bin/env python3
"""Generate a random smooth trajectory and save as JSON.

Produces a JSON with top-level key "poses" where each pose has
  - position: [x,y,z]
  - orientation: [qx,qy,qz,qw]  (quaternion, x,y,z,w)

Trajectory parameters (defaults):
  - 300 points
  - step length = 0.01 m between consecutive points
    - z range in [0.85, 1.3]
  - smooth, continuous roll/pitch/yaw

Usage:
  python linux_env_dev/generate_random_trajectory.py --out trajectoryToLearn/world_json/scene_1/traj_random.json

If no --out is provided, a timestamped file will be written under
trajectoryToLearn/world_json/scene_1/
"""

import os
import json
import argparse
import math
import time
import numpy as np


def euler_to_quat(roll, pitch, yaw):
    """Convert Euler angles (roll, pitch, yaw) to quaternion (x,y,z,w).
    Uses the z-y'-x'' (yaw-pitch-roll) convention.
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return [float(x), float(y), float(z), float(w)]


def generate_trajectory(n_points=300, step=0.01, z_min=0.85, z_max=1.3, seed=None):
    rng = np.random.RandomState(seed)

    # Start near origin in x,y; choose starting z inside range (use midpoint)
    # per requirement: initial z should be the minimum (start at z_min)
    start_z = float(z_min)
    pos = np.array([0.0, 0.0, start_z], dtype=float)

    positions = [pos.copy()]
    orientations = []

    # Create smooth yaw, pitch, roll sequences by integrating small deltas
    yaw = 0.0
    pitch = 0.0
    roll = 0.0

    # delta noise std controls curvature; set small values for continuity
    yaw_std = 0.05  # radians per step
    pitch_std = 0.005
    roll_std = 0.005

    for i in range(n_points - 1):
        # small smooth updates
        yaw += rng.normal(scale=yaw_std)
        pitch += rng.normal(scale=pitch_std)
        roll += rng.normal(scale=roll_std)

        # build direction vector from yaw/pitch
        cp = math.cos(pitch)
        dir_x = cp * math.cos(yaw)
        dir_y = cp * math.sin(yaw)
        dir_z = math.sin(pitch)
        dir_vec = np.array([dir_x, dir_y, dir_z], dtype=float)
        # normalize and step
        norm = np.linalg.norm(dir_vec)
        if norm <= 1e-12:
            dir_vec = np.array([1.0, 0.0, 0.0], dtype=float)
            norm = 1.0
        dir_vec = dir_vec / norm
        next_pos = positions[-1] + dir_vec * float(step)

        positions.append(next_pos)

    positions = np.vstack(positions)

    # adjust z-range by shifting entire trajectory in z so it fits [z_min, z_max]
    zmin_traj = float(np.min(positions[:, 2]))
    zmax_traj = float(np.max(positions[:, 2]))
    span = zmax_traj - zmin_traj
    target_span = float(z_max - z_min)
    if span < 1e-12:
        # flat in z; set to midpoint
        z_shift = (z_min + z_max) / 2.0 - zmin_traj
        positions[:, 2] += z_shift
    else:
        # scale and shift to fit range while preserving shape
        scale = target_span / span
        positions[:, 2] = (positions[:, 2] - zmin_traj) * scale + z_min

    # Recompute orientations per-segment: for each point we compute roll/pitch/yaw
    # using the forward direction (segment i = point i-1 -> i), store orientation at each point
    # For first point where there's no prior segment, reuse the next segment's orientation
    orientations = []
    for i in range(positions.shape[0]):
        if i == 0:
            # use direction from point 0->1
            vec = positions[1] - positions[0]
        else:
            vec = positions[i] - positions[i - 1]
        horiz = math.hypot(float(vec[0]), float(vec[1]))
        yaw_i = math.atan2(float(vec[1]), float(vec[0]))
        pitch_i = math.atan2(float(vec[2]), horiz)
        # roll is set small and continuous: we can make roll a low-frequency function
        roll_i = 0.01 * math.sin(0.005 * i)
        q = euler_to_quat(roll_i, pitch_i, yaw_i)
        orientations.append(q)

    poses = []
    import pdb; pdb.set_trace()
    for pnt, q in zip(positions.tolist(), orientations):
        poses.append({
            'position': [float(pnt[0]), float(pnt[1]), float(pnt[2])],
            'orientation': q,
        })

    return {'poses': poses}


def main():
    parser = argparse.ArgumentParser(description='Generate random smooth trajectory JSON')
    parser.add_argument('--out', type=str, default=None, help='Output JSON file path')
    parser.add_argument('--n', type=int, default=300, help='Number of points')
    parser.add_argument('--step', type=float, default=0.01, help='Step length between consecutive points (m)')
    parser.add_argument('--zmin', type=float, default=0.85, help='Minimum z')
    parser.add_argument('--zmax', type=float, default=1.3, help='Maximum z')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    args = parser.parse_args()

    out = args.out
    if out is None:
        # use Asia/Shanghai timezone so filenames use Beijing time (UTC+8)
        try:
            from datetime import datetime
            from zoneinfo import ZoneInfo
            ts = datetime.now(ZoneInfo("Asia/Shanghai")).strftime('%Y%m%d_%H%M%S')
        except Exception:
            # fallback to local time
            ts = time.strftime('%Y%m%d_%H%M%S')
        out_dir = os.path.join('trajectoryToLearn', 'world_json', 'scene_1')
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f'traj_random_{ts}.json')
    else:
        os.makedirs(os.path.dirname(out) or '.', exist_ok=True)

    traj = generate_trajectory(n_points=args.n, step=args.step, z_min=args.zmin, z_max=args.zmax, seed=args.seed)
    with open(out, 'w') as f:
        json.dump(traj, f, indent=2)

    print(f'Wrote trajectory with {args.n} poses to: {out}')


if __name__ == '__main__':
    main()
