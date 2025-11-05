#!/usr/bin/env python3
"""Visualize a trajectory JSON containing a top-level "poses" list with "position" and "orientation".

Usage:
    python linux_env_dev/visualize_trajectory.py /path/to/traj.json [--save] [--no-show]

Outputs:
    - By default shows an interactive matplotlib window (unless --no-show).
    - If --save is passed, saves a PNG next to the JSON with suffix _trajectory.png.

The script supports 2D (x,y) or 3D (x,y,z) visualization. Color encodes time index.
"""
import argparse
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def load_positions_from_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    if 'poses' not in data or not isinstance(data['poses'], list):
        raise ValueError("JSON must contain a top-level 'poses' list")
    positions = []
    for i, p in enumerate(data['poses']):
        if not isinstance(p, dict) or 'position' not in p:
            raise ValueError(f"pose at index {i} missing 'position' field")
        pos = p['position']
        if not (isinstance(pos, list) or isinstance(pos, tuple)):
            raise ValueError(f"pose.position at index {i} is not a list/tuple")
        positions.append(tuple(float(x) for x in pos))
    return np.array(positions, dtype=float)


def plot_trajectory(positions, title=None, save_path=None, show=True):
    n = positions.shape[0]
    dim = positions.shape[1]
    cmap = plt.get_cmap('viridis')

    # helper: safe normalization
    def _safe_norm(v):
        norm = np.linalg.norm(v)
        return v / (norm + 1e-12)

    # compute segment directions (unit vectors) for adjacent points
    if n >= 2:
        seg_dirs = []
        for i in range(n - 1):
            v = positions[i + 1] - positions[i]
            seg_dirs.append(_safe_norm(v))
        seg_dirs = np.vstack(seg_dirs)
    else:
        seg_dirs = np.zeros((0, dim))

    # quaternion helpers (w, x, y, z)
    def axis_angle_to_quat(axis, angle):
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        w = np.cos(angle / 2.0)
        xyz = axis * np.sin(angle / 2.0)
        return np.array([w, xyz[0], xyz[1], xyz[2]], dtype=float)

    def quat_to_euler_rpy(q):
        # input q = [w, x, y, z]
        w, x, y, z = q
        # roll (x-axis rotation)
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)
        # pitch (y-axis rotation)
        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch = np.arcsin(t2)
        # yaw (z-axis rotation)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)
        return np.array([roll, pitch, yaw], dtype=float)

    # compute absolute per-segment Euler angles (roll, pitch, yaw) for each segment
    # We'll align angles to position indices by storing the i-th segment's angles at euler_changes[i+1]
    # so index 0 remains zeros (no previous point).
    euler_changes = np.zeros((n, 3), dtype=float)
    if seg_dirs.shape[0] >= 1:
        # for each segment (i = 0..n-2), compute its direction's yaw and pitch
        for i in range(seg_dirs.shape[0]):
            v = seg_dirs[i]
            # yaw: rotation around Z to align X axis to projection on XY plane
            yaw = float(np.arctan2(v[1], v[0]))
            # pitch: rotation around Y (negative if using aviation conventions). Here we take pitch = atan2(z, sqrt(x^2+y^2))
            horiz_norm = float(np.linalg.norm(v[:2]))
            pitch = float(np.arctan2(v[2], horiz_norm)) if dim == 3 else 0.0
            # roll is ambiguous for a direction-only frame; set to zero
            roll = 0.0
            # store at index i+1 so the angle corresponds to the point after the segment
            euler_changes[i + 1] = np.array([roll, pitch, yaw], dtype=float)

    # plotting layout: 3 rows x 3 cols
    # top row: [trajectory | spacing vs idx | cumulative length vs idx]
    # second row: [roll | pitch | yaw]
    # third row: [x | y | z]
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 0.8, 0.8])

    # top-left: trajectory (3D or 2D)
    if dim == 3:
        ax_traj = fig.add_subplot(gs[0, 0], projection='3d')
        xs = positions[:, 0]
        ys = positions[:, 1]
        zs = positions[:, 2]
        ax_traj.plot(xs, ys, zs, linestyle='-', linewidth=1.5, color='gray', alpha=0.4)
        sc = ax_traj.scatter(xs, ys, zs, c=np.arange(n), cmap='viridis', s=20)
        ax_traj.scatter([xs[0]], [ys[0]], [zs[0]], color='green', marker='o', s=80, label='start')
        ax_traj.scatter([xs[-1]], [ys[-1]], [zs[-1]], color='red', marker='X', s=100, label='end')
        ax_traj.set_xlabel('X (m)')
        ax_traj.set_ylabel('Y (m)')
        ax_traj.set_zlabel('Z (m)')
    else:
        ax_traj = fig.add_subplot(gs[0, 0])
        xs = positions[:, 0]
        ys = positions[:, 1]
        ax_traj.plot(xs, ys, linestyle='-', linewidth=1.5, color='gray', alpha=0.4)
        sc = ax_traj.scatter(xs, ys, c=np.arange(n), cmap='viridis', s=20)
        ax_traj.scatter(xs[0], ys[0], color='green', marker='o', s=80, label='start')
        ax_traj.scatter(xs[-1], ys[-1], color='red', marker='X', s=100, label='end')
        ax_traj.set_xlabel('X (m)')
        ax_traj.set_ylabel('Y (m)')

    # compute adjacent distances and cumulative length
    if n >= 2:
        dists = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        cumlen = np.concatenate([[0.0], np.cumsum(dists)])
    else:
        dists = np.array([], dtype=float)
        cumlen = np.zeros((n,), dtype=float)

    # top-middle: spacing vs index (segment index 0..n-2)
    ax_spacing = fig.add_subplot(gs[0, 1])
    if dists.size > 0:
        ax_spacing.plot(np.arange(dists.shape[0]), dists, marker='.', linestyle='-', color='C3')
        ax_spacing.set_xlabel('segment index')
        ax_spacing.set_ylabel('distance (m)')
        ax_spacing.set_title('Adjacent Point Spacing')
        ax_spacing.grid(True)
    else:
        ax_spacing.text(0.5, 0.5, 'No spacing data', ha='center', va='center')

    # top-right: cumulative length vs point index (0..n-1)
    ax_cum = fig.add_subplot(gs[0, 2])
    if cumlen.size > 0:
        ax_cum.plot(np.arange(cumlen.shape[0]), cumlen, marker='o', linestyle='-', color='C4')
        ax_cum.set_xlabel('point index')
        ax_cum.set_ylabel('cumulative length (m)')
        ax_cum.set_title('Cumulative Length Along Trajectory')
        ax_cum.grid(True)
    else:
        ax_cum.text(0.5, 0.5, 'No length data', ha='center', va='center')

    if title:
        plt.suptitle(title)
    plt.legend(loc='best')
    plt.colorbar(sc, ax=ax_traj, label='time index')

    # second row: roll, pitch, yaw plots
    ax_roll = fig.add_subplot(gs[1, 0])
    ax_pitch = fig.add_subplot(gs[1, 1])
    ax_yaw = fig.add_subplot(gs[1, 2])

    t_idx = np.arange(n)
    roll = euler_changes[:, 0]
    pitch = euler_changes[:, 1]
    yaw = euler_changes[:, 2]

    ax_roll.plot(t_idx, roll, color='C0', linewidth=1.2)
    ax_roll.set_ylabel('roll (rad)')
    ax_roll.grid(True, alpha=0.3)

    ax_pitch.plot(t_idx, pitch, color='C1', linewidth=1.2)
    ax_pitch.set_ylabel('pitch (rad)')
    ax_pitch.grid(True, alpha=0.3)

    ax_yaw.plot(t_idx, yaw, color='C2', linewidth=1.2)
    ax_yaw.set_ylabel('yaw (rad)')
    ax_yaw.set_xlabel('step index')
    ax_yaw.grid(True, alpha=0.3)

    # third row: x, y, z coordinates over the trajectory
    ax_x = fig.add_subplot(gs[2, 0])
    ax_y = fig.add_subplot(gs[2, 1])
    ax_z = fig.add_subplot(gs[2, 2])

    if dim >= 1:
        x_vals = positions[:, 0]
    else:
        x_vals = np.zeros((n,), dtype=float)
    if dim >= 2:
        y_vals = positions[:, 1]
    else:
        y_vals = np.zeros((n,), dtype=float)
    if dim == 3:
        z_vals = positions[:, 2]
    else:
        z_vals = np.zeros((n,), dtype=float)

    ax_x.plot(t_idx, x_vals, color='C3', linewidth=1.0)
    ax_x.set_ylabel('X (m)')
    ax_x.set_xlabel('point index')
    ax_x.grid(True, alpha=0.3)

    ax_y.plot(t_idx, y_vals, color='C4', linewidth=1.0)
    ax_y.set_ylabel('Y (m)')
    ax_y.set_xlabel('point index')
    ax_y.grid(True, alpha=0.3)

    ax_z.plot(t_idx, z_vals, color='C5', linewidth=1.0)
    ax_z.set_ylabel('Z (m)')
    ax_z.set_xlabel('point index')
    ax_z.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"Saved trajectory plot to: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Visualize trajectory JSON')
    parser.add_argument('json_path', type=str, help='Path to trajectory JSON')
    parser.add_argument('--save', action='store_true', help='Save PNG next to JSON')
    parser.add_argument('--no-show', action='store_true', help='Do not show interactive window')
    args = parser.parse_args(argv)
    args.save = True
    args.no_show = True

    json_path = args.json_path
    if not os.path.exists(json_path):
        print(f"Path not found: {json_path}")
        sys.exit(2)

    # prepare output directory if saving
    visual_dir = os.path.join(os.path.dirname(__file__), 'visual_traj')
    if args.save:
        os.makedirs(visual_dir, exist_ok=True)

    # If the user passed a directory, process every .json file inside
    if os.path.isdir(json_path):
        files = sorted([os.path.join(json_path, f) for f in os.listdir(json_path) if f.lower().endswith('.json')])
        if not files:
            print(f"No JSON files found in directory: {json_path}")
            sys.exit(3)
        print(f"Found {len(files)} JSON files in {json_path}; processing...")
        # when processing multiple files, do not show interactive windows unless explicitly requested
        show_each = (not args.no_show) and len(files) == 1
        for f in files:
            try:
                positions = load_positions_from_json(f)
            except Exception as e:
                print(f"Failed to load {f}: {e}")
                continue
            # compute adjacent distances and report average
            if positions.shape[0] < 2:
                print(f"Trajectory {f} has fewer than 2 points; cannot compute adjacent spacing.")
            else:
                dists = np.linalg.norm(np.diff(positions, axis=0), axis=1)
                avg = float(np.mean(dists))
                print(f"Trajectory {os.path.basename(f)}: mean adjacent spacing = {avg:.6f} m (n={dists.shape[0]})")
            save_path = None
            if args.save:
                base_name = os.path.splitext(os.path.basename(f))[0]
                save_path = os.path.join(visual_dir, base_name + '_trajectory.png')
            title = os.path.basename(f)
            print(f"Plotting {f} -> {save_path or 'display'}")
            plot_trajectory(positions, title=title, save_path=save_path, show=show_each)
        print("Done.")
        return

    # single file
    try:
        positions = load_positions_from_json(json_path)
    except Exception as e:
        print(f"Failed to load positions from JSON: {e}")
        sys.exit(3)

    # compute adjacent distances and print mean spacing
    if positions.shape[0] < 2:
        print(f"Trajectory {os.path.basename(json_path)} has fewer than 2 points; cannot compute adjacent spacing.")
    else:
        dists = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        avg = float(np.mean(dists))
        print(f"Trajectory {os.path.basename(json_path)}: mean adjacent spacing = {avg:.6f} m (n={dists.shape[0]})")

    save_path = None
    if args.save:
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        save_path = os.path.join(visual_dir, base_name + '_trajectory.png')

    title = os.path.basename(json_path)
    plot_trajectory(positions, title=title, save_path=save_path, show=not args.no_show)


if __name__ == '__main__':
    main()
