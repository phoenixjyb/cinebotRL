#!/usr/bin/env python3
"""Measure gripper (left_gripper_link) Z extrema by sampling joint space and visualize samples.

Saves:
 - JSON results with best configs
 - NPZ with sampled positions and joint values
 - PNG visualizations: top-down scatter (x,y colored by z) and z histogram

Run from repo root:
  python linux_env_dev/measure_and_visualize_gripper_z.py --samples 10000
"""
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

try:
    import pybullet as p
    import pybullet_data
except Exception as e:
    raise RuntimeError("pybullet is required: pip install pybullet") from e


URDF = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assets_own', 'mobile_manipulator_little_xy_link.urdf'))


def find_link_joint_index(robot, link_name):
    # return the joint index whose child link name matches link_name
    for i in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, i)
        child = info[12]
        if isinstance(child, bytes):
            child = child.decode()
        if child == link_name:
            return i
    return None


def find_joint_idx_by_name(robot, name):
    for i in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, i)
        jname = info[1]
        if isinstance(jname, bytes):
            jname = jname.decode()
        if jname == name:
            return i
    return None


def get_limits(robot, joint_names):
    limits = {}
    for n in joint_names:
        idx = find_joint_idx_by_name(robot, n)
        if idx is None:
            raise KeyError(f"Joint {n} not found in URDF")
        info = p.getJointInfo(robot, idx)
        lower = info[8]
        upper = info[9]
        if lower is None or not np.isfinite(lower):
            lower = -10.0
        if upper is None or not np.isfinite(upper):
            upper = 10.0
        limits[n] = (float(lower), float(upper), int(idx))
    return limits


def sample_and_measure(n_samples=10000, include_base=False, seed=0, out_prefix='linux_env_dev/gripper_z'):
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    robot = p.loadURDF(URDF, useFixedBase=False)

    arm_joint_names = [
        'left_arm_joint1', 'left_arm_joint2', 'left_arm_joint3',
        'left_arm_joint4', 'left_arm_joint5', 'left_arm_joint6'
    ]
    base_joint_names = ['joint_x', 'joint_y', 'joint_theta'] if include_base else []
    joint_names = base_joint_names + arm_joint_names

    limits = get_limits(robot, joint_names)

    link_idx = find_link_joint_index(robot, 'left_gripper_link')
    if link_idx is None:
        # if direct child not found, try searching by name among link indices
        for li in range(p.getNumJoints(robot)):
            info = p.getJointInfo(robot, li)
            child = info[12]
            if isinstance(child, bytes):
                child = child.decode()
            if child == 'left_gripper_link':
                link_idx = li
                break
    if link_idx is None:
        p.disconnect()
        raise RuntimeError('left_gripper_link not found')

    names = list(limits.keys())
    lows = np.array([limits[n][0] for n in names], dtype=float)
    ups  = np.array([limits[n][1] for n in names], dtype=float)

    rng = np.random.default_rng(seed)

    sampled_pos = np.zeros((n_samples, 3), dtype=float)
    sampled_cfg = np.zeros((n_samples, len(names)), dtype=float)

    best_max = -np.inf; best_min = np.inf
    best_max_cfg = None; best_min_cfg = None
    best_max_idx = None
    best_min_idx = None

    for i in range(n_samples):
        s = rng.random(len(names)) * (ups - lows) + lows
        sampled_cfg[i, :] = s
        for val, n in zip(s, names):
            idx = limits[n][2]
            try:
                p.resetJointState(robot, idx, targetValue=float(val))
            except Exception:
                pass
        # forward kinematics
        st = p.getLinkState(robot, link_idx, computeForwardKinematics=True)
        pos = st[0]
        sampled_pos[i, :] = np.array(pos, dtype=float)
        z = pos[2]
        if z > best_max:
            best_max = z
            best_max_cfg = dict(zip(names, [float(x) for x in s]))
            best_max_idx = i
        if z < best_min:
            best_min = z
            best_min_cfg = dict(zip(names, [float(x) for x in s]))
            best_min_idx = i

    # compute x/y extrema indices
    xs = sampled_pos[:, 0]
    ys = sampled_pos[:, 1]
    zs = sampled_pos[:, 2]
    x_max_idx = int(np.nanargmax(xs)) if sampled_pos.shape[0] > 0 else None
    x_min_idx = int(np.nanargmin(xs)) if sampled_pos.shape[0] > 0 else None
    y_max_idx = int(np.nanargmax(ys)) if sampled_pos.shape[0] > 0 else None
    y_min_idx = int(np.nanargmin(ys)) if sampled_pos.shape[0] > 0 else None

    # save results
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    npz_path = out_prefix + '_samples.npz'
    np.savez_compressed(npz_path, pos=sampled_pos, cfg=sampled_cfg, names=np.array(names))

    res = {
        'best_max_z': float(best_max),
        'best_max_cfg': best_max_cfg,
        'best_min_z': float(best_min),
        'best_min_cfg': best_min_cfg,
        'x_max': float(xs[x_max_idx]) if x_max_idx is not None else None,
        'x_min': float(xs[x_min_idx]) if x_min_idx is not None else None,
        'y_max': float(ys[y_max_idx]) if y_max_idx is not None else None,
        'y_min': float(ys[y_min_idx]) if y_min_idx is not None else None,
        'x_max_idx': x_max_idx,
        'x_min_idx': x_min_idx,
        'y_max_idx': y_max_idx,
        'y_min_idx': y_min_idx,
        'n_samples': int(n_samples),
        'names': names,
        'npz': npz_path
    }
    json_path = out_prefix + '_result.json'
    with open(json_path, 'w') as f:
        json.dump(res, f, indent=2)

    # visualizations
    xs = sampled_pos[:, 0]
    ys = sampled_pos[:, 1]
    zs = sampled_pos[:, 2]

    # 3D scatter: show x/y/z samples in 3D space, color by Z for clarity
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    sc3 = ax.scatter(xs, ys, zs, c=zs, cmap='viridis', s=8, alpha=0.8)
    cb = fig.colorbar(sc3, ax=ax, pad=0.1)
    cb.set_label('Z (m)')

    # mark best max/min positions in 3D
    try:
        if best_max_idx is not None:
            ax.scatter([sampled_pos[best_max_idx, 0]], [sampled_pos[best_max_idx, 1]], [sampled_pos[best_max_idx, 2]],
                       s=80, c='red', marker='X', label=f'max Z {best_max:.3f}m')
        if best_min_idx is not None:
            ax.scatter([sampled_pos[best_min_idx, 0]], [sampled_pos[best_min_idx, 1]], [sampled_pos[best_min_idx, 2]],
                       s=80, c='green', marker='D', label=f'min Z {best_min:.3f}m')

        # mark x/y extrema
        if x_max_idx is not None:
            ax.scatter([sampled_pos[x_max_idx, 0]], [sampled_pos[x_max_idx, 1]], [sampled_pos[x_max_idx, 2]],
                       s=60, c='blue', marker='s', label=f'max X {xs[x_max_idx]:.3f}m')
        if x_min_idx is not None:
            ax.scatter([sampled_pos[x_min_idx, 0]], [sampled_pos[x_min_idx, 1]], [sampled_pos[x_min_idx, 2]],
                       s=60, c='cyan', marker='o', label=f'min X {xs[x_min_idx]:.3f}m')
        if y_max_idx is not None:
            ax.scatter([sampled_pos[y_max_idx, 0]], [sampled_pos[y_max_idx, 1]], [sampled_pos[y_max_idx, 2]],
                       s=60, c='magenta', marker='^', label=f'max Y {ys[y_max_idx]:.3f}m')
        if y_min_idx is not None:
            ax.scatter([sampled_pos[y_min_idx, 0]], [sampled_pos[y_min_idx, 1]], [sampled_pos[y_min_idx, 2]],
                       s=60, c='orange', marker='v', label=f'min Y {ys[y_min_idx]:.3f}m')

        if True:
            ax.legend(loc='best', fontsize='small')
    except Exception:
        pass

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Gripper 3D samples colored by Z (n={n_samples})')
    # set a good viewing angle
    ax.view_init(elev=30, azim=-60)
    plt.title(f'Gripper top-down samples colored by Z (n={n_samples})')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    out_scatter = out_prefix + '_topdown.png'
    plt.tight_layout()
    plt.savefig(out_scatter, dpi=200)
    plt.close(fig)

    # z histogram + mark extrema
    fig2 = plt.figure(figsize=(6, 4))
    plt.hist(zs, bins=80, color='C0', alpha=0.8)
    plt.axvline(best_max, color='r', linestyle='--', label=f'max {best_max:.3f}m')
    plt.axvline(best_min, color='g', linestyle='--', label=f'min {best_min:.3f}m')
    plt.legend()
    plt.title('Histogram of sampled gripper Z')
    plt.xlabel('Z (m)')
    plt.ylabel('Count')
    out_hist = out_prefix + '_z_hist.png'
    plt.tight_layout()
    plt.savefig(out_hist, dpi=200)
    plt.close(fig2)

    # x histogram
    figx = plt.figure(figsize=(6, 4))
    plt.hist(xs, bins=80, color='C2', alpha=0.8)
    if x_max_idx is not None:
        plt.axvline(xs[x_max_idx], color='b', linestyle='--', label=f'x_max {xs[x_max_idx]:.3f}m')
    if x_min_idx is not None:
        plt.axvline(xs[x_min_idx], color='c', linestyle='--', label=f'x_min {xs[x_min_idx]:.3f}m')
    plt.legend()
    plt.title('Histogram of sampled gripper X')
    plt.xlabel('X (m)')
    plt.ylabel('Count')
    out_xhist = out_prefix + '_x_hist.png'
    plt.tight_layout()
    plt.savefig(out_xhist, dpi=200)
    plt.close(figx)

    # y histogram
    figy = plt.figure(figsize=(6, 4))
    plt.hist(ys, bins=80, color='C3', alpha=0.8)
    if y_max_idx is not None:
        plt.axvline(ys[y_max_idx], color='m', linestyle='--', label=f'y_max {ys[y_max_idx]:.3f}m')
    if y_min_idx is not None:
        plt.axvline(ys[y_min_idx], color='orange', linestyle='--', label=f'y_min {ys[y_min_idx]:.3f}m')
    plt.legend()
    plt.title('Histogram of sampled gripper Y')
    plt.xlabel('Y (m)')
    plt.ylabel('Count')
    out_yhist = out_prefix + '_y_hist.png'
    plt.tight_layout()
    plt.savefig(out_yhist, dpi=200)
    plt.close(figy)

    # --- save pose images for extrema (place base at origin first) ---
    try:
        # Load samples (we saved npz earlier): keys are 'pos','cfg','names'
        npz = np.load(npz_path, allow_pickle=True)
        samples = npz['cfg'] if 'cfg' in npz else None
        names_np = npz['names'] if 'names' in npz else None
    except Exception:
        samples = None
        names_np = None

    # helper to set robot to a given sample and set base at origin
    def _apply_sample_and_save(idx, out_png_prefix):
        if samples is None:
            return
        if idx is None or idx < 0 or idx >= samples.shape[0]:
            return
        cfg = samples[idx]
        # reset base to origin (position 0,0,base_z) and yaw 0
        try:
            # find base link height (keep current z of base or use 0.2 if unknown)
            base_z = 0.2
            try:
                base_state = p.getBasePositionAndOrientation(robot)
                base_z = float(base_state[0][2])
            except Exception:
                pass
            # set base pose to origin
            p.resetBasePositionAndOrientation(robot, [0.0, 0.0, base_z], p.getQuaternionFromEuler([0.0, 0.0, 0.0]))
        except Exception:
            pass

        # apply joint values: map by names -> joint index stored in `limits`
        try:
            for jval, jname in zip(cfg, names):
                jidx = limits[jname][2]
                try:
                    p.resetJointState(robot, int(jidx), float(jval))
                except Exception:
                    pass
        except Exception:
            pass

        # try to use environment save if available
        out_png = out_png_prefix + '.png'
        try:
            # Always use a fixed camera that keeps the base at the bottom and upright.
            cam_w, cam_h = 800, 600
            try:
                # attempt to read base z; fallback to 0.2
                try:
                    base_state = p.getBasePositionAndOrientation(robot)
                    base_z = float(base_state[0][2])
                except Exception:
                    base_z = 0.2

                # fixed camera: place farther back and slightly higher so the full robot fits
                cam_distance = 2.0  # meters behind the base
                cam_elev = 2.0      # meters above base
                eye = [cam_distance, -2.0, base_z + cam_elev]
                target = [0.0, 0.0, base_z + 0.25]
                up = [0, 0, 1]
                view = p.computeViewMatrix(cameraEyePosition=eye, cameraTargetPosition=target, cameraUpVector=up)
                proj = p.computeProjectionMatrixFOV(fov=60.0, aspect=cam_w / cam_h, nearVal=0.01, farVal=100.0)
                img_tuple = p.getCameraImage(cam_w, cam_h, viewMatrix=view, projectionMatrix=proj)
                rgba = img_tuple[2]
                arr = np.reshape(np.array(rgba, dtype=np.uint8), (cam_h, cam_w, 4))
                rgb_arr = arr[:, :, :3]
                # save as-is (no vertical flip) so base remains at bottom
                plt.imsave(out_png, rgb_arr)
            except Exception:
                # best-effort: create an empty placeholder file
                open(out_png, 'wb').close()
        except Exception:
            try:
                open(out_png, 'wb').close()
            except Exception:
                pass

    # grab variables used during sampling (these exist in this scope)
    try:
        robot = locals().get('robot', globals().get('robot'))
        joint_ids = locals().get('joint_ids', globals().get('joint_ids'))
    except Exception:
        robot = globals().get('robot')
        joint_ids = globals().get('joint_ids')

    extrema_map = {
        'z_max': best_max_idx,
        'z_min': best_min_idx,
        'x_max': x_max_idx,
        'x_min': x_min_idx,
        'y_max': y_max_idx,
        'y_min': y_min_idx,
    }

    out_dir = os.path.dirname(npz_path)
    for name, idx in extrema_map.items():
        out_png_prefix = os.path.join(out_dir, f'gripper_pose_{name}')
        _apply_sample_and_save(idx, out_png_prefix)
    print(f"Saved extrema pose images to: {out_dir}")
    

    # For each extrema (x/y/z min/max), reset joint states to the sampled config and save a rendered image
    def _save_extrema_pose(idx, axis, bound_name):
        """Set robot to sampled joint config at index `idx`, render and save an image.
        Returns the output path if saved, else None.
        """
        if idx is None:
            return None

        # set joint states according to sampled_cfg
        cfg = sampled_cfg[idx]
        for jpos, n in zip(cfg, names):
            jidx = limits[n][2]
            try:
                p.resetJointState(robot, jidx, targetValue=float(jpos))
            except Exception:
                pass

        # step simulation a few times to ensure forward kinematics updated
        for _ in range(3):
            try:
                p.stepSimulation()
            except Exception:
                pass

        # target is the gripper position
        tgt = sampled_pos[idx]
        # choose a camera eye offset relative to the target so pose is visible
        distance = 0.9
        # angled view: offset diagonally behind/above
        eye = [tgt[0] + distance, tgt[1] - distance, tgt[2] + 0.6]
        up = [0, 0, 1]

        # camera image size
        cam_w, cam_h = 800, 600

        try:
            view = p.computeViewMatrix(cameraEyePosition=eye, cameraTargetPosition=tgt.tolist(), cameraUpVector=up)
            proj = p.computeProjectionMatrixFOV(fov=60.0, aspect=cam_w / cam_h, nearVal=0.01, farVal=100.0)
            # try hardware renderer first; fall back if not available
            try:
                img_tuple = p.getCameraImage(cam_w, cam_h, viewMatrix=view, projectionMatrix=proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
            except Exception:
                img_tuple = p.getCameraImage(cam_w, cam_h, viewMatrix=view, projectionMatrix=proj)
        except Exception:
            return None

        if not img_tuple or len(img_tuple) < 3:
            return None

        try:
            w_img, h_img, rgb, depth = img_tuple[0], img_tuple[1], img_tuple[2], img_tuple[3]
            arr = np.reshape(np.array(rgb, dtype=np.uint8), (h_img, w_img, 4))
            rgb_arr = arr[:, :, :3]
            rgb_arr = np.flipud(rgb_arr)
            out_pose = out_prefix + f'_pose_{axis}_{bound_name}.png'
            plt.imsave(out_pose, rgb_arr)
            return out_pose
        except Exception:
            return None

    pose_files = {}
    mappings = [
        (best_max_idx, 'z', 'max'), (best_min_idx, 'z', 'min'),
        (x_max_idx, 'x', 'max'), (x_min_idx, 'x', 'min'),
        (y_max_idx, 'y', 'max'), (y_min_idx, 'y', 'min')
    ]
    for idx, ax, bn in mappings:
        pf = _save_extrema_pose(idx, ax, bn)
        if pf:
            pose_files[f"{ax}_{bn}"] = pf

    p.disconnect()
    return res, npz_path, out_scatter, out_hist, json_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=10000)
    parser.add_argument('--include_base', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out_prefix', type=str, default='linux_env_dev/gripper_z')
    args = parser.parse_args()

    print('URDF used:', URDF)
    print('Sampling... (this may take some seconds)')
    res, npz, sc, hist, js = sample_and_measure(n_samples=args.samples, include_base=args.include_base,
                                                seed=args.seed, out_prefix=args.out_prefix)
    print('Done. Results:')
    print(json.dumps(res, indent=2))
    print('Saved:', npz, sc, hist, js)
