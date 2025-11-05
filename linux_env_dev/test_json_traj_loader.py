import os
import numpy as np
from linux_env_dev.pybullet_envs.json_trajectory_loader import JSONTrajectory


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sample = os.path.join(repo_root, "trajectoryToLearn", "world_json", "scene_1", "traj_1.json")
    if not os.path.exists(sample):
        # try alternative path relative
        sample = os.path.join(os.path.dirname(__file__), "..", "trajectoryToLearn", "world_json", "scene_1", "traj_1.json")
    sample = os.path.abspath(sample)
    print(f"Loading sample trajectory: {sample}")
    traj = JSONTrajectory(sample)
    import pdb; pdb.set_trace()

    # pick an example EE position near the start
    ee_pos = traj.get_position(5, 0.3) + np.array([0.05, -0.03, 0.0])
    print("Example ee_pos:", ee_pos)

    seg_idx, alpha, proj, orient = traj.project_onto_trajectory(ee_pos)
    print(f"Nearest projection -> segment {seg_idx}, alpha={alpha:.3f}, proj={proj}")

    # here step_dt is in units of waypoints; we previously used waypoint_dt=0.02
    # and step_dt=0.1s -> waypoints = 0.1 / 0.02 = 5 waypoints per lookahead
    look = traj.lookahead_from(seg_idx, alpha, steps=5, step_dt=5.0)
    print("Lookahead positions:")
    for i, (p, q) in enumerate(look):
        print(f"  +{i+1}: pos={p}, quat={q}")


if __name__ == '__main__':
    main()
