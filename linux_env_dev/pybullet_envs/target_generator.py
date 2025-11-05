import abc
import numpy as np
from typing import Tuple, Optional


class TargetGenerator(abc.ABC):
    """Abstract base for generating targets over time/steps/episodes.

    Note: step() may receive the current end-effector position via ee_pos when
    the environment supports asking the generator for a target based on current
    robot state (e.g. nearest-point lookup). Implementations should accept the
    optional ee_pos parameter even if they ignore it.
    """

    @abc.abstractmethod
    def reset(self, rng: Optional[np.random.Generator] = None):
        """Called at env.reset(); return initial target (x,y,z) as numpy array."""

    @abc.abstractmethod
    def step(self, step_count: int, ee_pos: Optional[np.ndarray] = None) -> np.ndarray:
        """Called every env.step(); returns current target (x,y,z).

        ee_pos: Optional current end-effector/base position (3,) in world frame.
        """


class FixedTarget(TargetGenerator):
    def __init__(self, target: Tuple[float, float, float]):
        self._target = np.array(target, dtype=np.float32)

    def reset(self, rng: Optional[np.random.Generator] = None):
        return self._target

    def step(self, step_count: int) -> np.ndarray:
        return self._target

class RandomTargetForEpisode(TargetGenerator):
    def __init__(self, low: Tuple[float, float, float], high: Tuple[float, float, float], seed: Optional[int] = None):
        self.low = np.array(low, dtype=float)
        self.high = np.array(high, dtype=float)
        if seed is None:
            import secrets
            seed = secrets.randbits(63)
        self.rng = np.random.default_rng(int(seed))
        self._current = None

    def reset(self, rng: Optional[np.random.Generator] = None):
        if rng is not None:
            self.rng = rng
        self._current = self.rng.uniform(self.low, self.high).astype(np.float32)
        return self._current

    def step(self, step_count: int, ee_pos: Optional[np.ndarray] = None) -> np.ndarray:
        if self._current is None:
            return self.reset()
        return self._current

class RandomTarget(TargetGenerator):
    def __init__(self, low: Tuple[float, float, float], high: Tuple[float, float, float], seed: Optional[int] = None,
                 respawn_interval: int = 200):
        self.low = np.array(low, dtype=float)
        self.high = np.array(high, dtype=float)
        self.respawn_interval = int(respawn_interval)
        self.rng = np.random.default_rng(seed)
        self._current = None

    def reset(self, rng: Optional[np.random.Generator] = None):
        if rng is not None:
            self.rng = rng
        self._current = self.rng.uniform(self.low, self.high).astype(np.float32)
        self._last_change = 0
        return self._current

    def step(self, step_count: int, ee_pos: Optional[np.ndarray] = None) -> np.ndarray:
        if self._current is None:
            self.reset()
        if (step_count - getattr(self, '_last_change', 0)) >= self.respawn_interval:
            self._current = self.rng.uniform(self.low, self.high).astype(np.float32)
            self._last_change = step_count
        return self._current


class CurriculumTarget(TargetGenerator):
    def __init__(self, start: Tuple[float, float, float], end: Tuple[float, float, float], duration_steps: int):
        self.start = np.array(start, dtype=float)
        self.end = np.array(end, dtype=float)
        self.duration_steps = max(1, int(duration_steps))
        self._current = None

    def reset(self, rng: Optional[np.random.Generator] = None):
        self._current = self.start.astype(np.float32)
        return self._current

    def step(self, step_count: int, ee_pos: Optional[np.ndarray] = None) -> np.ndarray:
        t = float(min(step_count, self.duration_steps)) / float(self.duration_steps)
        cur = (1.0 - t) * self.start + t * self.end
        return cur.astype(np.float32)


# JSON-based nearest-point target generator
# import pdb; pdb.set_trace()
from .json_trajectory_loader import JSONTrajectory



class JSONNearestTargetGenerator(TargetGenerator):
    """Use a recorded JSON trajectory and pick target as nearest point projection.

    This generator treats lookahead in units of waypoints (integer/fractional).
    If you have time-based lookahead, convert seconds -> waypoints externally
    (waypoints = seconds * sample_rate) and pass that as lookahead_dt.
    """

    def __init__(self, json_path: str, lookahead_steps: int = 0, lookahead_dt: float = 1.0):
        self.traj = JSONTrajectory(json_path)
        self.lookahead_steps = int(lookahead_steps)
        # lookahead_dt is number-of-waypoints per lookahead step (default 1.0)
        self.lookahead_dt = float(lookahead_dt)
        # last known ee_pos for cases where step called without ee_pos
        self._last_ee_pos = None

    def reset(self, rng: Optional[np.random.Generator] = None):
        # default initial target is first pose
        self._last_ee_pos = None
        pos = self.traj.get_position(0, 0.0).astype(np.float32)
        return pos
    
    def test_vel(self):
        # 生成100个3维数组，形成[100,3]
        poss = np.random.uniform(-1, 1, (100, 3)).astype(np.float32)
        old_l = []
        new_l = []
        
        import time
        t0 = time.time()
        for p in poss:
            seg_idx, _, proj_pos, _ = self.traj.project_onto_trajectory(np.asarray(p, dtype=float))
            old_l.append((seg_idx, proj_pos))
        t1 = time.time()
        print(f"Old method time: {t1 - t0:.6f}s")

        t0 = time.time()
        for p in poss:
            seg_idx_new, proj_pos_new = self.traj.project_onto_trajectory_fast(np.asarray(p, dtype=float))
            new_l.append((seg_idx_new, proj_pos_new))
        t1 = time.time()
        print(f"New method time: {t1 - t0:.6f}s")
        import pdb; pdb.set_trace()

    def get_projection(self, ee_pos: Optional[np.ndarray] = None) -> np.ndarray:
        # seg_idx, alpha, proj_pos, proj_orient = self.traj.project_onto_trajectory(np.asarray(ee_pos, dtype=float))
        seg_idx, proj_pos = self.traj.project_onto_trajectory_fast(np.asarray(ee_pos, dtype=float))
        return seg_idx, proj_pos.astype(np.float32)
    
    def step(self,):
        pass
    
    def get_remaining_traj_nums(self, cur_id):
        total_len = self.traj.get_position_len()
        return total_len - cur_id, (total_len - cur_id) / total_len

    def find_gap_id(self, cur_id, fut_gap):
        cur_p = self.traj.get_position(cur_id, 0.0)
        for i in range(cur_id, self.traj.get_position_len()):
            next_p = self.traj.get_position(i, 0.0)
            diff = next_p - cur_p
            if np.linalg.norm(diff) > fut_gap:
                return i - cur_id
        return self.traj.get_position_len() - 1 - cur_id

    def get_lookahead(self, traj_id, fut_gap, ahead_num):
        trajs = []
        gap_id = self.find_gap_id(traj_id, fut_gap)
        for i in range(ahead_num):
            fut_id = min(traj_id + gap_id * (i + 1), self.traj.get_position_len() - 1)
            trajs.append(self.traj.get_position(fut_id, 0.0).astype(np.float32))
        return trajs

    def get_lookahead_bk(self, ee_pos: Optional[np.ndarray] = None) -> np.ndarray:
        if ee_pos is None:
            ee_pos = self._last_ee_pos
        else:
            self._last_ee_pos = np.asarray(ee_pos, dtype=float)

        if ee_pos is None:
            # fallback to first pose
            return self.traj.get_position(0, 0.0).astype(np.float32)

        seg_idx, alpha, proj, _ = self.traj.project_onto_trajectory(np.asarray(ee_pos, dtype=float))
        if self.lookahead_steps <= 0:
            return proj.astype(np.float32)

        look = self.traj.lookahead_from(seg_idx, alpha, self.lookahead_steps, self.lookahead_dt)
        look_pos = [np.array(pos, dtype=np.float32) for (pos, _) in look]
        
        return look_pos

