import abc
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
import math


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

    json_paths may be a single path (string) passed as `json_path=` keyword
    for backward compatibility, or a list of paths. One path is selected
    randomly at construction (or deterministically if `seed` is provided).
    """

    def __init__(self, json_paths: Optional[List[str]] = None, json_txt: Optional[str] = None,
                 mode: str = "random", lookahead_steps: int = 0, lookahead_dt: float = 1.0):
        # config
        self.json_paths = json_paths
        self.json_txt = json_txt
        self.read_all_jsons()
        self.traj_mode = mode
        if self.traj_mode == "random":
            self.random_select_path()
        elif self.traj_mode == "seq":
            self.sel_idx = 0
            self.traj = JSONTrajectory(self.json_paths[self.sel_idx])
        else:
            raise ValueError(f"Unknown mode: {mode}")
        self.lookahead_steps = int(lookahead_steps)
        # lookahead_dt is number-of-waypoints per lookahead step (default 1.0)
        self.lookahead_dt = float(lookahead_dt)
        # last known ee_pos for cases where step called without ee_pos
        self.step_dt = 0.1 # 控制频率 10Hz
        self.cruise_speed = 0.2
        self.ramp_up_time = 2.0
        self.ramp_down_time = 2.0
        
        # traj info
        self._last_ee_pos = None
        self.len_per_point = None
        self.total_length = None

    def read_json_from_txt(self,):
        paths: List[str] = []
        if self.json_txt is None:
            raise ValueError("json_txt is None; cannot read paths.")
        with open(self.json_txt, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    paths.append(line)
        # print(f"{paths[:5]}")
        # import pdb; pdb.set_trace()
        self.json_paths = paths[:]
        # self.json_paths = ['linux_env_dev/new_json_50/cinematic_db_arc_left_push_arc_left_push_090.json']
    
    def read_all_jsons(self):
        if self.json_txt is not None:
            # txt读取优先，否则json
            self.read_json_from_txt() # 覆盖self.json_paths
        else:
            self.json_paths = self.json_paths
        print(f"Total {len(self.json_paths)} paths available")
    
    def random_select_path(self):
        paths: List[str] = self.json_paths
        # RNG for deterministic selection if seed provided
        _rng = np.random.default_rng()
        self.sel_idx = int(_rng.integers(0, len(paths)))
        selected_path = paths[self.sel_idx]

        self.traj = JSONTrajectory(selected_path)
    
    def get_traj_path(self,):
        return self.json_paths[self.sel_idx]

    def get_path_heading(self,):
        preview_dist = 0.1
        point_start = self.traj.positions[0]
        for i in range(1, self.traj.get_position_len()):
            point_10i = self.traj.positions[10 * i]
            vec = point_10i[0:2] - point_start[0:2]
            dist = np.linalg.norm(vec)
            if dist >= preview_dist:
                heading = math.atan2(float(vec[1]), float(vec[0]))
                return heading
        # import pdb; pdb.set_trace()
        return 0.0
    
    def reset(self, rng: Optional[np.random.Generator] = None):
        # default initial target is first pose
        self._last_ee_pos = None
        self.len_per_point = None
        self.total_length = None
        self.traj = None
        if self.traj_mode == "random":
            self.random_select_path()
            # print(f"Random to {self.sel_idx}th path")
        elif self.traj_mode == "seq":
            self.sel_idx = (self.sel_idx + 1) % len(self.json_paths)
            self.traj = JSONTrajectory(self.json_paths[self.sel_idx])
            print(f"Switch to {self.sel_idx}th path: {self.json_paths[self.sel_idx]}, "
                  f"final p = {self.traj.positions[-1]}")
        else:
            raise ValueError(f"Unknown mode: {self.traj_mode}")
        pos = self.traj.positions[0].astype(np.float32)
        return pos

    def cal_length_per_point_fast(self,):
        self.len_per_point, self.total_length = self.traj.len_per_point, self.traj.total_length

    def compute_s_from_step(self, step):
        """Compute the traveled arc-length s for a given discrete step using the
        configured speed profile. Returns s in meters (float).
        """
        step_dt = getattr(self, 'step_dt', 0.1)
        v_cruise = getattr(self, 'cruise_speed', 0.3)
        ramp_up = getattr(self, 'ramp_up_time', 1.0)
        ramp_down = getattr(self, 'ramp_down_time', 1.0)

        # ensure cumulative lengths computed
        if not hasattr(self, 'len_per_point'):
            self.cal_length_per_point_fast()

        total_len = self.total_length
        if total_len <= 0.0:
            raise ValueError("Total trajectory length is zero or negative.")

        T_total = total_len / max(1e-12, v_cruise) + 0.5 * (ramp_up + ramp_down)
        t = float(step) * float(step_dt)

        if t <= 0.0:
            s = 0.0
        elif t < ramp_up:
            s = v_cruise * (t * t) / (2.0 * ramp_up)
        elif t < (T_total - ramp_down):
            s_ramp = 0.5 * v_cruise * ramp_up
            s = s_ramp + v_cruise * (t - ramp_up)
        elif t < T_total:
            tau = t - (T_total - ramp_down)
            s = total_len - 0.5 * v_cruise * ((ramp_down - tau) ** 2) / max(1e-12, ramp_down)
        else:
            s = total_len

        return float(max(0.0, min(s, total_len)))

    def _s_to_idx_alpha(self, s):
        """Map arc-length s to (segment_index, alpha) on the trajectory."""
        if not hasattr(self, 'len_per_point'):
            self.cal_length_per_point_fast()
        lpp = self.len_per_point
        n = len(lpp)
        if n == 0:
            return 0, 0.0
        if s >= lpp[-1]:
            return n - 1, 0.0

        lo = 0
        hi = n - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if lpp[mid] <= s:
                if lpp[mid + 1] > s:
                    lo = mid
                    break
                lo = mid + 1
            else:
                hi = mid - 1
        i = lo
        if i >= n - 1:
            return n - 1, 0.0
        seg_start = lpp[i]
        seg_end = lpp[i + 1]
        seg_len = max(1e-12, float(seg_end - seg_start))
        alpha = float((s - seg_start) / seg_len)
        return int(i), float(alpha)


    def cal_target_given_speed_profile(self, step):
        """Return a target position (3,) for a given episode step index based on a
        time-based speed profile.

        Speed profile: start at 0, linearly ramp to cruise_speed over ramp_up_time,
        cruise at cruise_speed, then linearly ramp down to 0 over ramp_down_time so
        that the trajectory end is reached exactly at the end of motion.

        Arguments:
            step: integer step index (env.step count). We interpret time = step * step_dt.

        Returns: np.ndarray shape (3,) world position on the trajectory.
        """
        # defaults for timing/speed (seconds and m/s)
        step_dt = self.step_dt
        v_cruise = self.cruise_speed
        ramp_up = self.ramp_up_time
        ramp_down = self.ramp_down_time

        # ensure cumulative lengths computed
        if self.len_per_point is None:
            self.cal_length_per_point_fast()

        total_len = self.total_length
        if total_len <= 0.0:
            # degenerate: fallback to first pose
            raise ValueError("Total trajectory length is zero or negative.")

        # compute total time needed so that integral of v(t) equals total_len
        # integral over profile = v_cruise * (T - 0.5*(ramp_up + ramp_down))
        T_total = total_len / max(1e-12, v_cruise) + 0.5 * (ramp_up + ramp_down)

        t = float(step) * float(step_dt)
        # clamp t
        if t <= 0.0:
            s = 0.0
        elif t < ramp_up:
            # accelerating: v(t) = v_cruise * (t / ramp_up)
            s = v_cruise * (t * t) / (2.0 * ramp_up)
        elif t < (T_total - ramp_down):
            # cruise phase
            s_ramp = 0.5 * v_cruise * ramp_up
            s = s_ramp + v_cruise * (t - ramp_up)
        elif t < T_total:
            # deceleration phase
            # tau = time since start of decel
            tau = t - (T_total - ramp_down)
            # s = total_len - 0.5 * v_cruise * (ramp_down - tau)**2 / ramp_down
            s = total_len - 0.5 * v_cruise * ((ramp_down - tau) ** 2) / max(1e-12, ramp_down)
        else:
            s = total_len

        # clamp s
        s = float(max(0.0, min(s, total_len)))

        # map arc-length s to trajectory index and alpha
        # len_per_point[i] = cumulative length up to point i
        lpp = self.len_per_point
        n = len(lpp)
        if s >= lpp[-1]:
            return n - 1, self.traj.positions[n - 1].astype(np.float32)

        # binary search for index i where lpp[i] <= s < lpp[i+1]
        lo = 0
        hi = n - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if lpp[mid] <= s:
                if lpp[mid + 1] > s:
                    lo = mid
                    break
                lo = mid + 1
            else:
                hi = mid - 1
        i = lo
        # ensure bounds
        if i >= n - 1:
            return n - 1, self.traj.positions[n - 1].astype(np.float32)

        return i, self.traj.positions[i].astype(np.float32)
    
    def test_target_given_speed_profile(self,):
        """Sample targets over steps and save a plot of X/Y/Z vs step.

        Returns the path to the saved PNG.
        """
        steps = 500
        xs = []
        ys = []
        zs = []
        for i in range(0, steps):
            pos = self.cal_target_given_speed_profile(i)
            pos = np.asarray(pos, dtype=float)
            xs.append(float(pos[0]))
            ys.append(float(pos[1]))
            zs.append(float(pos[2]))

        steps_arr = np.arange(0, steps)

        # also collect ids (i + alpha) for each step
        ids = []
        for i_step in range(0, steps):
            s = self.compute_s_from_step(i_step)
            idx, alpha = self._s_to_idx_alpha(s)
            ids.append(float(idx) + float(alpha))

        fig, axs = plt.subplots(4, 1, figsize=(9, 8), constrained_layout=True)
        axs[0].plot(steps_arr, xs, color='tab:blue', label='target_x')
        axs[0].set_ylabel('X (m)')
        axs[0].legend()

        axs[1].plot(steps_arr, ys, color='tab:orange', label='target_y')
        axs[1].set_ylabel('Y (m)')
        axs[1].legend()

        axs[2].plot(steps_arr, zs, color='tab:green', label='target_z')
        axs[2].set_ylabel('Z (m)')
        axs[2].set_xlabel('env step')
        axs[2].legend()

        fig.suptitle('Target position vs step (speed profile)')

        # plot the fractional trajectory id (index + alpha)
        axs[3].plot(steps_arr, np.asarray(ids, dtype=float), color='tab:purple', label='traj_id (i + alpha)')
        axs[3].set_ylabel('traj id')
        axs[3].set_xlabel('env step')
        axs[3].legend()

        out_dir = os.path.join(os.getcwd(), 'linux_env_dev', 'plots')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'target_vs_step.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return out_path
        
    
    def test_vel(self):
        # 生成100个3维数组，形成[100,3]
        poss = np.random.uniform(-1, 1, (100, 3)).astype(np.float32)
        old_l = []
        new_l = []
        
        import time
        t0 = time.time()
        # for p in poss:
        #     seg_idx, _, proj_pos, _ = self.traj.project_onto_trajectory(np.asarray(p, dtype=float))
        #     old_l.append((seg_idx, proj_pos))
        for i in range(100):
            self.traj.get_position(i, 0.0)
        t1 = time.time()
        print(f"Old method time: {t1 - t0:.6f}s")

        t0 = time.time()
        # for p in poss:
        #     seg_idx_new, proj_pos_new = self.traj.project_onto_trajectory_fast(np.asarray(p, dtype=float))
        #     new_l.append((seg_idx_new, proj_pos_new))
        for i in range(100):
            self.traj.positions[i]
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
        cur_p = self.traj.positions[cur_id]
        for i in range(cur_id, self.traj.get_position_len()):
            next_p = self.traj.positions[i]
            diff = next_p - cur_p
            if np.linalg.norm(diff) > fut_gap:
                return i - cur_id
        return self.traj.get_position_len() - 1 - cur_id

    def get_lookahead(self, traj_id, fut_gap, ahead_num):
        trajs = []
        gap_id = self.find_gap_id(traj_id, fut_gap)
        for i in range(ahead_num):
            fut_id = min(traj_id + gap_id * (i + 1), self.traj.get_position_len() - 1)
            trajs.append(self.traj.positions[fut_id].astype(np.float32))
        return trajs

    def get_lookahead_bk(self, ee_pos: Optional[np.ndarray] = None) -> np.ndarray:
        if ee_pos is None:
            ee_pos = self._last_ee_pos
        else:
            self._last_ee_pos = np.asarray(ee_pos, dtype=float)

        if ee_pos is None:
            # fallback to first pose
            return self.traj.positions[0].astype(np.float32)

        seg_idx, alpha, proj, _ = self.traj.project_onto_trajectory(np.asarray(ee_pos, dtype=float))
        if self.lookahead_steps <= 0:
            return proj.astype(np.float32)

        look = self.traj.lookahead_from(seg_idx, alpha, self.lookahead_steps, self.lookahead_dt)
        look_pos = [np.array(pos, dtype=np.float32) for (pos, _) in look]
        
        return look_pos


class CurriculumJSONNearestTargetGenerator(TargetGenerator):
    """Curriculum wrapper around two JSONNearestTargetGenerator instances.

    Selects stage2 with probability `stage2_prob` at each reset and delegates
    all other methods/attributes to the active generator for the episode.
    """

    def __init__(
        self,
        stage1_txt: str,
        stage2_txt: str,
        *,
        mode: str = "random",
        stage2_prob: float = 0.0,
    ):
        self.stage1_txt = str(stage1_txt)
        self.stage2_txt = str(stage2_txt)
        self._rng = np.random.default_rng()

        self.stage1 = JSONNearestTargetGenerator(json_paths=[], json_txt=self.stage1_txt, mode=mode)
        self.stage2 = JSONNearestTargetGenerator(json_paths=[], json_txt=self.stage2_txt, mode=mode)

        self._active = self.stage1
        self._active_stage = 1
        self._last_target = None
        self.set_stage2_prob(stage2_prob)

    def set_stage2_prob(self, p: float):
        try:
            p = float(p)
        except Exception:
            p = 0.0
        self.stage2_prob = float(np.clip(p, 0.0, 1.0))

    def get_active_stage(self) -> int:
        return int(self._active_stage)

    @property
    def traj(self):
        return self._active.traj

    def reset(self, rng: Optional[np.random.Generator] = None):
        if rng is not None:
            self._rng = rng
        u = float(self._rng.random())
        if u < float(self.stage2_prob):
            self._active = self.stage2
            self._active_stage = 2
        else:
            self._active = self.stage1
            self._active_stage = 1
        out = self._active.reset(rng=self._rng)
        try:
            self._last_target = np.asarray(out, dtype=np.float32)
        except Exception:
            self._last_target = out
        return out

    def step(self, step_count: int, ee_pos: Optional[np.ndarray] = None) -> np.ndarray:
        fn = getattr(self._active, "step", None)
        if callable(fn):
            try:
                out = fn(step_count, ee_pos=ee_pos)
            except TypeError:
                try:
                    out = fn(step_count)
                except TypeError:
                    out = fn()
            if out is not None:
                self._last_target = np.asarray(out, dtype=np.float32)
                return self._last_target
        if self._last_target is not None:
            return np.asarray(self._last_target, dtype=np.float32)
        # fallback to first point of the current trajectory
        return self.traj.positions[0].astype(np.float32)

    def __getattr__(self, name):
        # Delegate unknown attrs/methods (e.g. traj helpers) to the active generator.
        return getattr(self._active, name)
