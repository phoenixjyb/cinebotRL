import json
import numpy as np
from typing import Tuple, Optional, List


def _ensure_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q.shape[0] != 4:
        raise ValueError("Quaternion must have 4 elements [w,x,y,z]")
    norm = np.linalg.norm(q)
    if norm == 0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return (q / norm).astype(float)


def quat_slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation for quaternions in [w,x,y,z] format."""
    q1 = _ensure_quat(q1)
    q2 = _ensure_quat(q2)
    dot = np.dot(q1, q2)
    # if dot < 0, slerp won't take shorter path — invert
    if dot < 0.0:
        q2 = -q2
        dot = -dot

    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        # quaternions are nearly identical — use linear interp
        res = q1 + t * (q2 - q1)
        return res / np.linalg.norm(res)

    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    theta = theta_0 * float(t)
    q_perp = q2 - q1 * dot
    q_perp /= np.linalg.norm(q_perp)
    return (q1 * np.cos(theta) + q_perp * np.sin(theta))


class JSONTrajectory:
    """Load a JSON trajectory with 'poses': [{'position':[x,y,z], 'orientation':[w,x,y,z]}, ...]

    Provides nearest-point (projection to polyline) lookup and stateless lookahead sampling.
    """

    def __init__(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)

        poses = data.get('poses', None)
        if poses is None:
            raise ValueError(f"JSON at {path} does not contain top-level 'poses' list")

        positions: List[List[float]] = []
        quats: List[List[float]] = []
        for p in poses:
            positions.append(list(map(float, p.get('position', [0.0, 0.0, 0.0]))))
            q = p.get('orientation', None)
            if q is None:
                # default no rotation
                q = [1.0, 0.0, 0.0, 0.0]
            quats.append(list(map(float, q)))

        self.positions = np.asarray(positions, dtype=float)
        self.orientations = np.asarray([_ensure_quat(np.asarray(q)) for q in quats], dtype=float)
        self.num_points = int(self.positions.shape[0])
        if self.num_points == 0:
            raise ValueError("Trajectory contains no poses")

    def get_position(self, idx: int, alpha: float = 0.0) -> np.ndarray:
        """Return interpolated position between idx and idx+1 by alpha in [0,1].
        If idx is the last index, returns last position regardless of alpha."""
        if idx >= self.num_points - 1 or idx == -1:
            return self.positions[-1].copy()
        a = self.positions[idx]
        b = self.positions[idx + 1]
        return (1.0 - alpha) * a + alpha * b

    def get_position_len(self):
        return self.positions.shape[0]

    def get_orientation(self, idx: int, alpha: float = 0.0) -> np.ndarray:
        if idx >= self.num_points - 1 or idx == -1:
            return self.orientations[-1].copy()
        return quat_slerp(self.orientations[idx], self.orientations[idx + 1], float(alpha))

    def project_onto_trajectory(self, point: np.ndarray) -> Tuple[int, float, np.ndarray, np.ndarray]:
        """Project a 3D point onto the polyline defined by trajectory positions.

        Returns (segment_idx, alpha, projected_point, projected_orientation)
        where projected point = interp(positions[segment_idx], positions[segment_idx+1], alpha)
        and projected_orientation = slerp(orients[segment_idx], orients[segment_idx+1], alpha).
        If the trajectory has a single point, returns (0, 0.0, positions[0], orientations[0]).
        """
        p = np.asarray(point, dtype=float)
        if self.num_points == 1:
            return 0, 0.0, self.positions[0].copy(), self.orientations[0].copy()

        best_d2 = float('inf')
        best_i = 0
        best_alpha = 0.0
        best_proj = self.positions[0]
        for i in range(self.num_points - 1):
            a = self.positions[i]
            b = self.positions[i + 1]
            v = b - a
            vv = np.dot(v, v)
            if vv == 0.0:
                alpha = 0.0
                proj = a
            else:
                w = p - a
                alpha = float(np.dot(w, v) / vv)
                if alpha < 0.0:
                    alpha = 0.0
                elif alpha > 1.0:
                    alpha = 1.0
                proj = a + alpha * v
            d2 = float(np.sum((p - proj) ** 2))
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
                best_alpha = alpha
                best_proj = proj

        proj_orient = self.get_orientation(best_i, best_alpha)
        return best_i, best_alpha, best_proj, proj_orient

    def project_onto_trajectory_fast(self, point: np.ndarray) -> Tuple[int, np.ndarray]:
        """Fast, vectorized projection of a 3D point onto the trajectory polyline.

        Returns (segment_idx, projected_point) where projected_point is the
        closest point (3-vector) on the polyline. This version avoids Python
        loops by using numpy broadcasting and is intended for speed when many
        projections are required.

        Note: If the trajectory contains a single point, returns (0, positions[0]).
        """
        p = np.asarray(point, dtype=float)
        if self.num_points == 1:
            return 0, self.positions[0].copy()

        # segment endpoints: a = positions[:-1], b = positions[1:]
        a = self.positions[:-1]    # (M,3)
        b = self.positions[1:]     # (M,3)
        v = b - a                  # (M,3) segment vectors

        # vector from segment start to point, broadcasted (M,3)
        w = p[None, :] - a

        # squared segment lengths (M,)
        vv = np.einsum('ij,ij->i', v, v)
        # dot product between w and v for each segment (M,)
        dot = np.einsum('ij,ij->i', w, v)

        # safe division: where vv==0 (degenerate segment), alpha -> 0
        alpha = np.zeros_like(dot, dtype=float)
        nonzero = vv > 0.0
        alpha[nonzero] = dot[nonzero] / vv[nonzero]
        alpha = np.clip(alpha, 0.0, 1.0)

        # projected points on each segment (M,3)
        proj = a + alpha[:, None] * v

        # squared distances from p to each projected point
        d2 = np.einsum('ij,ij->i', (p[None, :] - proj), (p[None, :] - proj))

        best_i = int(np.argmin(d2))
        best_proj = proj[best_i].copy()
        return best_i, best_proj

    def lookahead_from(self, seg_idx: int, alpha: float, steps: int, step_dt: float=1.0) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Statelessly sample future poses starting after the given (seg_idx, alpha).

        Here `step_dt` is interpreted in units of waypoints (so step_dt=1.0 advances
        by one waypoint per lookahead step). If your trajectories were recorded at
        a fixed sampling frequency f (Hz) and you want time-based lookahead (e.g.
        0.1s), convert seconds -> waypoints by: waypoints = seconds * f, and pass
        that as step_dt.

        Returns a list of (position, orientation) pairs of length `steps`.
        """
        if steps <= 0:
            return []
        # treat step_dt as waypoints to advance (no waypoint_dt support)
        current_f = float(seg_idx) + float(alpha)
        out = []
        for s in range(1, steps + 1):
            delta = float(s) * float(step_dt)
            f = current_f + delta
            if f >= (self.num_points - 1):
                idx = self.num_points - 1
                alpha_f = 0.0
            else:
                idx = int(np.floor(f))
                alpha_f = float(f - idx)
            pos = self.get_position(idx, alpha_f)
            orient = self.get_orientation(idx, alpha_f)
            out.append((pos.copy(), orient.copy()))
        return out
