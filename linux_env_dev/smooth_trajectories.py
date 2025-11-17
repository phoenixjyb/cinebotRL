import os
import json
import glob
import argparse
from pathlib import Path

import numpy as np

# try to import savgol_filter; fallback to None
try:
    from scipy.signal import savgol_filter  # type: ignore
    _HAS_SAVGOL = True
except Exception:
    _HAS_SAVGOL = False


def _savgol_or_moving_avg(seq: np.ndarray, window: int = 7, poly: int = 3, method: str = "auto") -> np.ndarray:
    """Smooth 1D sequence using Savitzky-Golay when available, else moving average."""
    n = seq.shape[0]
    if n <= 1:
        return seq.copy()
    if method == "savgol" and _HAS_SAVGOL:
        # ensure window is odd and <= n
        w = min(window, n if n % 2 == 1 else n - 1)
        if w < 3:
            return seq.copy()
        return savgol_filter(seq, window_length=w, polyorder=min(poly, w - 1))
    # fallback or explicit moving average
    w = min(window, n)
    if w < 1:
        return seq.copy()
    # use symmetric padding to avoid edge shrink
    pad = w // 2
    padded = np.pad(seq, (pad, pad), mode="edge")
    kernel = np.ones(w) / float(w)
    conv = np.convolve(padded, kernel, mode="valid")
    return conv[:n]


def smooth_positions(positions: np.ndarray, window: int = 7, poly: int = 3, method: str = "auto") -> np.ndarray:
    """Smooth Nx3 positions array."""
    if positions.ndim != 2 or positions.shape[1] < 3:
        return positions
    out = np.zeros_like(positions, dtype=float)
    for i in range(3):
        out[:, i] = _savgol_or_moving_avg(positions[:, i], window=window, poly=poly, method=method)
    return out


def smooth_quaternions(quats: np.ndarray, window: int = 7, poly: int = 3, method: str = "auto") -> np.ndarray:
    """Smooth Nx4 quaternion array by smoothing components then renormalizing (simple, robust)."""
    if quats.ndim != 2 or quats.shape[1] != 4:
        return quats
    out = np.zeros_like(quats, dtype=float)
    for i in range(4):
        out[:, i] = _savgol_or_moving_avg(quats[:, i], window=window, poly=poly, method=method)
    # renormalize to unit quaternions to keep valid rotations
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    out = out / norms
    return out


def process_file(in_path: Path, out_dir: Path, window: int, poly: int, method: str, verbose: bool = True, out_name: str = None) -> None:
    with open(in_path, "r") as fh:
        data = json.load(fh)

    # try several common pose containers: 'poses', 'trajectory', top-level list
    poses = None
    if isinstance(data, dict):
        if "poses" in data and isinstance(data["poses"], list):
            poses = data["poses"]
        elif "trajectory" in data and isinstance(data["trajectory"], list):
            poses = data["trajectory"]
    if poses is None and isinstance(data, list):
        poses = data

    if not poses or len(poses) == 0:
        if verbose:
            print(f"[SKIP] {in_path.name}: no poses found")
        return

    # Extract positions and quaternions where available
    positions = []
    quats = []
    pos_keys = ("position", "ee_pos", "pos", "position_xyz")
    ori_keys = ("orientation", "quat", "quaternion", "ee_quat")
    for p in poses:
        # p may be dict or list
        if isinstance(p, dict):
            # find position
            found_pos = None
            for k in pos_keys:
                if k in p and isinstance(p[k], (list, tuple)) and len(p[k]) >= 3:
                    found_pos = list(p[k])[:3]
                    break
            # fallback: nested keys
            if found_pos is None:
                # try common nested structure 'pose' -> 'position'
                if "pose" in p and isinstance(p["pose"], dict) and "position" in p["pose"]:
                    cand = p["pose"]["position"]
                    if isinstance(cand, (list, tuple)) and len(cand) >= 3:
                        found_pos = list(cand)[:3]
            positions.append(found_pos if found_pos is not None else [np.nan, np.nan, np.nan])

            # orientation
            found_ori = None
            for k in ori_keys:
                if k in p and isinstance(p[k], (list, tuple)) and len(p[k]) >= 4:
                    found_ori = list(p[k])[:4]
                    break
            quats.append(found_ori if found_ori is not None else None)
        else:
            # pose is a list/tuple of numbers
            if isinstance(p, (list, tuple)) and len(p) >= 3:
                positions.append(list(p[:3]))
                quats.append(None)
            else:
                positions.append([np.nan, np.nan, np.nan])
                quats.append(None)

    pos_arr = np.array(positions, dtype=float)
    # handle NaNs by simple forward/backward fill before smoothing
    nan_mask = np.isnan(pos_arr)
    if nan_mask.any():
        # forward fill then backward fill per axis
        for i in range(pos_arr.shape[1]):
            col = pos_arr[:, i]
            nans = np.isnan(col)
            if nans.all():
                col[:] = 0.0
            else:
                # forward fill
                idx = np.where(~nans)[0]
                first, last = idx[0], idx[-1]
                col[:first] = col[first]
                col[last + 1 :] = col[last]
                # linear interp for internal nans
                nans = np.isnan(col)
                if nans.any():
                    good = ~nans
                    col[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(good), col[good])
            pos_arr[:, i] = col

    pos_sm = smooth_positions(pos_arr, window=window, poly=poly, method=method)

    # translate trajectory so that its starting point is at [0, 0, 0.85]
    try:
        if pos_sm.shape[0] > 0:
            desired_start = np.array([0.0, 0.0, 0.85], dtype=float)
            cur_start = pos_sm[0, :3].astype(float)
            delta = desired_start - cur_start
            pos_sm = pos_sm + delta
            if verbose:
                print(f"  Translated trajectory by {delta.tolist()} so start -> {desired_start.tolist()}")
    except Exception:
        # if anything goes wrong, keep original smoothed positions
        pass

    # quaternions
    has_quat = any(q is not None for q in quats)
    quat_sm = None
    if has_quat:
        q_arr = np.array([q if q is not None else [1.0, 0.0, 0.0, 0.0] for q in quats], dtype=float)
        quat_sm = smooth_quaternions(q_arr, window=window, poly=poly, method=method)

    # write back smoothed values into data structure
    for idx, p in enumerate(poses):
        if isinstance(p, dict):
            # set position at existing key if present, else create 'position'
            written = False
            for k in pos_keys:
                if k in p:
                    p[k] = [float(x) for x in pos_sm[idx, :3]]
                    written = True
                    break
            if not written:
                # prefer 'position'
                p["position"] = [float(x) for x in pos_sm[idx, :3]]
            if has_quat and quat_sm is not None:
                written = False
                for k in ori_keys:
                    if k in p:
                        p[k] = [float(x) for x in quat_sm[idx, :4]]
                        written = True
                        break
                if not written:
                    p["orientation"] = [float(x) for x in quat_sm[idx, :4]]
        else:
            # list-like pose -> replace first 3 values
            if isinstance(poses[idx], (list, tuple)):
                old = list(poses[idx])
                # ensure length >=3
                if len(old) >= 3:
                    old[:3] = [float(x) for x in pos_sm[idx, :3]]
                    poses[idx] = old
                else:
                    poses[idx] = [float(x) for x in pos_sm[idx, :3]]

    # determine output path and save
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_name is None:
        out_path = out_dir / in_path.name
    else:
        out_path = out_dir / out_name
    with open(out_path, "w") as fh:
        json.dump(data, fh, indent=2)
    if verbose:
        print(f"[OK] Smoothed {in_path.name} -> {out_path.name} (len={pos_sm.shape[0]})")


def main():
    parser = argparse.ArgumentParser(description="Smooth recorded trajectory JSON files")
    parser.add_argument(
        "--input_dir", type=str, default="trajectoryToLearn", help="Input directory with JSON trajectories"
    )
    parser.add_argument(
        "--output_dir", type=str, default="linux_env_dev/new_json_50", help="Output directory to save smoothed JSONs"
    )
    parser.add_argument("--pattern", type=str, default="**/*.json", help="Glob pattern to find json files")
    parser.add_argument("--window", type=int, default=50, help="Smoothing window (odd preferred)")
    parser.add_argument("--poly", type=int, default=3, help="Polynomial order for Savitzky-Golay")
    parser.add_argument("--method", type=str, default="auto", choices=["auto", "savgol", "moving"], help="Smoothing method")
    parser.add_argument("--dry", action="store_true", help="Dry run; do not write files")
    args = parser.parse_args()

    # import pdb; pdb.set_trace()
    if args.input_dir[-4:] == "json":
        in_dir = Path(args.input_dir).parent
        files = [Path(args.input_dir)]
    else:
        in_dir = Path(args.input_dir)
        files = sorted(in_dir.glob(args.pattern), key=lambda p: p.name)
    out_dir_base = Path(args.output_dir)
    if not files:
        print(f"No files found in {in_dir} with pattern {args.pattern}")
        return

    print(f"Found {len(files)} files in {in_dir}; saving to {out_dir_base}")
    if args.method == "savgol" and not _HAS_SAVGOL:
        print("Warning: scipy not available; falling back to moving average")
    for f in files:
        try:
            # construct output name including parent directories as prefix
            try:
                rel = f.relative_to(in_dir)
                prefix = "_".join(rel.parent.parts) if str(rel.parent) not in (".", "") else ""
                out_name = f"{prefix + '_' if prefix else ''}{f.name}"
            except Exception:
                out_name = f.name

            if args.dry:
                print(f"[DRY] Would process: {f} -> {out_name}")
            else:
                process_file(f, out_dir_base, window=args.window, poly=args.poly, method=args.method, out_name=out_name)
        except Exception as e:
            print(f"[ERR] Failed to process {f.name}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()