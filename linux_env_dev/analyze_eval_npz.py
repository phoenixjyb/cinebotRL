#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


def _ema(x: np.ndarray, alpha: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    alpha = float(alpha)
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, x.shape[0]):
        y[i] = alpha * x[i] + (1.0 - alpha) * y[i - 1]
    return y


def main():
    parser = argparse.ArgumentParser(description="Analyze SB3 EvalCallback evaluations.npz")
    parser.add_argument("npz_path", type=str, help="Path to evaluations.npz")
    parser.add_argument("--ema", type=float, default=0.0, help="EMA smoothing alpha (0 disables)")
    parser.add_argument("--save", type=str, default=None, help="Save plot PNG to this path (optional)")
    parser.add_argument("--no-show", action="store_true", help="Do not show an interactive plot window")
    args = parser.parse_args()

    npz_path = Path(args.npz_path)
    if not npz_path.is_file():
        raise FileNotFoundError(f"Not found: {npz_path}")

    data = np.load(npz_path)
    timesteps = np.asarray(data["timesteps"], dtype=int)
    results = np.asarray(data["results"], dtype=float)  # (n_eval, n_episodes)

    mean_rew = np.mean(results, axis=1)
    std_rew = np.std(results, axis=1)

    best_idx = int(np.argmax(mean_rew)) if mean_rew.size else 0
    print(f"File: {npz_path}")
    print(f"Eval points: {len(timesteps)}  Episodes per eval: {results.shape[1] if results.ndim == 2 else 'unknown'}")
    if mean_rew.size:
        print(f"Best mean reward: {mean_rew[best_idx]:.2f} @ {timesteps[best_idx]}")
        print(f"Last mean reward: {mean_rew[-1]:.2f} ± {std_rew[-1]:.2f} @ {timesteps[-1]}")

    success = None
    if "successes" in data.files:
        successes = np.asarray(data["successes"], dtype=float)
        success = np.mean(successes, axis=1)
        best_s_idx = int(np.argmax(success)) if success.size else 0
        if success.size:
            print(f"Best success_rate: {success[best_s_idx]:.3f} @ {timesteps[best_s_idx]}")
            print(f"Last success_rate: {success[-1]:.3f} @ {timesteps[-1]}")

    extra_series = {}
    # Our extended EvalInfoMetricsCallback stores arrays named like "<key>s", e.g. ee_distances, traj_ids, ...
    for k in ("ee_distance", "traj_id", "remain_traj_ratio", "base_lin_vel_norm", "base_ang_vel_norm"):
        nk = f"{k}s"
        if nk in data.files:
            arr = np.asarray(data[nk], dtype=float)
            if arr.ndim == 2 and arr.shape[0] == timesteps.shape[0]:
                extra_series[k] = np.nanmean(arr, axis=1)

    if extra_series:
        for k, series in extra_series.items():
            if series.size:
                print(f"Last eval final_{k}_mean: {series[-1]:.4f}")

    try:
        import matplotlib.pyplot as plt

        if float(args.ema) > 0.0:
            mean_rew_plot = _ema(mean_rew, float(args.ema))
            if success is not None:
                success_plot = _ema(success, float(args.ema))
        else:
            mean_rew_plot = mean_rew
            success_plot = success

        n_rows = 1 + (1 if success is not None else 0) + (1 if "ee_distance" in extra_series else 0)
        fig, axes = plt.subplots(n_rows, 1, figsize=(10, 3.2 * n_rows), sharex=True)
        if n_rows == 1:
            axes = [axes]

        ax0 = axes[0]
        ax0.plot(timesteps, mean_rew_plot, label="eval/mean_reward", color="C0")
        ax0.fill_between(timesteps, mean_rew - std_rew, mean_rew + std_rew, color="C0", alpha=0.15, linewidth=0)
        ax0.set_ylabel("reward")
        ax0.grid(True, alpha=0.3)
        ax0.legend(loc="best")

        row = 1
        if success is not None:
            ax = axes[row]
            ax.plot(timesteps, success_plot, label="eval/success_rate", color="C2")
            ax.set_ylabel("success")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            row += 1

        if "ee_distance" in extra_series:
            ax = axes[row]
            ax.plot(timesteps, extra_series["ee_distance"], label="eval/final_ee_distance_mean", color="C3")
            ax.set_ylabel("meters")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")

        axes[-1].set_xlabel("timesteps")
        fig.tight_layout()

        if args.save:
            out = Path(args.save)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"Saved plot to: {out}")

        if not args.no_show:
            plt.show()
        plt.close(fig)
    except Exception as e:
        print(f"Plot skipped (matplotlib issue): {e}")


if __name__ == "__main__":
    main()

