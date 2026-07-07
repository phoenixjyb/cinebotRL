"""Compare Proto2 recovery policy evaluation outputs.

This script is intentionally a thin wrapper around evaluate_recovery_candidate.py.
It can either aggregate existing evaluator JSON files or run the evaluator for a
set of named checkpoints with identical gate settings.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
EVALUATOR = SCRIPT_DIR / "evaluate_recovery_candidate.py"

PRIMARY_METRICS = [
    "episode_reward_mean",
    "episode_length_mean",
    "ee_pos_error_mean_m",
    "ee_pos_error_p95_m",
    "ee_ori_error_mean_deg",
    "unreachable_zone_pct",
    "workspace_hard_exceed_pct",
    "obstacle_unsafe_pct",
    "obstacle_collision_pct",
    "base_target_dist_mean",
    "base_target_dist_max",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit policy regressions across recovery evaluator outputs.")
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "Named candidate. PATH may be an existing recovery_eval*.json, or a PPO checkpoint "
            "when --run_eval is set. Repeatable."
        ),
    )
    parser.add_argument("--baseline", default=None, help="Candidate name used for delta columns.")
    parser.add_argument("--output_dir", default=None, help="Audit output directory.")
    parser.add_argument("--run_eval", action="store_true", help="Run evaluate_recovery_candidate.py for checkpoint paths.")
    parser.add_argument("--headless", action="store_true", help="Pass --headless to evaluator when --run_eval is used.")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--num_episodes", type=int, default=16)
    parser.add_argument("--trajectory_stage", default="stage1_recovery")
    parser.add_argument("--max_trajectories", type=int, default=4)
    parser.add_argument("--min_trajectory_duration", type=float, default=5.0)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument(
        "--no_obstacles",
        action="store_true",
        help="Disable obstacles during evaluator reruns. Existing JSON inputs are not affected.",
    )
    return parser.parse_args()


def parse_candidate(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"--candidate must be NAME=PATH, got: {raw}")
    name, path = raw.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"empty candidate name in: {raw}")
    return name, Path(path.strip())


def latest_eval_json(output_dir: Path) -> Path:
    matches = sorted(output_dir.glob("recovery_eval_*.json"))
    if not matches:
        raise FileNotFoundError(f"no recovery_eval_*.json found in {output_dir}")
    return matches[-1]


def run_evaluator(name: str, checkpoint: Path, args: argparse.Namespace, output_root: Path) -> Path:
    output_dir = output_root / name
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(EVALUATOR),
        "--checkpoint",
        str(checkpoint),
        "--num_envs",
        str(args.num_envs),
        "--num_episodes",
        str(args.num_episodes),
        "--trajectory_stage",
        args.trajectory_stage,
        "--max_trajectories",
        str(args.max_trajectories),
        "--min_trajectory_duration",
        str(args.min_trajectory_duration),
        "--seed",
        str(args.seed),
        "--output_dir",
        str(output_dir),
    ]
    if args.headless:
        cmd.append("--headless")
    if args.deterministic:
        cmd.append("--deterministic")
    else:
        cmd.append("--no-deterministic")
    if args.no_obstacles:
        cmd.append("--no-enable_obstacles")

    log_path = output_dir / "evaluate.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"evaluator failed for {name}, see {log_path}")
    return latest_eval_json(output_dir)


def stat_mean(data: dict[str, Any], metric: str) -> float | None:
    if metric in ("episode_reward_mean", "episode_length_mean"):
        value = data.get(metric)
        return float(value) if isinstance(value, (int, float)) else None
    metrics = data.get("metrics", {})
    value = metrics.get(metric)
    if isinstance(value, dict) and isinstance(value.get("mean"), (int, float)):
        return float(value["mean"])
    return None


def summarize_episode_groups(data: dict[str, Any]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    groups = data.get("episode_group_summary", {})
    if not isinstance(groups, dict):
        return out
    for group_name, metrics in groups.items():
        if not isinstance(metrics, dict):
            continue
        compact: dict[str, float] = {}
        for metric in (
            "ee_pos_error_mean_m",
            "ee_ori_error_mean_deg",
            "unreachable_zone_pct",
            "workspace_hard_exceed_pct",
            "obstacle_collision_pct",
        ):
            value = metrics.get(metric)
            if isinstance(value, dict) and isinstance(value.get("mean"), (int, float)):
                compact[metric] = float(value["mean"])
        out[group_name] = compact
    return out


def classify_against_baseline(row: dict[str, Any], baseline_row: dict[str, Any] | None) -> str:
    if baseline_row is None:
        return "baseline"
    worse = 0
    better = 0
    for metric in ("ee_pos_error_mean_m", "ee_ori_error_mean_deg", "unreachable_zone_pct"):
        current = row.get(metric)
        baseline = baseline_row.get(metric)
        if not isinstance(current, (int, float)) or not isinstance(baseline, (int, float)):
            continue
        delta = current - baseline
        tolerance = max(abs(baseline) * 0.05, 1e-6)
        if delta > tolerance:
            worse += 1
        elif delta < -tolerance:
            better += 1
    if worse and not better:
        return "regressed"
    if better and not worse:
        return "improved"
    if worse and better:
        return "mixed"
    return "similar"


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = ["name", "source_json", "checkpoint", "mode", "episodes_completed", "steps", "verdict"]
    for metric in PRIMARY_METRICS:
        if metric not in fieldnames:
            fieldnames.append(metric)
        delta = f"delta_vs_baseline_{metric}"
        if any(delta in row for row in rows):
            fieldnames.append(delta)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if not args.candidate:
        print("at least one --candidate NAME=PATH is required", file=sys.stderr)
        return 2

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "evaluation_results" / "policy_regression_audit" / timestamp
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    group_summaries: dict[str, Any] = {}
    for raw in args.candidate:
        name, source = parse_candidate(raw)
        if not source.is_absolute():
            source = PROJECT_ROOT / source
        if source.suffix.lower() == ".json":
            json_path = source
        elif args.run_eval:
            json_path = run_evaluator(name, source, args, output_root)
        else:
            raise ValueError(f"{source} is not JSON; pass --run_eval to evaluate checkpoints")
        data = json.loads(json_path.read_text(encoding="utf-8"))
        row: dict[str, Any] = {
            "name": name,
            "source_json": str(json_path),
            "checkpoint": data.get("checkpoint"),
            "mode": data.get("mode"),
            "episodes_completed": data.get("episodes_completed"),
            "steps": data.get("steps"),
        }
        for metric in PRIMARY_METRICS:
            value = stat_mean(data, metric)
            if value is not None:
                row[metric] = value
        rows.append(row)
        group_summaries[name] = summarize_episode_groups(data)

    baseline_row = None
    if args.baseline:
        baseline_row = next((row for row in rows if row["name"] == args.baseline), None)
        if baseline_row is None:
            raise ValueError(f"baseline candidate not found: {args.baseline}")
        for row in rows:
            if row is baseline_row:
                continue
            for metric in PRIMARY_METRICS:
                current = row.get(metric)
                baseline = baseline_row.get(metric)
                if isinstance(current, (int, float)) and isinstance(baseline, (int, float)):
                    row[f"delta_vs_baseline_{metric}"] = float(current) - float(baseline)
    for row in rows:
        row["verdict"] = classify_against_baseline(row, baseline_row if row is not baseline_row else None)

    csv_path = output_root / "policy_regression_summary.csv"
    json_path = output_root / "policy_regression_summary.json"
    write_csv(rows, csv_path)
    json_path.write_text(
        json.dumps(
            {
                "timestamp": timestamp,
                "baseline": args.baseline,
                "rows": rows,
                "episode_group_summary": group_summaries,
                "settings": {
                    "run_eval": args.run_eval,
                    "num_envs": args.num_envs,
                    "num_episodes": args.num_episodes,
                    "trajectory_stage": args.trajectory_stage,
                    "max_trajectories": args.max_trajectories,
                    "min_trajectory_duration": args.min_trajectory_duration,
                    "deterministic": args.deterministic,
                    "seed": args.seed,
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    for row in rows:
        print(
            "{name}: verdict={verdict}, ee_pos={ee:.4f}, unreachable={unreach:.2f}, ori={ori:.2f}".format(
                name=row["name"],
                verdict=row["verdict"],
                ee=float(row.get("ee_pos_error_mean_m", float("nan"))),
                unreach=float(row.get("unreachable_zone_pct", float("nan"))),
                ori=float(row.get("ee_ori_error_mean_deg", float("nan"))),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
