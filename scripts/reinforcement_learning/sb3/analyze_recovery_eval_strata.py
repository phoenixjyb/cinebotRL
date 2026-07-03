"""Analyze enriched recovery-eval episode details by trajectory strata."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


FRACTION_BINS = (
    (0.00, 0.45, "<0.45"),
    (0.45, 0.50, "0.45-0.50"),
    (0.50, 0.55, "0.50-0.55"),
    (0.55, 0.60, "0.55-0.60"),
    (0.60, 0.65, "0.60-0.65"),
    (0.65, 0.70, "0.65-0.70"),
    (0.70, math.inf, ">=0.70"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stratify enriched Proto2 recovery evaluations.")
    parser.add_argument("inputs", nargs="+", help="Evaluation JSON files or directories containing recovery_eval_*.json.")
    parser.add_argument("--output_json", default=None, help="Optional JSON summary path.")
    parser.add_argument("--output_md", default=None, help="Optional Markdown summary path.")
    parser.add_argument("--top_n", type=int, default=15, help="Number of worst trajectory files to report.")
    return parser.parse_args()


def expand_inputs(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.glob("recovery_eval_*.json")))
        else:
            files.append(path)
    return files


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def percentile(values: list[float], pct: float) -> float | None:
    clean = sorted(v for v in values if math.isfinite(v))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    rank = (len(clean) - 1) * pct / 100.0
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return clean[lo]
    weight = rank - lo
    return clean[lo] * (1.0 - weight) + clean[hi] * weight


def fraction_bin(value: Any) -> str:
    fraction = finite_float(value)
    if fraction is None:
        return "unknown"
    for low, high, label in FRACTION_BINS:
        if low <= fraction < high:
            return label
    return "unknown"


def summarize_episode_group(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    def values(key: str) -> list[float]:
        return [v for item in episodes if (v := finite_float(item.get(key))) is not None]

    unreachable = values("unreachable_zone_pct")
    base_mean = values("base_target_dist_mean")
    base_max = values("base_target_dist_max")
    hard = values("workspace_hard_exceed_pct")
    soft = values("workspace_soft_exceed_pct")
    unsafe = values("obstacle_unsafe_pct")
    collision = values("obstacle_collision_pct")
    reward = values("episode_reward")
    length = values("episode_length")
    clearance = values("obstacle_clearance_min")

    return {
        "episodes": len(episodes),
        "unreachable_mean": mean(unreachable) if unreachable else None,
        "unreachable_p95": percentile(unreachable, 95),
        "unreachable_max": max(unreachable) if unreachable else None,
        "base_target_dist_mean": mean(base_mean) if base_mean else None,
        "base_target_dist_max_mean": mean(base_max) if base_max else None,
        "workspace_hard_mean": mean(hard) if hard else None,
        "workspace_soft_mean": mean(soft) if soft else None,
        "obstacle_unsafe_max": max(unsafe) if unsafe else None,
        "obstacle_collision_max": max(collision) if collision else None,
        "obstacle_clearance_min": min(clearance) if clearance else None,
        "episode_reward_mean": mean(reward) if reward else None,
        "episode_length_mean": mean(length) if length else None,
    }


def grouped_summary(episodes: list[dict[str, Any]], key_fn) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        buckets[str(key_fn(episode))].append(episode)
    return {key: summarize_episode_group(items) for key, items in sorted(buckets.items())}


def load_eval(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"{path}: could not read JSON: {exc}"
    details = payload.get("episode_details")
    if not isinstance(details, list) or not details:
        return None, f"{path}: no episode_details found; rerun with enriched evaluator first"
    return payload, None


def render_float(value: Any) -> str:
    numeric = finite_float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:.4f}"


def render_table(title: str, rows: dict[str, dict[str, Any]], limit: int | None = None) -> list[str]:
    ordered = sorted(
        rows.items(),
        key=lambda item: (
            -(finite_float(item[1].get("unreachable_mean")) or -1.0),
            -(finite_float(item[1].get("unreachable_p95")) or -1.0),
        ),
    )
    if limit is not None:
        ordered = ordered[:limit]
    lines = [
        f"## {title}",
        "",
        "| group | episodes | unreachable mean | unreachable p95 | unreachable max | base mean | hard ws mean | unsafe max | collision max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, stats in ordered:
        lines.append(
            "| "
            + " | ".join(
                [
                    key,
                    str(stats["episodes"]),
                    render_float(stats.get("unreachable_mean")),
                    render_float(stats.get("unreachable_p95")),
                    render_float(stats.get("unreachable_max")),
                    render_float(stats.get("base_target_dist_mean")),
                    render_float(stats.get("workspace_hard_mean")),
                    render_float(stats.get("obstacle_unsafe_max")),
                    render_float(stats.get("obstacle_collision_max")),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def main() -> int:
    args = parse_args()
    paths = expand_inputs(args.inputs)
    payloads: list[dict[str, Any]] = []
    skipped: list[str] = []
    for path in paths:
        payload, error = load_eval(path)
        if payload is None:
            skipped.append(error or f"{path}: skipped")
        else:
            payload["_source_file"] = str(path)
            payloads.append(payload)

    if not payloads:
        for item in skipped:
            print(f"SKIP: {item}")
        print("No enriched evaluation JSONs found.")
        return 2

    per_eval: list[dict[str, Any]] = []
    all_episodes: list[dict[str, Any]] = []
    for payload in payloads:
        episodes = payload["episode_details"]
        for episode in episodes:
            episode["_source_file"] = payload["_source_file"]
        all_episodes.extend(episodes)
        per_eval.append(
            {
                "source_file": payload["_source_file"],
                "checkpoint": payload.get("checkpoint"),
                "mode": payload.get("mode"),
                "episodes_completed": payload.get("episodes_completed"),
                "overall": summarize_episode_group(episodes),
                "by_category": grouped_summary(episodes, lambda item: item.get("trajectory_category", "unknown")),
                "by_start_fraction": grouped_summary(episodes, lambda item: fraction_bin(item.get("waypoint_fraction"))),
                "by_trajectory_file": grouped_summary(episodes, lambda item: item.get("trajectory_file", "unknown")),
            }
        )

    combined = {
        "inputs": [str(path) for path in paths],
        "skipped": skipped,
        "total_enriched_evals": len(payloads),
        "total_episodes": len(all_episodes),
        "combined": {
            "overall": summarize_episode_group(all_episodes),
            "by_source": grouped_summary(all_episodes, lambda item: item.get("_source_file", "unknown")),
            "by_category": grouped_summary(all_episodes, lambda item: item.get("trajectory_category", "unknown")),
            "by_start_fraction": grouped_summary(all_episodes, lambda item: fraction_bin(item.get("waypoint_fraction"))),
            "by_trajectory_file": grouped_summary(all_episodes, lambda item: item.get("trajectory_file", "unknown")),
        },
        "per_eval": per_eval,
    }

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Recovery Stratified Failure Analysis",
        "",
        f"- Enriched evals: {combined['total_enriched_evals']}",
        f"- Episodes: {combined['total_episodes']}",
    ]
    if skipped:
        lines.append(f"- Skipped inputs: {len(skipped)}")
    lines.append("")
    lines.extend(render_table("Combined Source Comparison", combined["combined"]["by_source"]))
    lines.extend(render_table("Combined Category Buckets", combined["combined"]["by_category"]))
    lines.extend(render_table("Combined Start-Fraction Buckets", combined["combined"]["by_start_fraction"]))
    lines.extend(render_table("Worst Trajectory Files", combined["combined"]["by_trajectory_file"], args.top_n))
    if skipped:
        lines.extend(["## Skipped Inputs", ""])
        lines.extend(f"- {item}" for item in skipped)
        lines.append("")
    markdown = "\n".join(lines)

    if args.output_md:
        out_md = Path(args.output_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
