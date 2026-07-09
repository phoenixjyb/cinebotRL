#!/usr/bin/env python3
"""Run or summarize the five-case full-horizon no-obstacle rollout gate.

The gate is intentionally narrow: it compares candidate rollout coverage against
the dense-BC five-case baseline and rejects policies that regress known-good
full-start tracking, especially case 0028.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
RENDER_SCRIPT = SCRIPT_DIR / "record_rendered_recovery_rollout.py"


@dataclass(frozen=True)
class GateCase:
    case_id: str
    steps: int
    raw_seconds: float


GATE_CASES: tuple[GateCase, ...] = (
    GateCase("0001", 530, 25.12368904166667),
    GateCase("0020", 1040, 50.83595604166666),
    GateCase("0028", 765, 37.193848625),
    GateCase("0050", 785, 38.017315125),
    GateCase("0079", 370, 17.436073833333335),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None, help="Candidate PPO checkpoint.")
    parser.add_argument("--vec_normalize", default=None, help="VecNormalize pkl for candidate checkpoint.")
    parser.add_argument("--disable_vec_normalize", action="store_true")
    parser.add_argument("--summary_only", action="store_true", help="Only summarize existing rendered metadata.")
    parser.add_argument(
        "--existing_output_dir",
        default=None,
        help="Existing rendered case directory to summarize. Implies --summary_only.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Gate output directory. Defaults to evaluation_results/full_horizon_five_case_gate/<timestamp>.",
    )
    parser.add_argument("--name_prefix", default="candidate", help="Per-case output folder prefix.")
    parser.add_argument("--trajectory_stage", default="stage_gik_no_obstacle79_nominal")
    parser.add_argument("--min_trajectory_duration", type=float, default=5.0)
    parser.add_argument("--episode_length_s", type=float, default=60.0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--camera_eye", default="3.2,-5.4,4.0")
    parser.add_argument("--camera_target", default="0.45,-0.05,1.0")
    parser.add_argument("--render_polish", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--base_action_slew_limit", type=float, default=0.0)
    parser.add_argument(
        "--baseline_dir",
        default="evaluation_results/videos_rendered/no_obstacle79_dense01_full_5cases_20260709",
        help="Rendered dense-BC baseline directory containing per-case rendered_rollout_meta.json files.",
    )
    parser.add_argument("--baseline_prefix", default="dense01_bc80_noobs")
    parser.add_argument(
        "--baseline_tolerance",
        type=float,
        default=0.02,
        help="Allowed coverage regression versus baseline for non-hard cases.",
    )
    parser.add_argument(
        "--hard_case",
        default="0028",
        help="Case that must not regress beyond --hard_case_tolerance.",
    )
    parser.add_argument(
        "--hard_case_tolerance",
        type=float,
        default=0.01,
        help="Allowed coverage regression for the hard case.",
    )
    parser.add_argument("--stop_on_done", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def case_by_id(case_id: str) -> GateCase:
    for item in GATE_CASES:
        if item.case_id == case_id:
            return item
    raise KeyError(case_id)


def load_case_rows(folder: Path, prefix: str) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for meta_path in sorted(folder.glob(f"{prefix}_*_full/rendered_rollout_meta.json")):
        case_id = meta_path.parent.name.split("_")[-2]
        if case_id not in {case.case_id for case in GATE_CASES}:
            continue
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        gate_case = case_by_id(case_id)
        fps = float(data.get("fps") or 20)
        steps_executed = int(data.get("steps_executed") or 0)
        executed_seconds = steps_executed / fps
        coverage = executed_seconds / gate_case.raw_seconds
        rows[case_id] = {
            "case": case_id,
            "meta_path": str(meta_path),
            "steps_requested": int(data.get("steps_requested") or gate_case.steps),
            "steps_executed": steps_executed,
            "fps": fps,
            "executed_seconds": executed_seconds,
            "raw_seconds": gate_case.raw_seconds,
            "coverage": coverage,
            "done_count": int(data.get("done_count") or 0),
            "selected_trajectories": data.get("selected_trajectories", []),
        }
    return rows


def run_case(args: argparse.Namespace, output_dir: Path, gate_case: GateCase) -> None:
    if not args.checkpoint:
        raise ValueError("--checkpoint is required unless --summary_only or --existing_output_dir is used")

    cmd = [
        sys.executable,
        str(RENDER_SCRIPT),
        "--checkpoint",
        str(resolve_project_path(args.checkpoint)),
        "--trajectory_stage",
        args.trajectory_stage,
        "--trajectory_file_contains",
        gate_case.case_id,
        "--max_trajectories",
        "1",
        "--min_trajectory_duration",
        str(args.min_trajectory_duration),
        "--no-random_start_waypoint",
        "--reset_base_to_trajectory_start",
        "--reset_anchor_target_blend",
        "0.0",
        "--no-enable_obstacles",
        "--steps",
        str(gate_case.steps),
        "--episode_length_s",
        str(args.episode_length_s),
        "--fps",
        str(args.fps),
        "--camera_eye",
        args.camera_eye,
        "--camera_target",
        args.camera_target,
        "--output_dir",
        str(output_dir),
        "--name",
        f"{args.name_prefix}_{gate_case.case_id}_full",
    ]
    if args.vec_normalize:
        cmd.extend(["--vec_normalize", str(resolve_project_path(args.vec_normalize))])
    if args.disable_vec_normalize:
        cmd.append("--disable_vec_normalize")
    if args.headless:
        cmd.append("--headless")
    if args.stop_on_done:
        cmd.append("--stop_on_done")
    else:
        cmd.append("--no-stop_on_done")
    if args.render_polish:
        cmd.append("--render_polish")
    else:
        cmd.append("--no-render_polish")
    if args.base_action_slew_limit > 0.0:
        cmd.extend(["--base_action_slew_limit", str(args.base_action_slew_limit)])

    print(f"[five-case-gate] running case {gate_case.case_id}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def evaluate_rows(
    candidate_rows: dict[str, dict[str, object]],
    baseline_rows: dict[str, dict[str, object]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], str]:
    out: list[dict[str, object]] = []
    verdict = "PASS"
    missing = []
    for gate_case in GATE_CASES:
        case_id = gate_case.case_id
        candidate = candidate_rows.get(case_id)
        baseline = baseline_rows.get(case_id)
        if candidate is None:
            missing.append(case_id)
            verdict = "FAIL"
            continue
        baseline_coverage = float(baseline["coverage"]) if baseline else 0.0
        candidate_coverage = float(candidate["coverage"])
        tolerance = args.hard_case_tolerance if case_id == args.hard_case else args.baseline_tolerance
        delta = candidate_coverage - baseline_coverage
        case_verdict = "PASS" if delta >= -float(tolerance) else "FAIL"
        if case_verdict == "FAIL":
            verdict = "FAIL"
        out.append(
            {
                "case": case_id,
                "verdict": case_verdict,
                "candidate_coverage": candidate_coverage,
                "baseline_coverage": baseline_coverage,
                "delta_vs_baseline": delta,
                "tolerance": tolerance,
                "candidate_steps": int(candidate["steps_executed"]),
                "candidate_seconds": float(candidate["executed_seconds"]),
                "baseline_steps": int(baseline["steps_executed"]) if baseline else None,
                "done_count": int(candidate["done_count"]),
                "meta_path": candidate["meta_path"],
            }
        )
    if missing:
        out.append({"case": ",".join(missing), "verdict": "FAIL", "reason": "missing candidate metadata"})
    return out, verdict


def write_outputs(output_dir: Path, rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    summary_path = output_dir / "five_case_gate_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = output_dir / "five_case_gate_summary.csv"
    fieldnames = [
        "case",
        "verdict",
        "candidate_coverage",
        "baseline_coverage",
        "delta_vs_baseline",
        "tolerance",
        "candidate_steps",
        "candidate_seconds",
        "baseline_steps",
        "done_count",
        "meta_path",
        "reason",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# Five-Case Full-Horizon Gate: {summary['verdict']}",
        "",
        "| case | verdict | candidate | baseline | delta | steps |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if "candidate_coverage" not in row:
            lines.append(f"| {row.get('case')} | FAIL | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            "| {case} | {verdict} | {candidate:.1%} | {baseline:.1%} | {delta:+.1%} | {steps} |".format(
                case=row["case"],
                verdict=row["verdict"],
                candidate=float(row["candidate_coverage"]),
                baseline=float(row["baseline_coverage"]),
                delta=float(row["delta_vs_baseline"]),
                steps=int(row["candidate_steps"]),
            )
        )
    (output_dir / "five_case_gate_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.existing_output_dir:
        args.summary_only = True
        output_dir = resolve_project_path(args.existing_output_dir)
    else:
        output_dir = resolve_project_path(
            args.output_dir or f"evaluation_results/full_horizon_five_case_gate/{timestamp}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.summary_only:
        for gate_case in GATE_CASES:
            run_case(args, output_dir, gate_case)

    baseline_dir = resolve_project_path(args.baseline_dir)
    baseline_rows = load_case_rows(baseline_dir, args.baseline_prefix)
    candidate_rows = load_case_rows(output_dir, args.name_prefix)
    rows, verdict = evaluate_rows(candidate_rows, baseline_rows, args)
    summary = {
        "verdict": verdict,
        "output_dir": str(output_dir),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "vec_normalize": str(args.vec_normalize) if args.vec_normalize else None,
        "disable_vec_normalize": bool(args.disable_vec_normalize),
        "baseline_dir": str(baseline_dir),
        "baseline_prefix": args.baseline_prefix,
        "name_prefix": args.name_prefix,
        "baseline_tolerance": float(args.baseline_tolerance),
        "hard_case": args.hard_case,
        "hard_case_tolerance": float(args.hard_case_tolerance),
        "rows": rows,
    }
    write_outputs(output_dir, rows, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
