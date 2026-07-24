#!/usr/bin/env python3
"""Hash and probe learned-policy render media without judging robot visuals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from rl_platform.tasks.two_wheel_balance import (
    riser_model_based_learned_render_contract as contract,
)


RENDER_CONFIG = contract.RENDER_CONFIG
REPRESENTATIVE_CASES = contract.REPRESENTATIVE_CASES


def identity(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def case_paths(values: list[str]) -> dict[int, Path]:
    result = {}
    for value in values:
        case_text, separator, path_text = value.partition("=")
        if not separator:
            raise ValueError(f"expected CASE=PATH: {value}")
        case = int(case_text)
        if case in result:
            raise ValueError(f"duplicate case: {case}")
        result[case] = Path(path_text).resolve()
    if sorted(result) != REPRESENTATIVE_CASES:
        raise ValueError("render cases do not match the representative set")
    return result


def probe(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,duration:"
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise ValueError(f"ffprobe found no video stream: {path}")
    stream = streams[0]
    numerator, denominator = str(stream["avg_frame_rate"]).split("/", 1)
    fps = float(numerator) / float(denominator)
    duration = float(stream.get("duration") or payload["format"]["duration"])
    return {
        "codec": str(stream["codec_name"]),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "duration_s": duration,
    }


def audit(args: argparse.Namespace) -> dict:
    videos = case_paths(args.case_video)
    rollouts = case_paths(args.case_rollout)
    video_rows = []
    rollout_rows = []
    checks = {}
    for case in REPRESENTATIVE_CASES:
        video = videos[case]
        rollout = rollouts[case]
        metadata = probe(video)
        payload = json.loads(rollout.read_text(encoding="utf-8"))
        results = payload.get("results")
        checks[str(case)] = {
            "video_exists": video.is_file() and video.stat().st_size > 0,
            "codec_supported": metadata["codec"] in {"h264", "hevc"},
            "width_bounded": metadata["width"] >= RENDER_CONFIG["minimum_width"],
            "height_bounded": metadata["height"] >= RENDER_CONFIG["minimum_height"],
            "fps_bounded": metadata["fps"] >= 24.0,
            "duration_bounded": metadata["duration_s"]
            >= RENDER_CONFIG["minimum_duration_s"],
            "rollout_case": payload.get("cases") == [case],
            "rollout_passed": payload.get("passed") is True
            and isinstance(results, list)
            and len(results) == 1
            and results[0].get("case") == case
            and results[0].get("passed") is True,
            "rollout_source": payload.get("trajectory_command_source")
            == "model_based_planner_plus_torchscript_residual",
            "rollout_profile": payload.get("tracking_profile")
            == RENDER_CONFIG["tracking_profile"],
            "rollout_command_base": payload.get("policy_command_base")
            == "model_based_planner",
            "rollout_scales": payload.get("residual_action_scales")
            == RENDER_CONFIG["residual_action_scales"],
            "rollout_control_ownership": all(
                payload.get(name) == value
                for name, value in RENDER_CONFIG[
                    "control_ownership"
                ].items()
            ),
        }
        video_rows.append({"case": case, **identity(video), **metadata})
        rollout_rows.append({"case": case, **identity(rollout)})
    passed = all(all(value.values()) for value in checks.values())
    return {
        "schema": "cinebotrl_two_wheel_riser_learned_render_media_manifest_v1",
        "admission": identity(args.admission),
        "preflight": identity(args.preflight),
        "policy": identity(args.policy),
        "source_all79_report": identity(args.all79_report),
        "cases": REPRESENTATIVE_CASES,
        "rollout_gates": rollout_rows,
        "videos": video_rows,
        "media_checks": checks,
        "manual_visual_review_required": True,
        "runtime_started": True,
        "recording_started": True,
        "training_started": False,
        "ppo_authorized": False,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--all79-report", type=Path, required=True)
    parser.add_argument("--case-video", action="append", default=[])
    parser.add_argument("--case-rollout", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
