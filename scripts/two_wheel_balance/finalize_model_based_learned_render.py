#!/usr/bin/env python3
"""Finalize learned-render evidence from media and explicit visual review."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rl_platform.tasks.two_wheel_balance import (
    riser_model_based_learned_render_contract as contract,
)


REPRESENTATIVE_CASES = contract.REPRESENTATIVE_CASES


VISUAL_CHECKS = {
    "robot_asset_intact",
    "riser_motion_visible",
    "camera_and_gimbal_visible",
    "wheel_ground_contact_plausible",
    "no_detached_links",
    "no_abnormal_oscillation",
}


def identity(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def finalize(media_path: Path, review_path: Path) -> dict:
    media = _load(media_path)
    review = _load(review_path)
    checks = review.get("visual_checks")
    if (
        media.get("schema")
        != "cinebotrl_two_wheel_riser_learned_render_media_manifest_v1"
        or media.get("passed") is not True
        or media.get("cases") != REPRESENTATIVE_CASES
        or review.get("schema")
        != "cinebotrl_two_wheel_riser_learned_render_visual_review_v1"
        or review.get("cases") != REPRESENTATIVE_CASES
        or review.get("videos") != media.get("videos")
        or not isinstance(review.get("reviewer"), str)
        or not review["reviewer"].strip()
        or not isinstance(review.get("reviewed_at_utc"), str)
        or not review["reviewed_at_utc"].endswith("Z")
        or not isinstance(checks, dict)
        or set(checks) != VISUAL_CHECKS
        or not all(value is True for value in checks.values())
        or review.get("passed") is not True
    ):
        raise ValueError("learned render visual review is incomplete or mismatched")
    return {
        "schema": "cinebotrl_two_wheel_riser_learned_render_audit_v2",
        "policy": media["policy"],
        "source_all79_report": media["source_all79_report"],
        "render_admission": media["admission"],
        "render_preflight": media["preflight"],
        "media_manifest": identity(media_path),
        "visual_review": identity(review_path),
        "cases": REPRESENTATIVE_CASES,
        "rollout_gates": media["rollout_gates"],
        "videos": media["videos"],
        "visual_checks": checks,
        "passed": True,
        "training_started": False,
        "ppo_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media-manifest", type=Path, required=True)
    parser.add_argument("--visual-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = finalize(args.media_manifest, args.visual_review)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
