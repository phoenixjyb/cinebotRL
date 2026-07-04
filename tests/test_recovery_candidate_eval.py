"""Tests for the Proto2 recovery candidate evaluation helpers.

These tests avoid Isaac Sim and can be run directly:

    python -X utf8 tests/test_recovery_candidate_eval.py
"""

from __future__ import annotations

from argparse import Namespace
import importlib.util
from pathlib import Path
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = REPO_ROOT / "scripts" / "reinforcement_learning" / "sb3" / "evaluate_recovery_candidate.py"
spec = importlib.util.spec_from_file_location("evaluate_recovery_candidate", EVALUATOR_PATH)
assert spec is not None and spec.loader is not None
evaluate_recovery_candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate_recovery_candidate)

_metric_stats = evaluate_recovery_candidate._metric_stats
_resolve_trajectory_manifest = evaluate_recovery_candidate._resolve_trajectory_manifest
summarize_episode_details = evaluate_recovery_candidate.summarize_episode_details


def _write_manifest(root: Path, lines: list[str]) -> Path:
    manifest = root / "manifest.txt"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        path = root / line
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"poses": []}', encoding="utf-8")
    return manifest


def test_resolve_trajectory_manifest_filters_and_limits() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = _write_manifest(
            root,
            [
                "# comment",
                "cinematic_db/crane_down/crane_down_000.json",
                "cinematic_db/crane_down/crane_down_019.json",
                "cinematic_db/crane_up/crane_up_001.json",
                "cinematic_db/handheld_subtle/handheld_subtle_004.json",
            ],
        )
        output_dir = root / "out"
        args = Namespace(
            trajectory_dir=str(root),
            trajectory_manifest=str(manifest),
            trajectory_stage="stage1_recovery",
            trajectory_category=["crane_down"],
            trajectory_file_contains=["019"],
            max_trajectories=1,
        )

        trajectory_dir, resolved_manifest, selected = _resolve_trajectory_manifest(args, output_dir)

        assert trajectory_dir == root
        assert resolved_manifest == output_dir / "resolved_feasibility_manifest.txt"
        assert len(selected) == 1
        assert selected[0].replace("\\", "/").endswith("cinematic_db/crane_down/crane_down_019.json")
        assert "crane_down_019.json" in resolved_manifest.read_text(encoding="utf-8")


def test_metric_stats_ignores_none_and_nonfinite() -> None:
    stats = _metric_stats([1.0, None, float("nan"), 3.0])
    assert stats["count"] == 2
    assert stats["mean"] == 2.0
    assert stats["max"] == 3.0


def test_summarize_episode_details_groups_by_category_and_file() -> None:
    summary = summarize_episode_details(
        [
            {
                "trajectory_category": "crane_down",
                "trajectory_file": "crane_down_000.json",
                "ee_pos_error_mean_m": 0.1,
                "ee_pos_error_p95_m": 0.2,
                "ee_ori_error_mean_deg": 5.0,
                "unreachable_zone_pct": 0.0,
                "workspace_hard_exceed_pct": 0.0,
                "obstacle_unsafe_pct": 0.0,
                "obstacle_collision_pct": 0.0,
            },
            {
                "trajectory_category": "crane_down",
                "trajectory_file": "crane_down_000.json",
                "ee_pos_error_mean_m": 0.3,
                "ee_pos_error_p95_m": 0.4,
                "ee_ori_error_mean_deg": 15.0,
                "unreachable_zone_pct": 10.0,
                "workspace_hard_exceed_pct": 20.0,
                "obstacle_unsafe_pct": 0.0,
                "obstacle_collision_pct": 0.0,
            },
        ]
    )

    category = summary["category:crane_down"]
    file_summary = summary["file:crane_down_000.json"]
    assert category["ee_pos_error_mean_m"]["count"] == 2
    assert category["ee_pos_error_mean_m"]["mean"] == 0.2
    assert file_summary["unreachable_zone_pct"]["max"] == 10.0
    assert file_summary["obstacle_collision_pct"]["mean"] == 0.0


def main() -> None:
    test_resolve_trajectory_manifest_filters_and_limits()
    test_metric_stats_ignores_none_and_nonfinite()
    test_summarize_episode_details_groups_by_category_and_file()


if __name__ == "__main__":
    main()
