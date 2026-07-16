import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/two_wheel_balance/validate_riser_exact_source_manifest.py"


def _item(case: int, count: int = 5) -> dict:
    return {
        "case": case,
        "source_pose_count": count,
        "source_timestamp_count": count,
        "retargeted_waypoint_state_count": count,
        "transition_count": count - 1,
        "source_duration_s": 1.0,
        "ordered_target_geometry_preserved": True,
        "source_timestamps_preserved": True,
        "initialization_separated": True,
        "trajectory_integrity_passed": True,
        "quality_gate_passed": True,
        "valid_for_training": True,
    }


def _run(tmp_path: Path, manifest: dict) -> tuple[subprocess.CompletedProcess, dict]:
    source = tmp_path / "manifest.json"
    output = tmp_path / "audit.json"
    source.write_text(json.dumps(manifest), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(source),
            "--expected-count",
            "3",
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result, json.loads(output.read_text(encoding="utf-8"))


def test_exact_source_manifest_requires_every_integrity_and_quality_gate(tmp_path) -> None:
    manifest = {
        "trajectory_integrity_contract": "exact_source_v1",
        "valid_for_training": True,
        "quality_gate_passed": True,
        "case_count": 3,
        "items": [_item(case) for case in range(1, 4)],
    }
    result, audit = _run(tmp_path, manifest)
    assert result.returncode == 0
    assert audit["passed"]
    assert audit["training_authorized"]
    assert not audit["ppo_authorized"]


def test_integrity_only_canaries_are_rejected_for_training(tmp_path) -> None:
    manifest = {
        "trajectory_integrity_contract": "exact_source_v1",
        "valid_for_training": False,
        "case_count": 3,
        "items": [
            {
                "episode_index": case,
                "source_pose_count": 5,
                "reference_pose_count": 5,
                "state_count": 5,
                "action_count": 4,
                "duration_s": 1.0,
                "trajectory_integrity_passed": True,
                "valid_for_training": False,
            }
            for case in range(1, 4)
        ],
    }
    result, audit = _run(tmp_path, manifest)
    assert result.returncode == 6
    assert not audit["passed"]
    assert not audit["training_authorized"]
    assert not audit["top_checks"]["package_valid_for_training"]


def test_exact_source_manifest_rejects_resampling_or_initialization_blending(tmp_path) -> None:
    items = [_item(case) for case in range(1, 4)]
    items[1]["retargeted_waypoint_state_count"] = 4
    items[2]["initialization_separated"] = False
    manifest = {
        "trajectory_integrity_contract": "exact_source_v1",
        "valid_for_training": True,
        "quality_gate_passed": True,
        "case_count": 3,
        "items": items,
    }
    result, audit = _run(tmp_path, manifest)
    assert result.returncode == 6
    assert not audit["rows"][1]["checks"]["waypoint_state_count_preserved"]
    assert not audit["rows"][2]["checks"]["initialization_separated"]
