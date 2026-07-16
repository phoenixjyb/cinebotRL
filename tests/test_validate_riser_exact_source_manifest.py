import hashlib
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/two_wheel_balance/validate_riser_exact_source_manifest.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_reference_package(tmp_path: Path, count: int = 3) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    items = []
    for case in range(1, count + 1):
        episode = tmp_path / f"episode_{case:04d}"
        episode.mkdir()
        source = episode / "source.json"
        source.write_text(
            json.dumps(
                {
                    "poses": [
                        {
                            "position": [0.0, 0.0, 0.9],
                            "orientation": [0.0, 0.0, 0.0, 1.0],
                            "time": 0.0,
                        },
                        {
                            "position": [0.1, 0.0, 0.9],
                            "orientation": [0.0, 0.0, 0.0, 1.0],
                            "time": 1.0,
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        items.append(
            {
                "episode_index": case,
                "bundled_source_json": f"episode_{case:04d}/source.json",
                "source_json_sha256": _sha(source),
                "source_pose_count": 2,
                "source_duration_s": 1.0,
                "trajectory_integrity_contract": "exact_source_v1",
                "integrity_passed": True,
                "quality_qualified_teacher": False,
                "valid_for_training": False,
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "trajectory_integrity_contract": "exact_source_v1",
                "episode_count": count,
                "integrity_passed": True,
                "quality_qualified_teacher": False,
                "valid_for_training": False,
                "items": items,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _training_item(case: int, count: int = 5) -> dict:
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
        "source_json_sha256": "a" * 64,
        "plan_sha256": "b" * 64,
    }


def _run(
    tmp_path: Path, manifest: Path, mode: str, expected_hash: str | None = None
) -> tuple[subprocess.CompletedProcess, dict]:
    output = tmp_path / f"{mode}_audit.json"
    command = [
        sys.executable,
        str(SCRIPT),
        "--manifest",
        str(manifest),
        "--expected-count",
        "3",
        "--mode",
        mode,
        "--output",
        str(output),
    ]
    if expected_hash is not None:
        command.extend(("--expected-manifest-sha256", expected_hash))
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert output.is_file(), result.stderr
    return result, json.loads(output.read_text(encoding="utf-8"))


def test_reference_ingest_accepts_integrity_only_exact_source_package(tmp_path) -> None:
    manifest = _write_reference_package(tmp_path)
    result, audit = _run(tmp_path, manifest, "reference_ingest", _sha(manifest))
    assert result.returncode == 0
    assert audit["passed"]
    assert audit["reference_ingest_authorized"]
    assert not audit["training_authorized"]


def test_reference_ingest_rejects_hash_mismatch_and_copied_old_name(tmp_path) -> None:
    manifest = _write_reference_package(tmp_path)
    result, audit = _run(tmp_path, manifest, "reference_ingest", "0" * 64)
    assert result.returncode == 6
    assert not audit["manifest_hash_matches_expected"]

    source = tmp_path / "episode_0002/source.json"
    source.write_text(source.read_text() + "\n", encoding="utf-8")
    result, audit = _run(tmp_path, manifest, "reference_ingest")
    assert result.returncode == 6
    assert not audit["rows"][1]["checks"]["source_hash_matches"]


def test_reference_ingest_rejects_missing_reordered_or_retimed_anchors(tmp_path) -> None:
    manifest = _write_reference_package(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["items"][0]["episode_index"] = 2
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    result, audit = _run(tmp_path, manifest, "reference_ingest")
    assert result.returncode == 6
    assert not audit["top_checks"]["contiguous_cases"]

    manifest = _write_reference_package(tmp_path / "retimed")
    source = tmp_path / "retimed/episode_0001/source.json"
    payload = json.loads(source.read_text())
    payload["poses"][1]["time"] = 0.5
    source.write_text(json.dumps(payload), encoding="utf-8")
    package = json.loads(manifest.read_text())
    package["items"][0]["source_json_sha256"] = _sha(source)
    manifest.write_text(json.dumps(package), encoding="utf-8")
    result, audit = _run(tmp_path / "retimed", manifest, "reference_ingest")
    assert result.returncode == 6
    assert not audit["rows"][0]["checks"]["source_time_preserved"]


def test_reference_only_package_is_rejected_from_training(tmp_path) -> None:
    manifest = _write_reference_package(tmp_path)
    result, audit = _run(tmp_path, manifest, "training")
    assert result.returncode == 6
    assert not audit["training_authorized"]
    assert not audit["top_checks"]["package_valid_for_training"]


def test_training_manifest_requires_mapping_quality_and_clean_lineage(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    payload = {
        "trajectory_integrity_contract": "exact_source_v1",
        "valid_for_training": True,
        "quality_gate_passed": True,
        "quarantined_lineage_absent": True,
        "case_count": 3,
        "items": [_training_item(case) for case in range(1, 4)],
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    result, audit = _run(tmp_path, manifest, "training")
    assert result.returncode == 0
    assert audit["training_authorized"]

    payload["items"][1]["retargeted_waypoint_state_count"] = 4
    payload["items"][2]["initialization_separated"] = False
    payload["quarantined_lineage_absent"] = False
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    result, audit = _run(tmp_path, manifest, "training")
    assert result.returncode == 6
    assert not audit["rows"][1]["checks"]["waypoint_state_count_preserved"]
    assert not audit["rows"][2]["checks"]["initialization_separated"]
    assert not audit["top_checks"]["quarantined_lineage_absent"]
