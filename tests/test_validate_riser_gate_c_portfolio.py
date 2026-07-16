import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/two_wheel_balance/validate_riser_gate_c_portfolio.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portfolio(tmp_path: Path, count: int = 3) -> Path:
    items = []
    for case in range(1, count + 1):
        plan = tmp_path / f"case_{case:04d}_exact_source_riser_playback_v1.npz"
        plan.write_bytes(f"plan-{case}".encode())
        items.append(
            {
                "case": case,
                "file": plan.name,
                "plan_sha256": _sha(plan),
                "trajectory_integrity_passed": True,
                "source_timestamps_preserved": True,
                "ordered_target_geometry_preserved": True,
                "initialization_separated": True,
                "kinematic_gate_passed": case != 3,
                "quality_gate_passed": False,
                "valid_for_training": False,
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "cinebotrl_two_wheel_riser_exact_source_plan_portfolio_v1",
                "trajectory_integrity_contract": "exact_source_v1",
                "source_manifest_sha256": "a" * 64,
                "case_count": count,
                "exact_source_pass_count": count,
                "kinematic_accepted_count": 2,
                "kinematic_accepted_cases": [1, 2],
                "kinematic_rejected_count": 1,
                "kinematic_rejected_cases": [3],
                "minimum_gate_c_candidates": 2,
                "gate_c_candidate_count_sufficient": True,
                "gate_c_dynamic_quality_started": False,
                "quality_gate_passed": False,
                "valid_for_training": False,
                "training_started": False,
                "ppo_started": False,
                "quarantined_lineage_absent": True,
                "base_manifest_training_started": False,
                "recovery_manifest_training_started": False,
                "items": items,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _run(tmp_path: Path, manifest: Path, cases: str, expected_hash: str | None = None):
    output = tmp_path / "audit.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(manifest),
            "--expected-manifest-sha256",
            expected_hash or _sha(manifest),
            "--expected-source-manifest-sha256",
            "a" * 64,
            "--expected-count",
            "3",
            "--minimum-candidates",
            "2",
            "--cases",
            cases,
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result, json.loads(output.read_text())


def test_gate_c_admits_only_requested_kinematic_passes(tmp_path: Path) -> None:
    manifest = _portfolio(tmp_path)
    result, audit = _run(tmp_path, manifest, "1,2")
    assert result.returncode == 0
    assert audit["gate_c_execution_authorized"]
    assert not audit["residual_capture_authorized"]
    assert not audit["bc_authorized"]
    assert not audit["ppo_authorized"]


def test_gate_c_rejects_rejected_case_or_tampered_plan(tmp_path: Path) -> None:
    manifest = _portfolio(tmp_path)
    result, audit = _run(tmp_path, manifest, "3")
    assert result.returncode == 6
    assert not audit["top_checks"]["requested_cases_admitted"]

    plan = tmp_path / "case_0001_exact_source_riser_playback_v1.npz"
    plan.write_bytes(b"tampered")
    result, audit = _run(tmp_path, manifest, "1")
    assert result.returncode == 6
    assert not audit["rows"][0]["checks"]["plan_hash_matches"]


def test_gate_c_rejects_manifest_hash_or_training_flags(tmp_path: Path) -> None:
    manifest = _portfolio(tmp_path)
    result, audit = _run(tmp_path, manifest, "1", "0" * 64)
    assert result.returncode == 6
    assert not audit["top_checks"]["manifest_hash_matches"]

    payload = json.loads(manifest.read_text())
    payload["training_started"] = True
    manifest.write_text(json.dumps(payload))
    result, audit = _run(tmp_path, manifest, "1")
    assert result.returncode == 6
    assert not audit["top_checks"]["training_disabled"]
