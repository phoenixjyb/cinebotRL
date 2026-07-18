import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/two_wheel_balance/validate_riser_smoothed_gate_c_canary.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(tmp_path: Path) -> Path:
    items = []
    for case in range(1, 4):
        plan = tmp_path / f"case_{case:04d}_smoothed_riser_plan_v1.npz"
        plan.write_bytes(f"plan-{case}".encode())
        accepted = case != 3
        items.append(
            {
                "case": case,
                "file": plan.name,
                "plan_sha256": _sha(plan),
                "source_json_sha256": "b" * 64,
                "execution_source_duration_ratio": 1.5,
                "checks": {"integrity": accepted},
                "kinematic_checks": {"rates": accepted},
                "timing_transition_kinematic_gate_passed": accepted,
                "valid_for_training": False,
                "passed": accepted,
            }
        )
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema": "cinebotrl_two_wheel_riser_smoothed_plan_export_v1",
                "plan_schema": "cinebotrl_two_wheel_riser_smoothed_plan_v1",
                "source_manifest_sha256": "a" * 64,
                "source_package_case_count": 3,
                "requested_cases": [1, 2, 3],
                "attempted_cases": [1, 2, 3],
                "passed_cases": [1, 2],
                "rejected_cases": [3],
                "minimum_passes_required": 2,
                "minimum_pass_count_met": True,
                "portfolio_gate_passed": True,
                "tracked_state_clean": True,
                "code_commit": "c" * 40,
                "upstream_commit": "c" * 40,
                "isaac_started": False,
                "residual_capture_started": False,
                "bc_started": False,
                "ppo_started": False,
                "differential_session_work_started": False,
                "valid_for_training": False,
                "items": items,
            }
        )
    )
    return path


def _run(tmp_path: Path, manifest: Path, case: int):
    output = tmp_path / "admission.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(manifest),
            "--expected-manifest-sha256",
            _sha(manifest),
            "--expected-source-manifest-sha256",
            "a" * 64,
            "--expected-planner-commit",
            "c" * 40,
            "--expected-count",
            "3",
            "--minimum-candidates",
            "2",
            "--case",
            str(case),
            "--output",
            str(output),
        ],
        check=False,
    )
    return result, json.loads(output.read_text())


def test_smoothed_gate_c_admits_only_a_hash_bound_passing_plan(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    result, admission = _run(tmp_path, manifest, 2)
    assert result.returncode == 0
    assert admission["gate_c_execution_authorized"]
    assert admission["selected_plan"]["case"] == 2
    assert not admission["residual_capture_authorized"]
    assert not admission["bc_authorized"]
    assert not admission["ppo_authorized"]


def test_smoothed_gate_c_rejects_failed_or_tampered_plan(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    result, admission = _run(tmp_path, manifest, 3)
    assert result.returncode == 6
    assert not admission["top_checks"]["requested_case_admitted"]

    (tmp_path / "case_0002_smoothed_riser_plan_v1.npz").write_bytes(b"tampered")
    result, admission = _run(tmp_path, manifest, 2)
    assert result.returncode == 6
    assert not admission["rows"][1]["checks"]["plan_hash_matches"]


def test_smoothed_gate_c_rejects_learning_or_lineage_change(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["bc_started"] = True
    payload["upstream_commit"] = "d" * 40
    manifest.write_text(json.dumps(payload))
    result, admission = _run(tmp_path, manifest, 2)
    assert result.returncode == 6
    assert not admission["top_checks"]["runtime_and_learning_not_started"]
    assert not admission["top_checks"]["planner_commit_bound"]
