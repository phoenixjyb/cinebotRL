import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/two_wheel_balance/audit_riser_motion_direction_compatibility.py"


def test_compatibility_audit_preserves_healthy_tracking_commands(
    tmp_path: Path,
) -> None:
    trace = [
        {
            "step": step,
            "target_base_xy_yaw": [0.05, 0.05, 0.1],
            "actual_base_xy_yaw": [0.0, 0.0, 0.0],
            "phase_feedforward_v_mps": 0.01,
            "phase_feedforward_wz_rad_s": 0.0,
            "vx_reference_mps": 0.09,
            "wz_reference_rad_s": 0.195,
        }
        for step in (1, 201)
    ]
    payload = {
        "passed": True,
        "results": [
            {
                "case": 1,
                "passed": True,
                "peak_base_xy_error_m": 0.08,
                "checks": {"position": True, "attitude": True},
                "executed_residual_dataset": None,
                "trace": trace,
            }
        ],
    }
    case_json = tmp_path / "case.json"
    case_json.write_text(json.dumps(payload), encoding="utf-8")
    case_hash = hashlib.sha256(case_json.read_bytes()).hexdigest()
    output = tmp_path / "summary.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--case-json",
            str(case_json),
            "--expected-sha256",
            case_hash,
            "--output",
            str(output),
        ],
        check=False,
    )
    summary = json.loads(output.read_text(encoding="utf-8"))

    assert result.returncode == 0
    assert summary["passed"]
    assert summary["healthy_cases"] == [1]
    assert summary["rows"][0]["recovery_active_sample_count"] == 0
    assert summary["rows"][0]["candidate_command_delta_max"] == 0.0
    assert not summary["candidate_dynamically_validated"]
    assert not summary["valid_for_training"]
