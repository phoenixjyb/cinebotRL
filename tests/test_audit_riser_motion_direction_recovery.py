import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/two_wheel_balance/audit_riser_motion_direction_recovery.py"


def test_motion_direction_audit_localizes_reverse_steering_cancellation(
    tmp_path: Path,
) -> None:
    trace = []
    for step in (1, 201):
        trace.append(
            {
                "step": step,
                "elapsed_s": (step - 1) / 200.0,
                "phase_time_s": (step - 1) / 200.0,
                "position_error_m": 0.4,
                "base_xy_error_m": 0.3,
                "target_base_xy_yaw": [-0.2, 0.2, -0.25],
                "actual_base_xy_yaw": [0.0, 0.0, 0.0],
                "phase_feedforward_v_mps": 0.01,
                "phase_feedforward_wz_rad_s": 0.0,
                "vx_reference_mps": -0.31,
                "wz_reference_rad_s": 0.0,
            }
        )
    payload = {
        "results": [
            {
                "case": 74,
                "dynamic_quality_passed": False,
                "executed_residual_dataset": None,
                "checks": {
                    "completed_reference": False,
                    "position_p95_bounded": False,
                    "position_max_bounded": False,
                    "proxy_servo_error_bounded": True,
                    "proxy_saturation_bounded": True,
                    "no_termination": True,
                },
                "trace": trace,
            }
        ]
    }
    case_json = tmp_path / "case.json"
    case_json.write_text(json.dumps(payload), encoding="utf-8")
    case_hash = hashlib.sha256(case_json.read_bytes()).hexdigest()
    output = tmp_path / "audit.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--case-json",
            str(case_json),
            "--expected-case-sha256",
            case_hash,
            "--output",
            str(output),
        ],
        check=False,
    )
    summary = json.loads(output.read_text(encoding="utf-8"))

    assert result.returncode == 0
    assert summary["passed"]
    assert summary["direction_conflict_sample_count"] == 2
    assert summary["bad_position_direction_conflict_sample_count"] == 2
    assert summary["peak_position_error"]["candidate_yaw_rate_rad_s"] < -0.2
    assert summary["candidate_scope"] == "base_error_gated_recovery_only"
    assert not summary["candidate_dynamically_validated"]
    assert not summary["valid_for_training"]
