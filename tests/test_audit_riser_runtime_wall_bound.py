import json
import os
from pathlib import Path

from scripts.two_wheel_balance.audit_riser_runtime_wall_bound import audit


def write_json(path: Path, payload: dict, mtime: float) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    os.utime(path, (mtime, mtime))


def test_wall_bound_uses_host_envelope_not_virtual_step_time(
    tmp_path: Path,
) -> None:
    admission = tmp_path / "admission.json"
    gate = tmp_path / "gate.json"
    final_status = tmp_path / "final_status.json"
    timeout_final = tmp_path / "timeout_final.json"
    write_json(admission, {}, 1000.0)
    write_json(
        gate,
        {
            "results": [
                {
                    "case": 30,
                    "completed_steps": 10_000,
                    "wall_duration_s": 50.0,
                    "passed": True,
                }
            ]
        },
        1398.0,
    )
    write_json(final_status, {"passed": True}, 1400.0)
    write_json(
        timeout_final,
        {
            "playback_exit_code": 124,
            "maximum_wall_duration_s": 900.0,
            "dynamic_qualification_passed": False,
            "case78_validation_admitted": False,
        },
        2000.0,
    )

    result = audit(
        admission,
        gate,
        final_status,
        timeout_final,
        maximum_steps=100_000,
        margin_s=900.0,
        rounding_s=300,
    )

    assert result["reference_virtual_step_duration_s"] == 50.0
    assert result["reference_host_filesystem_envelope_s"] == 400.0
    assert result["conservative_policy_step_rate_hz"] == 25.0
    assert result["estimated_maximum_loop_wall_s"] == 4000.0
    assert result["proposed_maximum_wall_duration_s"] == 5100
    assert result["audit_passed"] is True
    assert result["runtime_retry_authorized"] is False


def test_wall_bound_fails_closed_without_positive_host_envelope(
    tmp_path: Path,
) -> None:
    admission = tmp_path / "admission.json"
    gate = tmp_path / "gate.json"
    final_status = tmp_path / "final_status.json"
    timeout_final = tmp_path / "timeout_final.json"
    write_json(admission, {}, 1000.0)
    write_json(
        gate,
        {
            "results": [
                {
                    "case": 30,
                    "completed_steps": 10_000,
                    "wall_duration_s": 50.0,
                    "passed": True,
                }
            ]
        },
        1000.0,
    )
    write_json(final_status, {"passed": True}, 1000.0)
    write_json(
        timeout_final,
        {
            "playback_exit_code": 124,
            "maximum_wall_duration_s": 900.0,
            "dynamic_qualification_passed": False,
            "case78_validation_admitted": False,
        },
        1000.0,
    )

    result = audit(
        admission,
        gate,
        final_status,
        timeout_final,
        maximum_steps=100_000,
        margin_s=900.0,
        rounding_s=300,
    )

    assert result["audit_passed"] is False
    assert result["proposed_maximum_wall_duration_s"] is None
    assert result["runtime_retry_authorized"] is False

