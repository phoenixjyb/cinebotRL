import json
from pathlib import Path

import pytest

from scripts.two_wheel_balance import audit_case78_camera_recovery_candidate as audit_module


def synthetic_gate() -> dict:
    trace = [
        {
            "progress_scale": 0.5,
            "position_error_m": 0.16,
            "camera_lever_arm_correction_saturated": True,
        }
        for _ in range(10)
    ]
    return {
        "results": [
            {
                "case": 78,
                "completed_phase_time_s": 192.0,
                "execution_duration_s": 192.0,
                "termination": None,
                "checks": {
                    "position_p95_bounded": False,
                    "position_max_bounded": True,
                    "pitch_bounded": True,
                },
                "position_error_p95_m": 0.162,
                "position_error_max_m": 0.23,
                "completed_steps": 40_000,
                "trace": trace,
            }
        ]
    }


def test_camera_recovery_candidate_is_bounded(monkeypatch, tmp_path: Path) -> None:
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps(synthetic_gate()) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        audit_module,
        "CANONICAL_GATE_SHA256",
        audit_module.sha256_file(gate),
    )

    result = audit_module.audit(gate)

    assert result["failed_checks"] == ["position_p95_bounded"]
    assert result["candidate_progress_scale_mean"] == pytest.approx(0.2)
    assert result["projected_candidate_steps"] <= 115_381
    assert result["candidate_supported_for_bounded_canary"] is True
    assert result["runtime_authorized"] is False


def test_camera_recovery_candidate_rejects_broad_failure(
    monkeypatch, tmp_path: Path
) -> None:
    payload = synthetic_gate()
    payload["results"][0]["checks"]["pitch_bounded"] = False
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        audit_module,
        "CANONICAL_GATE_SHA256",
        audit_module.sha256_file(gate),
    )

    result = audit_module.audit(gate)

    assert result["checks"]["only_position_p95_failed"] is False
    assert result["candidate_supported_for_bounded_canary"] is False
