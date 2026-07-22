import json
from pathlib import Path

from scripts.two_wheel_balance import audit_case78_recovery_outcome as module


def gate(*, p95: float, scale: float, recovery: bool) -> dict:
    trace = [
        {
            "phase_time_s": float(index),
            "position_error_m": scale * (0.10 + index * 0.005),
        }
        for index in range(5)
    ]
    return {
        "results": [
            {
                "case": 78,
                "source_duration_s": 4.0,
                "execution_duration_s": 4.0,
                "completed_phase_time_s": 4.0,
                "termination": None,
                "checks": {
                    "position_p95_bounded": False,
                    "pitch_bounded": True,
                },
                "position_error_p95_m": p95,
                "position_error_max_m": p95 + 0.05,
                "completed_steps": 1000 if not recovery else 1100,
                "camera_recovery_activation_ratio": 0.0 if not recovery else 0.1,
                "dynamic_quality_passed": False,
                "trace": trace,
            }
        ]
    }


def test_recovery_outcome_rejects_worse_official_p95(
    monkeypatch, tmp_path: Path
) -> None:
    baseline = tmp_path / "baseline.json"
    recovery = tmp_path / "recovery.json"
    baseline.write_text(
        json.dumps(gate(p95=0.16, scale=1.0, recovery=False)) + "\n"
    )
    recovery.write_text(
        json.dumps(gate(p95=0.17, scale=1.1, recovery=True)) + "\n"
    )
    monkeypatch.setattr(module, "BASELINE_SHA256", module.sha256_file(baseline))
    monkeypatch.setattr(module, "RECOVERY_SHA256", module.sha256_file(recovery))

    result = module.audit(baseline, recovery, phase_step_s=0.5)

    assert result["audit_passed"] is True
    assert result["official_time_weighted"]["position_p95_delta_m"] > 0.0
    assert result["phase_aligned"]["position_p95_delta_m"] > 0.0
    assert result["camera_recovery_candidate_rejected"] is True
    assert result["camera_recovery_candidate_admitted"] is False
    assert result["runtime_authorized"] is False


def test_phase_improvement_does_not_admit_failed_dynamic_gate(
    monkeypatch, tmp_path: Path
) -> None:
    baseline = tmp_path / "baseline.json"
    recovery = tmp_path / "recovery.json"
    baseline.write_text(
        json.dumps(gate(p95=0.16, scale=1.0, recovery=False)) + "\n"
    )
    recovery.write_text(
        json.dumps(gate(p95=0.17, scale=0.9, recovery=True)) + "\n"
    )
    monkeypatch.setattr(module, "BASELINE_SHA256", module.sha256_file(baseline))
    monkeypatch.setattr(module, "RECOVERY_SHA256", module.sha256_file(recovery))

    result = module.audit(baseline, recovery, phase_step_s=0.5)

    assert result["audit_passed"] is True
    assert result["findings"]["phase_aligned_p95_improved"] is True
    assert result["findings"]["recovery_dynamic_gate_passed"] is False
    assert result["camera_recovery_candidate_rejected"] is True


def test_quantile_interpolates_sorted_values() -> None:
    assert module.quantile([0.0, 1.0, 2.0], 0.75) == 1.5


def test_phase_series_rejects_empty_or_unordered_trace() -> None:
    try:
        module.phase_series([])
    except ValueError as exc:
        assert str(exc) == "phase trace is empty"
    else:
        raise AssertionError("empty phase trace was accepted")

    try:
        module.phase_series(
            [
                {"phase_time_s": 1.0, "position_error_m": 0.1},
                {"phase_time_s": 0.5, "position_error_m": 0.2},
            ]
        )
    except ValueError as exc:
        assert str(exc) == "phase trace is not ordered"
    else:
        raise AssertionError("unordered phase trace was accepted")
