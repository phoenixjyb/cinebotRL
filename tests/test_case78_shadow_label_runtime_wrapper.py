from pathlib import Path


def test_shadow_label_runtime_is_one_case_bounded_and_trace_only() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/two_wheel_balance/run_case78_shadow_label_measurement.sh"
    ).read_text(encoding="utf-8")
    assert "--cases 78" in source
    assert "timeout --signal=TERM --kill-after=30s 5400" in source
    assert "--runtime-heartbeat-interval-steps 2000" in source
    assert "--maximum-camera-lever-arm-correction-m 0.10" in source
    assert "--residual-action-scales 0.35,0.40,0.10" in source
    assert "--shadow-teacher-trace-dir" in source
    assert "--dataset-dir" not in source
    assert "--raw-teacher-dir" not in source
    assert "--residual-policy" not in source
    assert "--zero-policy-action" not in source
    assert 'rm -f "$AUTHORIZATION_FILE"' in source
    assert "assert_gpu_free" in source
    assert "wait_for_gpu_release" in source
