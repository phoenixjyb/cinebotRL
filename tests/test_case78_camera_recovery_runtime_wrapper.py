from pathlib import Path


def test_camera_recovery_runtime_wrapper_is_exact_and_no_capture() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/two_wheel_balance/run_case78_camera_recovery_canary.sh"
    ).read_text(encoding="utf-8")
    assert "timeout --signal=TERM --kill-after=30s 5400" in source
    assert "--cases 78" in source
    assert "--enable-camera-error-recovery-governor" in source
    assert "--camera-recovery-error-start-m 0.13" in source
    assert "--camera-recovery-error-full-m 0.155" in source
    assert "--minimum-camera-recovery-scale 0.20" in source
    assert "--runtime-heartbeat-interval-steps 2000" in source
    assert "--dataset-dir" not in source
    assert "--raw-teacher-dir" not in source
    assert "--policy-trace-dir" not in source
    assert "--shadow-teacher-trace-dir" not in source
    assert 'rm -f "$AUTHORIZATION_FILE"' in source

