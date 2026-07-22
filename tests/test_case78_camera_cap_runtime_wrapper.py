from pathlib import Path


def test_camera_cap_runtime_wrapper_is_one_case_bounded_and_no_learning() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/two_wheel_balance/run_case78_camera_cap_canary.sh"
    ).read_text(encoding="utf-8")
    assert "--cases 78" in source
    assert "timeout --signal=TERM --kill-after=30s 5400" in source
    assert "--runtime-heartbeat-interval-steps 2000" in source
    assert "--maximum-camera-lever-arm-correction-m 0.10" in source
    assert "--enable-camera-error-recovery-governor" not in source
    assert "--dataset-dir" not in source
    assert "--policy" not in source
    assert 'rm -f "$AUTHORIZATION_FILE"' in source
    assert "assert_gpu_free" in source
    assert "wait_for_gpu_release" in source
