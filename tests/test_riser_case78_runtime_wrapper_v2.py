from pathlib import Path


def test_v2_runtime_wrapper_is_one_case_heartbeat_only() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/two_wheel_balance/run_riser_case78_dynamic_canary_v2.sh"
    ).read_text(encoding="utf-8")
    assert "timeout --signal=TERM --kill-after=30s 5400" in source
    assert "--cases 78" in source
    assert "--runtime-heartbeat" in source
    assert "--runtime-heartbeat-interval-steps 2000" in source
    assert "--dataset-dir" not in source
    assert "--raw-teacher-dir" not in source
    assert "--policy-trace-dir" not in source
    assert "--shadow-teacher-trace-dir" not in source
    assert 'rm -f "$AUTHORIZATION_FILE"' in source


def test_v2_runtime_wrapper_uses_fresh_literal_namespace() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/two_wheel_balance/run_riser_case78_dynamic_canary_v2.sh"
    ).read_text(encoding="utf-8")
    assert "20260721_case78_dynamic_qualification_v2_heartbeat_exclusive" in source
    assert "\\$NAMESPACE" not in source

