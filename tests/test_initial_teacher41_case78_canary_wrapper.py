from pathlib import Path


WRAPPER = (
    Path(__file__).parents[1]
    / "scripts/two_wheel_balance/run_initial_teacher41_case78_canary.sh"
)


def test_wrapper_runs_learned_first_then_zero_only_on_dynamic_pass() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    assert "CASE=78" in source
    assert source.index('--residual-policy "$POLICY_WIN"') < source.index(
        "--zero-policy-action"
    )
    assert 'LEARNED_DYNAMIC_PASSED="$(python3' in source
    assert '"$LEARNED_DYNAMIC_PASSED" == 1' in source
    assert "--maximum-regression-fraction 0.05" in source
    assert "--minimum-zero-improvement-fraction 0.05" in source


def test_wrapper_pins_case78_camera_contract_and_has_no_capture_path() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    assert "--maximum-camera-lever-arm-correction-m 0.10" in source
    assert "--residual-action-scales 0.35,0.40,0.10" in source
    assert "CPU_CONTRACT_SHA256=" in source
    assert "PLAN_SHA256=" in source
    assert "TEACHER_GATE_SHA256=" in source
    assert "POLICY_SHA256=" in source
    assert "--dataset-dir" not in source
    assert "--raw-teacher-dir" not in source
    assert "--policy-trace-dir" not in source
    assert "--shadow-teacher-trace-dir" not in source


def test_wrapper_consumes_token_before_first_isaac_and_is_exclusive() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    assert source.index('rm -f "$AUTHORIZATION_FILE"') < source.index(
        "timeout --signal=TERM --kill-after=30s 5400"
    )
    assert "stat -c '%a'" in source
    assert "AUTHORIZATION_SHA256=" in source
    assert "wait_gpu_free" in source
    assert 'HEAD" == "$UPSTREAM' in source
