from pathlib import Path


WRAPPER = (
    Path(__file__).parents[1]
    / "scripts/two_wheel_balance/run_initial_teacher41_case8_canary.sh"
)


def test_wrapper_is_one_case_zero_then_learned_and_fail_closed() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    assert "CASE=8" in source
    assert "--zero-policy-action" in source
    assert "--residual-policy \"$POLICY_WIN\"" in source
    assert source.index("--zero-policy-action") < source.index(
        "--residual-policy \"$POLICY_WIN\""
    )
    assert "--maximum-regression-fraction 0.05" in source
    assert "--minimum-zero-improvement-fraction 0.05" in source
    assert "case78_authorized\": False" in source
    assert "--dataset-dir" not in source
    assert "--raw-teacher-dir" not in source
    assert "--policy-trace-dir" not in source


def test_wrapper_consumes_one_use_token_before_isaac() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    token_delete = source.index('rm -f "$AUTHORIZATION_FILE"')
    first_playback = source.index('timeout --signal=TERM --kill-after=30s 600')
    assert token_delete < first_playback
    assert "stat -c '%a'" in source
    assert "AUTHORIZATION_SHA256=" in source
    assert "wait_gpu_free" in source
    assert "HEAD\" == \"$UPSTREAM" in source


def test_wrapper_pins_cpu_admission_plan_teacher_policy_and_controller() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    for name in (
        "CPU_CONTRACT_SHA256",
        "CPU_ADMISSION_SHA256",
        "PLAN_SHA256",
        "TEACHER_GATE_SHA256",
        "POLICY_SHA256",
        "PLAYBACK_SHA256",
        "ROLLOUT_GATE_SHA256",
        "GAINS_SHA256",
    ):
        assert f'{name}="' in source
    assert "--controller-wz-kp 1.05" in source
    assert "--camera-lever-arm-compensation-gain 1.0" in source
    assert "--maximum-camera-lever-arm-correction-m 0.05" in source
    assert "--residual-action-scales 0.35,0.40,0.10" in source
