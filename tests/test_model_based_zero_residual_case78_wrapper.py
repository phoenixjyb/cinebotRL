import os
from pathlib import Path
import subprocess


WRAPPER = (
    Path(__file__).parents[1]
    / "scripts/two_wheel_balance/run_model_based_zero_residual_case78_canary.sh"
)


def test_wrapper_is_case78_model_based_and_capture_closed() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    assert "CASE=78" in source
    assert "--policy-command-base model_based_planner" in source
    assert "--residual-action-scales 0.05,0.05,0.02" in source
    assert "--maximum-camera-lever-arm-correction-m 0.10" in source
    assert "--zero-policy-action" in source
    assert '--residual-policy "$POLICY_WIN"' in source
    assert source.index("--zero-policy-action") < source.index(
        '--residual-policy "$POLICY_WIN"'
    )
    assert "--dataset-dir" not in source
    assert "--raw-teacher-dir" not in source
    assert "--policy-trace-dir" not in source
    assert "--shadow-teacher-trace-dir" not in source


def test_execute_rejects_absent_authorization_before_any_output() -> None:
    result = subprocess.run(
        ["bash", str(WRAPPER), "--execute"],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        check=False,
    )
    assert result.returncode == 4
    assert "runtime authorization is not issued" in result.stderr


def test_wrapper_pins_contract_plan_teacher_policy_and_finalizer() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    for name in (
        "CPU_CONTRACT_SHA256",
        "PLAN_SHA256",
        "TEACHER_SHA256",
        "POLICY_SHA256",
        "PLAYBACK_SHA256",
        "FINALIZER_SHA256",
        "GAINS_SHA256",
    ):
        assert f'{name}="' in source
    assert 'AUTHORIZATION_SHA256=""' in source
    assert "timeout --signal=TERM --kill-after=30s 5400" in source
    assert "wait_gpu_free" in source
    assert 'HEAD" == "$UPSTREAM' in source


def test_future_token_is_consumed_before_isaac_and_next_stages_stay_closed() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    token_delete = source.index('rm -f "$AUTHORIZATION_FILE"')
    first_playback = source.index("timeout --signal=TERM --kill-after=30s 5400")
    assert token_delete < first_playback
    assert '[[ ! -L "$AUTHORIZATION_FILE" ]]' in source
    assert "stat -c '%a'" in source
    assert '"dataset_creation_authorized": False' in source
    assert '"case16_22_32_authorized": False' in source
    assert '"holdout_opened": False' in source
    assert '"ppo_authorized": False' in source
