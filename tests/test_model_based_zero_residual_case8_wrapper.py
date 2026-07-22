import os
from pathlib import Path
import subprocess


WRAPPER = (
    Path(__file__).parents[1]
    / "scripts/two_wheel_balance/run_model_based_zero_residual_case8_canary.sh"
)


def test_wrapper_is_model_based_two_rollout_and_capture_closed() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    assert "CASE=8" in source
    assert "--policy-command-base model_based_planner" in source
    assert "--residual-action-scales 0.05,0.05,0.02" in source
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
    assert "one-use token is absent" in result.stderr


def test_wrapper_pins_inputs_and_consumes_future_token_before_isaac() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    for name in (
        "CPU_CONTRACT_SHA256",
        "CPU_ADMISSION_SHA256",
        "PLAN_SHA256",
        "TEACHER_SHA256",
        "POLICY_SHA256",
        "PLAYBACK_SHA256",
        "FINALIZER_SHA256",
        "GAINS_SHA256",
    ):
        assert f'{name}="' in source
    assert (
        'AUTHORIZATION_SHA256="af3cf7748bafb522acaa2827553d49a939771e135c61c367c42d492ecc5a96c0"'
        in source
    )
    token_delete = source.index('rm -f "$AUTHORIZATION_FILE"')
    first_playback = source.index("timeout --signal=TERM --kill-after=30s 600")
    assert token_delete < first_playback
    assert "wait_gpu_free" in source
    assert 'HEAD" == "$UPSTREAM' in source
    assert "stat -c '%a'" in source
    assert '[[ ! -L "$AUTHORIZATION_FILE" ]]' in source


def test_preflight_declares_learning_and_case78_closed() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    assert '"runtime_authorization_hash_issued": True' in source
    assert '"runtime_token_consumed": False' in source
    assert '"dataset_creation_authorized": False' in source
    assert '"case78_authorized": False' in source
    assert '"holdout_opened": False' in source
    assert '"ppo_authorized": False' in source
