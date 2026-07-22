from pathlib import Path


def test_shadow_label_preflight_has_no_runtime_path() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/two_wheel_balance/run_case78_shadow_label_preflight.sh"
    ).read_text(encoding="utf-8")
    assert '[[ "$MODE" != --execute ]]' in source
    assert "runtime_authorization_not_issued" in source
    assert "validate_case78_shadow_label_contract.py" in source
    assert "timeout " not in source
    assert "--shadow-teacher-trace-dir" not in source
    assert "RISER_CASE78_SHADOW_ACTION_SCALES" in source
    assert "assert_gpu_free" in source


def test_shadow_label_preflight_rejects_machine_overrides() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/two_wheel_balance/run_case78_shadow_label_preflight.sh"
    ).read_text(encoding="utf-8")
    for variable in (
        "RISER_ROOT",
        "RISER_WIN_ROOT",
        "ISAAC_PYTHON",
        "RISER_CASE78_SHADOW_AUTHORIZATION",
        "RISER_CASE78_SHADOW_NAMESPACE",
        "RISER_CASE78_SHADOW_CONTRACT",
        "RISER_CASE78_SHADOW_PLAN",
        "RISER_CASE78_SHADOW_OUTPUT",
        "RISER_CASE78_SHADOW_TIMEOUT",
        "RISER_CASE78_SHADOW_HEARTBEAT",
        "RISER_CASE78_SHADOW_ACTION_SCALES",
    ):
        assert variable in source
