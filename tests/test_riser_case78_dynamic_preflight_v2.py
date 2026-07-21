from pathlib import Path


def test_v2_preflight_wrapper_has_no_runtime_path() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/two_wheel_balance/run_riser_case78_dynamic_preflight_v2.sh"
    ).read_text(encoding="utf-8")
    assert '[[ "$MODE" != --execute ]]' in source
    assert "runtime_authorization_not_issued" in source
    assert "runtime_heartbeat.json" not in source
    assert "smoke_riser_reference_playback.py" not in source
    assert "timeout " not in source

