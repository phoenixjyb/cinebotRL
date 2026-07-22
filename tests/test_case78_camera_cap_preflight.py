from pathlib import Path


def test_camera_cap_preflight_has_no_runtime_path() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/two_wheel_balance/run_case78_camera_cap_preflight.sh"
    ).read_text(encoding="utf-8")
    assert '[[ "$MODE" != --execute ]]' in source
    assert "runtime_authorization_not_issued" in source
    assert "smoke_riser_reference_playback.py" not in source
    assert "timeout " not in source
    assert "RISER_CASE78_CAMERA_CAP" in source
