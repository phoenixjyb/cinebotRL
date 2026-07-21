from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/two_wheel_balance/run_riser_case78_dynamic_canary.sh"


def test_runtime_wrapper_is_one_case_deterministic_only() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    assert "--cases 78" in source
    assert "timeout --signal=TERM --kill-after=30s 900" in source
    assert "--controller-wz-kp 1.05" in source
    assert "--maximum-duration-scale 3.0" in source
    assert "--enable-camera-lever-arm-compensation" in source
    assert "--residual-policy" not in source
    assert "--zero-policy-action" not in source
    assert "--dataset-dir" not in source
    assert "--raw-teacher-dir" not in source
    assert "--policy-trace-dir" not in source
    assert "--shadow-teacher-trace-dir" not in source
    assert source.index('rm -f "$AUTHORIZATION_FILE"') < source.index("$PLAYBACK")
    assert (
        'OUTPUT_WIN="${WIN_ROOT}\\\\artifacts\\\\two_wheel_riser'
        '\\\\${NAMESPACE}"'
    ) in source
    assert r"\$NAMESPACE" not in source
    assert "summarize_riser_case78_dynamic_canary.py" in source
