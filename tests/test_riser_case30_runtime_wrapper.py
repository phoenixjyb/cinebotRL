from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/two_wheel_balance/run_riser_case30_perturbation_canary.sh"


def test_runtime_wrapper_is_one_case_measurement_only() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    assert "--cases 30" in source
    assert "--deterministic-wrench-profile" in source
    assert "--shadow-teacher-trace-dir" in source
    assert "--dataset-dir" not in source
    assert "--raw-teacher-dir" not in source
    assert "timeout --signal=TERM --kill-after=30s 600" in source
    assert source.index('rm -f "$AUTHORIZATION_FILE"') < source.index("$PLAYBACK")
    assert "summarize_riser_case30_perturbation_canary.py" in source
    assert "case_0030.json" in source
    assert (
        'OUTPUT_WIN="${WIN_ROOT}\\\\artifacts\\\\two_wheel_riser'
        '\\\\${NAMESPACE}"'
    ) in source
    assert r"\$NAMESPACE" not in source
