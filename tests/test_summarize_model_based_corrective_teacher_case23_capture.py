import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/two_wheel_balance/summarize_model_based_corrective_teacher_case23_capture.py"
SPEC = importlib.util.spec_from_file_location("case23_capture_summary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_case23_finalizer_passes_exact_archive_contract(monkeypatch, tmp_path) -> None:
    observed = {}

    def fake_summarize(root, admission_path, **kwargs):
        observed.update(kwargs)
        return {"passed": True}

    monkeypatch.setattr(MODULE, "summarize_capture", fake_summarize)
    result = MODULE.summarize(
        tmp_path,
        tmp_path / "admission.json",
        runtime_commit="a" * 40,
        playback_exit_code=0,
        gpu_release_passed=True,
    )
    assert result["passed"] is True
    assert observed["expected_case"] == 23
    assert observed["expected_namespace"] == MODULE.NAMESPACE
    assert observed["capture_name"] == MODULE.CAPTURE_NAME
    assert observed["plan_identity_name"] == "case23_plan"
    assert observed["runtime_commit"] == "a" * 40
    assert observed["playback_exit_code"] == 0
    assert observed["gpu_release_passed"] is True
