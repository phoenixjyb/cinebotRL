import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/two_wheel_balance/summarize_riser_gate_c_canary.py"


def _gate(path: Path, case: int, passed: bool) -> None:
    path.write_text(
        json.dumps(
            {
                "passed": passed,
                "results": [
                    {
                        "case": case,
                        "passed": passed,
                        "source_duration_s": 1.0,
                        "execution_duration_s": 3.0,
                        "completed_steps": 10,
                        "classification": (
                            None
                            if passed
                            else "action_envelope_zero_clipping_rejection"
                        ),
                        "stage": "dynamic_gate",
                    }
                ],
            }
        )
    )


def test_summary_stops_at_first_reject_and_keeps_training_closed(tmp_path: Path) -> None:
    (tmp_path / "gates").mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "admission.json").write_text("{}")
    _gate(tmp_path / "gates/case_0001.json", 1, True)
    _gate(tmp_path / "gates/case_0002.json", 2, False)
    output = tmp_path / "summary.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(tmp_path),
            "--git-commit",
            "a" * 40,
            "--cases",
            "1,2,3",
            "--output",
            str(output),
        ],
        check=False,
    )
    summary = json.loads(output.read_text())
    assert result.returncode == 0
    assert summary["dynamically_passed_cases"] == [1]
    assert summary["first_dynamic_reject"]["case"] == 2
    assert summary["not_started_cases"] == [3]
    assert summary["source_execution_timing_separated"]
    assert not summary["residual_capture_started"]
    assert not summary["bc_started"]
    assert not summary["ppo_started"]
    assert not summary["passed"]
