import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/two_wheel_balance/summarize_riser_gate_c_canary.py"


def _gate(
    path: Path, case: int, passed: bool, *, label_envelope_passed: bool = True
) -> None:
    path.write_text(
        json.dumps(
            {
                "passed": passed,
                "dynamic_quality_passed": passed,
                "training_started": False,
                "ppo_authorized": False,
                "trajectory_command_source": "deterministic_teacher",
                "residual_policy": None,
                "controller_profile": "structural_robust_v1",
                "tracking_profile": "riser_recovery_direction_v4",
                "tracking_direction_recovery_error_range_m": [0.2, 0.4],
                "riser_thermal_force_contract": (
                    "leadshine_400w_first_order_monitor_v1"
                ),
                "cases": [case],
                "passed_case_count": 1 if passed else 0,
                "results": [
                    {
                        "case": case,
                        "passed": passed,
                        "source_duration_s": 1.0,
                        "execution_duration_s": 3.0,
                        "completed_steps": 10,
                        "dynamic_quality_passed": passed,
                        "residual_label_envelope_passed": label_envelope_passed,
                        "residual_label_admission_passed": (
                            passed and label_envelope_passed
                        ),
                        "riser_thermal_load_max": 0.4,
                        "riser_effort_max_n": 100.0,
                        "checks": {
                            "riser_thermal_force_observed": True,
                            "riser_thermal_load_bounded": True,
                            "riser_peak_force_bounded": True,
                        },
                        "executed_residual_dataset": None,
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
    assert summary["gate_rows"][0]["riser_thermal_load_max"] == 0.4


def test_dynamic_pass_is_independent_of_label_envelope(tmp_path: Path) -> None:
    (tmp_path / "gates").mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "admission.json").write_text("{}")
    _gate(
        tmp_path / "gates/case_0074.json",
        74,
        True,
        label_envelope_passed=False,
    )
    output = tmp_path / "summary.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(tmp_path),
            "--git-commit",
            "a" * 40,
            "--cases",
            "74",
            "--output",
            str(output),
        ],
        check=True,
    )
    summary = json.loads(output.read_text())
    assert summary["passed"]
    assert summary["dynamic_quality_passed"]
    assert not summary["residual_label_envelope_passed"]
    assert not summary["residual_label_admission_passed"]
    assert not summary["valid_for_training"]


def test_runtime_contract_rejects_wrong_tracking_profile(tmp_path: Path) -> None:
    (tmp_path / "gates").mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "admission.json").write_text("{}")
    gate = tmp_path / "gates/case_0074.json"
    _gate(gate, 74, True)
    payload = json.loads(gate.read_text())
    payload["tracking_profile"] = "riser_motion_direction_v3"
    gate.write_text(json.dumps(payload))
    output = tmp_path / "summary.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(tmp_path),
            "--git-commit",
            "a" * 40,
            "--cases",
            "74",
            "--output",
            str(output),
        ],
        check=True,
    )
    summary = json.loads(output.read_text())
    assert summary["dynamic_quality_passed"]
    assert not summary["runtime_contract_passed"]
    assert summary["first_dynamic_reject"]["classification"] == (
        "runtime_contract_rejection"
    )
    assert not summary["passed"]


def test_runtime_contract_rejects_missing_thermal_force_gate(tmp_path: Path) -> None:
    (tmp_path / "gates").mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "admission.json").write_text("{}")
    gate = tmp_path / "gates/case_0074.json"
    _gate(gate, 74, True)
    payload = json.loads(gate.read_text())
    payload["results"][0]["checks"]["riser_thermal_load_bounded"] = False
    gate.write_text(json.dumps(payload))
    output = tmp_path / "summary.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(tmp_path),
            "--git-commit",
            "a" * 40,
            "--cases",
            "74",
            "--output",
            str(output),
        ],
        check=True,
    )
    summary = json.loads(output.read_text())
    assert summary["dynamic_quality_passed"]
    assert not summary["runtime_contract_passed"]
    assert summary["first_dynamic_reject"]["classification"] == (
        "runtime_contract_rejection"
    )
