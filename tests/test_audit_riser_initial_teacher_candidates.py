import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/two_wheel_balance/audit_riser_initial_teacher_candidates.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(root: Path, *, bad_plan_hash: bool = False) -> tuple[Path, Path]:
    portfolio = root / "portfolio"
    evidence = root / "evidence"
    portfolio.mkdir()
    evidence.mkdir()
    items = []
    source_hash = "a" * 64
    for case in (1, 2):
        plan = portfolio / f"case_{case:04d}_smoothed_riser_plan_v1.npz"
        np.savez_compressed(plan, case=np.array(case))
        items.append(
            {
                "case": case,
                "file": plan.name,
                "plan_sha256": _sha256(plan),
                "passed": True,
            }
        )
        run = evidence / f"run_{case}"
        (run / "gates").mkdir(parents=True)
        gate = run / "gates" / f"case_{case:04d}.json"
        result = {
            "case": case,
            "dynamic_quality_passed": True,
            "thermal_admission_passed": True,
            "controller_evidence_passed": True,
            "termination": None,
            "executed_residual_dataset": None,
            "raw_residual_label_applied_to_commands": False,
            "residual_action_abs_max": [0.0, 0.0, 0.0],
            "raw_residual_command_abs_max": [0.31 if case == 2 else 0.2, 0.1, 0.01],
        }
        gate.write_text(
            json.dumps(
                {
                    "dynamic_quality_passed": True,
                    "thermal_admission_passed": True,
                    "controller_evidence_passed": True,
                    "results": [result],
                }
            ),
            encoding="utf-8",
        )
        admission = run / "admission.json"
        admission.write_text(
            json.dumps(
                {
                    "passed": True,
                    "requested_cases": [case],
                    "source_manifest_sha256": source_hash,
                    "selected_plan": {
                        "case": case,
                        "plan_sha256": (
                            "b" * 64 if bad_plan_hash and case == 2 else _sha256(plan)
                        ),
                    },
                    "runtime_commit": "c" * 40,
                }
            ),
            encoding="utf-8",
        )
        summary = {
            "dynamic_quality_passed": True,
            "thermal_admission_passed": True,
            "runtime_contract_passed": True,
            "source_execution_timing_separated": True,
            "admission_sha256": _sha256(admission),
            "residual_capture_started": False,
            "bc_started": False,
            "ppo_started": False,
            "thresholds_relaxed": False,
            "actions_clipped": False,
            "gate_rows": [
                {
                    "case": case,
                    "gate": str(gate),
                    "gate_sha256": _sha256(gate),
                    "passed": True,
                    "dynamic_quality_passed": True,
                    "thermal_admission_passed": True,
                    "runtime_contract_passed": True,
                    "controller_profile": "structural_robust_v1",
                    "tracking_profile": "test",
                    "source_duration_s": 1.0,
                    "execution_duration_s": 1.5,
                    "completed_steps": 10,
                    "residual_label_envelope_passed": case == 1,
                }
            ],
        }
        (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    manifest = portfolio / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "cinebotrl_two_wheel_riser_smoothed_plan_export_v1",
                "source_manifest_sha256": source_hash,
                "items": items,
            }
        ),
        encoding="utf-8",
    )
    return manifest, evidence


def _run(tmp_path: Path, *, bad_plan_hash: bool = False) -> tuple[subprocess.CompletedProcess, dict]:
    manifest, evidence = _write_fixture(tmp_path, bad_plan_hash=bad_plan_hash)
    output = tmp_path / "selection.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--portfolio-manifest",
            str(manifest),
            "--evidence-root",
            str(evidence),
            "--cases",
            "1,2",
            "--minimum-teacher-cases",
            "2",
            "--minimum-train-cases",
            "0",
            "--minimum-validation-cases",
            "1",
            "--minimum-holdout-cases",
            "1",
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result, json.loads(output.read_text(encoding="utf-8"))


def test_selects_physical_pass_despite_legacy_label_overflow(tmp_path: Path) -> None:
    result, payload = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert payload["passed"]
    assert payload["selected_cases"] == [1, 2]
    assert payload["legacy_residual_label_envelope_passed_count"] == 1
    assert payload["legacy_residual_label_envelope_rejected_cases"] == [2]
    assert payload["raw_residual_command_abs_max"] == [0.31, 0.1, 0.01]
    assert payload["selection_count_met"]
    assert payload["fresh_homogeneous_capture_required"]
    assert not payload["capture_gate_passed"]
    assert not payload["bc_authorized"]
    assert not payload["ppo_authorized"]
    assert not payload["valid_for_training"]


def test_rejects_evidence_for_a_different_plan_hash(tmp_path: Path) -> None:
    result, payload = _run(tmp_path, bad_plan_hash=True)
    assert result.returncode == 2
    assert payload["selected_cases"] == [1]
    assert payload["missing_cases"] == [2]
    assert not payload["selection_count_met"]
    assert not payload["passed"]
